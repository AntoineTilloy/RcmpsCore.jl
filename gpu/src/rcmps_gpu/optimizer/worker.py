"""Fresh-interpreter generic GPU execution; invoked only by explicit --run-gpu."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys

from .. import __version__
from ..io import Integration, StateRecord, read_portable, write_portable
from .cli import RUNTIME, digest, launcher_source_sha256, runtime_source_sha256, verify_runtime

PRODUCER = f"rcmps_gpu {__version__}, optimizer worker"


def compile_dimension(D, cp, directory=None):
    from .runtime.persistent_trajectory_backend import trajectory_source, geometry, trajectory_ptx_report
    from .runtime.generic_layout import compilation_options, configure_shared_memory
    info = geometry(D // 2)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    kernel = cp.RawKernel(trajectory_source(D // 2), "persistent_trajectory",
                          options=compilation_options(D // 2), enable_cooperative_groups=True)
    kernel.compile()
    dynamic = configure_shared_memory(kernel, props, info)
    resident = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        kernel.kernel.ptr, info["threads"], dynamic))
    capacity = resident * int(props["multiProcessorCount"])
    if info["blocks"] > capacity:
        raise RuntimeError(f"D={D} requires {info['blocks']} resident blocks; this GPU can hold {capacity}. "
                           "The current one-kernel row layout cannot run safely at this size.")
    name = props["name"]
    ptx = trajectory_ptx_report(D // 2, cp, directory)
    return dict(**info, GPU=name.decode() if isinstance(name, bytes) else name,
                SM=[int(props["major"]), int(props["minor"])],
                registers=int(kernel.num_regs), local_bytes=int(kernel.local_size_bytes),
                dynamic_shared_bytes=dynamic, active_blocks_per_SM=resident,
                resident_capacity=capacity, full_grid_residency_verified=True,
                compiled=True, PTX=ptx, numerical_launch_performed=False, actual_GPU_qualified=False)


def write_reference(directory, record, plan, **extra):
    """Write the run's initial state, the reference directory of the runtime: a portable
    state, format 1 (rcmps_gpu.io), plus what runtime/sweep_driver.py reads besides R, K and
    rho: saved_gradient.bin (zero without a gradient), saved_energy, saved_gradient_norm and
    their saved_*_available flags, and precond_reg (from [manifold], which starts the
    optimizer's regularization; 1e-4 without one). An energy, gradient or gradient norm
    carried by the initial state are thus those the first evaluation is compared with:
    unless it reproduces the energy to 1e-8 and the gradient to 3e-6 (metric norm), the
    optimizer is not started, as for the research exports of Julia states. A state without
    energy or gradient gets the run's integration settings as [integration]."""
    import numpy as np
    controls = plan["controls"]
    if record.integration is None and record.energy is None and record.gradient is None:
        record = replace(record, integration=Integration(
            controls["quadrature"], controls["window"], controls["atol"], controls["rtol"]))
    runtime = dict(saved_energy=0.0 if record.energy is None else record.energy,
                   saved_energy_available=record.energy is not None,
                   saved_gradient_available=record.gradient is not None,
                   saved_gradient_norm=0.0 if record.gradient_norm is None else record.gradient_norm,
                   saved_gradient_norm_available=record.gradient_norm is not None)
    if record.manifold is not None:
        runtime["precond_reg"] = record.manifold.precond_reg
    gradient = np.zeros_like(record.R) if record.gradient is None else record.gradient
    write_portable(directory, record, extra={**extra, **runtime}, extra_matrices=dict(saved_gradient=gradient))


def random_reference(plan, directory):
    import numpy as np
    from .runtime.numerics import stationary_state
    d, n = plan["D"], plan["D"] // 2
    rng = np.random.default_rng(plan["seed"])
    def draw():
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2)
    k1, k2 = draw(), draw()
    zero = np.zeros((n, n), dtype=np.complex128)
    r = np.block([[zero, draw()], [draw(), zero]]) / math.sqrt(d)
    k = np.block([[k1 + k1.conj().T, zero], [zero, k2 + k2.conj().T]]) / math.sqrt(d)
    q = -1j * k - r.conj().T @ r / 2
    rho, stats = stationary_state(q, r, tolerance=plan["controls"]["stationary_tolerance"])
    write_reference(directory, StateRecord(D=d, model="phi4", coupling=plan["coupling"], R=r, K=k, rho=rho,
                                           producer=PRODUCER, source=f"random Z2 state, seed {plan['seed']}"),
                    plan, seed=plan["seed"], symmetry="Z2",
                    random_generator="NumPy PCG64; complex standard normal; scaled by 1/sqrt(D)")
    return stats


def initial_reference(plan, directory):
    """The --initial state (any dialect of rcmps_gpu.io.read_portable) as the run's initial/;
    rho is solved for if the directory has none (the statistics are returned, else None)."""
    from .runtime.numerics import stationary_state
    path = Path(plan["initial"]["path"])
    record = read_portable(path)
    if set(record.files) != set(plan["initial"]["sha256"]):
        raise ValueError("initial state changed after planning")
    stats = None
    if record.rho is None:
        q = -1j * record.K - record.R.conj().T @ record.R / 2
        rho, stats = stationary_state(q, record.R, tolerance=plan["controls"]["stationary_tolerance"])
        record = replace(record, rho=rho)
    write_reference(directory, replace(record, producer=PRODUCER, source=record.source or path.name), plan)
    return stats


def qualify(r, k, rho, coupling, args, cp, timeout, directory):
    """Compare complete fast E/g with independent CuPy propagation and kinetic."""
    import numpy as np
    from .runtime import overlap_sweep_driver as driver
    from .runtime import persistent_trajectory_overlap as fast
    from .runtime.ode_backend import evaluate_quartic
    from .runtime.numerics import metric
    from .runtime.warm_sweep_driver import binding
    class CuPyQuartic:
        def __init__(self, xp, inputs, coupling, settings):
            self.xp, self.inputs, self.coupling, self.settings = xp, inputs, coupling, settings
            self.setup_seconds = 0.
        def __call__(self):
            q, r, rho = self.inputs
            s = self.settings
            return evaluate_quartic(r, q, rho, self.coupling, xp=self.xp,
                half_window=s.window, n_quad=s.quadrature, atol=s.atol, rtol=s.rtol,
                max_steps=s.max_steps, packed_z2=True, use_graph=False)
        def close(self):
            pass
    receipt = dict(D=len(r), passed=False, numerical_GPU_launch_performed=True)
    arrays = {}
    try:
        with driver.OverlapObjective() as coordinator:
            actual = fast.evaluate_complete(coordinator, r, k, rho, coupling, args, cp,
                mode="trajectory", kinetic="cooperative", max_device_seconds=timeout)
            receipt.update(fast_energy=actual["value"], fast_stats=actual["stats"])
            for name in ("gradient", "covector", "environment"):
                arrays["fast_" + name] = actual[name]
            with binding(fast, "ORIGINAL_PREPARED", CuPyQuartic):
                oracle = fast.evaluate_complete(coordinator, r, k, rho, coupling, args, cp,
                    mode="cached", kinetic="baseline", max_device_seconds=timeout)
            receipt.update(oracle_energy=oracle["value"], oracle_stats=oracle["stats"])
            for name in ("gradient", "covector", "environment"):
                arrays["oracle_" + name] = oracle[name]
        difference = actual["gradient"] - oracle["gradient"]
        receipt.update(energy_difference=abs(actual["value"] - oracle["value"]),
            metric_gradient_difference=math.sqrt(max(0., metric(rho, difference, difference, args.metric_reg))),
            raw_comparison=fast.compare(oracle, actual, rho, args.metric_reg),
            fast_stats=actual["stats"], oracle_stats=oracle["stats"])
        receipt["passed"] = bool(fast.passed(receipt["raw_comparison"]) and
            all(result["stats"]["kinetic"]["maximum_relative_residual"] <= args.krylov_tolerance and
                result["stats"]["kinetic"]["fallback_solves"] == 0 for result in (actual, oracle)))
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (directory / "initial_qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if arrays:
            np.savez(directory / "initial_qualification.npz", R=r, K=k, rho=rho, **arrays)
    if not receipt["passed"]:
        raise RuntimeError("initial complete E/g qualification failed; optimizer was not launched")
    return receipt


def main(argv=None):
    import numpy as np
    plan_path = Path((argv or sys.argv[1:])[0])
    plan = json.loads(plan_path.read_bytes())
    output = Path(plan["output"])
    if plan["schema_version"] != 3 or Path(plan["runtime"]) != RUNTIME:
        raise ValueError(f"plan written by another launcher version or installation: schema "
                         f"{plan['schema_version']}, runtime {plan['runtime']}; this worker is schema 3, runtime {RUNTIME}")
    if runtime_source_sha256() != plan["runtime_source_sha256"]:
        raise ValueError("runtime changed after planning")
    launcher = launcher_source_sha256()
    for name in sorted(set(launcher) | set(plan["launcher_source_sha256"])):
        if launcher.get(name) != plan["launcher_source_sha256"].get(name):
            raise ValueError("launcher source changed after planning: " + name)
    if verify_runtime() != plan["runtime_provenance"]:
        raise ValueError("runtime provenance changed after planning")
    for path, expected in plan["resume_sha256"].items():
        if digest(path) != expected:
            raise ValueError("resume checkpoint changed after planning")
    if plan["initial"]:
        for name, expected in plan["initial"]["sha256"].items():
            if digest(Path(plan["initial"]["path"]) / name) != expected:
                raise ValueError("initial state changed after planning")
    from .runtime import trajectory_sweep_driver as trajectory
    from .runtime import numerics
    from .runtime.warm_sweep_driver import binding
    # Numerical imports must all come from this package's runtime directory, not old experiments.
    for name in plan["runtime_source_sha256"]:
        module = sys.modules.get(f"{__package__}.runtime.{Path(name).stem}")
        if module is not None and Path(module.__file__).resolve().parent != RUNTIME:
            raise ValueError("mixed numerical runtimes: " + name)
    snapshot = trajectory.verify_checkpoint(plan["resume_from"]) if plan["resume_from"] else None
    status = dict(state="preflight", converged=False, final_consistency_passed=False)
    def save():
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    try:
        import cupy as cp
        with cp.cuda.Device(plan["device"]):
            status["compile"] = compile_dimension(plan["D"], cp, output / "ptx")
            (output / "compile.json").write_text(json.dumps(status["compile"], indent=2) + "\n")
            if plan["compile_only"]:
                status["state"] = "compiled_without_numerical_launch"
                save()
                return 0
            # A resumed run keeps the reference of its checkpoint; a new run writes initial/.
            if plan["resume_from"]:
                reference = Path(plan["initial"]["path"])
            else:
                reference = output / "initial"
                status["initial_stationary_solve"] = (initial_reference if plan["initial"] else random_reference)(
                    plan, reference)
            matrices, metadata = numerics.load_reference(reference)
            state = snapshot["state"] if snapshot else matrices
            r, k, rho = (state[n] for n in ("R", "K", "rho"))
            if any(v.shape != (plan["D"], plan["D"]) or v.dtype != np.complex128 or
                   not np.isfinite(v).all() for v in (r, k, rho)):
                raise ValueError("initial state requires finite complex128 matrices with the requested D")
            options = [str(reference), "--backend", "cupy", "--mode", "hybrid", "--packed-z2",
                "--quartic-backend", "persistent_trajectory", "--arnoldi", "gpu", "--projection", "hessenberg_gpu",
                "--kinetic", "cooperative", "--controller", "robust", "--warm-start-line-search",
                "--gradient-tolerance", str(plan["gradient_target"]), "--max-device-seconds", str(plan["max_device_seconds"]),
                "--iterations", str(plan["max_iterations"] + (int(snapshot["state"]["iteration"]) if snapshot else 0)),
                "--output", str(output / "optimization.json")]
            for name, value in plan["controls"].items():
                options.extend(["--" + name.replace("_", "-"), str(value)])
            if plan["resume_from"]:
                options.extend(["--resume-from", plan["resume_from"]])
            original_parse = trajectory.ORIGINAL_PARSE
            def parse(arguments=None):
                args = original_parse(arguments)
                args.max_seconds = plan["max_seconds"]
                return args
            base = parse([str(reference), "--backend", "cupy", "--mode", "hybrid", "--packed-z2",
                "--quartic-backend", "persistent_trajectory", "--arnoldi", "gpu", "--projection", "hessenberg_gpu"])
            for name, value in plan["controls"].items():
                setattr(base, name, value)
            status["state"] = "initial_qualification"
            save()
            qualify(r, k, rho, metadata["coupling"], base, cp, plan["max_device_seconds"], output)
            status["state"] = "optimizing"
            save()
            with binding(trajectory, "ORIGINAL_PARSE", parse):
                trajectory.main(options)
            stopped = json.loads((output / "optimization.json").read_bytes())
            status.update(state="stopped", converged=bool(stopped["converged"]),
                stop_reason=stopped["stop_reason"], iteration=len(stopped["history"]),
                gradient_norm=stopped["final_gradient_norm"], energy=stopped["final_energy"])
            save()
            from . import validate_trajectory_sweep
            validate_trajectory_sweep.main([str(output / "optimization.json"), str(reference),
                                           "--output", str(output / "final_validation.json")])
            status.update(state="completed", final_consistency_passed=True)
            save()
            if runtime_source_sha256() != plan["runtime_source_sha256"] or verify_runtime() != plan["runtime_provenance"]:
                raise ValueError("runtime changed during the run")
            print(json.dumps(status), flush=True)
            return 0
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
