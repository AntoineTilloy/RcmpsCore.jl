"""Fresh-interpreter generic GPU execution; invoked only by explicit --run-gpu."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from .cli import RUNTIME, digest, runtime_source_sha256


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
    directory.mkdir(exist_ok=False)
    for name, value in (("R", r), ("K", k), ("Q", q), ("rho", rho),
                        ("saved_gradient", np.zeros_like(r))):
        value.ravel(order="F").astype("<c16").tofile(directory / (name + ".bin"))
    (directory / "metadata.toml").write_text(
        f'D = {d}\ncoupling = {plan["coupling"]!r}\nseed = {plan["seed"]}\n'
        'symmetry = "Z2"\nrandom_generator = "NumPy PCG64; complex standard normal"\n'
        'saved_energy = 0.0\nsaved_gradient_norm = 0.0\n'
        'saved_energy_available = false\nsaved_gradient_available = false\n'
        'saved_gradient_norm_available = false\nCPU_objective_evaluated = false\nCPU_optimizer_run = false\n')
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
    if plan["schema_version"] != 2 or Path(plan["runtime"]) != RUNTIME:
        raise ValueError("plan written by another launcher version or installation")
    if runtime_source_sha256() != plan["runtime_source_sha256"]:
        raise ValueError("runtime changed after planning")
    for name, expected in plan["launcher_source_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("launcher source changed after planning: " + name)
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
            reference = Path(plan["initial"]["path"]) if plan["initial"] else output / "initial"
            if not plan["initial"]:
                status["initial_stationary_solve"] = random_reference(plan, reference)
            matrices, metadata = numerics.load_reference(reference)
            state = snapshot["state"] if snapshot else matrices
            r, k, rho = (state[n] for n in ("R", "K", "rho"))
            if any(v.shape != (plan["D"], plan["D"]) or v.dtype != np.complex128 or
                   not np.isfinite(v).all() for v in (r, k, rho)):
                raise ValueError("initial state requires finite complex128 matrices with the requested D")
            options = [str(reference), "--backend", "cupy", "--mode", "hybrid", "--packed-z2",
                "--quartic-backend", "persistent_rows_general", "--arnoldi", "gpu", "--projection", "hessenberg_gpu",
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
                "--quartic-backend", "persistent_rows_general", "--arnoldi", "gpu", "--projection", "hessenberg_gpu"])
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
            if runtime_source_sha256() != plan["runtime_source_sha256"]:
                raise ValueError("runtime changed during the run")
            print(json.dumps(status), flush=True)
            return 0
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
