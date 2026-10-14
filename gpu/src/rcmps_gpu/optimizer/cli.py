"""Compile/run the fastest FP64 Z2 implementation at a user-supplied D.

Without --run-gpu this command only describes/checks sources and input bytes.
--emit-cuda generates the dimension-specialized trajectory without CUDA.
No rental, SSH, network or CPU objective/optimizer operations are implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generic_gpu import build_runtime
from generic_gpu.layout import describe

HERE = Path(__file__).resolve().parent
DEFAULTS = dict(window=30., quadrature=512, nodes=512, atol=1e-13, rtol=1e-12,
                maxdim=1024, metric_reg=1e-3, krylov_tolerance=1e-12,
                stationary_tolerance=1e-13, max_steps=1_000_000,
                line_search_energy_atol=1e-14)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_ledger(directory, D, coupling):
    directory = Path(directory).resolve()
    metadata = tomllib.loads((directory / "metadata.toml").read_text())
    if metadata["D"] != D or float(metadata["coupling"]) != coupling:
        raise ValueError("initial state D/coupling differs from the requested model")
    names = ("R.bin", "K.bin", "rho.bin", "saved_gradient.bin", "metadata.toml")
    for name in names[:-1]:
        if (directory / name).stat().st_size != 16 * D * D:
            raise ValueError("wrong complex128 matrix byte count: " + name)
    return dict(path=str(directory), sha256={name: digest(directory / name) for name in names})


def prepare(args, manifest):
    previous = None
    if args.resume_from:
        previous = json.loads(args.resume_from.read_bytes())
        if not previous.get("stop_reason"):
            raise ValueError("resume requires a stopped checkpoint, not a live run")
        if previous["reference"]["D"] != args.D:
            raise ValueError("cannot resume an optimizer at a different D")
        expected = previous["runs"][-1]["source_sha256"]
        if not expected or any(manifest["runtime_source_sha256"].get(n) != h for n, h in expected.items()):
            raise ValueError("resume requires this exact generic runtime; sealed historical runs remain unchanged")
    controls = {}
    for name, default in DEFAULTS.items():
        value = getattr(args, name)
        prior = previous["controls"][name] if previous else default
        if previous and value is not None and value != prior:
            raise ValueError("resume cannot change objective controls: " + name)
        controls[name] = prior if value is None else value
    for name, value in controls.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("control must be positive and finite: " + name)
    if not 0 < controls["window"] <= 40 or controls["quadrature"] < 8:
        raise ValueError("the source expansion requires window <= 40 and quadrature >= 8")
    if controls["rtol"] < 100 * sys.float_info.epsilon or controls["krylov_tolerance"] > 1e-12:
        raise ValueError("unsupported ODE precision or relaxed kinetic tolerance")
    if (args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)
            or args.max_iterations < 1 or args.device < 0 or args.seed < 0
            or not math.isfinite(args.gradient_target) or args.gradient_target <= 0
            or not math.isfinite(args.max_device_seconds) or args.max_device_seconds <= 0):
        raise ValueError("invalid optimizer, seed, device or per-trajectory timeout")
    coupling = (float(previous["reference"]["coupling"]) if previous else 2.7) if args.coupling is None else args.coupling
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError("phi4 coupling must be finite and nonnegative")
    initial = args.initial.resolve() if args.initial else None
    if previous:
        prior = Path(previous["controls"]["reference"]).resolve()
        if initial and initial != prior:
            raise ValueError("resume initial reference differs from the saved checkpoint")
        initial = prior
    elif initial and args.coupling is None:
        coupling = float(tomllib.loads((initial / "metadata.toml").read_text())["coupling"])
        if not math.isfinite(coupling) or coupling < 0:
            raise ValueError("invalid initial coupling")
    source_files = ("run.py", "worker.py", "build_runtime.py", "layout.py")
    plan = dict(schema_version=1, D=args.D, coupling=coupling, seed=args.seed,
                runtime=str(args.runtime.resolve()), runtime_manifest_sha256=digest(args.runtime / "manifest.json"),
                launcher_source_sha256={n: digest(HERE / n) for n in source_files},
                layout=describe(args.D, controls["quadrature"]), controls=controls,
                max_seconds=args.max_seconds, max_iterations=args.max_iterations,
                gradient_target=args.gradient_target, max_device_seconds=args.max_device_seconds,
                device=args.device, compile_only=args.compile_only,
                output=str(args.output.resolve()) if args.output else None,
                resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                resume_sha256={str(p.resolve()): digest(p) for p in (args.resume_from, args.resume_from.with_suffix(".npz"))}
                    if previous else {},
                initial=input_ledger(initial, args.D, coupling) if initial else None,
                optimization_wall_time_limit_is_explicit=args.max_seconds is not None,
                actual_GPU_qualified=False, CPU_objective_or_optimizer=False)
    return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--D", type=int, required=True, help="full even Z2 bond dimension")
    p.add_argument("--coupling", type=float)
    p.add_argument("--seed", type=int, default=0, help="NumPy PCG64 Z2 initialization, when --initial is absent")
    p.add_argument("--initial", type=Path, help="existing portable initial export")
    p.add_argument("--resume-from", type=Path, help="stopped generic JSON checkpoint; keep L-BFGS history")
    p.add_argument("--output", type=Path, help="fresh run directory")
    p.add_argument("--runtime", type=Path, default=build_runtime.TARGET)
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--compile-only", action="store_true", help="compile/resource check without a numerical launch")
    p.add_argument("--emit-cuda", type=Path, help="write generated CUDA on the CPU, without --run-gpu")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-iterations", type=int, default=20000, help="additional iterations for a resumed invocation")
    p.add_argument("--max-seconds", type=float, help="optional invocation limit; no wall-time cap by default")
    p.add_argument("--gradient-target", type=float, default=1e-7)
    p.add_argument("--max-device-seconds", type=float, default=300., help="one trajectory's device watchdog, not run wall time")
    for name in DEFAULTS:
        p.add_argument("--" + name.replace("_", "-"), type=int if isinstance(DEFAULTS[name], int) else float)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        describe(args.D)
        if args.emit_cuda and args.run_gpu:
            raise ValueError("--emit-cuda is a CPU-only action")
        if args.run_gpu and not args.output:
            raise ValueError("--run-gpu requires a fresh --output directory")
        if args.output and args.output.exists():
            raise ValueError("output already exists; choose a fresh run directory")
        if args.emit_cuda and args.emit_cuda.exists():
            raise ValueError("CUDA output already exists; choose a new file")
        manifest = build_runtime.build(args.runtime)
        plan = prepare(args, manifest)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if args.emit_cuda:
        # The source-generator imports NumPy/SciPy/Triton, but never CuPy/CUDA.
        code = "import sys;sys.path.insert(0,sys.argv[1]);from persistent_trajectory_backend import trajectory_source;print(trajectory_source(int(sys.argv[2])),end='')"
        result = subprocess.run([sys.executable, "-c", code, str(args.runtime.resolve()), str(args.D // 2)],
                                check=True, capture_output=True, text=True)
        args.emit_cuda.parent.mkdir(parents=True, exist_ok=True)
        with args.emit_cuda.open("x") as stream:
            stream.write(result.stdout)
        plan["emitted_cuda"] = str(args.emit_cuda.resolve())
        plan["emitted_cuda_sha256"] = digest(args.emit_cuda)
    if not args.run_gpu:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(HERE / "worker.py"), str((args.output / "plan.json").resolve())],
                          env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
