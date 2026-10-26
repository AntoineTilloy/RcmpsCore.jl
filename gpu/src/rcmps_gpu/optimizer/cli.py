"""Compile/run the fastest FP64 implementation at a user-supplied D, for Z2 states (packed,
the default) or states without symmetry (--symmetry none: the unpacked channels).

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
import site
import subprocess
import sys

from .. import __version__
from .. import provenance
from ..io import read_portable
from .layout import SYMMETRIES, TILE_CHOICES, describe

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
RUNTIME = HERE / "runtime"
# Every other file that runs with the runtime, relative to the rcmps_gpu package; plan.json
# records their sha256 and the worker requires them unchanged.
LAUNCHER_SOURCES = ("__init__.py", "io.py", "provenance.py", "optimizer/__init__.py", "optimizer/__main__.py",
                    "optimizer/cli.py", "optimizer/worker.py", "optimizer/layout.py",
                    "optimizer/validate_sweep.py", "optimizer/validate_trajectory_sweep.py",
                    "optimizer/provenance.json", "optimizer/runtime/__init__.py")
# sha256 of runtime/manifest.json as imported, i.e. of json.dumps(manifest, indent=2,
# sort_keys=True) + newline for the manifest held by provenance.json.
IMPORTED_MANIFEST_SHA256 = "dda1fdbfa9f435f04958e418379f269b2381cc3e58d9b09bb81c7f42bdd6beea"
# Modules that run on every import of a runtime module: only a docstring and constants.
INERT_SOURCES = ("__init__.py", "optimizer/__init__.py", "optimizer/runtime/__init__.py")
DEFAULTS = dict(window=30., quadrature=512, nodes=512, atol=1e-13, rtol=1e-12,
                maxdim=1024, metric_reg=1e-3, krylov_tolerance=1e-12,
                stationary_tolerance=1e-13, max_steps=1_000_000,
                line_search_energy_atol=1e-14)
# The controller's line searches (runtime/robust_optimizer.py LINE_SEARCHES): the bounded Armijo
# backtracking, or RcmpsCore's Hager-Zhang search with energy tolerance WOLFE_ENERGY_ATOL.
LINE_SEARCHES = ("armijo", "wolfe")
WOLFE_ENERGY_ATOL = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_source_sha256():
    """sha256 of every numerical runtime module, keyed by file name as in the checkpoints."""
    return {path.name: digest(path) for path in sorted(RUNTIME.glob("*.py")) if path.name != "__init__.py"}


def launcher_source_sha256():
    return {name: digest(PACKAGE / name) for name in LAUNCHER_SOURCES}


def verify_runtime():
    """Refuse (ValueError) a runtime that is not the imported one with the edits recorded in
    provenance.json, as the research launcher refused one that was not the derivation of the
    sealed H100 candidate; return what plan.json records of the check."""
    raw = (HERE / "provenance.json").read_bytes()
    record = json.loads(raw)
    manifest = record["manifest"]
    if (provenance.sha256((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()) != IMPORTED_MANIFEST_SHA256
            or record["imported"]["manifest_sha256"] != IMPORTED_MANIFEST_SHA256):
        raise ValueError("provenance.json: the imported runtime manifest changed")
    names = runtime_source_sha256()
    try:
        edited = provenance.verify(RUNTIME, manifest["runtime_source_sha256"], record["edits"], names)
    except ValueError as error:
        raise ValueError(f"runtime differs from its recorded provenance: {error}") from None
    for name in INERT_SOURCES:
        if not provenance.inert(PACKAGE / name):
            raise ValueError(f"rcmps_gpu/{name} must hold only a docstring and constants")
    return dict(imported_manifest_sha256=IMPORTED_MANIFEST_SHA256, provenance_sha256=provenance.sha256(raw),
                edits=[entry["commit_subject"] for entry in record["edits"]], edited_files=edited,
                imported_bytes_with_recorded_edits=True)


def child_process(*arguments):
    """Command and environment of a child interpreter that imports this copy of rcmps_gpu.

    -P keeps the working directory off the child's sys.path, where another copy could shadow
    this one; the directory that holds this copy goes first on PYTHONPATH, unless it is one
    of the interpreter's site directories, which the child searches as the parent does."""
    root = str(PACKAGE.parent)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    sites = {str(Path(path).resolve()) for path in (*site.getsitepackages(), site.getusersitepackages())}
    if root not in sites:
        environment["PYTHONPATH"] = os.pathsep.join([root, *filter(None, [os.environ.get("PYTHONPATH")])])
    return [sys.executable, "-P", *arguments], environment


def package_identity():
    """Version, and the git commit when these files are tracked in a checkout (else None)."""
    def git(*arguments):
        return subprocess.run(("git",) + arguments, cwd=HERE, capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
    try:
        git("ls-files", "--error-unmatch", "cli.py")
        commit = git("rev-parse", "HEAD") or None
        modified = bool(git("status", "--porcelain", "--untracked-files=no", "--", ".."))
    except (OSError, subprocess.SubprocessError):
        commit = modified = None
    return dict(name="rcmps-gpu", version=__version__, git_commit=commit, git_tracked_files_modified=modified)


def z2_blocks(R, K):
    """Whether R is odd and K even, exactly, under the parity with the first D/2 basis states
    even: the structure that the packed Z2 channels require (no projection is made)."""
    D = len(R)
    if D % 2:
        return False
    d = D // 2
    return not any(block.any() for block in (R[:d, :d], R[d:, d:], K[:d, d:], K[d:, :d]))


def input_ledger(directory, D, coupling, symmetry=None):
    """Check an initial state directory, any dialect of rcmps_gpu.io.read_portable (format 1,
    as initial/ and optimized_state/ of a run, or a research export), and hash the files the
    worker will read from it. With symmetry z2 (a new run) the matrices must have the exact
    block structure of the packed channels; a resume (symmetry None) keeps its checkpoint's."""
    directory = Path(directory).resolve()
    record = read_portable(directory, arrays=False)
    if record.D != D or record.coupling != coupling:
        raise ValueError("initial state D/coupling differs from the requested model")
    if record.model != "phi4":
        raise ValueError(f"the optimizer implements the phi4 model, not {record.model!r} (initial state)")
    record = read_portable(directory)   # sizes and finiteness of the matrices
    if symmetry == "z2" and not z2_blocks(record.R, record.K):
        stated = f" (its metadata states symmetry = {record.symmetry!r})" if record.symmetry else ""
        raise ValueError(f"the initial state is not Z2 symmetric in the packed basis (R odd, K even, the first "
                         f"D/2 states even){stated}: use --symmetry none")
    return dict(path=str(directory), sha256={name: digest(directory / name) for name in record.files})


def prepare(args, runtime_sha256):
    previous = None
    if args.resume_from:
        previous = json.loads(args.resume_from.read_bytes())
        if not previous.get("stop_reason"):
            raise ValueError("resume requires a stopped checkpoint, not a live run")
        if previous["reference"]["D"] != args.D:
            raise ValueError("cannot resume an optimizer at a different D")
        if bool(previous["controls"].get("reduced")) != (args.symmetry == "none"):
            raise ValueError("cannot resume an optimizer with another --symmetry (the checkpoint's is "
                             + ("none" if previous["controls"].get("reduced") else "z2") + ")")
        expected = previous["runs"][-1]["source_sha256"]
        if not expected or any(runtime_sha256.get(n) != h for n, h in expected.items()):
            raise ValueError("resume requires this exact generic runtime; sealed historical runs remain unchanged")
    # A checkpoint that records no line search ran the Armijo backtracking.
    prior_search = (previous["controls"].get("robust_controls") or {}) if previous else {}
    line_search = prior_search.get("line_search", "armijo") if previous else args.line_search or "armijo"
    if args.line_search not in (None, line_search):
        raise ValueError(f"cannot resume an optimizer with another --line-search (the checkpoint's is {line_search})")
    if line_search == "armijo" and args.wolfe_energy_atol is not None:
        raise ValueError("--wolfe-energy-atol requires --line-search wolfe")
    wolfe_energy_atol = None
    if line_search == "wolfe":
        wolfe_energy_atol = (args.wolfe_energy_atol if args.wolfe_energy_atol is not None
                             else prior_search.get("wolfe_energy_atol", WOLFE_ENERGY_ATOL))
        if not math.isfinite(wolfe_energy_atol) or wolfe_energy_atol < 0:
            raise ValueError("--wolfe-energy-atol must be finite and nonnegative")
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
        coupling = read_portable(initial, arrays=False).coupling
        if not math.isfinite(coupling) or coupling < 0:
            raise ValueError("invalid initial coupling")
    plan = dict(schema_version=6, D=args.D, symmetry=args.symmetry, coupling=coupling, seed=args.seed, tiles=args.tiles,
                line_search=line_search, wolfe_energy_atol=wolfe_energy_atol,
                package=package_identity(),
                runtime=str(RUNTIME), runtime_source_sha256=dict(runtime_sha256),
                runtime_provenance=verify_runtime(),
                launcher_source_sha256=launcher_source_sha256(),
                layout=describe(args.D, controls["quadrature"], args.tiles, args.symmetry), controls=controls,
                max_seconds=args.max_seconds, max_iterations=args.max_iterations,
                gradient_target=args.gradient_target, max_device_seconds=args.max_device_seconds,
                device=args.device, compile_only=args.compile_only,
                output=str(args.output.resolve()) if args.output else None,
                resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                resume_sha256={str(p.resolve()): digest(p) for p in (args.resume_from, args.resume_from.with_suffix(".npz"))}
                    if previous else {},
                initial=input_ledger(initial, args.D, coupling, None if previous else args.symmetry) if initial else None,
                optimization_wall_time_limit_is_explicit=args.max_seconds is not None,
                actual_GPU_qualified=False, CPU_objective_or_optimizer=False)
    return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--D", type=int, required=True, help="bond dimension (even for --symmetry z2)")
    p.add_argument("--symmetry", choices=SYMMETRIES, default="z2",
                   help="z2 (the default): Z2-symmetric states, R odd and K even, in the packed parity blocks; "
                        "none: states without symmetry, full complex R and K (the unpacked channels)")
    p.add_argument("--coupling", type=float)
    p.add_argument("--seed", type=int, default=0, help="NumPy PCG64 initialization (Z2 or not, by --symmetry), "
                   "when --initial is absent")
    p.add_argument("--initial", type=Path, help="initial state, a portable state directory (rcmps_gpu.io): the "
                   "initial/ or optimized_state/ of a run, or an output of gpu/julia/export_state.jl or "
                   "prepare_initial.jl")
    p.add_argument("--resume-from", type=Path, help="stopped generic JSON checkpoint; keep L-BFGS history")
    p.add_argument("--output", type=Path, help="fresh run directory")
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--compile-only", action="store_true", help="compile/resource check without a numerical launch")
    p.add_argument("--emit-cuda", type=Path, help="write generated CUDA on the CPU, without --run-gpu")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-iterations", type=int, default=20000, help="additional iterations for a resumed invocation")
    p.add_argument("--max-seconds", type=float, help="optional invocation limit; no wall-time cap by default")
    p.add_argument("--gradient-target", type=float, default=1e-7)
    p.add_argument("--line-search", choices=LINE_SEARCHES,
                   help="armijo (the default for a new run): bounded Armijo backtracking, which accepts on energy "
                        "differences only; wolfe: the Hager-Zhang search of RcmpsCore's SafeguardedLineSearch (exact "
                        "or approximate Wolfe conditions, from the slope along the step), then the Armijo backtracking "
                        "if it fails. A resume keeps its checkpoint's choice and refuses the other")
    p.add_argument("--wolfe-energy-atol", type=float,
                   help=f"with --line-search wolfe: the energy increase an approximate Wolfe step may show "
                        f"(default {WOLFE_ENERGY_ATOL:g}, RcmpsCore's energy_atol; a resume keeps its checkpoint's)")
    p.add_argument("--max-device-seconds", type=float, default=300., help="one trajectory's device watchdog, not run wall time")
    p.add_argument("--tiles", choices=TILE_CHOICES, default="auto",
                   help="tile product of the trajectory kernel: mma (FP64 tensor cores, compute capability >= 8.0), "
                        "scalar (FP64 FMAs with warp shuffles, any GPU with cooperative launch), or auto (the "
                        "default: mma from compute capability 8.0 on, scalar below)")
    for name in DEFAULTS:
        p.add_argument("--" + name.replace("_", "-"), type=int if isinstance(DEFAULTS[name], int) else float)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        describe(args.D, symmetry=args.symmetry)
        if args.emit_cuda and args.run_gpu:
            raise ValueError("--emit-cuda is a CPU-only action")
        if args.run_gpu and not args.output:
            raise ValueError("--run-gpu requires a fresh --output directory")
        if args.output and args.output.exists():
            raise ValueError("output already exists; choose a fresh run directory")
        if args.emit_cuda and args.emit_cuda.exists():
            raise ValueError("CUDA output already exists; choose a new file")
        plan = prepare(args, runtime_source_sha256())
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if args.emit_cuda:
        # The source generator imports NumPy/SciPy, but never CuPy or CUDA.
        code = ("import sys;from pathlib import Path;from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as t;"
                "Path(t.__file__).resolve().parent==Path(sys.argv[1]) or sys.exit('--emit-cuda: rcmps_gpu imported from '+t.__file__);"
                "print((t.unpacked_trajectory_source if sys.argv[4]=='none' else t.trajectory_source)"
                "(int(sys.argv[2]),sys.argv[3]),end='')")
        # Without a device, auto writes the tensor-core source (the golden source of the plan).
        tiles = "scalar" if args.tiles == "scalar" else "mma"
        size = args.D if args.symmetry == "none" else args.D // 2
        command, environment = child_process("-c", code, str(RUNTIME), str(size), tiles, args.symmetry)
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        if result.returncode:
            lines = result.stderr.strip().splitlines()
            p.error("CUDA source generation failed: " + (lines[-1] if lines else f"exit status {result.returncode}"))
        args.emit_cuda.parent.mkdir(parents=True, exist_ok=True)
        with args.emit_cuda.open("x") as stream:
            stream.write(result.stdout)
        plan["emitted_cuda"] = str(args.emit_cuda.resolve())
        plan["emitted_cuda_sha256"] = digest(args.emit_cuda)
        plan["emitted_cuda_tiles"] = tiles
    if not args.run_gpu:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    command, environment = child_process("-m", "rcmps_gpu.optimizer.worker", str((args.output / "plan.json").resolve()))
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
