"""Qualification of the GPU optimizer on one NVIDIA GPU: tile products, compilation, runs.

    python gpu_optimizer_check.py --output NEW_DIR [--stages unit trajectory compile run]
        [--D 8 16 32 64] [--tiles auto [scalar]] [--iterations 3] [--device 0] [--symmetry z2|none]

Stages (all by default), each for every D and every tile path of --tiles (auto, mma,
scalar; auto is what the device selects: mma from compute capability 8.0 on, scalar below):

  unit        The tile product alone, one warp per 8 x 8 x 4 tile, on random tiles (spread
              exponents), ties, cancellations, signed zeros and subnormal products: the scalar
              tiles (SCALAR_TILE of the trajectory kernel) and, from compute capability 8.0, the
              tensor-core tiles (MMA_TILE), each compared bitwise with the host model
              chain_model (FMAs in k order, correctly rounded) and with each other. Whether this
              GPU's FP64 tensor cores compute that chain decides whether the two paths can give
              bitwise identical trajectories.
  trajectory  PersistentTrajectoryDOP853, the kernel of a run, on the random state of the
              optimizer (seed --seed; Z2, or with --symmetry none a state without symmetry on
              the unpacked channels) for each tile path: steps and samples of two launches
              (determinism), bitwise comparison of the paths, and at D <= 8 the host oracle
              integrate_trajectory_numpy; device time per trajectory. A grid that is not
              resident at any register cap of the worker is a result ("refused"), as in the
              compile stage, not a failure.
  compile     rcmps-gpu-optimize --compile-only: the worker's tile selection, compilation,
              cooperative residency check and PTX audit. A size the device cannot hold is a
              result ("resident": false, with the worker's message), not a failure.
  run         rcmps-gpu-optimize --run-gpu --max-iterations N: compilation, the initial
              qualification of the complete energy and gradient against the independent CuPy
              propagation, N accepted L-BFGS iterations and the final five-row validation. When
              two tile paths ran at one D, their first evaluations and histories are compared.

The compile and run stages start the command line of the package in a child process (its
run directory under NEW_DIR/<stage>/D<D>_<tiles>, its output in <directory>.log), with the
model and integration settings of this script (--coupling, --window, --quadrature, --atol,
--rtol; the defaults are those of the command line) and its --seed, --device, --symmetry and
--max-device-seconds; nothing here re-implements the numerics. With --symmetry none the run
directories are NEW_DIR/<stage>/D<D>_<tiles>_none. Every result is one JSON line on stdout; the exit status is
1 if a check failed (a refused size or a refused --tiles mma below 8.0 is not a failure).
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

LANE = np.arange(32)
ROW, K, COL = LANE >> 2, LANE & 3, 2 * (LANE & 3)
FAILED = []


def emit(**row):
    if row.get("ok") is False:
        FAILED.append(row)
    print(json.dumps(row, default=float), flush=True)


def fma(a, b, c):
    """a b + c rounded once to the nearest double; an exact zero is -0.0 only when the product
    is a zero of negative sign and c is -0.0 (IEEE 754, round to nearest)."""
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        product_negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
        if (a == 0 or b == 0) and c == 0 and product_negative and math.copysign(1.0, c) < 0:
            return -0.0
        return 0.0
    return float(exact)


def chain_model(A, B, C):
    """A B + C (8 x 4, 4 x 8, 8 x 8) with every element a chain of FMAs in k order 0..3: the
    arithmetic of SCALAR_TILE (tests/optimizer/test_tiles.py checks that)."""
    D = np.empty((8, 8))
    for i in range(8):
        for j in range(8):
            value = float(C[i, j])
            for k in range(4):
                value = fma(float(A[i, k]), float(B[k, j]), value)
            D[i, j] = value
    return D


def unit_cases(rng, random=64):
    """Tiles (A, B, C) of the unit stage: random with spread exponents, a tie (1 plus two
    products of 2^-53: the chain gives 1, a single rounding 1 + 2^-52), cancellation
    (C = -A B to the last bit), signed zeros and subnormal products."""
    cases = []
    for _ in range(random):
        A, B, C = rng.standard_normal((8, 4)), rng.standard_normal((4, 8)), rng.standard_normal((8, 8))
        A *= 2.0 ** rng.integers(-30, 30, size=A.shape)
        cases.append((A, B, C))
    A, B = np.zeros((8, 4)), np.zeros((4, 8))
    A[:, :2], B[:2, :] = 2.0 ** -27, 2.0 ** -26
    cases.append((A, B, np.ones((8, 8))))
    A, B = rng.standard_normal((8, 4)), rng.standard_normal((4, 8))
    cases.append((A, B, -(A @ B)))
    A, B, C = np.zeros((8, 4)), np.ones((4, 8)), np.full((8, 8), -0.0)
    A[:4] = -0.0
    cases.append((A, B, C))
    A, B = rng.standard_normal((8, 4)) * 2.0 ** -540, rng.standard_normal((4, 8)) * 2.0 ** -500
    cases.append((A, B, rng.standard_normal((8, 8)) * 2.0 ** -1060))
    return cases


UNIT_KERNEL = r'''
extern "C" __global__ void tiles(const double* a,const double* b,const double* c,double* scalar,double* mma,int use_mma){
    const int lane=threadIdx.x&31,t=blockIdx.x;
    const double av=a[32*t+lane],bv=b[32*t+lane];
    double c0=c[64*t+2*lane],c1=c[64*t+2*lane+1];
    double s0=c0,s1=c1;
    scalar_mma(av,bv,s0,s1);
    scalar[64*t+2*lane]=s0;scalar[64*t+2*lane+1]=s1;
#if USE_MMA
    tensor_mma(av,bv,c0,c1);
    mma[64*t+2*lane]=c0;mma[64*t+2*lane+1]=c1;
#endif
}
'''


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)


def bitwise(x, y):
    """Same shape, type and bytes (signed zeros and NaN payloads included)."""
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype
                and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes())


def stage_unit(cp, args, capability):
    from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as backend
    native = capability >= (8, 0)
    source = (f"#define USE_MMA {int(native)}\n" + backend.SCALAR_TILE.replace("void mma(", "void scalar_mma(")
              + (backend.MMA_TILE.replace("void mma(", "void tensor_mma(") if native else "") + UNIT_KERNEL)
    kernel = cp.RawKernel(source, "tiles", options=("--std=c++17",))
    cases = unit_cases(np.random.default_rng(args.seed))
    a = np.stack([A[ROW, K] for A, _, _ in cases])
    b = np.stack([B[K, ROW] for _, B, _ in cases])
    c = np.stack([np.stack([C[ROW, COL], C[ROW, COL + 1]], axis=1).reshape(-1) for _, _, C in cases])
    out = [cp.zeros(c.shape, dtype=cp.float64) for _ in range(2)]
    kernel((len(cases),), (32,), (cp.asarray(a), cp.asarray(b), cp.asarray(c), out[0], out[1], np.int32(native)))
    host = [cp.asnumpy(x).reshape(len(cases), 32, 2) for x in out]
    names = ["random"] * (len(cases) - 4) + ["tie", "cancellation", "signed_zero", "subnormal"]
    tally = {}
    for i, (A, B, C) in enumerate(cases):
        want = chain_model(A, B, C)
        want = np.stack([want[ROW, COL], want[ROW, COL + 1]], axis=1)
        row = tally.setdefault(names[i], dict(tiles=0, scalar_equals_chain=0, **(
            dict(mma_equals_chain=0, mma_equals_scalar=0) if native else {})))
        row["tiles"] += 1
        row["scalar_equals_chain"] += bool(np.array_equal(bits(host[0][i]), bits(want)))
        if native:
            row["mma_equals_chain"] += bool(np.array_equal(bits(host[1][i]), bits(want)))
            row["mma_equals_scalar"] += bool(np.array_equal(bits(host[1][i]), bits(host[0][i])))
    scalar_ok = all(row["scalar_equals_chain"] == row["tiles"] for row in tally.values())
    emit(stage="unit", compute_capability=list(capability), tensor_cores=native, cases=tally,
         scalar_tiles_are_the_fma_chain=scalar_ok,
         tensor_cores_are_the_fma_chain=(all(row["mma_equals_chain"] == row["tiles"] for row in tally.values())
                                         if native else None), ok=scalar_ok)


def random_state(args, D, directory):
    """The random initial state of a run with --seed and --symmetry (worker.random_reference)."""
    from rcmps_gpu.optimizer import cli, worker
    from rcmps_gpu.optimizer.runtime import numerics
    controls = dict(cli.DEFAULTS, quadrature=args.quadrature, window=args.window)
    worker.random_reference(dict(D=D, seed=args.seed, coupling=args.coupling, controls=controls,
                                 symmetry=args.symmetry), directory)
    matrices, _ = numerics.load_reference(directory)
    r, k, rho = (matrices[name] for name in ("R", "K", "rho"))
    return -1j * k - r.conj().T @ r / 2, r, rho


def stage_trajectory(cp, args, capability, D, paths):
    from rcmps_gpu.optimizer.runtime.ode_backend import _PackedZ2QuarticRHS, _UnpackedQuarticRHS, quadrature
    from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as backend
    from rcmps_gpu.optimizer.runtime.generic_layout import register_caps, select_tiles
    q, r, rho = random_state(args, D, Path(args.output) / "trajectory" / f"D{D}{suffix(args)}_initial")
    unpacked = args.symmetry == "none"
    layout, d = (_UnpackedQuarticRHS, D) if unpacked else (_PackedZ2QuarticRHS, D // 2)
    points, _ = quadrature(args.window, args.quadrature)
    nodes = np.r_[points, math.sqrt(args.window)]
    samples = {}
    for requested in paths:
        row = dict(stage="trajectory", D=D, symmetry=args.symmetry, tiles_requested=requested)
        try:
            tiles = select_tiles(requested, *capability)
        except ValueError as error:
            emit(**row, refused=str(error))
            continue
        row["tiles"] = tiles
        if tiles in samples:
            emit(**row, same_as_previous_path=True)
            continue
        try:
            rhs = layout(cp.asarray(q), cp.asarray(r), cp.asarray(rho), cp)
            # The register caps of the worker, in its order: the first whose grid is resident.
            for cap in register_caps(d):
                try:
                    solver = backend.PersistentTrajectoryDOP853(rhs, nodes, atol=args.atol, rtol=args.rtol,
                                                                max_device_seconds=args.max_device_seconds,
                                                                tiles=tiles, register_cap=cap)
                    break
                except RuntimeError as error:
                    if "unsafe cooperative trajectory grid" not in str(error) or cap == register_caps(d)[-1]:
                        raise
            row["register_cap"] = cap
            with solver:
                first, stats = solver.run()
                first = cp.asnumpy(first)
                again, stats_again = solver.run()
                again = cp.asnumpy(again)
            samples[tiles] = (first, stats)
            row.update(accepted=stats["accepted"], rejected=stats["rejected"], attempts=stats["attempts"],
                       device_milliseconds=[stats["device_milliseconds"], stats_again["device_milliseconds"]],
                       registers=stats["occupancy"]["registers"],
                       resident_capacity=stats["occupancy"]["resident_capacity"], blocks=stats["occupancy"]["blocks"],
                       repeat_bitwise=bitwise(first, again))
            if D <= 8:
                host = layout(q, r, rho, np)
                oracle, counts = backend.integrate_trajectory_numpy(host, nodes, atol=args.atol, rtol=args.rtol)
                row.update(oracle_attempts=counts["attempts"], oracle_same_steps=counts["attempts"] == stats["attempts"],
                           oracle_relative_difference=float(np.abs(first - oracle).max() / np.abs(oracle).max()))
            # Rounding differences (contraction, summation order) can move a step; a bug cannot stay below 1e-9.
            row["ok"] = row["repeat_bitwise"] and row.get("oracle_relative_difference", 0.0) < 1e-9
        except Exception as error:      # report and continue with the next path
            message = f"{type(error).__name__}: {error}"[:2000]
            row.update(dict(refused=message, ok=True) if refused(message) else dict(error=message, ok=False))
        emit(**row)
    if len(samples) == 2:
        (a, sa), (b, sb) = samples["mma"], samples["scalar"]
        emit(stage="trajectory_paths", D=D, symmetry=args.symmetry, same_steps=(sa["attempts"], sa["accepted"]) == (sb["attempts"], sb["accepted"]),
             samples_bitwise=bitwise(a, b),
             max_relative_difference=float(np.abs(a - b).max() / np.abs(a).max()) if a.shape == b.shape else None,
             speed_ratio_scalar_over_mma=sb["device_milliseconds"] / sa["device_milliseconds"])


def suffix(args):
    return "_none" if args.symmetry == "none" else ""


def optimize(args, stage, D, tiles, *options):
    """rcmps-gpu-optimize in a child process: (run directory, exit status, seconds)."""
    directory = Path(args.output) / stage / f"D{D}_{tiles}{suffix(args)}"
    command = [sys.executable, "-m", "rcmps_gpu.optimizer", "--D", str(D), "--run-gpu", "--output", str(directory),
               "--symmetry", args.symmetry, "--tiles", tiles, "--device", str(args.device), "--seed", str(args.seed),
               "--coupling", repr(args.coupling), "--window", repr(args.window), "--quadrature", str(args.quadrature),
               "--atol", repr(args.atol), "--rtol", repr(args.rtol),
               "--max-device-seconds", repr(args.max_device_seconds), *options]
    started = time.perf_counter()
    with (Path(args.output) / stage / f"D{D}_{tiles}{suffix(args)}.log").open("w") as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    return directory, status, time.perf_counter() - started


def load(path):
    path = Path(path)
    return json.loads(path.read_bytes()) if path.exists() else None


def compile_summary(compiled):
    return {key: compiled.get(key) for key in ("tiles", "SM", "GPU", "threads", "blocks", "registers", "local_bytes",
                                                "dynamic_shared_bytes", "active_blocks_per_SM", "resident_capacity",
                                                "compiler_options", "register_cap", "register_caps_tried",
                                                "kernel_run_time_limit")} | dict(PTX=compiled.get("PTX"))


# The errors of a size this device cannot hold or a tile path it cannot run: the worker's
# (status.json) and those of PersistentTrajectoryDOP853 and configure_shared_memory.
REFUSALS = ("resident blocks", "unsafe cooperative trajectory grid", "--tiles mma needs", "exceeds device limit",
            "thread-block limit")


def refused(message):
    """`message` if it is a refusal (REFUSALS), else None."""
    return message if any(text in message for text in REFUSALS) else None


def refusal(status):
    """The worker's error if it refused the size or the tile path for this device, else None."""
    return refused((status or {}).get("error", ""))


def stage_compile(args, D, paths):
    for tiles in paths:
        directory, code, seconds = optimize(args, "compile", D, tiles, "--compile-only")
        status = load(directory / "status.json")
        row = dict(stage="compile", D=D, symmetry=args.symmetry, tiles_requested=tiles, exit_status=code, seconds=seconds)
        if status and status.get("state") == "compiled_without_numerical_launch":
            emit(**row, resident=True, **compile_summary(status["compile"]), ok=True)
        elif refusal(status):
            emit(**row, resident=False, refused=refusal(status), ok=True)
        else:
            emit(**row, error=(status or {}).get("error", f"no status.json (see {directory}.log)"), ok=False)


def stage_run(args, D, paths):
    runs = {}
    for tiles in paths:
        directory, code, seconds = optimize(args, "run", D, tiles, "--max-iterations", str(args.iterations),
                                            *(("--max-seconds", str(args.max_seconds)) if args.max_seconds else ()))
        status = load(directory / "status.json") or {}
        row = dict(stage="run", D=D, symmetry=args.symmetry, tiles_requested=tiles, exit_status=code, seconds=seconds,
                   state=status.get("state"))
        if refusal(status):
            emit(**row, refused=refusal(status), ok=True)
            continue
        qualification = load(directory / "initial_qualification.json") or {}
        checkpoint = load(directory / "optimization.json") or {}
        validation = load(directory / "final_validation.json") or {}
        history = checkpoint.get("history", [])
        row.update(tiles=(status.get("compile") or {}).get("tiles", {}).get("resolved"),
                   registers=(status.get("compile") or {}).get("registers"),
                   resident_capacity=(status.get("compile") or {}).get("resident_capacity"),
                   qualification_passed=qualification.get("passed"),
                   energy_difference=qualification.get("energy_difference"),
                   metric_gradient_difference=qualification.get("metric_gradient_difference"),
                   iterations=len(history), stop_reason=checkpoint.get("stop_reason"),
                   final_energy=checkpoint.get("final_energy"), final_gradient_norm=checkpoint.get("final_gradient_norm"),
                   final_phi=checkpoint.get("final_phi"),
                   validation_passed=validation.get("passed"), error=status.get("error"),
                   ok=status.get("state") == "completed" and bool(validation.get("passed")))
        emit(**row)
        if row["tiles"]:
            runs.setdefault(row["tiles"], directory)
    if len(runs) == 2:
        compare_runs(D, runs["mma"], runs["scalar"])


def compare_runs(D, first, second):
    """First evaluation and history of two runs of the same state with the two tile paths."""
    row = dict(stage="run_paths", D=D)
    try:
        with np.load(first / "initial_qualification.npz") as a, np.load(second / "initial_qualification.npz") as b:
            row["first_gradient_bitwise"] = bitwise(a["fast_gradient"], b["fast_gradient"])
        qa, qb = load(first / "initial_qualification.json"), load(second / "initial_qualification.json")
        row["first_energy_difference"] = qb["fast_energy"] - qa["fast_energy"]
        ha, hb = load(first / "optimization.json")["history"], load(second / "optimization.json")["history"]
        energies = [(x.get("energy"), y.get("energy")) for x, y in zip(ha, hb)]
        row.update(iterations=[len(ha), len(hb)], history_energies_equal=all(x == y for x, y in energies),
                   max_history_energy_difference=max((abs(x - y) for x, y in energies if x is not None and y is not None),
                                                     default=None))
        with np.load(first / "optimization.npz") as a, np.load(second / "optimization.npz") as b:
            row["final_state_bitwise"] = all(bitwise(a[n], b[n]) for n in ("R", "K", "gradient"))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:2000]
    emit(**row)


def parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new directory")
    parser.add_argument("--stages", nargs="+", choices=("unit", "trajectory", "compile", "run"),
                        default=["unit", "trajectory", "compile", "run"])
    parser.add_argument("--D", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--tiles", nargs="+", choices=("auto", "mma", "scalar"), default=["auto"])
    parser.add_argument("--symmetry", choices=("z2", "none"), default="z2",
                        help="z2: the packed Z2 channels (the default); none: the unpacked channels")
    parser.add_argument("--iterations", type=int, default=3, help="accepted iterations of a run")
    parser.add_argument("--max-seconds", type=float, default=None, help="wall-time limit of a run")
    parser.add_argument("--max-device-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coupling", type=float, default=2.7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--quadrature", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    return parser


def main():
    args = parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for stage in ("compile", "run", "trajectory"):
        (output / stage).mkdir()
    paths = list(dict.fromkeys(args.tiles))
    import cupy as cp
    with cp.cuda.Device(args.device):
        props = cp.cuda.runtime.getDeviceProperties(args.device)
        capability = (int(props["major"]), int(props["minor"]))
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        emit(stage="device", GPU=name, compute_capability=list(capability),
             multiprocessors=int(props["multiProcessorCount"]), cooperative_launch=bool(props.get("cooperativeLaunch")),
             shared_memory_per_block_optin=int(props.get("sharedMemPerBlockOptin", 0)),
             kernel_run_time_limit=bool(props.get("kernelExecTimeoutEnabled", False)),
             cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()), driver=int(cp.cuda.runtime.driverGetVersion()),
             cupy=cp.__version__, numpy=np.__version__, python=sys.version.split()[0])
        if "unit" in args.stages:
            stage_unit(cp, args, capability)
        for D in args.D:
            if "trajectory" in args.stages:
                stage_trajectory(cp, args, capability, D, paths)
    for D in args.D:
        if "compile" in args.stages:
            stage_compile(args, D, paths)
        if "run" in args.stages:
            stage_run(args, D, paths)
    emit(stage="summary", failed=len(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
