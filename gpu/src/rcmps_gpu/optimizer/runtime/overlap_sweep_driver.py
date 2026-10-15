"""Opt-in FP64 stream overlap around unchanged sweep/controller functions.

The default original controller retains the ordinary sweep L-BFGS controller
and cold stationary solve. ``--controller robust`` selects the existing
resolution-aware controller and accepted-density stationary correction. Both
choices preserve the physical objective, FP64 precision and strict kinetic
residual. Quartic solver preparation is serialized and included per objective;
only its propagation/pullback overlaps the independent kinetic calculation.

Example, through trajectory_sweep_driver, which supplies the quartic solver
(overlap_benchmark.PreparedQuartic no longer builds one):
  trajectory_sweep_driver.py results/reference_D64_initial --backend cupy
  --mode hybrid --packed-z2 --quartic-backend persistent_trajectory
  --arnoldi gpu --projection hessenberg_gpu --window 20 --iterations 2000
  --gradient-tolerance 3e-5 --output results/overlap_full/optimization.json

Importing this module does not import CuPy or initialize CUDA. A run owns two
persistent host workers and separate nonblocking streams, which are closed
after joined work on every exit. No backend, optimizer or CPU package is edited.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from . import overlap_benchmark as overlap
from . import robust_sweep_driver as robust
from . import sweep_driver as driver
from . import warm_sweep_driver as warm
from .numerics import SolverFailure, adj
from .warm_sweep_driver import binding, digest


OVERLAP_SOURCES = (*robust.ROBUST_SOURCES, "overlap_benchmark.py", "overlap_sweep_driver.py")
BASE_PARSE_ARGS = robust.BASE_PARSE_ARGS
BASE_OPTIMIZE_WITH_PROGRESS = warm.BASE_OPTIMIZE_WITH_PROGRESS


def validate_overlap_arguments(args):
    driver.validate_arguments(args)
    if (args.backend != "cupy" or args.mode != "hybrid" or not args.packed_z2 or
            args.quartic_backend != "persistent_trajectory"):
        raise ValueError("overlap requires CuPy hybrid packed Z2 and persistent_trajectory")
    if (args.arnoldi != "gpu" or args.projection != "hessenberg_gpu" or
            args.krylov_ceiling is not None or not math.isfinite(args.krylov_tolerance) or
            not 0 < args.krylov_tolerance <= 1e-12):
        raise ValueError("overlap requires GPU Arnoldi/Hessenberg, true residual <=1e-12 and no fallback")
    if args.reduced or args.fused_rhs:
        raise ValueError("overlap cannot change the packed cached objective layout")


def source_hashes(args):
    hashes = warm.BASE_SOURCE_HASHES(args)
    hashes.update({name: digest(Path(__file__).with_name(name).read_bytes())
                   for name in OVERLAP_SOURCES})
    return hashes


def verify_checkpoint(path, requested_args=None):
    """Accept only complete original/warm/robust/overlap source sets.

    The unchanged warm verifier performs stable-generation, exact source,
    reference, finite complex128 state and L-BFGS history checks. Historical
    hashes remain historical; the caller records this invocation separately.
    """
    original_report = Path(path).read_bytes()
    report = json.loads(original_report)
    args = argparse.Namespace(**report["controls"])
    expected = (report["runs"][-1] if report.get("runs") else report)["source_sha256"]
    extras = set(expected) - set(warm.BASE_SOURCE_HASHES(args))
    known = (set(), set(warm.WARM_SOURCES), set(robust.ROBUST_SOURCES), set(OVERLAP_SOURCES))
    if extras not in known:
        raise ValueError("checkpoint has unknown or incomplete overlap/robust/warm source extras")
    files = tuple(name for name in OVERLAP_SOURCES if name in extras) or warm.WARM_SOURCES
    with binding(warm, "WARM_SOURCES", files):
        snapshot = warm.verify_resume_checkpoint(path, requested_args)
    if digest(original_report) != snapshot["checkpoint_sha256"][Path(path).name]:
        raise ValueError("checkpoint changed between source preflight and stable-generation validation")
    snapshot["extras"] = tuple(name for name in OVERLAP_SOURCES if name in extras)
    snapshot["source_sha256"] = expected
    snapshot["overlap"] = extras == set(OVERLAP_SOURCES)
    if snapshot["overlap"]:
        validate_overlap_arguments(snapshot["args"])
        recorded = getattr(snapshot["args"], "overlap_controls", {})
        if recorded.get("enabled") is not True or recorded.get("controller") not in ("original", "robust"):
            raise ValueError("overlap checkpoint lacks its explicit enabled/controller controls")
    elif getattr(snapshot["args"], "overlap_controls", {}).get("enabled"):
        raise ValueError("overlap controls require the complete overlap source set")
    return snapshot


def matching_source_hashes(snapshot, args):
    hashes = warm.BASE_SOURCE_HASHES(args)
    hashes.update({name: digest(Path(__file__).with_name(name).read_bytes())
                   for name in snapshot["extras"]})
    return hashes


class OverlapObjective:
    """Run-owned coordinator; repeated calls reuse workers/streams, never quartic solvers."""
    def __init__(self):
        self.cp = None
        self.executor = None
        self.streams = None
        self.combine = None
        self.device = None
        self.closed = False
        self._lock = threading.Lock()

    def _initialize(self, cp):
        if self.cp is not None:
            if cp is not self.cp or int(cp.cuda.Device().id) != self.device:
                raise OverlapFailure("a run cannot change its CUDA module/device")
            return
        self.device = int(cp.cuda.Device().id)
        self.cp = cp
        self.streams = {}
        for name in ("kinetic", "quartic"):
            self.streams[name] = cp.cuda.Stream(non_blocking=True)
        self.combine = cp.cuda.Stream(non_blocking=True)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rcmps-overlap")

    def __call__(self, r, k, rho, coupling, args, xp):
        if not self._lock.acquire(blocking=False):
            raise OverlapFailure("one objective coordinator cannot evaluate concurrent states")
        try:
            if self.closed:
                raise OverlapFailure("objective coordinator is closed")
            validate_overlap_arguments(args)
            if xp.__name__ != "cupy":
                raise ValueError("overlapped objective is GPU-only")
            matrices = (r, k, rho)
            if (r.ndim != 2 or r.shape[0] != r.shape[1] or
                    any(x.shape != r.shape or x.dtype != np.complex128 or not np.isfinite(x).all()
                        for x in matrices)):
                raise SolverFailure("objective requires equally shaped finite complex128 matrices")
            self._initialize(xp)
            q = -1j*k-adj(r)@r/2
            original = tuple(np.array(x, dtype=np.complex128, copy=True) for x in (q, r, rho))
            # No prior evaluation is active, and the quartic solver preparation below
            # starts only after this producer stream has finished its uploads.
            xp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            inputs = tuple(xp.asarray(x, dtype=xp.complex128) for x in original)
            inputs_ready = xp.cuda.Event()
            inputs_ready.record(xp.cuda.get_current_stream())
            packed_args = SimpleNamespace(window=args.window, quadrature=args.quadrature,
                atol=args.atol, rtol=args.rtol, max_steps=args.max_steps)
            with self.combine:
                self.combine.wait_event(inputs_ready)
                prepared = overlap.PreparedQuartic(xp, inputs, coupling, packed_args)
                ready = xp.cuda.Event()
                ready.record(self.combine)
            q_gpu, r_gpu, rho_gpu = inputs
            calls = dict(kinetic=lambda: overlap.kinetic_raw(q_gpu, r_gpu, rho_gpu, args.nodes,
                xp=xp, tolerance=args.krylov_tolerance, maxdim=args.maxdim,
                arnoldi_backend=args.arnoldi, fallback_tolerance=None,
                projection_backend=args.projection), quartic=prepared)
            try:
                results = overlap.dispatch_pair(calls, self.executor, parallel=True,
                    worker=overlap.gpu_worker(xp, self.device, self.streams, ready, args.krylov_tolerance))
            finally:
                # dispatch_pair joins both futures, including a failing pair.
                prepared.close()
            with self.combine:
                for result in results.values():
                    self.combine.wait_event(result.end_event)
                raw = overlap.raw_sum(results, xp)
                covector, environment = (xp.asnumpy(raw[name]) for name in ("covector", "environment"))
                actual = tuple(xp.asnumpy(array) for array in inputs)
            overlap.require_immutable_inputs(original, actual)
            overlap.require_immutable_inputs(original, (q, r, rho))
            if any(x.dtype != np.complex128 or not np.isfinite(x).all() for x in (covector, environment)):
                raise SolverFailure("finished raw covector/environment requires finite complex128")
            observable_seconds = time.perf_counter()-started
            finish_begin = time.perf_counter()
            # Calling the driver's binding preserves the independent FD
            # checker's ordinary/tighter boundary audit without another sum.
            gradient, boundary = driver.finish_gradient(q, r, rho, covector, environment,
                                                       metric_reg=args.metric_reg)
            boundary_seconds = time.perf_counter()-finish_begin
            if gradient.dtype != np.complex128 or not np.isfinite(gradient).all():
                raise SolverFailure("finished metric gradient requires finite complex128")
            intervals = {name: dict(start_ms=float(xp.cuda.get_elapsed_time(ready, result.start_event)),
                end_ms=float(xp.cuda.get_elapsed_time(ready, result.end_event)))
                for name, result in results.items()}
            thread_ids = {name: int(result.thread_id) for name, result in results.items()}
            stream_pointers = {name: int(result.stream_pointer) for name, result in results.items()}
            if len(set(thread_ids.values())) != 2 or len(set(stream_pointers.values())) != 2:
                raise OverlapFailure("overlap requires two host workers and distinct CUDA streams")
            stats = dict(kinetic=results["kinetic"].raw["stats"], quartic=results["quartic"].raw["stats"],
                observable_seconds=float(observable_seconds), cpu_boundary_metric_seconds=float(boundary_seconds),
                boundary=boundary, total_seconds=float(time.perf_counter()-started),
                overlap=dict(enabled=True, worker_thread_ids=thread_ids, stream_pointers=stream_pointers,
                    component_wall_seconds={name: float(result.wall_seconds) for name, result in results.items()},
                    stream_intervals=intervals,
                    stream_interval_overlap_ms=float(max(0., min(x["end_ms"] for x in intervals.values())
                        - max(x["start_ms"] for x in intervals.values()))),
                    serial_quartic_preparation_seconds=float(prepared.setup_seconds),
                    setup_included=True, immutable_inputs_verified=True, implicit_fallbacks=0))
            return float(raw["value"]), gradient, stats
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=False)
            if self.cp is not None:
                with self.cp.cuda.Device(self.device):
                    for stream in (*((self.streams or {}).values()), self.combine):
                        if stream is not None:
                            stream.synchronize()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *errors):
        self.close()


OverlapFailure = overlap.OverlapFailure


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--controller", choices=("original", "robust"), default="original")
    parser.add_argument("--resume-from")
    parser.add_argument("--reset-lbfgs", action="store_true")
    parser.add_argument("--secant-gradient-floor", type=float)
    parser.add_argument("--max-recoveries", type=int)
    cli, forwarded = parser.parse_known_args(argv)
    base_args = BASE_PARSE_ARGS(forwarded)
    validate_overlap_arguments(base_args)
    if cli.reset_lbfgs and not cli.resume_from:
        raise ValueError("--reset-lbfgs requires --resume-from and a distinct fresh --output")
    if cli.controller == "original" and (cli.secant_gradient_floor is not None or cli.max_recoveries is not None):
        raise ValueError("robust controls require --controller robust")
    snapshot = None
    if cli.resume_from or base_args.resume:
        snapshot = verify_checkpoint(cli.resume_from or base_args.output, base_args)
        if base_args.resume and snapshot["overlap"]:
            if snapshot["args"].overlap_controls["controller"] != cli.controller:
                raise ValueError("in-place resume cannot change controller; use an explicit --resume-from fork")
    elif Path(base_args.output).exists() or Path(base_args.output).with_suffix(".npz").exists():
        raise ValueError("fresh overlap run requires a new JSON/NPZ output")
    recorded = dict(enabled=True, controller=cli.controller, persistent_host_workers=2,
                    separate_nonblocking_streams=True, serial_quartic_preparation=True)

    def parse(arguments=None):
        out = BASE_PARSE_ARGS(arguments)
        validate_overlap_arguments(out)
        out.overlap_controls = recorded
        if cli.resume_from or base_args.resume:
            out.resume = True
        return out

    with OverlapObjective() as coordinator, binding(driver, "objective", coordinator), \
            binding(driver, "source_hashes", source_hashes):
        if cli.controller == "robust":
            # The frozen robust wrapper's small parser abbreviates --resume
            # as --resume-from. Resume has already been parsed and verified
            # here; the scoped parse above carries its exact boolean through.
            robust_argv = [argument for argument in forwarded if argument != "--resume"]
            if cli.resume_from:
                robust_argv += ["--resume-from", cli.resume_from]
            if cli.reset_lbfgs:
                robust_argv += ["--reset-lbfgs"]
            prior = getattr(snapshot["args"], "robust_controls", {}) if snapshot else {}
            for key in ("secant_gradient_floor", "max_recoveries"):
                value = getattr(cli, key)
                if value is None:
                    value = prior.get(key, 1e-10 if key == "secant_gradient_floor" else 2)
                robust_argv += ["--"+key.replace("_", "-"), str(value)]
            with binding(robust, "BASE_PARSE_ARGS", parse), binding(robust, "source_hashes", source_hashes), \
                    binding(robust, "verify_checkpoint", verify_checkpoint):
                return robust.main(robust_argv)
        if cli.resume_from:
            robust.fork_checkpoint(snapshot, base_args.output, base_args, reset_lbfgs=cli.reset_lbfgs)
        with binding(driver, "parse_args", parse), \
                binding(driver, "optimize_with_progress", BASE_OPTIMIZE_WITH_PROGRESS):
            return driver.main(forwarded)


if __name__ == "__main__":
    main()
