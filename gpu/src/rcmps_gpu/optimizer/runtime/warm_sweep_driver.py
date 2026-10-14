"""Opt-in stationary correction around the unchanged dimension-sweep driver.

Only trial stationary solves inside the optimizer generator use the last
accepted density. Objectives, gradients, line search and L-BFGS code are the
original functions. Each generator advancement restores the original solver
before yielding an event. Resume preflight runs before CUDA is initialized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import numpy as np

import optimizer
import sweep_driver as driver
from stationary_correction import stationary_state_correction


WARM_SOURCES = ("warm_sweep_driver.py", "stationary_correction.py")
BASE_SOURCE_HASHES = driver.source_hashes
BASE_OPTIMIZE_WITH_PROGRESS = driver.optimize_with_progress
# Physical/objective controls must be identical when changing stationary
# initialization. Stopping, line-search and invocation limits remain explicit
# in the ordinary driver's per-run controls/history.
OBJECTIVE_CONTROLS = ("reference", "backend", "mode", "quartic_backend", "window",
    "quadrature", "atol", "rtol", "nodes", "krylov_tolerance", "maxdim", "arnoldi",
    "projection", "krylov_ceiling", "max_steps", "metric_reg", "stationary_tolerance",
    "graph", "dense_graph", "device_controller", "reduced", "packed_z2", "fused_rhs")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes(args):
    hashes = BASE_SOURCE_HASHES(args)
    hashes.update({name: digest(Path(__file__).with_name(name).read_bytes())
                   for name in WARM_SOURCES})
    return hashes


@contextmanager
def binding(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def verify_resume_checkpoint(path, requested_args=None):
    """Verify a stable stopped generation and exact core/known-extra hashes.

    The original 13-source backend is always checked directly using the
    original driver function. Known warm extras are accepted only as a complete
    pair with their exact current hashes; arbitrary extra/missing files fail.
    This accepts original checkpoints for the first opt-in continuation and
    warm checkpoints for subsequent continuation and independent validation.
    """
    path = Path(path).resolve()
    npz_path = path.with_suffix(".npz")
    first = path.read_bytes()
    before = npz_path.stat()
    npz_bytes = npz_path.read_bytes()
    second = path.read_bytes()
    after = npz_path.stat()
    signature = lambda stat: (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if first != second or signature(before) != signature(after):
        raise ValueError("checkpoint changed during read; require a stopped stable generation")
    report = json.loads(first)
    if not report.get("stop_reason") and not report.get("converged"):
        raise ValueError("warm continuation/validation requires a stopped checkpoint")
    controls = dict(report["controls"])
    args = argparse.Namespace(**controls)
    driver.validate_arguments(args)
    if vars(args) != controls:
        raise ValueError("checkpoint controls differ from validated latest-run controls")
    latest = report["runs"][-1] if report.get("runs") else report
    if latest.get("controls", controls) != controls:
        raise ValueError("latest run and checkpoint controls differ")
    expected = latest["source_sha256"]
    core = BASE_SOURCE_HASHES(args)
    actual_core = {name: expected.get(name) for name in core}
    if actual_core != core:
        names = [name for name in core if actual_core[name] != core[name]]
        raise ValueError("checkpoint core source hashes differ: " + ", ".join(names))
    extras = set(expected) - set(core)
    if extras not in (set(), set(WARM_SOURCES)):
        raise ValueError("checkpoint has unknown or incomplete warm source extras")
    if extras and expected != source_hashes(args):
        raise ValueError("checkpoint warm source hashes differ")
    if requested_args is not None:
        mismatches = []
        for key in OBJECTIVE_CONTROLS:
            old, new = getattr(args, key), getattr(requested_args, key)
            same = Path(old).resolve() == Path(new).resolve() if key == "reference" else old == new
            if not same:
                mismatches.append(key)
        if mismatches:
            raise ValueError("warm resume changes objective controls: " + ", ".join(mismatches))
    reference = Path(args.reference)
    for name, expected_digest in report["reference_sha256"].items():
        filename = name if "." in name else name + ".bin"
        if digest((reference / filename).read_bytes()) != expected_digest:
            raise ValueError("checkpoint reference hash differs: " + filename)
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as saved:
        state = {name: saved[name].copy() for name in saved.files}
    iteration = int(state["iteration"])
    energy = float(state["energy"])
    if (iteration != len(report["history"]) or not np.isfinite(energy) or
            ("final_energy" in report and energy != float(report["final_energy"]))):
        raise ValueError("JSON/NPZ checkpoint generations or final energies differ")
    r = state["R"]
    arrays = [state[key] for key in ("R", "K", "rho", "gradient")]
    if (r.ndim != 2 or r.shape[0] != r.shape[1] or len(r) != report["reference"]["D"] or
            any(a.dtype != np.complex128 or a.shape != r.shape or not np.isfinite(a).all()
                for a in arrays)):
        raise ValueError("checkpoint state requires matching finite complex128 matrices")
    pairs = [state[key] for key in ("lbfgs_steps", "lbfgs_changes")]
    inverse = state["lbfgs_inverse_curvatures"]
    if (any(a.dtype != np.complex128 or a.ndim != 3 or a.shape[1:] != r.shape or
            not np.isfinite(a).all() for a in pairs) or
            pairs[0].shape != pairs[1].shape or inverse.shape != (len(pairs[0]),) or
            not np.isfinite(inverse).all() or np.any(inverse <= 0)):
        raise ValueError("invalid complex128 L-BFGS checkpoint history")
    return dict(report=report, state=state, args=args, warm=bool(extras), path=path,
                checkpoint_sha256={path.name: digest(first), npz_path.name: digest(npz_bytes)})


def accepted_density_events(events, rho, *, iteration=0, solve=stationary_state_correction):
    """Advance the unmodified generator with a scoped accepted-density solver.

    Rejected trials never become warm-start priors. The yielded state remains
    unchanged, and closing or a failed generator advancement restores globals.
    """
    prior = np.array(rho, dtype=np.complex128, copy=True)
    prior_iteration = int(iteration)

    def stationary(q, r, *, tolerance):
        result, stats = solve(q, r, prior, tolerance=tolerance)
        stats = dict(stats, warm_prior_iteration=prior_iteration)
        return result, stats

    try:
        while True:
            with binding(optimizer, "stationary_state", stationary):
                try:
                    event = next(events)
                except StopIteration:
                    return
            if event["type"] == "accepted":
                prior = event["state"]["rho"].copy()
                prior_iteration = event["state"]["iteration"]
            yield event
    finally:
        # close() cannot evaluate a new trial, but may close nested generators.
        events.close()


def optimize_with_progress(r, k, rho, energy, gradient, objective_callable, controls,
                           resume_state=None, **options):
    events = BASE_OPTIMIZE_WITH_PROGRESS(r, k, rho, energy, gradient,
                                         objective_callable, controls, resume_state, **options)
    resume = resume_state or {}
    return accepted_density_events(events, resume.get("rho", rho),
                                   iteration=resume.get("iteration", 0))


def main(argv=None):
    args = driver.parse_args(argv)
    if args.resume:
        verify_resume_checkpoint(args.output, args)
    elif Path(args.output).exists() or Path(args.output).with_suffix(".npz").exists():
        raise ValueError("fresh warm run requires a new output; use --resume for a verified stopped checkpoint")
    with binding(driver, "source_hashes", source_hashes), \
            binding(driver, "optimize_with_progress", optimize_with_progress):
        return driver.main(argv)


if __name__ == "__main__":
    main()
