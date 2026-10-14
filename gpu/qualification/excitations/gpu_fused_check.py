"""Qualification of the fused trajectory kernel on a GPU.

For each case the Z2-packed action system of one tangent is integrated three ways:
with the host-controlled NumPy integrator (ode.integrate_host, exact source), with
the fused kernel using native FP64 tensor-core products, and with the fused kernel
using scalar tiles. The samples at the quadrature nodes and the resulting action
are compared. Every result is one JSON line on stdout.

    python gpu_fused_check.py [--state STATE] [--random D [D ...]] [--window 16] [--n-quad 64]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coordinates  # noqa: E402
import fused  # noqa: E402
import kinetic  # noqa: E402
import models  # noqa: E402
import ode  # noqa: E402
import packed  # noqa: E402
import potential  # noqa: E402
import scaled  # noqa: E402
import states  # noqa: E402


def emit(**row):
    print(json.dumps(row, default=float), flush=True)


def tangent(ss, sector, momentum, seed):
    d = ss.D
    h = d // 2
    rng = np.random.default_rng(seed)
    m = np.zeros((d, d), dtype=np.complex128)

    def block():
        return rng.standard_normal((h, h)) + 1j * rng.standard_normal((h, h))

    if sector == "odd":
        m[:h, :h], m[h:, h:] = block(), block()
    else:
        m[:h, h:], m[h:, :h] = block(), block()
    m /= np.linalg.norm(m)
    frame = kinetic.TangentFrame(ss, momentum, scaled.stack([m]), tol=1e-13)
    return tuple(np.ascontiguousarray(x[:, 0, :]) for x in (frame.wh, frame.vh, frame.rw))


def check(cp, state, label, args, sector, momentum, scaling):
    ss = scaled.ScaledState(state, scaling)
    spec = models.monomial_spec(4)
    system = packed.PackedActionSystem(ss, momentum, spec, tangent(ss, sector, momentum, 5))
    points, weights = models.quadrature(state.integrator)
    endpoint = math.sqrt(state.integrator.max_point)
    row = dict(stage="fused", state=label, D=state.D, sector=sector, momentum=momentum, scaling=scaling,
               blocks=len(system.blocks))
    reference = None
    if state.D <= args.reference_limit:
        started, statistics = time.perf_counter(), {}
        reference = ode.integrate_host(system, points, endpoint, state.integrator, statistics=statistics)
        row.update(host_seconds=time.perf_counter() - started, host_attempts=statistics["attempts"])
    results = {}
    for use_mma in (True, False):
        name = "mma" if use_mma else "scalar"
        try:
            trajectory = fused.FusedTrajectory(cp, system.d, len(system.blocks), use_mma=use_mma)
            samples = trajectory.integrate(system, points, endpoint, state.integrator)
            again = trajectory.integrate(system, points, endpoint, state.integrator)   # warm timing
            results[name] = cp.asnumpy(again)
            last = trajectory.last
            row[name] = dict(milliseconds=last["device_milliseconds"], attempts=last["attempts"],
                             rejected=last["rejected"], dense_steps=last["dense_steps"], rhs_calls=last["rhs_calls"],
                             microseconds_per_rhs=1e3 * last["device_milliseconds"] / last["rhs_calls"],
                             registers=trajectory.occupancy["registers"],
                             repeat_difference=float(np.abs(cp.asnumpy(samples) - results[name]).max()))
            if reference is not None:
                scale = np.abs(reference).max()
                row[name]["max_difference_from_host"] = float(np.abs(results[name] - reference).max())
                row[name]["relative"] = float(np.abs(results[name] - reference).max() / scale)
                f_ref, a_ref, _ = system.pullback(reference, points, weights)
                f, a, _ = system.pullback(results[name], points, weights)
                row[name]["action_relative_difference"] = float(np.linalg.norm(f - f_ref) / np.linalg.norm(f_ref))
                row[name]["boundary_relative_difference"] = float(np.linalg.norm(a - a_ref) / max(np.linalg.norm(a_ref), 1e-300))
        except Exception as error:
            row[name] = dict(error=f"{type(error).__name__}: {error}"[:1500])
    if len(results) == 2:
        row["mma_vs_scalar"] = float(np.abs(results["mma"] - results["scalar"]).max())
    emit(**row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", nargs="*", default=[])
    parser.add_argument("--random", type=int, nargs="*", default=[4, 6, 8, 12])
    parser.add_argument("--window", type=float, default=16.0)
    parser.add_argument("--n-quad", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--reference-limit", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="one configuration per state")
    args = parser.parse_args()
    import cupy as cp
    integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window, atol=args.atol, rtol=args.rtol)
    cases = [(states.random_state(d, scaling=0.5, seed=17 + d, z2=True), f"random{d}") for d in args.random]
    for path in args.state:
        path = Path(path)
        loaded = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
        cases.append((coordinates.make_coordinates(loaded, sector="odd").state, path.name))
    for state, label in cases:
        state.integrator = integrator
        configurations = ([("odd", 0.0, "density")] if args.quick else
                          [("odd", 0.0, "none"), ("odd", 0.0, "density"), ("even", 0.4, "density")])
        for sector, momentum, scaling in configurations:
            check(cp, state, label, args, sector, momentum, scaling)


if __name__ == "__main__":
    main()
