"""Qualification of the fused trajectory kernel on a GPU.

For each case the Z2-packed action system of one tangent (for a state without symmetry,
--general, the unpacked system of the sector all, packed.UnpackedActionSystem) is integrated
with the host-controlled NumPy integrator (ode.integrate_host, exact source) and with the
fused kernel for each tile path (--tiles; default every path of the device: native FP64
tensor-core products and scalar tiles from compute capability 8.0 on, scalar tiles below),
twice. A path passes ("ok") when its two launches give the same bytes and, against the host
integrator (D <= --reference-limit), the samples at the quadrature nodes and the resulting
action differ by at most --tolerance (relative; 1e-9, the bound of the optimizer's
trajectory check: rounding can move a step, a bug cannot stay below it; the A100 gave 5e-12
and 4e-11 on the D = 16 fixture). A kernel this device cannot hold is reported as refused,
not failed. The two paths are compared with each other (largest difference, same bytes,
same steps) as a measurement. Every result is one JSON line on stdout; the exit status is 1
if a path failed.

    python gpu_fused_check.py [--state STATE] [--random D [D ...]] [--general D [D ...]] [--window 16]
        [--n-quad 64] [--tiles mma scalar] [--tolerance 1e-9]
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from rcmps_gpu.excitations import coordinates
from rcmps_gpu.excitations import fused
from rcmps_gpu.excitations import kinetic
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations import packed
from rcmps_gpu.excitations import potential
from rcmps_gpu.excitations import scaled
from rcmps_gpu.excitations import states
from gpu_bringup import bitwise, describe, device_tiles, emit, finish, refusal


def tangent(ss, sector, momentum, seed):
    d = ss.D
    h = d // 2
    rng = np.random.default_rng(seed)
    m = np.zeros((d, d), dtype=np.complex128)

    def block():
        return rng.standard_normal((h, h)) + 1j * rng.standard_normal((h, h))

    if sector == "all":
        m = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    elif sector == "odd":
        m[:h, :h], m[h:, h:] = block(), block()
    else:
        m[:h, h:], m[h:, :h] = block(), block()
    m /= np.linalg.norm(m)
    frame = kinetic.TangentFrame(ss, momentum, scaled.stack([m]), tol=1e-13)
    return tuple(np.ascontiguousarray(x[:, 0, :]) for x in (frame.wh, frame.vh, frame.rw))


def check(cp, state, label, args, sector, momentum, scaling, paths):
    ss = scaled.ScaledState(state, scaling)
    spec = models.monomial_spec(4)
    layout = packed.UnpackedActionSystem if sector == "all" else packed.PackedActionSystem
    system = layout(ss, momentum, spec, tangent(ss, sector, momentum, 5))
    points, weights = models.quadrature(state.integrator)
    endpoint = math.sqrt(state.integrator.max_point)
    row = dict(stage="fused", state=label, D=state.D, sector=sector, momentum=momentum, scaling=scaling,
               blocks=len(system.blocks))
    reference = None
    if state.D <= args.reference_limit:
        started, statistics = time.perf_counter(), {}
        reference = ode.integrate_host(system, points, endpoint, state.integrator, statistics=statistics)
        row.update(host_seconds=time.perf_counter() - started, host_attempts=statistics["attempts"])
    results, attempts = {}, {}
    for name in paths:
        try:
            trajectory = fused.FusedTrajectory(cp, system.d, len(system.blocks), use_mma=fused.USE_MMA[name])
        except Exception as error:
            message = refusal(error)
            row[name] = dict(refused=message, ok=True) if message else dict(error=describe(error), ok=False)
            continue
        try:
            # integrate returns the kernel's sample buffer, which the next launch overwrites: copy it first.
            first = cp.asnumpy(trajectory.integrate(system, points, endpoint, state.integrator))
            first_attempts = trajectory.last["attempts"]
            again = cp.asnumpy(trajectory.integrate(system, points, endpoint, state.integrator))   # warm timing
            results[name], last = again, trajectory.last
            attempts[name] = last["attempts"]
            entry = dict(milliseconds=last["device_milliseconds"], attempts=last["attempts"],
                         rejected=last["rejected"], dense_steps=last["dense_steps"], rhs_calls=last["rhs_calls"],
                         microseconds_per_rhs=1e3 * last["device_milliseconds"] / last["rhs_calls"],
                         registers=trajectory.occupancy["registers"],
                         compiler_options=trajectory.occupancy["compiler_options"],
                         repeat_difference=float(np.abs(first - again).max()), repeat_bitwise=bitwise(first, again),
                         repeat_same_steps=first_attempts == last["attempts"])
            failed = [] if entry["repeat_bitwise"] else ["repeat_bitwise"]
            if reference is not None:
                scale = np.abs(reference).max()
                entry["same_steps_as_host"] = last["attempts"] == row["host_attempts"]
                entry["max_difference_from_host"] = float(np.abs(again - reference).max())
                entry["relative"] = float(np.abs(again - reference).max() / scale)
                f_ref, a_ref, _ = system.pullback(reference, points, weights)
                f, a, _ = system.pullback(again, points, weights)
                entry["action_relative_difference"] = float(np.linalg.norm(f - f_ref) / np.linalg.norm(f_ref))
                entry["boundary_relative_difference"] = float(np.linalg.norm(a - a_ref) / max(np.linalg.norm(a_ref), 1e-300))
                failed += [key for key in ("relative", "action_relative_difference")
                           if not entry[key] <= args.tolerance]     # NaN fails
            entry.update(failed_checks=failed, ok=not failed)
        except Exception as error:
            entry = dict(error=describe(error), ok=False)
        row[name] = entry
    if len(results) == 2:
        row["mma_vs_scalar"] = float(np.abs(results["mma"] - results["scalar"]).max())
        row["mma_vs_scalar_bitwise"] = bitwise(results["mma"], results["scalar"])
        row["mma_vs_scalar_same_steps"] = attempts["mma"] == attempts["scalar"]
    emit(**row, tolerance=args.tolerance, ok=all(row[name]["ok"] for name in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", nargs="*", default=[])
    parser.add_argument("--random", type=int, nargs="*", default=[4, 6, 8, 12])
    parser.add_argument("--general", type=int, nargs="*", default=[],
                        help="bond dimensions of random states without symmetry, checked in the sector all")
    parser.add_argument("--window", type=float, default=16.0)
    parser.add_argument("--n-quad", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--reference-limit", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest relative difference from the host integrator (samples and action)")
    parser.add_argument("--quick", action="store_true", help="one configuration per state")
    parser.add_argument("--tiles", nargs="+", choices=fused.TILES, default=None,
                        help="tile paths (default: every path of the device)")
    args = parser.parse_args()
    import cupy as cp
    paths, capability = device_tiles(cp, args.tiles)
    emit(stage="device", compute_capability=capability, tiles=paths)
    integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window, atol=args.atol, rtol=args.rtol)
    cases = [(states.random_state(d, scaling=0.5, seed=17 + d, z2=True), f"random{d}") for d in args.random]
    for path in args.state:
        path = Path(path)
        loaded = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
        cases.append((coordinates.make_coordinates(loaded, sector="odd").state, path.name))
    cases += [(states.random_state(d, scaling=0.5, seed=17 + d), f"general{d}") for d in args.general]
    for state, label in cases:
        state.integrator = integrator
        sectors = ("all", "all") if label.startswith("general") else ("odd", "even")
        configurations = ([(sectors[0], 0.0, "density")] if args.quick else
                          [(sectors[0], 0.0, "none"), (sectors[0], 0.0, "density"), (sectors[1], 0.4, "density")])
        for sector, momentum, scaling in configurations:
            check(cp, state, label, args, sector, momentum, scaling, paths)
    finish()


if __name__ == "__main__":
    main()
