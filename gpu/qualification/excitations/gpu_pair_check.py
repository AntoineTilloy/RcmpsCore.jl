"""Qualification of the fused kernel on the packed diagonal bilinear system (packed_pair.py).

For a random white tangent W the quartic term of <Phi(W)|H|Phi(W)> is computed twice with
the fused kernel for each tile path (--tiles; default every path of the device: tensor-core
and scalar tiles from compute capability 8.0 on, scalar tiles below) and, with --reference
(small D), with the host-controlled NumPy integrator on the same packed system and on the
unpacked four-block system. A path passes ("ok") when its two values have the same bytes
and, with --reference, differ from both host values by at most --tolerance (relative; 1e-9,
the bound of the optimizer's trajectory check; the A100 gave 4.4e-15 and 6.2e-15 at
D = 16). A kernel this device cannot hold is reported as refused, not failed. The two paths
are compared with each other as a measurement. Every result is one JSON line on stdout; the
exit status is 1 if a path failed.

    python gpu_pair_check.py --state STATE [--reference] [--window 30] [--n-quad 512] [--tiles mma scalar]
        [--tolerance 1e-9]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from rcmps_gpu.excitations import bilinear
from rcmps_gpu.excitations import fused
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations import packed_pair
from rcmps_gpu.excitations import scaled
from rcmps_gpu.excitations import states
from gpu_bringup import bitwise, describe, device_tiles, emit, finish, load, refusal, white_tangent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--scaling", default="density")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--n-quad", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--tiles", nargs="+", choices=fused.TILES, default=None,
                        help="tile paths (default: every path of the device)")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest relative difference from the host values (with --reference)")
    args = parser.parse_args()
    import cupy as cp
    paths, capability = device_tiles(cp, args.tiles)
    c = load(args.state)
    s = c.state
    s.integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window, atol=args.atol, rtol=args.rtol)
    spec = models.monomial_spec(4)
    host_state = scaled.ScaledState(s, args.scaling)
    wh, _ = white_tangent(c, host_state, 1)
    # tol: the packed system freezes the channel-0 environments, so its value follows their
    # Poisson residual; at 1e-12 that, not the kernel, would set the comparison below.
    element = bilinear.BilinearElement(host_state, args.momentum, [(1.0, spec)], integrator=s.integrator,
                                       integrate=ode.integrate_host, tol=1e-14)
    pair = element.pair(wh, wh)
    row = dict(stage="pair", D=s.D, scaling=args.scaling, momentum=args.momentum, overlap=complex(pair.overlap).real,
               compute_capability=capability, tiles=paths)
    values, attempts = {}, {}
    for name in paths:
        try:
            integrate = fused.fused_integrator(cp, use_mma=fused.USE_MMA[name])
            started = time.perf_counter()
            value = complex(packed_pair.diagonal_observable(pair, spec, s.integrator, integrate))
            first = time.perf_counter() - started
            started = time.perf_counter()
            again = complex(packed_pair.diagonal_observable(pair, spec, s.integrator, integrate))
            kernel = next(iter(integrate.kernels.values()))
        except Exception as error:
            message = refusal(error)
            row[name] = dict(refused=message, ok=True) if message else dict(error=describe(error), ok=False)
            continue
        values[name], attempts[name] = value, kernel.last["attempts"]
        row[name] = dict(value=[value.real, value.imag], seconds=time.perf_counter() - started,
                         first_seconds=first, repeat_difference=abs(again - value),
                         repeat_bitwise=bitwise(np.complex128(again), np.complex128(value)),
                         milliseconds=kernel.last["device_milliseconds"], attempts=kernel.last["attempts"],
                         rhs_calls=kernel.last["rhs_calls"], registers=kernel.occupancy["registers"],
                         blocks=kernel.occupancy["blocks"], capacity=kernel.occupancy["resident_capacity"],
                         compiler_options=kernel.occupancy["compiler_options"])
    if len(values) == 2:
        row["mma_vs_scalar"] = abs(values["mma"] - values["scalar"])
        row["mma_vs_scalar_bitwise"] = bitwise(np.complex128(values["mma"]), np.complex128(values["scalar"]))
        row["mma_vs_scalar_same_steps"] = attempts["mma"] == attempts["scalar"]
    references = {}
    if args.reference:
        started = time.perf_counter()
        references["host_packed"] = complex(packed_pair.diagonal_observable(pair, spec, s.integrator, ode.integrate_host))
        row["host_packed"] = dict(value=[references["host_packed"].real, references["host_packed"].imag],
                                  seconds=time.perf_counter() - started)
        started = time.perf_counter()
        references["host_unpacked"] = complex(element.observable(pair, spec))
        row["host_unpacked"] = dict(value=[references["host_unpacked"].real, references["host_unpacked"].imag],
                                    seconds=time.perf_counter() - started)
        if values:      # the relative differences of the first tile path, as before; every path's below
            for key, reference in references.items():
                row[key]["relative_difference"] = abs(values[next(iter(values))] - reference) / abs(reference)
    for name, value in values.items():
        failed = [] if row[name]["repeat_bitwise"] else ["repeat_bitwise"]
        for key, reference in references.items():
            row[name][f"relative_difference_from_{key}"] = difference = abs(value - reference) / abs(reference)
            if not difference <= args.tolerance:        # NaN fails
                failed.append(f"relative_difference_from_{key}")
        row[name].update(failed_checks=failed, ok=not failed)
    emit(**row, tolerance=args.tolerance, ok=all(row[name]["ok"] for name in paths))
    finish()


if __name__ == "__main__":
    main()
