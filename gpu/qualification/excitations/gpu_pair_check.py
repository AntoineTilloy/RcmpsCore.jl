"""Qualification of the fused kernel on the packed diagonal bilinear system (packed_pair.py).

For a random white tangent W the quartic term of <Phi(W)|H|Phi(W)> is computed with the
fused kernel for each tile path (--tiles; default every path of the device: tensor-core and
scalar tiles from compute capability 8.0 on, scalar tiles below) and, at small D, with the
host-controlled NumPy integrator on the same packed system and on the unpacked four-block
system (compared with the first tile path). Every result is one JSON line on stdout.

    python gpu_pair_check.py --state STATE [--reference] [--window 30] [--n-quad 512] [--tiles mma scalar]
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
from gpu_bringup import device_tiles, emit, load, white_tangent


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
    values = {}
    for name in paths:
        integrate = fused.fused_integrator(cp, use_mma=fused.USE_MMA[name])
        started = time.perf_counter()
        values[name] = packed_pair.diagonal_observable(pair, spec, s.integrator, integrate)
        first = time.perf_counter() - started
        started = time.perf_counter()
        again = packed_pair.diagonal_observable(pair, spec, s.integrator, integrate)
        kernel = next(iter(integrate.kernels.values()))
        row[name] = dict(value=[values[name].real, values[name].imag], seconds=time.perf_counter() - started,
                         first_seconds=first, repeat_difference=abs(again - values[name]),
                         milliseconds=kernel.last["device_milliseconds"], attempts=kernel.last["attempts"],
                         rhs_calls=kernel.last["rhs_calls"], registers=kernel.occupancy["registers"],
                         blocks=kernel.occupancy["blocks"], capacity=kernel.occupancy["resident_capacity"])
    if len(values) == 2:
        row["mma_vs_scalar"] = abs(values["mma"] - values["scalar"])
    first = values[paths[0]]
    if args.reference:
        started = time.perf_counter()
        packed_value = packed_pair.diagonal_observable(pair, spec, s.integrator, ode.integrate_host)
        row["host_packed"] = dict(value=[packed_value.real, packed_value.imag], seconds=time.perf_counter() - started,
                                  relative_difference=abs(first - packed_value) / abs(packed_value))
        started = time.perf_counter()
        unpacked = element.observable(pair, spec)
        row["host_unpacked"] = dict(value=[unpacked.real, unpacked.imag], seconds=time.perf_counter() - started,
                                    relative_difference=abs(first - unpacked) / abs(unpacked))
    emit(**row)


if __name__ == "__main__":
    main()
