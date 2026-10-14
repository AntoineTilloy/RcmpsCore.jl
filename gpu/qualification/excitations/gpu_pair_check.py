"""Qualification of the fused kernel on the packed diagonal bilinear system (packed_pair.py).

For a random white tangent W the quartic term of <Phi(W)|H|Phi(W)> is computed with the
fused kernel (tensor-core and scalar tiles) and, at small D, with the host-controlled
NumPy integrator on the same packed system and on the unpacked four-block system.
Every result is one JSON line on stdout.

    python gpu_pair_check.py --state STATE [--reference] [--window 30] [--n-quad 512]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bilinear  # noqa: E402
import fused  # noqa: E402
import models  # noqa: E402
import ode  # noqa: E402
import packed_pair  # noqa: E402
import scaled  # noqa: E402
import states  # noqa: E402
from gpu_bringup import emit, load, white_tangent  # noqa: E402


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
    args = parser.parse_args()
    import cupy as cp
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
    row = dict(stage="pair", D=s.D, scaling=args.scaling, momentum=args.momentum, overlap=complex(pair.overlap).real)
    values = {}
    for use_mma in (True, False):
        name = "mma" if use_mma else "scalar"
        integrate = fused.fused_integrator(cp, use_mma=use_mma)
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
    row["mma_vs_scalar"] = abs(values["mma"] - values["scalar"])
    if args.reference:
        started = time.perf_counter()
        packed_value = packed_pair.diagonal_observable(pair, spec, s.integrator, ode.integrate_host)
        row["host_packed"] = dict(value=[packed_value.real, packed_value.imag], seconds=time.perf_counter() - started,
                                  relative_difference=abs(values["mma"] - packed_value) / abs(packed_value))
        started = time.perf_counter()
        unpacked = element.observable(pair, spec)
        row["host_unpacked"] = dict(value=[unpacked.real, unpacked.imag], seconds=time.perf_counter() - started,
                                    relative_difference=abs(values["mma"] - unpacked) / abs(unpacked))
    emit(**row)


if __name__ == "__main__":
    main()
