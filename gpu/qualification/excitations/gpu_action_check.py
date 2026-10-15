"""Complete Hamiltonian action on a GPU with the fused trajectory kernel: accuracy and timing.

    python gpu_action_check.py --state STATE [--reference] [--scaling density] [--momentum 0] [--tiles auto]
        [--tolerance 1e-9]
With --reference the same action is recomputed on the CPU with NumPy, the unpacked
equations and the host-controlled integrator (half a minute to minutes at D=16; do not use
beyond D=24). The check passes ("ok") when the repeated action of the first tangent differs
from the first one by at most --tolerance (relative; also reported: whether the bytes are
the same) and, with --reference, the GPU action differs from the CPU one by at most
--tolerance (relative; 1e-9, the bound of the optimizer's trajectory check; the A100 gave
3.1e-11 at D = 16). Every result is one JSON line on stdout; the exit status is 1 if a check
failed.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rcmps_gpu.excitations import coordinates
from rcmps_gpu.excitations import fused
from rcmps_gpu.excitations import hamiltonian
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations import packed
from rcmps_gpu.excitations import scaled
from rcmps_gpu.excitations import states
from gpu_bringup import bitwise, emit, finish, host, load, white_tangent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--scaling", default="density")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--n-quad", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--basis-bytes", type=float, default=6e9)
    parser.add_argument("--tiles", choices=fused.TILE_CHOICES, default="auto", help="tile path of the fused kernel")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="largest relative difference of a repeated action and from the CPU reference")
    args = parser.parse_args()
    import cupy as cp
    c = load(args.state)
    s = c.state
    s.integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window, atol=args.atol, rtol=args.rtol)
    terms = models.phi4_terms(s.coupling)
    started = time.perf_counter()
    ss = scaled.ScaledState(s, args.scaling, xp=cp, max_basis_bytes=args.basis_bytes)
    integrate = fused.fused_integrator(cp, use_mma=fused.USE_MMA[args.tiles])
    action = hamiltonian.HamiltonianAction(ss, args.momentum, terms, integrator=s.integrator, kinetic_nodes=args.nodes,
                                           tol=args.tol, integrate=integrate, system=packed.PackedActionSystem)
    cp.cuda.get_current_stream().synchronize()
    setup = time.perf_counter() - started
    wh1, x1 = white_tangent(c, ss, 1)
    wh2, x2 = white_tangent(c, ss, 2)
    timings, results = [], []
    for wh in [wh1, wh2] + [wh1] * max(0, args.repeat - 1):
        started = time.perf_counter()
        f = action(cp.asarray(wh))
        cp.cuda.get_current_stream().synchronize()
        timings.append(time.perf_counter() - started)
        results.append(host(f))
    kernel = next(iter(integrate.kernels.values()))
    lam = np.sqrt(np.real(np.diag(s.rho)))
    white = [(f * (ss.s_host / lam)[None, :])[c.rows, c.cols] for f in results[:2]]
    v1, v2 = x1[c.rows, c.cols], x2[c.rows, c.cols]
    h12, h21 = np.vdot(v1, white[1]), np.vdot(v2, white[0])
    repeat = float(np.linalg.norm(results[-1] - results[0]) / np.linalg.norm(results[0])) if args.repeat > 1 else None
    emit(stage="action", D=s.D, scaling=args.scaling, momentum=args.momentum, setup_seconds=setup,
         action_seconds=timings, trajectory=kernel.last, registers=kernel.occupancy["registers"],
         tiles=kernel.tiles, tiles_requested=kernel.tiles_requested,
         compiler_options=kernel.occupancy["compiler_options"], compute_capability=kernel.occupancy["compute_capability"],
         kinetic_stats=action.kinetic.stats, rayleigh_1=complex(np.vdot(v1, white[0])).real,
         imaginary_1=complex(np.vdot(v1, white[0])).imag, hermiticity_defect=abs(h12 - np.conj(h21)),
         repeat_difference=repeat, repeat_bitwise=bitwise(results[-1], results[0]) if args.repeat > 1 else None,
         tolerance=args.tolerance, ok=repeat is None or repeat <= args.tolerance)
    if args.reference:
        started = time.perf_counter()
        reference_state = scaled.ScaledState(s, args.scaling)
        reference = hamiltonian.HamiltonianAction(reference_state, args.momentum, terms, integrator=s.integrator,
                                                  kinetic_nodes=args.nodes, tol=args.tol, integrate=ode.integrate_host)
        f = reference(wh1)
        difference = float(np.linalg.norm(results[0] - f) / np.linalg.norm(f))
        emit(stage="action_reference", D=s.D, seconds=time.perf_counter() - started, relative_difference=difference,
             white_difference=float(np.linalg.norm(white[0] - (f * (ss.s_host / lam)[None, :])[c.rows, c.cols])),
             tolerance=args.tolerance, ok=difference <= args.tolerance)
    finish()


if __name__ == "__main__":
    main()
