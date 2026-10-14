"""Sensitivity of the Hamiltonian action of a stored excitation vector to the discretization.

The action is the sum of a resolvent part (kinetic terms, theta quadrature with a fixed
number of nodes) and an ODE part (phi^4, quadrature of the trajectory over a window).
Each part is evaluated on the stored vector for several settings and compared with its
finest one, so that an unconverged setting shows up as a difference of the action (and
as a fresh residual of the vector) before it shows up in the energy. Every result is
one JSON line on stdout.

    python gpu_settings_check.py STATE VECTORS.npz [--kinetic-nodes 256 512 1024]
        [--ode 512:30:1e-13:1e-12 1024:40:1e-14:1e-13 2048:40:1e-14:1e-13]

VECTORS.npz is the vectors.npz of run.py for the same state and sector; an --ode entry is
n_quad:window:atol:rtol. The last entry of each list is the reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eigsolve  # noqa: E402
import fused  # noqa: E402
import hamiltonian  # noqa: E402
import kinetic  # noqa: E402
import models  # noqa: E402
import packed  # noqa: E402
import states  # noqa: E402


def emit(**row):
    print(json.dumps(row, default=float), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state")
    parser.add_argument("vectors")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--sector", default="auto")
    parser.add_argument("--kinetic-nodes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--ode", nargs="+",
                        default=["512:30:1e-13:1e-12", "1024:30:1e-13:1e-12", "1024:40:1e-14:1e-13",
                                 "2048:40:1e-14:1e-13"])
    parser.add_argument("--tol", type=float, default=1e-14)
    parser.add_argument("--basis-bytes", type=float, default=6e9)
    args = parser.parse_args()
    import cupy as cp

    path = Path(args.state)
    state = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
    basis, _ = eigsolve.make_basis(state, {}, sector=args.sector, scaling="density", xp=cp,
                                   max_basis_bytes=args.basis_bytes)
    ss = basis.ss
    with np.load(args.vectors) as stored:
        x = basis.coordinates(basis.hat(np.array(stored["vectors"][0])))
    x = x / np.linalg.norm(x)
    wh = basis.tangent(x)
    terms = models.phi4_terms(state.coupling)
    resolved = kinetic.kinetic_terms(terms)
    others = [term for term in terms if term not in resolved]
    integrate = fused.fused_integrator(cp)
    emit(stage="context", D=state.D, dimension=len(basis), momentum=args.momentum,
         density_condition=basis.chart.density_condition, tol=args.tol)

    def part(name, setting, build):
        started = time.perf_counter()
        action = build()
        cp.cuda.get_current_stream().synchronize()
        setup = time.perf_counter() - started
        started = time.perf_counter()
        f = basis.covector(action(wh))
        seconds = time.perf_counter() - started
        value = np.vdot(x, f)
        emit(stage=name, setting=setting, energy=value.real, imaginary=value.imag, norm=float(np.linalg.norm(f)),
             setup_seconds=setup, action_seconds=seconds)
        return f

    resolvent = {}
    for nodes in args.kinetic_nodes:
        resolvent[nodes] = part("resolvent", nodes, lambda: hamiltonian.HamiltonianAction(
            ss, args.momentum, resolved, integrator=state.integrator, kinetic_nodes=nodes, tol=args.tol))
    ode = {}
    for entry in args.ode:
        n_quad, window, atol, rtol = entry.split(":")
        integrator = states.IntegratorParameters(n_quad=int(n_quad), max_point=float(window), atol=float(atol),
                                                 rtol=float(rtol))
        ode[entry] = part("ode", entry, lambda: hamiltonian.HamiltonianAction(
            ss, args.momentum, others, integrator=integrator, tol=args.tol, integrate=integrate,
            system=packed.PackedActionSystem))
    finest = (args.kinetic_nodes[-1], args.ode[-1])
    for name, table, reference in (("resolvent", resolvent, finest[0]), ("ode", ode, finest[1])):
        for setting, f in table.items():
            d = f - table[reference]
            emit(stage=f"{name}_difference", setting=setting, reference=reference, action=float(np.linalg.norm(d)),
                 energy=float(np.vdot(x, d).real))
    for nodes, fk in resolvent.items():
        for entry, fp in ode.items():
            f = fk + fp
            energy = float(np.vdot(x, f).real)
            emit(stage="total", kinetic_nodes=nodes, ode=entry, energy=energy,
                 residual=float(np.linalg.norm(f - energy * x)))


if __name__ == "__main__":
    main()
