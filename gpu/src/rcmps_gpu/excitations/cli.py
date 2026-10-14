"""Tangent-space excitation energies of an RCMPS ground state on a GPU (or the CPU).

    python run.py STATE --output DIR [--momentum 0.0] [--nev 1] [--sector auto] [--backend fused]

STATE is an rcmps_store checkpoint (.jld2) or a portable export directory of the GPU
optimizer (R.bin, K.bin, rho.bin, metadata.toml). The model is phi^4 at the coupling
stored with the state. The output directory must be new; it receives result.json
(energies, validation, history, timings, solver statistics, settings and source hashes),
vectors.npz (white coefficients and tangents W in the gauge of the state), log.jsonl and,
during the search, checkpoint.npz (current Ritz vectors after every action). `--start FILE`
restarts from either file, for the same state and sector: from the tangents W of a
vectors.npz, or from the white coefficients of a checkpoint.npz. The coefficients refer
to the eigenvectors of rho as computed in that run, so a checkpoint is to be reused in
the same software environment and with the same metric cutoff.

Backends: `fused` (GPU, one cooperative FP64 kernel per trajectory; Z2 states and one
parity sector only, so `--sector all` is refused), `host` (host-controlled integrator on
the device; slow, any state and sector) and `numpy` (CPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eigsolve  # noqa: E402
import models  # noqa: E402
import ode  # noqa: E402
import states  # noqa: E402
from davidson import DavidsonOptions  # noqa: E402

SOURCES = ("states.py", "models.py", "coordinates.py", "davidson.py", "scaled.py", "kinetic.py", "potential.py",
           "ode.py", "hamiltonian.py", "bilinear.py", "bilinear_fused.py", "packed.py", "packed_pair.py", "fused.py",
           "eigsolve.py", "run.py")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {k: jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return repr(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state")
    parser.add_argument("--output", required=True)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--nev", type=int, default=1)
    parser.add_argument("--sector", default="auto", choices=("auto", "odd", "even", "all"))
    parser.add_argument("--backend", default="fused", choices=("fused", "host", "numpy"))
    parser.add_argument("--scaling", default="density", choices=("density", "none"))
    parser.add_argument("--metric-tol", type=float, default=0.0)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--n-quad", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--kinetic-nodes", type=int, default=256)
    parser.add_argument("--preconditioner-nodes", type=int, default=8)
    parser.add_argument("--preconditioner-tol", type=float, default=1e-10)
    parser.add_argument("--solver-tol", type=float, default=1e-14,
                        help="relative residual of the GMRES solves of the actions (energies inherit about "
                             "tol * ||H||, so 1e-12 already costs 1e-8 at D=64)")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-7, help="Davidson residual target")
    parser.add_argument("--maxops", type=int, default=200)
    parser.add_argument("--maxspace", type=int, default=None)
    parser.add_argument("--retain", type=int, default=None, help="Ritz vectors kept at a restart (default nev)")
    parser.add_argument("--thick-restart", action="store_true",
                        help="recombine stored actions at a restart instead of recomputing them")
    parser.add_argument("--start", default=None,
                        help="vectors.npz or checkpoint.npz of an earlier run on the same state and sector")
    parser.add_argument("--stall-residual-limit", type=float, default=None)
    parser.add_argument("--stall-energy-atol", type=float, default=1e-9)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--refined-window", type=float, default=None)
    parser.add_argument("--refined-n-quad", type=int, default=None)
    parser.add_argument("--refined-atol", type=float, default=None)
    parser.add_argument("--refined-rtol", type=float, default=None)
    parser.add_argument("--refined-kinetic-nodes", type=int, default=None)
    parser.add_argument("--basis-bytes", type=float, default=6e9, help="bound on one GMRES Krylov basis (GPU)")
    parser.add_argument("--scalar-tiles", action="store_true", help="fused kernel without tensor-core products")
    parser.add_argument("--host-bilinear", action="store_true",
                        help="bilinear validation ODEs on the host-controlled integrator (reference, slow)")
    args = parser.parse_args()
    if args.backend == "fused" and args.sector == "all":
        # packed.PackedActionSystem needs tangents of one Z2 class; the search would fail at its second action.
        parser.error("--backend fused integrates one parity sector (odd or even) of a Z2 state; "
                     "use --backend host or numpy for --sector all")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    log_file = (output / "log.jsonl").open("w")
    started = time.time()

    def log(message):
        row = dict(seconds=round(time.time() - started, 3), message=message)
        print(json.dumps(row), flush=True)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

    path = Path(args.state)
    state = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
    state.integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window,
                                                   atol=args.atol, rtol=args.rtol)
    if state.model != "phi4":
        raise SystemExit(f"only the phi4 model is supported, not {state.model!r}")
    terms = models.phi4_terms(state.coupling)
    device = bilinear_element = None
    if args.backend == "numpy":
        xp, integrate, system, basis_bytes = np, ode.integrate_host, None, None
    else:
        import cupy as xp
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        device = dict(compute_capability=[int(props["major"]), int(props["minor"])],
                      multiprocessors=int(props["multiProcessorCount"]), memory=int(xp.cuda.Device().mem_info[1]),
                      cupy=xp.__version__, driver=int(xp.cuda.runtime.driverGetVersion()))
        basis_bytes = args.basis_bytes
        if args.backend == "fused":
            import fused
            import packed
            integrate, system = fused.fused_integrator(xp, use_mma=not args.scalar_tiles), packed.PackedActionSystem
            if not args.host_bilinear:
                # The bilinear validation energies also go through the fused kernel.
                import bilinear_fused
                bilinear_element = bilinear_fused.bound(integrate)
        else:
            integrate, system = ode.integrate_host, None
    refined = None
    if any(v is not None for v in (args.refined_window, args.refined_n_quad, args.refined_atol, args.refined_rtol)):
        base = eigsolve.refined_integration(state.integrator)
        refined = states.IntegratorParameters(
            n_quad=args.refined_n_quad or base.n_quad, max_point=args.refined_window or base.max_point,
            atol=args.refined_atol or base.atol, rtol=args.refined_rtol or base.rtol)
    options = DavidsonOptions(tol=args.tol, maxops=args.maxops,
                              maxspace=args.maxspace or max(8, 2 * args.nev + 2), retain=args.retain,
                              refresh=not args.thick_restart, stall_residual_limit=args.stall_residual_limit,
                              stall_energy_atol=args.stall_energy_atol)
    settings = dict(vars(args), coupling=state.coupling, D=state.D, saved_energy=state.e0, device=device,
                    numpy=np.__version__,
                    source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES})
    log(f"state {path.name} D={state.D} g={state.coupling} backend={args.backend} scaling={args.scaling}")
    if state.D > 64 and args.kinetic_nodes < 512:
        # Measured on the g = 2.7 states (README): the 256-node resolvent quadrature is off by
        # 2e-7 in the action at D = 96 and 1e-5 at D = 128; 512 nodes are converged to 3e-10.
        log(f"warning: kinetic_nodes={args.kinetic_nodes} is not converged beyond D = 64; use --kinetic-nodes 512")
    start = "auto"
    if args.start is not None:
        with np.load(args.start) as stored:
            start = ([np.array(w) for w in stored["vectors"]] if "vectors" in stored
                     else [np.array(x) for x in np.atleast_2d(stored["coefficients"])])
        log(f"start from {args.start} ({len(start)} vectors)")

    def checkpoint(row, u, au):
        # Current Ritz vectors, so that a failure after the search (validation) loses nothing.
        temporary = output / "checkpoint.tmp.npz"
        np.savez(temporary, coefficients=u.T, energies=np.array(row["energies"]),
                 residuals=np.array(row["residuals"]), count=row["count"])
        temporary.replace(output / "checkpoint.npz")

    result = eigsolve.excitation_eigsolve(
        state, args.momentum, terms, nev=args.nev, sector=args.sector, scaling=args.scaling,
        metric_tol=args.metric_tol, kinetic_nodes=args.kinetic_nodes,
        preconditioner_nodes=args.preconditioner_nodes, preconditioner_tol=args.preconditioner_tol,
        davidson_options=options, validate=not args.no_validate, refined=refined,
        refined_kinetic_nodes=args.refined_kinetic_nodes, solver_tol=args.solver_tol, xp=xp,
        integrate=integrate, system=system, bilinear_element=bilinear_element, chunk=args.chunk, log=log,
        callback=checkpoint, start=start, max_basis_bytes=basis_bytes)
    arrays = dict(coefficients=np.asarray(result["coefficients"]), vectors=np.asarray(result["vectors"]))
    np.savez(output / "vectors.npz", **arrays)
    summary = {k: v for k, v in result.items() if k not in ("vectors", "coefficients", "coordinates", "basis")}
    if args.backend == "fused":
        summary["kernels"] = {"_".join(map(str, key)): dict(occupancy=k.occupancy, last=k.last)
                              for key, k in integrate.kernels.items()}
    summary.update(settings=settings, wall_seconds=time.time() - started)
    (output / "result.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    log(f"energies {result['energies']} validated {result.get('validated_energies')} accepted {result['accepted']} "
        f"stop {result['stop_reason']} numops {result['numops']}")
    log_file.close()


if __name__ == "__main__":
    main()
