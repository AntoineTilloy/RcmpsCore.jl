"""Tangent-space excitation energies of an RCMPS ground state on a GPU (or the CPU).

    python -m rcmps_gpu.excitations STATE --output DIR [--momentum 0.0] [--nev 1] [--sector auto] [--backend fused]

STATE is an rcmps_store checkpoint (.jld2) or a portable state directory (rcmps_gpu.io,
format 1 or a research export). The model is phi^4 at the coupling stored with the
state. The output directory must be new; it receives result.json
(energies, validation, history, timings, solver statistics, settings and source hashes;
`settings` holds the options with the values used, kinetic_nodes, maxspace, retain and
thick_restart/fresh_restarts resolved, and `settings.arguments` the parsed command line),
vectors.npz (white coefficients and tangents W in the gauge of the state), log.jsonl and,
during the search, checkpoint.npz (current Ritz vectors after every action). `--start FILE`
restarts from either file, for the same state and sector: from the tangents W of a
vectors.npz, or from the white coefficients of a checkpoint.npz. The coefficients refer
to the eigenvectors of rho as computed in that run, so a checkpoint is to be reused in
the same software environment and with the same metric cutoff.

Backends: `fused` (GPU, one cooperative FP64 kernel per trajectory; Z2 states and one
parity sector only, so `--sector all` is refused), `host` (host-controlled integrator on
the device; slow, any state and sector) and `numpy` (CPU).

Tiles of the fused kernel. `--tiles auto` (the default) computes the matrix products on the
FP64 tensor cores from compute capability 8.0 on and with scalar tiles of the same layout
(warp shuffles, FP64 FMAs) below; `--tiles mma` is refused below 8.0, `--tiles scalar`
(also `--scalar-tiles`) runs on any GPU with cooperative launch. Before the output
directory is created, the kernels of the solve are compiled and their cooperative grids
checked: the action kernel, and the kernel of the bilinear validation unless --no-validate
or --host-bilinear (it needs more shared memory per block). result.json records the
resolved tiles (settings.tiles, settings.tiles_requested) and each kernel's occupancy.

Davidson schedule. The default is that of the A100 mass and sector runs of the README
(not of its comparison of scaled and unscaled variables, which used the CPU schedule), the
Davidson part of the Julia D <= 48 research configuration (docs/excitations.md): a search
space of maxspace = min(160, n) vectors, n the sector dimension, with thick restarts
(stored actions recombined) keeping retain = min(max(12, nev), maxspace - 1) Ritz vectors
(eigsolve.research_davidson_options). That is Julia's retain=12 wherever Julia accepts it
(nev <= 12, n >= 13); where Julia's check_davidson refuses retain 12, retain is nev
(nev > 12) or maxspace - 1 (n <= 12). n is computed before the output directory is
created, by eigsolve.sector_dimension with the --sector and --metric-tol of the solve, so
it is the dimension the solver uses. nev >= maxspace (for n <= 160: nev = n) leaves no
admissible retain and is a usage error. `--fresh-restarts` selects instead the library
schedule of the CPU code (maxspace max(8, 2 nev + 2), retain nev, every restart recomputes
the actions). `--maxspace` and `--retain` replace the respective rule of either schedule
(with only --maxspace, the research retain follows it). `--thick-restart` is the default
and still accepted, but an older command line (whose defaults were maxspace
max(8, 2 nev + 2), retain nev and fresh restarts) keeps its options only if it gave all of
--maxspace, --retain and --thick-restart, as the A100 runs did. One without --thick-restart
now needs --fresh-restarts; one with --thick-restart needs the --maxspace and --retain it
left out, at those former defaults. The stall rule is unchanged: the DavidsonOptions
default (residual limit max(1e-5, 10 tol), `--stall-energy-atol` 1e-9), as in the A100
runs, not the stall_residual_limit=1e-6 and stall_energy_atol=1e-10 of the Julia research
configuration.

Kinetic nodes. `--kinetic-nodes` defaults to eigsolve.default_kinetic_nodes(D): 256 up to
D = 64 and 512 beyond (256 nodes are off by 2e-7 at D = 96 and 1e-5 at D = 128 in the
resolvent part of the action; gpu/README.md, "Kinetic nodes at large D"). An explicit value below
that default is logged as a warning. Node counts were measured up to D = 128 only: beyond,
512 nodes are used and a note is logged; the refined check of the validation (twice the
nodes unless --refined-kinetic-nodes) is what would expose an unconverged quadrature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from .. import provenance  # noqa: E402
from . import eigsolve  # noqa: E402
from . import fused  # noqa: E402
from . import models  # noqa: E402
from . import ode  # noqa: E402
from . import states  # noqa: E402
from .davidson import check_davidson  # noqa: E402

SOURCES = ("states.py", "models.py", "coordinates.py", "davidson.py", "scaled.py", "kinetic.py", "potential.py",
           "ode.py", "hamiltonian.py", "bilinear.py", "bilinear_fused.py", "packed.py", "packed_pair.py", "fused.py",
           "eigsolve.py", "cli.py")
# The other files that run with them, relative to the rcmps_gpu package: the kernel's
# coefficient table, the state file reader, and the package modules on the import path.
SHARED_SOURCES = ("optimizer/runtime/jax_source.py", "__init__.py", "io.py", "provenance.py",
                  "excitations/__init__.py", "excitations/__main__.py", "excitations/provenance.json",
                  "optimizer/__init__.py", "optimizer/runtime/__init__.py")
# sha256 of json.dumps(table, indent=2, sort_keys=True) + newline for the table
# settings.source_sha256 of the A100 final2 result.json files (keyed run.py for cli.py),
# held by provenance.json as qualified.source_sha256.
QUALIFIED_SOURCES_SHA256 = "68c84931cca9aad270078af8e4a46ef7e8c4b680673a80b2fc2080429d5baa35"
MEASURED_D = 128   # largest D at which kinetic node counts were measured (gpu/README.md)


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


def source_provenance():
    """Whether SOURCES are the A100-qualified bytes with the edits recorded in provenance.json
    reverted; result.json records the answer (an unverified source is logged, not refused)."""
    raw = (HERE / "provenance.json").read_bytes()
    record = json.loads(raw)
    qualified = record["qualified"]
    result = dict(qualified_runs=qualified["runs"], provenance_sha256=hashlib.sha256(raw).hexdigest(),
                  edits=[entry["commit_subject"] for entry in record["edits"]])
    try:
        table = qualified["source_sha256"]
        if hashlib.sha256((json.dumps(table, indent=2, sort_keys=True) + "\n").encode()).hexdigest() \
                != QUALIFIED_SOURCES_SHA256:
            raise ValueError("provenance.json: the table of qualified sources changed")
        imported = {qualified["renamed"].get(name, name): digest for name, digest in table.items()}
        result.update(verified=True, edited_files=provenance.verify(HERE, imported, record["edits"], SOURCES))
    except (ValueError, KeyError) as error:
        result.update(verified=False, error=str(error))
    return result


def make_parser():
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
    parser.add_argument("--kinetic-nodes", type=int, default=None,
                        help="theta nodes of the kinetic resolvent quadrature (default: 256 for D <= 64, "
                             "512 beyond; eigsolve.default_kinetic_nodes)")
    parser.add_argument("--preconditioner-nodes", type=int, default=8)
    parser.add_argument("--preconditioner-tol", type=float, default=1e-10)
    parser.add_argument("--solver-tol", type=float, default=1e-14,
                        help="relative residual of the GMRES solves of the actions (energies inherit about "
                             "tol * ||H||, so 1e-12 already costs 1e-8 at D=64)")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-7, help="Davidson residual target")
    parser.add_argument("--maxops", type=int, default=200)
    parser.add_argument("--maxspace", type=int, default=None,
                        help="largest search space (default: min(160, n), n the sector dimension; with "
                             "--fresh-restarts max(8, 2 nev + 2))")
    parser.add_argument("--retain", type=int, default=None,
                        help="Ritz vectors kept at a restart (default: min(max(12, nev), maxspace - 1); with "
                             "--fresh-restarts nev)")
    restarts = parser.add_mutually_exclusive_group()
    restarts.add_argument("--thick-restart", action="store_true",
                          help="recombine stored actions at a restart (the default). An older command line keeps "
                               "its options only if it also gave --maxspace and --retain; formerly --thick-restart "
                               "alone meant maxspace max(8, 2 nev + 2) and retain nev")
    restarts.add_argument("--fresh-restarts", action="store_true",
                          help="library schedule of the CPU code: maxspace max(8, 2 nev + 2), retain nev, every "
                               "restart recomputes the actions")
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
    tiles = parser.add_mutually_exclusive_group()
    tiles.add_argument("--tiles", choices=fused.TILE_CHOICES, default=None,
                       help="tile products of the fused kernel: mma (FP64 tensor cores, compute capability >= 8.0), "
                            "scalar (warp shuffles and FP64 FMAs, any GPU with cooperative launch) or auto (the "
                            "default: mma from compute capability 8.0 on, scalar below)")
    tiles.add_argument("--scalar-tiles", action="store_true", help="the same as --tiles scalar")
    parser.add_argument("--host-bilinear", action="store_true",
                        help="bilinear validation ODEs on the host-controlled integrator (reference, slow)")
    return parser


def resolve_defaults(args, state):
    """Kinetic nodes, sector dimension n, Davidson options and the retain they imply
    (check_davidson: nev for retain None) of parsed arguments for a loaded state, with the
    log lines that report them; ValueError for a request that
    cannot run. n comes from eigsolve.sector_dimension with the sector and metric cutoff
    that main passes to excitation_eigsolve (default parity and gauge on both sides)."""
    d = state.D
    default_nodes = eigsolve.default_kinetic_nodes(d)
    nodes = default_nodes if args.kinetic_nodes is None else eigsolve.check_kinetic_nodes(args.kinetic_nodes)
    if args.refined_kinetic_nodes is not None:
        eigsolve.check_kinetic_nodes(args.refined_kinetic_nodes)
    n = eigsolve.sector_dimension(state, sector=args.sector, metric_tol=args.metric_tol)
    if not 1 <= args.nev <= n:
        raise ValueError(f"--nev {args.nev} must lie between 1 and the sector dimension {n}")
    controls = dict(tol=args.tol, maxops=args.maxops, stall_residual_limit=args.stall_residual_limit,
                    stall_energy_atol=args.stall_energy_atol)
    if args.fresh_restarts:
        schedule = "fresh-restarts"
        options = eigsolve.library_davidson_options(args.nev, maxspace=args.maxspace, retain=args.retain,
                                                    **controls)
    else:
        schedule = "research"
        options = eigsolve.research_davidson_options(n, args.nev, maxspace=args.maxspace, retain=args.retain,
                                                     **controls)
    retain = check_davidson(options, args.nev)
    refined_nodes = 2 * nodes if args.refined_kinetic_nodes is None else args.refined_kinetic_nodes
    messages = [f"kinetic_nodes={nodes} (default for D = {d})" if args.kinetic_nodes is None
                else f"kinetic_nodes={nodes} (given; the default for D = {d} is {default_nodes})"]
    if nodes < default_nodes:
        messages.append(f"warning: kinetic_nodes={nodes} is below the default {default_nodes} for D = {d}; "
                        f"measured: 256 nodes are converged up to D = 64, 512 up to D = 128 "
                        f"(gpu/README.md, 'Kinetic nodes at large D')")
    if d > MEASURED_D:
        messages.append(f"note: kinetic node counts are measured only up to D = {MEASURED_D}; at D = {d} the "
                        f"refined check of the validation ({refined_nodes} nodes) would expose an unconverged "
                        f"quadrature" + (", and --no-validate skips it" if args.no_validate else ""))
    messages.append(f"davidson schedule {schedule}: maxspace={options.maxspace} retain={retain} "
                    f"refresh={options.refresh} tol={options.tol:g} maxops={options.maxops} (sector dimension {n})")
    return dict(kinetic_nodes=nodes, default_kinetic_nodes=default_nodes, dimension=n, schedule=schedule,
                davidson=options, retain=retain, messages=messages)


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.backend == "fused" and args.sector == "all":
        # packed.PackedActionSystem needs tangents of one Z2 class; the search would fail at its second action.
        parser.error("--backend fused integrates one parity sector (odd or even) of a Z2 state; "
                     "use --backend host or numpy for --sector all")

    started = time.time()
    path = Path(args.state)
    state = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
    state.integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window,
                                                   atol=args.atol, rtol=args.rtol)
    if state.model != "phi4":
        raise SystemExit(f"only the phi4 model is supported, not {state.model!r}")
    try:
        # Before the output directory exists: an impossible schedule leaves nothing behind.
        resolved = resolve_defaults(args, state)
    except ValueError as error:
        parser.error(str(error))

    terms = models.phi4_terms(state.coupling)
    tiles = "scalar" if args.scalar_tiles else args.tiles or "auto"
    device = bilinear_element = None
    kernels = {}
    if args.backend == "numpy":
        xp, integrate, system, basis_bytes = np, ode.integrate_host, None, None
    else:
        import cupy as xp
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        device = dict(compute_capability=[int(props["major"]), int(props["minor"])],
                      multiprocessors=int(props["multiProcessorCount"]), memory=int(xp.cuda.Device().mem_info[1]),
                      cupy=xp.__version__, driver=int(xp.cuda.runtime.driverGetVersion()),
                      kernel_run_time_limit=bool(props.get("kernelExecTimeoutEnabled", False)))
        basis_bytes = args.basis_bytes
        if args.backend == "fused":
            from . import packed
            integrate, system = fused.fused_integrator(xp, use_mma=fused.USE_MMA[tiles]), packed.PackedActionSystem
            # Every kernel of the solve now, before the output directory exists: a GPU that
            # cannot hold one is refused before the search, not after it.
            names = ("action",) if args.no_validate or args.host_bilinear else ("action", "pair")
            try:
                kernels = fused.preflight(integrate, state.D // 2, names)
            except (ValueError, RuntimeError) as error:
                parser.error(f"{error}" + ("; --host-bilinear validates without it, on the host-controlled integrator"
                                           if str(error).startswith("the pair kernel") else ""))
            if not args.host_bilinear:
                # The bilinear validation energies also go through the fused kernel.
                from . import bilinear_fused
                bilinear_element = bilinear_fused.bound(integrate)
        else:
            integrate, system = ode.integrate_host, None

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    log_file = (output / "log.jsonl").open("w")

    def log(message):
        row = dict(seconds=round(time.time() - started, 3), message=message)
        print(json.dumps(row), flush=True)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

    refined = None
    if any(v is not None for v in (args.refined_window, args.refined_n_quad, args.refined_atol, args.refined_rtol)):
        base = eigsolve.refined_integration(state.integrator)
        refined = states.IntegratorParameters(
            n_quad=args.refined_n_quad or base.n_quad, max_point=args.refined_window or base.max_point,
            atol=args.refined_atol or base.atol, rtol=args.refined_rtol or base.rtol)
    options = resolved["davidson"]
    # The values used, as the result.json files of the A100 runs recorded them (integer node
    # count, thick_restart True for thick restarts); the raw arguments are kept under `arguments`.
    arguments = dict(vars(args))
    settings = dict(arguments, kinetic_nodes=resolved["kinetic_nodes"], maxspace=options.maxspace,
                    retain=resolved["retain"], thick_restart=not options.refresh, fresh_restarts=options.refresh,
                    arguments=arguments, coupling=state.coupling, D=state.D, saved_energy=state.e0, device=device,
                    tiles=kernels["action"].tiles if kernels else None,
                    tiles_requested=tiles if args.backend == "fused" else None,
                    numpy=np.__version__, davidson_schedule=resolved["schedule"],
                    sector_dimension=resolved["dimension"], default_kinetic_nodes=resolved["default_kinetic_nodes"],
                    source_sha256={**{name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES},
                                   **{name: hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest()
                                      for name in SHARED_SOURCES}},
                    source_provenance=source_provenance())
    log(f"state {path.name} D={state.D} g={state.coupling} backend={args.backend} scaling={args.scaling}")
    if not settings["source_provenance"]["verified"]:
        log(f"warning: the sources are not the qualified ones with their recorded edits: "
            f"{settings['source_provenance']['error']}")
    for message in resolved["messages"]:
        log(message)
    for name, kernel in kernels.items():
        o = kernel.occupancy
        log(f"{name} kernel: tiles {o['tiles']} (requested {o['tiles_requested']}, compute capability "
            f"{'.'.join(map(str, o['compute_capability']))}), {o['blocks']} thread blocks of {o['resident_capacity']} "
            f"resident, {o['registers']} registers, options {' '.join(o['compiler_options'])}")
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
        metric_tol=args.metric_tol, kinetic_nodes=resolved["kinetic_nodes"],
        preconditioner_nodes=args.preconditioner_nodes, preconditioner_tol=args.preconditioner_tol,
        davidson_options=options, validate=not args.no_validate, refined=refined,
        refined_kinetic_nodes=args.refined_kinetic_nodes, solver_tol=args.solver_tol, xp=xp,
        integrate=integrate, system=system, bilinear_element=bilinear_element, chunk=args.chunk, log=log,
        callback=checkpoint, start=start, max_basis_bytes=basis_bytes)
    if result["dimension"] != resolved["dimension"]:
        # Not expected: both come from the same make_coordinates call on the same state.
        log(f"warning: the solver used dimension {result['dimension']}, the schedule was set for "
            f"{resolved['dimension']}")
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
