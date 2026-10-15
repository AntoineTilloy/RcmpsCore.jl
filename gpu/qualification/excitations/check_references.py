"""Reference excitation levels on a GPU, through the production command line.

    python check_references.py --output NEW_DIR [--level NAME ...] [--backend fused|host|numpy]
        [--references FILE] [--fixtures DIR]

Recomputes levels of test/fixtures/excitation_references.toml (by default every level at
[settings.gpu_comparison]: the three D = 16 and the three D = 32 levels) with
rcmps_gpu.excitations.cli and compares them with the stored Julia values. Each solve (state,
settings, sector, momentum, nev) runs the command line into NEW_DIR/<solve>/ (result.json,
vectors.npz, log.jsonl) with the arguments that reproduce its [settings] table: the common
keys, the `python` table and, for a GPU backend, the `gpu` table. Settings the command line
does not expose (gauge, parity, start, validation and acceptance limits, refine steps, the
other Davidson controls, maxiters) must be at its defaults; that is checked before any solve.

Every result is one JSON line on stdout (the command line's own log goes to stderr): per level
the computed validated, four validation and Ritz energies, their differences from the stored
Julia values with the level's tolerances, and the differences from the stored A100 values
(gpu_*, informative: those runs used earlier sources, see gpu_sources_sha256). The last line
is a summary. Exit status 1 if any level is out of tolerance or not accepted.

Not run by the test suite: it needs a GPU (about a minute per D = 16 or D = 32 solve on an
A100). --backend numpy runs on the CPU; with --level D4_odd_p0.4_1 D4_odd_p0.4_2 it checks the
script itself in about 15 seconds.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

from rcmps_gpu.excitations import cli
from rcmps_gpu.excitations import eigsolve
from rcmps_gpu.excitations import states
from rcmps_gpu.excitations.davidson import DavidsonOptions

REPOSITORY = Path(__file__).resolve().parents[3]
FOUR = ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")
# Settings the command line does not expose, and the values it uses.
FIXED = dict(gauge="density", parity="auto", start="auto", validate=True, validation_tol=1e-7,
             acceptance_residual=1e-5, refine_steps=0, preconditioner="kinetic")
DAVIDSON_FIXED = ("hermitian_tol", "stall_window", "stall_minops", "stall_rtol")


def emit(**row):
    print(json.dumps(row), flush=True)


def check_fixed(name, settings):
    """ValueError unless the settings the command line cannot set are at its defaults."""
    for key, value in FIXED.items():
        if settings[key] != value:
            raise ValueError(f"[settings.{name}] {key} = {settings[key]!r}: the command line uses {value!r}")
    defaults = DavidsonOptions()
    for key in DAVIDSON_FIXED:
        if settings["davidson"][key] != getattr(defaults, key):
            raise ValueError(f"[settings.{name}.davidson] {key} = {settings['davidson'][key]!r}: the command line "
                             f"uses {getattr(defaults, key)!r}")
    maxiters = states.IntegratorParameters().maxiters
    for table in ("integration", "refined_integration"):
        if settings[table]["maxiters"] != maxiters:
            raise ValueError(f"[settings.{name}] {table}.maxiters must be {maxiters}")


def arguments(state_path, settings, sector, momentum, nev, n, backend, output):
    """The command line of one solve."""
    i, f, d, py = settings["integration"], settings["refined_integration"], settings["davidson"], settings["python"]
    argv = [str(state_path), "--output", str(output), "--momentum", repr(momentum), "--nev", str(nev),
            "--sector", sector, "--backend", backend, "--scaling", py["scaling"],
            "--metric-tol", repr(settings["metric_tol"]),
            "--window", repr(i["max_point"]), "--n-quad", str(i["n_quad"]), "--atol", repr(i["atol"]),
            "--rtol", repr(i["rtol"]), "--kinetic-nodes", str(settings["kinetic_nodes"]),
            "--preconditioner-nodes", str(settings["preconditioner_nodes"]),
            "--preconditioner-tol", repr(py["preconditioner_tol"]), "--solver-tol", repr(py["solver_tol"]),
            "--chunk", str(py["chunk"]), "--tol", repr(d["tol"]), "--maxops", str(d["maxops"]),
            "--maxspace", str(min(d["maxspace_cap"], n)), "--retain", str(d["retain"]),
            "--fresh-restarts" if d["refresh"] else "--thick-restart",
            "--stall-residual-limit", repr(d["stall_residual_limit"]),
            "--stall-energy-atol", repr(d["stall_energy_atol"]),
            "--refined-window", repr(f["max_point"]), "--refined-n-quad", str(f["n_quad"]),
            "--refined-atol", repr(f["atol"]), "--refined-rtol", repr(f["rtol"]),
            "--refined-kinetic-nodes", str(settings["refined_kinetic_nodes"])]
    if backend != "numpy" and "gpu" in settings:   # without a gpu table: the command line's defaults
        gpu = settings["gpu"]
        argv += ["--basis-bytes", repr(gpu["max_basis_bytes"])]
        if not gpu["tensor_cores"]:
            argv.append("--scalar-tiles")
        if gpu["bilinear"] != "fused":
            argv.append("--host-bilinear")
    return argv


def compare(entry, result):
    """Differences of level entry["level"] of a result.json from the stored Julia and A100 values."""
    k = entry["level"] - 1
    v = result["validation"][k]
    computed = dict(validated_energy=result["validated_energies"][k], four_energies=[v[q] for q in FOUR],
                    ritz_energy=result["energies"][k])
    julia = {"validated_energy": computed["validated_energy"] - entry["validated_energy"],
             **{q: value - stored for q, value, stored in zip(FOUR, computed["four_energies"], entry["four_energies"])},
             "ritz_energy": computed["ritz_energy"] - entry["ritz_energy"]}
    within = all(abs(value) <= entry["tolerance"] for key, value in julia.items() if key != "ritz_energy") and \
        abs(julia["ritz_energy"]) <= entry["ritz_tolerance"]
    a100 = None
    if "gpu_validated_energy" in entry:
        a100 = {"validated_energy": computed["validated_energy"] - entry["gpu_validated_energy"],
                **{q: value - stored for q, value, stored in zip(FOUR, computed["four_energies"],
                                                                 entry["gpu_four_energies"])},
                "ritz_energy": computed["ritz_energy"] - entry["gpu_ritz_energy"]}
    accepted = bool(result["converged"] and v["accepted"])
    return dict(computed=computed, difference_julia=julia, tolerance=entry["tolerance"],
                ritz_tolerance=entry["ritz_tolerance"], within_tolerance=within, accepted=accepted,
                difference_a100=a100, a100_run=entry.get("gpu_run"), numops=result["numops"],
                stored_numops=entry.get("numops"), ok=within and accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="new directory for the runs")
    parser.add_argument("--level", nargs="+", default=None, help="level names (default: settings gpu_comparison)")
    parser.add_argument("--backend", default="fused", choices=("fused", "host", "numpy"))
    parser.add_argument("--references", default=None,
                        help="reference file (default: excitation_references.toml in the fixtures directory)")
    parser.add_argument("--fixtures", default=None,
                        help="fixtures directory (default: RCMPS_TEST_FIXTURES, else test/fixtures of the repository)")
    args = parser.parse_args()
    fixtures = Path(args.fixtures or os.environ.get("RCMPS_TEST_FIXTURES") or REPOSITORY / "test" / "fixtures")
    references_path = Path(args.references) if args.references else fixtures / "excitation_references.toml"
    with references_path.open("rb") as handle:
        references = tomllib.load(handle)
    if (references["format"], references["format_version"]) != ("rcmps-excitation-references", 1):
        parser.error(f"{references_path}: unsupported format")
    if args.level is None:
        entries = [e for e in references["levels"] if e["settings"] == "gpu_comparison"]
    else:
        by_name = {e["name"]: e for e in references["levels"]}
        unknown = [name for name in args.level if name not in by_name]
        if unknown:
            parser.error(f"unknown levels: {', '.join(unknown)}")
        entries = [by_name[name] for name in args.level]
    groups = {}
    for e in entries:
        if "four_energies" not in e or "ritz_tolerance" not in e:
            parser.error(f"level {e['name']} has no recomputable record")
        groups.setdefault((e["state"], e["settings"], e["sector"], e["momentum"], e["nev"]), []).append(e)
    for key in groups:
        try:
            check_fixed(key[1], references["settings"][key[1]])
        except ValueError as error:
            parser.error(str(error))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    passed = 0
    for (state_name, settings_name, sector, momentum, nev), group in groups.items():
        state_entry, settings = references["states"][state_name], references["settings"][settings_name]
        path = fixtures / state_entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != state_entry["sha256"]:
            parser.error(f"{path}: sha256 {digest}, the references record {state_entry['sha256']}")
        n = eigsolve.sector_dimension(states.load_jld2(path), sector=sector, parity=settings["parity"],
                                      gauge=settings["gauge"], metric_tol=settings["metric_tol"])
        run = output / f"{state_name}_{sector}_p{momentum:g}_nev{nev}"
        argv = arguments(path, settings, sector, momentum, nev, n, args.backend, run)
        emit(stage="solve", state=state_name, settings=settings_name, sector=sector, momentum=momentum, nev=nev,
             backend=args.backend, output=run.name, arguments=argv[1:])
        with contextlib.redirect_stdout(sys.stderr):
            cli.main(argv)
        result = json.loads((run / "result.json").read_text())
        for entry in group:
            row = compare(entry, result)
            passed += row["ok"]
            emit(stage="level", name=entry["name"], wall_seconds=result["wall_seconds"], **row)
    emit(stage="summary", levels=len(entries), passed=passed, backend=args.backend)
    sys.exit(0 if passed == len(entries) else 1)


if __name__ == "__main__":
    main()
