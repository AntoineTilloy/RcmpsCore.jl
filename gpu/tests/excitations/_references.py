"""The reference excitation levels shared with the Julia package:
test/fixtures/excitation_references.toml (FIXTURES of _paths).

Each [[levels]] entry is one level of one excitation_eigsolve solve, defined by its state
([states]), settings ([settings]), sector, momentum and nev; the file's `about` and [fields]
describe every entry, and benchmark/excitation_references.jl regenerates the Julia values.
`solve` runs such a solve with eigsolve.excitation_eigsolve on the CPU (NumPy backend).
Without the file (an sdist without the repository fixtures) REFERENCES is None.
"""
from __future__ import annotations

import hashlib
import tomllib

from rcmps_gpu.excitations import eigsolve
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import states
from rcmps_gpu.excitations.davidson import DavidsonOptions
from rcmps_gpu.excitations.states import IntegratorParameters
from _paths import FIXTURES

PATH = FIXTURES / "excitation_references.toml"
FOUR = ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")
DAVIDSON = ("tol", "maxops", "retain", "refresh", "hermitian_tol", "stall_window", "stall_minops", "stall_rtol",
            "stall_residual_limit", "stall_energy_atol")


def load():
    """The parsed reference file, or None when it is not available."""
    if not PATH.exists():
        return None
    with PATH.open("rb") as handle:
        return tomllib.load(handle)


REFERENCES = load()


def level(name):
    """The [[levels]] entry called `name`, or None without the reference file."""
    if REFERENCES is None:
        return None
    matches = [entry for entry in REFERENCES["levels"] if entry["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"{len(matches)} reference levels are called {name!r}")
    return matches[0]


def state_path(name):
    return FIXTURES / REFERENCES["states"][name]["file"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solves(entries):
    """Entries grouped by solve, in file order: {(state, settings, sector, momentum, nev): [entries]}."""
    groups = {}
    for entry in entries:
        key = (entry["state"], entry["settings"], entry["sector"], entry["momentum"], entry["nev"])
        groups.setdefault(key, []).append(entry)
    return groups


def load_state(name):
    """The state of [states.name] as states.load_jld2 reads it, after checking its sha256, D,
    model and coupling against the file."""
    entry = REFERENCES["states"][name]
    path = state_path(name)
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"{path.name}: sha256 {digest}, the references record {entry['sha256']}")
    state = states.load_jld2(path)
    if (state.D, state.model, state.coupling) != (entry["D"], entry["model"], entry["coupling"]):
        raise ValueError(f"{path.name} is not the state [states.{name}] describes")
    return state


def integrator(table):
    return IntegratorParameters(n_quad=table["n_quad"], max_point=table["max_point"], atol=table["atol"],
                                rtol=table["rtol"], maxiters=table["maxiters"])


def davidson_options(table, n):
    """DavidsonOptions of a [settings.*.davidson] table for a sector of dimension n."""
    return DavidsonOptions(maxspace=min(table["maxspace_cap"], n), **{key: table[key] for key in DAVIDSON})


def solve(state, settings, sector, momentum, nev, **controls):
    """excitation_eigsolve on `state` with the [settings] table `settings` (its common keys and
    its `python` table); `controls` are passed on (xp, integrate, log, ...)."""
    python = settings["python"]
    state.integrator = integrator(settings["integration"])
    n = eigsolve.sector_dimension(state, sector=sector, parity=settings["parity"], gauge=settings["gauge"],
                                  metric_tol=settings["metric_tol"])
    if not settings["validate"]:
        raise ValueError("reference levels are validated")
    return eigsolve.excitation_eigsolve(
        state, momentum, models.phi4_terms(state.coupling), nev=nev, sector=sector, parity=settings["parity"],
        gauge=settings["gauge"], scaling=python["scaling"], metric_tol=settings["metric_tol"],
        kinetic_nodes=settings["kinetic_nodes"], preconditioner=settings["preconditioner"],
        preconditioner_nodes=settings["preconditioner_nodes"], preconditioner_tol=python["preconditioner_tol"],
        start=settings["start"], davidson_options=davidson_options(settings["davidson"], n), validate=True,
        refined=integrator(settings["refined_integration"]), refined_kinetic_nodes=settings["refined_kinetic_nodes"],
        refine_steps=settings["refine_steps"], validation_tol=settings["validation_tol"],
        acceptance_residual=settings["acceptance_residual"], solver_tol=python["solver_tol"], chunk=python["chunk"],
        **controls)


def differences(entry, result):
    """{quantity: (computed - stored, tolerance)} for level entry["level"] of `result`: the
    validated energy and the four validation energies at `tolerance`, the Ritz energy at
    `ritz_tolerance`."""
    k = entry["level"] - 1
    validation = result["validation"][k]
    rows = {"validated_energy": (result["validated_energies"][k] - entry["validated_energy"], entry["tolerance"])}
    for name, stored in zip(FOUR, entry["four_energies"]):
        rows[name] = (validation[name] - stored, entry["tolerance"])
    rows["ritz_energy"] = (result["energies"][k] - entry["ritz_energy"], entry["ritz_tolerance"])
    return rows
