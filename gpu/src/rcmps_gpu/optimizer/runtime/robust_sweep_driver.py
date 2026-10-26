"""Explicit checkpoint fork and bounded resolution-aware warm optimization.

--resume-from OLD --reset-lbfgs --output NEW preserves the old checkpoint,
scientific arrays and counters, clears L-BFGS pairs, and resets next_step to
--initial-step. The unchanged sweep objective is used by the new isolated
robust controller. All core, warm and robust sources are recorded separately.
--line-search wolfe (with --wolfe-energy-atol) selects its Hager-Zhang search;
robust_controls records the choice, which a resume or fork cannot change.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from . import sweep_driver as driver
from . import warm_sweep_driver as warm
from .warm_sweep_driver import binding, digest
from . import robust_optimizer


ROBUST_SOURCES = (*warm.WARM_SOURCES, "robust_sweep_driver.py", "robust_optimizer.py")
BASE_PARSE_ARGS = driver.parse_args


def source_hashes(args):
    hashes = warm.BASE_SOURCE_HASHES(args)
    hashes.update({name:digest(Path(__file__).with_name(name).read_bytes())
                   for name in ROBUST_SOURCES})
    return hashes


def verify_checkpoint(path, requested_args=None):
    # Only complete known source sets are accepted. The warm verifier then
    # performs the stable-generation, exact core, matrix/history and reference
    # checks, with the appropriate explicit extra-file list scoped here.
    report = json.loads(Path(path).read_text())
    args = argparse.Namespace(**report["controls"])
    expected = (report["runs"][-1] if report.get("runs") else report)["source_sha256"]
    extras = set(expected)-set(warm.BASE_SOURCE_HASHES(args))
    known = (set(), set(warm.WARM_SOURCES), set(ROBUST_SOURCES))
    if extras not in known:
        raise ValueError("checkpoint has unknown or incomplete robust/warm source extras")
    files = ROBUST_SOURCES if extras == set(ROBUST_SOURCES) else warm.WARM_SOURCES
    with binding(warm, "WARM_SOURCES", files):
        snapshot = warm.verify_resume_checkpoint(path, requested_args)
    snapshot["extras"] = tuple(name for name in ROBUST_SOURCES if name in extras)
    snapshot["source_sha256"] = expected
    return snapshot


def matching_source_hashes(snapshot, args):
    hashes = warm.BASE_SOURCE_HASHES(args)
    hashes.update({name:digest(Path(__file__).with_name(name).read_bytes())
                   for name in snapshot["extras"]})
    return hashes


def fork_checkpoint(snapshot, output, args, *, reset_lbfgs):
    output = Path(output)
    if output.resolve() in (snapshot["path"], snapshot["path"].with_suffix(".npz")):
        raise ValueError("resume-from requires a distinct new output; original checkpoint stays unchanged")
    if output.exists() or output.with_suffix(".npz").exists():
        raise ValueError("checkpoint fork requires a new JSON/NPZ output")
    state = {key:value.copy() for key,value in snapshot["state"].items()}
    old_count = len(state["lbfgs_steps"])
    old_step = float(state["next_step"])
    if reset_lbfgs:
        shape = (0, *state["R"].shape)
        state["lbfgs_steps"] = np.empty(shape, dtype=np.complex128)
        state["lbfgs_changes"] = np.empty(shape, dtype=np.complex128)
        state["lbfgs_inverse_curvatures"] = np.empty(0, dtype=np.float64)
        state["next_step"] = np.asarray(args.initial_step, dtype=np.float64)
    report = snapshot["report"]
    report.setdefault("robust_forks", []).append(dict(
        original_checkpoint=str(snapshot["path"]),
        original_checkpoint_sha256=snapshot["checkpoint_sha256"],
        reset_lbfgs=bool(reset_lbfgs), old_history_size=old_count,
        new_history_size=len(state["lbfgs_steps"]), old_next_step=old_step,
        new_next_step=float(state["next_step"]), iteration=int(state["iteration"]),
        optimizer_evaluations=int(state["optimizer_evaluations"]),
        scientific_array_sha256={name:digest(state[name].tobytes(order="F"))
                                 for name in ("R","K","rho","gradient")}))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = output.with_suffix(".npz.tmp")
    temporary_json = output.with_suffix(".json.tmp")
    with temporary_npz.open("xb") as stream:
        np.savez(stream, **state)
    temporary_json.write_text(json.dumps(report, indent=2)+"\n")
    temporary_npz.replace(output.with_suffix(".npz"))
    temporary_json.replace(output)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-from")
    parser.add_argument("--reset-lbfgs", action="store_true")
    parser.add_argument("--secant-gradient-floor", type=float, default=1e-10)
    parser.add_argument("--max-recoveries", type=int, default=2)
    parser.add_argument("--line-search", choices=robust_optimizer.LINE_SEARCHES)
    parser.add_argument("--wolfe-energy-atol", type=float)
    cli, forwarded = parser.parse_known_args(argv)
    args = BASE_PARSE_ARGS(forwarded)
    if cli.reset_lbfgs and not cli.resume_from:
        raise ValueError("--reset-lbfgs requires --resume-from and a distinct fresh --output")
    snapshot = None
    if cli.resume_from:
        snapshot = verify_checkpoint(cli.resume_from, args)
    elif args.resume:
        snapshot = verify_checkpoint(args.output, args)
    elif Path(args.output).exists() or Path(args.output).with_suffix(".npz").exists():
        raise ValueError("fresh robust run requires a new output")
    # A resume keeps the line search of its checkpoint (Armijo for one that
    # records none) and, unless given, its Wolfe energy tolerance.
    prior = getattr(snapshot["args"], "robust_controls", {}) if snapshot else {}
    line_search = prior.get("line_search", "armijo") if snapshot else cli.line_search or "armijo"
    if cli.line_search not in (None, line_search):
        raise ValueError(f"resume cannot change the line search (the checkpoint's is {line_search})")
    if line_search == "armijo" and cli.wolfe_energy_atol is not None:
        raise ValueError("--wolfe-energy-atol requires --line-search wolfe")
    wolfe_energy_atol = None
    if line_search == "wolfe":
        wolfe_energy_atol = cli.wolfe_energy_atol if cli.wolfe_energy_atol is not None else \
            prior.get("wolfe_energy_atol", robust_optimizer.WOLFE_ENERGY_ATOL)
    robustness = robust_optimizer.RobustnessControls(
        secant_gradient_floor=cli.secant_gradient_floor, max_recoveries=cli.max_recoveries,
        line_search=line_search, wolfe_energy_atol=wolfe_energy_atol)
    if cli.resume_from:
        fork_checkpoint(snapshot, args.output, args, reset_lbfgs=cli.reset_lbfgs)
        args.resume = True
    recorded = dict(asdict(robustness), refresh_curvatures=True,
                    resume_from=cli.resume_from, reset_lbfgs=cli.reset_lbfgs)
    if line_search == "wolfe":
        recorded["hager_zhang"] = asdict(robust_optimizer.HAGER_ZHANG)

    def parse(arguments=None):
        out = BASE_PARSE_ARGS(arguments)
        if cli.resume_from:
            out.resume = True
        out.robust_controls = recorded
        return out

    def optimize_with_progress(r,k,rho,energy,gradient,objective_callable,controls,
                               resume_state=None,**options):
        controls = replace(controls, refresh_curvatures=True)
        events = robust_optimizer.optimize_with_progress(r,k,rho,energy,gradient,
            objective_callable,controls,resume_state,robustness=robustness,**options)
        resume = resume_state or {}
        return warm.accepted_density_events(events,resume.get("rho",rho),
                                            iteration=resume.get("iteration",0))

    with binding(driver,"parse_args",parse), binding(driver,"source_hashes",source_hashes), \
            binding(driver,"optimize_with_progress",optimize_with_progress):
        return driver.main(forwarded)


if __name__=="__main__":
    main()
