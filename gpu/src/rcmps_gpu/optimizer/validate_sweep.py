"""GPU-only accuracy refinement of a stopped dimension-sweep checkpoint.

Requires a stopped JSON/NPZ generation; accuracy and convergence are distinct. Never runs a CPU objective
or optimizer, and does not overwrite the accepted state or its L-BFGS history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from numerics import load_reference, metric
import sweep_driver as driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("initial_reference")
    parser.add_argument("--comparison-reference")
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    checkpoint = Path(cli.checkpoint)
    report = json.loads(checkpoint.read_text())
    if not report.get("stop_reason"):
        raise ValueError("Optimization is still running")
    state = np.load(checkpoint.with_suffix(".npz"))
    if int(state["iteration"]) != len(report["history"]):
        raise ValueError("JSON/NPZ checkpoint generations differ")
    args = driver.parse_args([cli.initial_reference])
    vars(args).update(report["controls"])
    if args.backend != "cupy":
        raise ValueError("Final validation requires the GPU backend")
    driver.validate_arguments(args)
    hashes = driver.source_hashes(args)
    if hashes != report["runs"][-1]["source_sha256"]:
        raise ValueError("Current sources differ from the final optimizer invocation")
    _, cpu_metadata = load_reference(cli.initial_reference)
    comparison_metadata = load_reference(cli.comparison_reference)[1] if cli.comparison_reference else None
    if comparison_metadata and comparison_metadata["D"] != cpu_metadata["D"]:
        raise ValueError("Comparison reference dimension differs")
    r, k, rho, cached_gradient = (state[name] for name in ("R", "K", "rho", "gradient"))
    if len(r) != cpu_metadata["D"]:
        raise ValueError("CPU reference dimension differs")
    import cupy as cp
    target = args.gradient_tolerance
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    validation = dict(
        iteration=int(state["iteration"]), gradient_target=target,
        cached_energy=float(state["energy"]),
        cached_gradient_norm=float(np.sqrt(max(0., metric(rho, cached_gradient, cached_gradient, args.metric_reg)))),
        saved_cpu_energy=comparison_metadata["saved_energy"] if comparison_metadata else None,
        optimization_target_reached=bool(report.get("converged")),
        optimization_stop_reason=report["stop_reason"],
        checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (checkpoint, checkpoint.with_suffix(".npz"))},
        source_sha256=hashes,
        validation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evaluations=[], cpu_objective_rerun=False)
    gradients = {}
    standard_gradient = None
    standard_energy = None
    for name, nodes in (("standard_cold", args.nodes), ("standard_warm_1", args.nodes),
                        ("standard_warm_2", args.nodes), ("refined_512", 512),
                        ("refined_1024", 1024)):
        refined = name.startswith("refined")
        controls = argparse.Namespace(**vars(args))
        controls.nodes = nodes
        if refined:
            controls.window = 40.
            controls.quadrature = 1024
            controls.atol = 1e-14
            controls.rtol = 1e-13
        begin = time.perf_counter()
        energy, gradient, stats = driver.objective(r, k, rho, cpu_metadata["coupling"], controls, cp)
        seconds = time.perf_counter() - begin
        if standard_gradient is None:
            standard_energy, standard_gradient = energy, gradient
        difference = gradient - standard_gradient
        cached_difference = gradient - cached_gradient
        row = dict(name=name, controls={key: vars(controls)[key] for key in
                   ("nodes", "window", "quadrature", "atol", "rtol")}, seconds=seconds,
                   energy=energy, gradient_norm=float(np.sqrt(max(0., metric(rho, gradient, gradient, args.metric_reg)))),
                   energy_difference_standard=abs(energy-standard_energy),
                   energy_difference_cached=abs(energy-float(state["energy"])),
                   energy_difference_saved_cpu=abs(energy-comparison_metadata["saved_energy"]) if comparison_metadata else None,
                   gradient_metric_difference_standard=float(np.sqrt(max(0., metric(rho, difference, difference, args.metric_reg)))),
                   gradient_metric_difference_cached=float(np.sqrt(max(0., metric(rho, cached_difference, cached_difference, args.metric_reg)))),
                   stats=stats)
        row["gradient_target_reached"] = bool(row["gradient_norm"] <= target)
        row["passed"] = bool(np.isfinite(energy) and np.isfinite(gradient).all() and
                             row["gradient_metric_difference_cached"] <= 3e-7 and
                             row["energy_difference_cached"] <= 1e-8 and
                             stats["kinetic"]["maximum_relative_residual"] <= controls.krylov_tolerance and
                             stats["kinetic"]["fallback_solves"] == 0)
        gradients[name] = gradient
        validation["evaluations"].append(row)
        validation["passed"] = all(x["passed"] for x in validation["evaluations"])
        output.write_text(json.dumps(validation, indent=2)+"\n")
        np.savez(output.with_suffix(".npz"), R=r, K=k, rho=rho, **gradients)
        print(json.dumps({key: value for key, value in row.items() if key != "stats"}), flush=True)
    validation["standard_warm_median_seconds"] = float(np.median(
        [x["seconds"] for x in validation["evaluations"] if x["name"].startswith("standard_warm")]))
    output.write_text(json.dumps(validation, indent=2)+"\n")
    if not validation["passed"]:
        raise RuntimeError("Final GPU accuracy refinement failed; inspect the saved report")
    # Portable Julia import: no NPZ dependency is added to the CPU package.
    export = output.parent / "optimized_state"
    export.mkdir(exist_ok=True)
    for name, matrix in (("R", r), ("K", k), ("rho", rho),
                         ("gradient", gradients["refined_1024"])):
        matrix.ravel(order="F").astype("<c16").tofile(export / (name+".bin"))
    final = validation["evaluations"][-1]
    metadata = dict(D=len(r), coupling=cpu_metadata["coupling"],
                    energy=final["energy"], gradient_norm=final["gradient_norm"],
                    gradient_target=target, gradient_target_reached=bool(final["gradient_target_reached"]),
                    metric_reg=args.metric_reg,
                    precond_reg=float(state["precond_reg"]),
                    window=40., quadrature=1024, atol=1e-14, rtol=1e-13)
    (export / "metadata.toml").write_text(
        "validation_passed = true\n" +
        "".join(f"{key} = {str(value).lower() if isinstance(value,bool) else repr(value)}\n" for key, value in metadata.items()))
    print(f"Validated portable state: {export}", flush=True)


if __name__ == "__main__":
    main()
