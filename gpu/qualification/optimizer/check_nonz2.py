"""The GPU objective on states without Z2 symmetry against RcmpsCore (Julia).

    python check_nonz2.py --output NEW_DIR DIR... [--tiles auto] [--device 0] [--packed-crosscheck]

Each DIR is a portable state directory with the energy and gradient of RcmpsCore at the
integration settings of the GPU optimizer, as nonz2_references.jl writes them: `energy` and
gradient.bin with 512 kinetic angular nodes (the GPU's), gradient_1024.bin and
julia_energy_1024 with 1024, `phi` = phi_monomial(s, 1). For each state, with R, K and rho as
stored, the worker's initial qualification (worker.qualify, symmetry none) evaluates the
complete energy and gradient of a run: the trajectory kernel on the unpacked channels with the
tiles this device selects, the cooperative GPU Arnoldi kinetic with 512 nodes and the boundary
and metric solves, and the same with the independent oracle (host-driven CuPy integration of
_ReducedQuarticRHS, baseline kinetic). One JSON line per state: the GPU energy minus Julia's,
the metric norm of the gradient difference (against the 512- and 1024-node Julia gradients),
the GPU <phi> (ode_backend.phi_expectation) minus Julia's, and the fast-oracle differences.
With --packed-crosscheck, a state whose matrices are Z2 symmetric in the packed basis is also
evaluated on the packed channels (symmetry z2), and the two GPU results are compared.

The exit status is 1 if an evaluation failed or the GPU differs from Julia by more than 1e-9
in energy, 1e-7 in the metric gradient norm or 1e-10 in <phi>.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

FAILED = []


def emit(**row):
    if row.get("ok") is False:
        FAILED.append(row)
    print(json.dumps(row, default=float), flush=True)


def arguments(reference, unpacked):
    """The objective controls of a run (worker.main's `base`), with the command line defaults."""
    from rcmps_gpu.optimizer import cli
    from rcmps_gpu.optimizer.runtime import trajectory_sweep_driver as trajectory
    base = trajectory.ORIGINAL_PARSE([str(reference), "--backend", "cupy", "--mode", "hybrid",
                                      "--reduced" if unpacked else "--packed-z2", "--quartic-backend",
                                      "persistent_trajectory", "--arnoldi", "gpu", "--projection", "hessenberg_gpu"])
    for name, value in cli.DEFAULTS.items():
        setattr(base, name, value)
    return base


def evaluate(cp, record, reference, output, args, symmetry):
    """The fast GPU energy and gradient of `record` (worker.qualify) and its receipt."""
    from rcmps_gpu.optimizer import worker
    compiled = worker.compile_dimension(record.D, cp, None, args.tiles, symmetry)
    tiles, cap = compiled["tiles"]["resolved"], compiled["register_cap"]
    output.mkdir(parents=True)
    receipt = worker.qualify(record.R, record.K, record.rho, record.coupling, arguments(reference, symmetry == "none"),
                             cp, args.max_device_seconds, output, tiles, cap)
    with np.load(output / "initial_qualification.npz") as saved:
        gradient = saved["fast_gradient"].copy()
    return receipt, gradient, compiled


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path, help="new directory")
    p.add_argument("--tiles", choices=("auto", "mma", "scalar"), default="auto")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-device-seconds", type=float, default=300.)
    p.add_argument("--packed-crosscheck", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from rcmps_gpu.io import read_portable
    from rcmps_gpu.optimizer.cli import z2_blocks
    from rcmps_gpu.optimizer.runtime.numerics import metric
    from rcmps_gpu.optimizer.runtime.ode_backend import phi_expectation
    import cupy as cp
    with cp.cuda.Device(args.device):
        for i, directory in enumerate(args.directories):
            row = dict(state=str(directory), index=i)
            try:
                record = read_portable(directory)
                if record.rho is None or record.gradient is None or record.energy is None:
                    raise ValueError("the directory needs rho, an energy and a gradient")
                integration = record.integration
                if (integration.n_quad, integration.max_point, integration.atol, integration.rtol) != (512, 30., 1e-13, 1e-12):
                    raise ValueError(f"the Julia values are not at the GPU's integration settings: {integration}")
                metric_reg = record.manifold.metric_reg if record.manifold else 1e-3
                norm = lambda x: math.sqrt(max(0., metric(record.rho, x, x, metric_reg)))
                receipt, gradient, compiled = evaluate(cp, record, directory, args.output / f"state{i:02d}", args, "none")
                row.update(D=record.D, coupling=record.coupling, tiles=compiled["tiles"]["resolved"],
                           register_cap=compiled["register_cap"], blocks=compiled["blocks"],
                           julia_energy=record.energy, julia_gradient_norm=record.gradient_norm,
                           gpu_energy=receipt["fast_energy"], energy_difference=receipt["fast_energy"] - record.energy,
                           gradient_metric_difference=norm(gradient - record.gradient),
                           gradient_relative_difference=norm(gradient - record.gradient) / max(norm(record.gradient), 1e-300),
                           fast_oracle_energy_difference=receipt["energy_difference"],
                           fast_oracle_gradient_difference=receipt["metric_gradient_difference"],
                           trajectory_steps=receipt["fast_stats"]["quartic"].get("accepted"),
                           trajectory_device_milliseconds=receipt["fast_stats"]["quartic"].get("device_milliseconds"),
                           objective_seconds=receipt["fast_stats"].get("total_seconds"))
                extra = directory / "gradient_1024.bin"
                if extra.exists():
                    fine = np.fromfile(extra, dtype="<c16").reshape((record.D, record.D), order="F")
                    row.update(julia_energy_1024=record.metadata.get("julia_energy_1024"),
                               gradient_metric_difference_1024=norm(gradient - fine),
                               julia_512_1024_gradient_difference=norm(fine - record.gradient))
                phi = phi_expectation(record.R, record.rho, integration.max_point)
                if record.phi is not None:
                    row.update(julia_phi=record.phi, gpu_phi=phi, phi_difference=phi - record.phi,
                               julia_phi_closed=record.metadata.get("julia_phi_closed"))
                row["ok"] = bool(receipt["passed"] and abs(row["energy_difference"]) <= 1e-9
                                 and row["gradient_metric_difference"] <= 1e-7
                                 and (record.phi is None or abs(row["phi_difference"]) <= 1e-10))
                if args.packed_crosscheck and z2_blocks(record.R, record.K):
                    packed, packed_gradient, _ = evaluate(cp, record, directory, args.output / f"state{i:02d}_z2", args, "z2")
                    row.update(packed_minus_unpacked_energy=packed["fast_energy"] - receipt["fast_energy"],
                               packed_unpacked_gradient_difference=norm(packed_gradient - gradient))
            except Exception as error:
                row.update(error=f"{type(error).__name__}: {error}"[:2000], ok=False)
            emit(**row)
    emit(stage="summary", failed=len(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
