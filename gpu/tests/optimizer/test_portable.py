"""Initial and final states of the optimizer in the portable format 1 (rcmps_gpu.io).

The worker writes initial/ and the validator optimized_state/ in format 1, with what the
runtime reads besides; --initial accepts either, and the research dialects. CPU only: the
validator runs in a child interpreter with a stand-in cupy module and objective.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import rcmps_gpu
from rcmps_gpu import io
from rcmps_gpu.optimizer import cli, worker
from rcmps_gpu.optimizer.runtime import numerics

RUNTIME_FILES = ("R.bin", "K.bin", "rho.bin", "saved_gradient.bin", "metadata.toml")   # sweep_driver.main hashes them


def plan(*arguments):
    return cli.prepare(cli.parser().parse_args(["--D", "4", *arguments]), cli.runtime_source_sha256())


def legacy_random_reference(plan, directory):
    """worker.random_reference before format 1 (research generic_gpu/worker.py), less Q.bin and
    metadata.toml: the matrices the runtime reads must be unchanged."""
    d, n = plan["D"], plan["D"] // 2
    rng = np.random.default_rng(plan["seed"])

    def draw():
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2)
    k1, k2 = draw(), draw()
    zero = np.zeros((n, n), dtype=np.complex128)
    r = np.block([[zero, draw()], [draw(), zero]]) / math.sqrt(d)
    k = np.block([[k1 + k1.conj().T, zero], [zero, k2 + k2.conj().T]]) / math.sqrt(d)
    q = -1j * k - r.conj().T @ r / 2
    rho, _ = numerics.stationary_state(q, r, tolerance=plan["controls"]["stationary_tolerance"])
    directory.mkdir()
    for name, value in (("R", r), ("K", k), ("rho", rho), ("saved_gradient", np.zeros_like(r))):
        value.ravel(order="F").astype("<c16").tofile(directory / (name + ".bin"))


VALIDATOR = r'''
import json, sys, types
from pathlib import Path
from unittest import mock
import numpy as np
import rcmps_gpu
assert rcmps_gpu.__file__ == sys.argv[1], rcmps_gpu.__file__
sys.modules["cupy"] = types.ModuleType("cupy")   # imported by the validator, used by the objective only
from rcmps_gpu.io import read_portable
from rcmps_gpu.optimizer import validate_sweep
from rcmps_gpu.optimizer.runtime import sweep_driver as driver
run = Path(sys.argv[2]); reference = run / "initial"
state = read_portable(reference)
args = driver.parse_args([str(reference), "--backend", "cupy", "--mode", "hybrid", "--packed-z2",
                          "--quartic-backend", "persistent_trajectory", "--gradient-tolerance", "1e-5"])
gradient = 1e-7 * (np.arange(16).reshape(4, 4) + 1j)
energy = -0.125
np.savez(run / "optimization.npz", R=state.R, K=state.K, rho=state.rho, gradient=gradient, energy=energy,
         iteration=2, precond_reg=3e-6)
report = dict(stop_reason="gradient target", converged=True, history=[{}, {}], controls=vars(args),
              runs=[dict(source_sha256=driver.source_hashes(args))])
(run / "optimization.json").write_text(json.dumps(report))
calls = []
def objective(r, k, rho, coupling, controls, cp):
    calls.append((controls.nodes, controls.window, controls.quadrature, controls.atol, controls.rtol, coupling))
    return energy, gradient, dict(kinetic=dict(maximum_relative_residual=0.0, fallback_solves=0))
argv = ["validate_sweep", str(run / "optimization.json"), str(reference), "--output", str(run / "final_validation.json")]
phi_by_ode = validate_sweep.phi_by_ode   # on the CPU here: the stand-in cupy has no arrays
with mock.patch.object(driver, "objective", objective), mock.patch.object(sys, "argv", argv), \
        mock.patch.object(validate_sweep, "phi_by_ode", lambda r, k, rho, controls, xp: phi_by_ode(r, k, rho, controls, np)):
    validate_sweep.main()
print(json.dumps(dict(calls=calls, gradient=[[[z.real, z.imag] for z in row] for row in gradient.tolist()])))
'''


class PortableStates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_random_initial_state_is_format_1_with_what_the_runtime_reads(self):
        p = plan("--seed", "7", "--coupling", "1.25")
        initial = self.root / "run" / "initial"
        initial.parent.mkdir()
        stats = worker.random_reference(p, initial)
        self.assertGreaterEqual(stats["matvecs"], 1)
        legacy_random_reference(p, self.root / "legacy")
        for name in ("R.bin", "K.bin", "rho.bin", "saved_gradient.bin"):
            self.assertEqual((initial / name).read_bytes(), (self.root / "legacy" / name).read_bytes(), name)
        self.assertEqual(sorted(path.name for path in initial.iterdir()), sorted(RUNTIME_FILES))
        record = io.read_portable(initial)
        self.assertEqual((record.origin, record.D, record.model, record.coupling, record.mass),
                         ("portable", 4, "phi4", 1.25, 1.0))
        self.assertEqual((record.energy, record.gradient, record.gradient_norm, record.manifold), (None,) * 4)
        self.assertEqual(record.integration, io.Integration(512, 30.0, 1e-13, 1e-12))
        self.assertTrue(record.producer.startswith(f"rcmps_gpu {rcmps_gpu.__version__}"))
        matrices, metadata = numerics.load_reference(initial)   # as runtime/sweep_driver.py reads it
        self.assertEqual(sorted(matrices), ["K", "R", "rho", "saved_gradient"])
        self.assertFalse(matrices["saved_gradient"].any())
        self.assertEqual((metadata["D"], metadata["coupling"], metadata["seed"]), (4, 1.25, 7))
        self.assertEqual({key: metadata[key] for key in metadata if key.startswith("saved_")},
                         dict(saved_energy=0.0, saved_energy_available=False, saved_gradient_available=False,
                              saved_gradient_norm=0.0, saved_gradient_norm_available=False))
        self.assertNotIn("precond_reg", metadata)   # the runtime's default 1e-4

    def test_initial_and_optimized_state_are_valid_initial_states(self):
        run = self.root / "run"
        run.mkdir()
        worker.random_reference(plan("--seed", "3"), run / "initial")
        # optimized_state/ written by the validator code path (validate_sweep.main)
        result = subprocess.run([sys.executable, "-c", VALIDATOR, rcmps_gpu.__file__, str(run)],
                                env=dict(os.environ, OPENBLAS_NUM_THREADS="1"), capture_output=True, text=True,
                                timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        child = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual([tuple(call[1:5]) for call in child["calls"]],
                         [(30.0, 512, 1e-13, 1e-12)] * 3 + [(40.0, 1024, 1e-14, 1e-13)] * 2)
        gradient = np.array(child["gradient"])[..., 0] + 1j * np.array(child["gradient"])[..., 1]
        final = io.read_portable(run / "optimized_state")
        source = io.read_portable(run / "initial")
        for name in ("R", "K", "rho"):
            self.assertEqual(getattr(final, name).tobytes(), getattr(source, name).tobytes(), name)
        self.assertEqual(final.gradient.tobytes(), gradient.tobytes())
        self.assertEqual((final.origin, final.model, final.coupling, final.energy, final.validation_passed),
                         ("portable", "phi4", 2.7, -0.125, True))
        self.assertEqual(final.gradient_norm, math.sqrt(numerics.metric(final.rho, gradient, gradient, 1e-3)))
        self.assertEqual(final.integration, io.Integration(1024, 40.0, 1e-14, 1e-13))
        self.assertEqual(final.manifold, io.Manifold(1e-3, 3e-6))
        self.assertEqual((final.metadata["gradient_target"], final.metadata["gradient_target_reached"]), (1e-5, True))
        self.assertEqual(final.source, "optimization.json iteration 2, evaluation refined_1024")
        self.assertEqual(sorted(path.name for path in (run / "optimized_state").iterdir()),
                         ["K.bin", "R.bin", "gradient.bin", "metadata.toml", "rho.bin"])
        # both are accepted as --initial; the new run writes its own initial/ from them
        for name, directory in (("initial", run / "initial"), ("optimized", run / "optimized_state")):
            p = plan("--initial", str(directory))
            record = io.read_portable(directory)
            self.assertEqual(p["coupling"], record.coupling)
            self.assertEqual(sorted(p["initial"]["sha256"]), sorted(record.files))
            target = self.root / name / "initial"
            target.parent.mkdir()
            self.assertIsNone(worker.initial_reference(p, target))
            copied = io.read_portable(target)
            for matrix in ("R", "K", "rho"):
                self.assertEqual(getattr(copied, matrix).tobytes(), getattr(record, matrix).tobytes())
            self.assertEqual((copied.energy, copied.gradient_norm, copied.integration, copied.manifold),
                             (record.energy, record.gradient_norm, record.integration, record.manifold))
            matrices, metadata = numerics.load_reference(target)
            self.assertTrue(all((target / file).exists() for file in RUNTIME_FILES))
            if name == "initial":
                self.assertFalse(metadata["saved_energy_available"] or metadata["saved_gradient_available"])
                self.assertNotIn("gradient", matrices)
            else:   # the first evaluation of the run must reproduce the validated energy and gradient
                self.assertEqual((metadata["saved_energy"], metadata["saved_energy_available"],
                                  metadata["saved_gradient_available"], metadata["saved_gradient_norm"],
                                  metadata["precond_reg"]), (-0.125, True, True, final.gradient_norm, 3e-6))
                self.assertEqual(matrices["saved_gradient"].tobytes(), gradient.tobytes())
                self.assertEqual(matrices["gradient"].tobytes(), gradient.tobytes())

    def test_research_initial_and_state_without_rho(self):
        p = plan("--seed", "5")
        worker.random_reference(p, self.root / "initial")
        source = io.read_portable(self.root / "initial")
        legacy = self.root / "legacy"
        legacy_random_reference(p, legacy)
        (legacy / "metadata.toml").write_text(
            "D = 4\ncoupling = 2.7\nsaved_energy = 0.0\nsaved_energy_available = false\n"
            "saved_gradient_available = false\nsaved_gradient_norm = 0.0\nsaved_gradient_norm_available = false\n")
        self.assertEqual(plan("--initial", str(legacy))["initial"]["sha256"].keys(),
                         {"metadata.toml", "R.bin", "K.bin", "rho.bin"})
        bare = io.write_portable(self.root / "bare", io.StateRecord(D=4, model="phi4", coupling=2.7,
                                                                     R=source.R, K=source.K))
        p = plan("--initial", str(bare))
        (self.root / "run").mkdir()
        stats = worker.initial_reference(p, self.root / "run" / "initial")
        self.assertGreaterEqual(stats["matvecs"], 1)
        self.assertLess(np.abs(io.read_portable(self.root / "run" / "initial").rho - source.rho).max(), 1e-13)

    def test_refusals(self):
        worker.random_reference(plan(), self.root / "initial")
        state = io.read_portable(self.root / "initial")
        other = io.write_portable(self.root / "schwinger", io.StateRecord(
            D=4, model="schwinger", coupling=2.7, R=state.R, K=state.K))
        with self.assertRaisesRegex(ValueError, "implements the phi4 model, not 'schwinger'"):
            plan("--initial", str(other))
        with self.assertRaisesRegex(ValueError, "D/coupling"):
            plan("--initial", str(self.root / "initial"), "--coupling", "2.0")
        p = plan("--initial", str(self.root / "initial"))
        (self.root / "initial" / "gradient.bin").write_bytes((self.root / "initial" / "R.bin").read_bytes())
        with self.assertRaisesRegex(ValueError, "initial state changed after planning"):
            worker.initial_reference(p, self.root / "run")
        self.assertFalse((self.root / "run").exists())
        (self.root / "initial" / "K.bin").write_bytes(b"\0" * 16)
        with self.assertRaisesRegex(ValueError, "K.bin has 16 bytes"):
            plan("--initial", str(self.root / "initial"))


if __name__ == "__main__":
    unittest.main()
