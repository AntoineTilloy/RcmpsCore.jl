"""The generic optimizer end to end on the CPU: rcmps-gpu-optimize --run-gpu at D = 8.

cli.main plans the run, worker.main compiles, qualifies the first evaluation against the
CuPy oracle, optimizes through trajectory_sweep_driver and validates the result through
validate_trajectory_sweep; then a second run resumes from the first checkpoint. Everything
runs as written except the leaves that need a GPU (a NumPy-backed cupy, the CPU oracle of
the trajectory kernel, the CPU kinetic solvers): see worker_on_cpu.py, which runs in a child
interpreter. This checks that the parts of the runtime the worker reaches fit together (for
example the backend label it passes and the labels the drivers accept), not the GPU kernels.
About 15 s per run on one thread.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import rcmps_gpu
from rcmps_gpu import io

HELPER = Path(__file__).with_name("worker_on_cpu.py")


def optimize(output, *arguments):
    """worker_on_cpu.py: rcmps-gpu-optimize --output OUTPUT --run-gpu *arguments on the CPU."""
    result = subprocess.run([sys.executable, str(HELPER), rcmps_gpu.__file__, str(output), *arguments],
                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"), capture_output=True, text=True,
                            timeout=600)
    return result


def load(path):
    return json.loads(Path(path).read_bytes())


class WorkerOnCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def check_run(self, run, iterations):
        plan, status = load(run / "plan.json"), load(run / "status.json")
        self.assertEqual((status["state"], status["final_consistency_passed"]), ("completed", True), status)
        self.assertEqual(status["iteration"], iterations)
        self.assertEqual((status["compile"]["numerical_launch_performed"], status["compile"]["actual_GPU_qualified"]),
                         (False, False))
        self.assertTrue(load(run / "initial_qualification.json")["passed"])
        checkpoint = load(run / "optimization.json")
        self.assertEqual(checkpoint["controls"]["quartic_backend"], "persistent_trajectory")
        self.assertEqual(checkpoint["runs"][-1]["source_sha256"], plan["runtime_source_sha256"])
        self.assertEqual(len(plan["runtime_source_sha256"]), 22)
        validation = load(run / "final_validation.json")
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["source_sha256"], plan["runtime_source_sha256"])
        state = io.read_portable(run / "optimized_state")
        self.assertEqual((state.origin, state.D, state.model, state.validation_passed), ("portable", 8, "phi4", True))
        self.assertEqual(state.energy, validation["evaluations"][-1]["energy"])
        return plan, checkpoint

    def test_new_run_then_resume(self):
        first = self.root / "first"
        result = optimize(first, "--D", "8", "--max-iterations", "2")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(first, 2)
        self.assertIsNone(plan["initial"])
        self.assertEqual(Path(checkpoint["controls"]["reference"]), first / "initial")
        self.assertEqual(io.read_portable(first / "initial").source, "random Z2 state, seed 0")

        second = self.root / "second"
        result = optimize(second, "--D", "8", "--max-iterations", "1", "--resume-from", str(first / "optimization.json"))
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(second, 3)
        self.assertEqual(Path(plan["initial"]["path"]), first / "initial")   # a resume keeps its reference
        self.assertFalse((second / "initial").exists())
        self.assertEqual([run["starting_iteration"] for run in checkpoint["runs"]], [0, 2])


if __name__ == "__main__":
    unittest.main()
