"""The generic optimizer end to end on the CPU: rcmps-gpu-optimize --run-gpu at D = 8.

cli.main plans the run, worker.main compiles, qualifies the first evaluation against the
CuPy oracle, optimizes through trajectory_sweep_driver and validates the result through
validate_trajectory_sweep; then a second run resumes from the first checkpoint. Everything
runs as written except the leaves that need a GPU (a NumPy-backed cupy, the CPU oracle of
the trajectory kernel, the CPU kinetic solvers): see worker_on_cpu.py, which runs in a child
interpreter. This checks that the parts of the runtime the worker reaches fit together (for
example the backend label it passes and the labels the drivers accept), not the GPU kernels.
About 15 s per run on one thread.

Tiles: the first run is on a device of compute capability 8.0, where --tiles auto selects the
tensor-core tiles; the resume is on a device of compute capability 7.0, where it selects the
scalar tiles, which the checkpoint records for that run and the validation inherits; and
--tiles mma there is refused by the worker before anything is compiled or written. Register
caps: on the resume's device a kernel above 100 registers is not resident, so the worker
compiles with cap 96, the first of its ladder (none, 128, 96, 64) that fits, and every
evaluation of that run, qualification and validation included, uses it.
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


def optimize(output, *arguments, capability="8.0", registers=255):
    """worker_on_cpu.py: rcmps-gpu-optimize --output OUTPUT --run-gpu *arguments on the CPU,
    on a stand-in device of that compute capability where a kernel above `registers`
    registers is not resident."""
    result = subprocess.run([sys.executable, str(HELPER), rcmps_gpu.__file__, str(output), *arguments],
                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", RCMPS_CPU_COMPUTE_CAPABILITY=capability,
                                     RCMPS_CPU_MAX_REGISTERS=str(registers)),
                            capture_output=True, text=True, timeout=600)
    return result


def load(path):
    return json.loads(Path(path).read_bytes())


class WorkerOnCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def check_run(self, run, iterations, tiles, cap=None):
        plan, status = load(run / "plan.json"), load(run / "status.json")
        self.assertEqual((status["state"], status["final_consistency_passed"]), ("completed", True), status)
        self.assertEqual(status["iteration"], iterations)
        self.assertEqual((status["compile"]["numerical_launch_performed"], status["compile"]["actual_GPU_qualified"]),
                         (False, False))
        self.assertEqual((plan["schema_version"], plan["tiles"], plan["layout"]["tiles_requested"]), (4, "auto", "auto"))
        self.assertEqual(load(run / "compile.json"), status["compile"])
        self.assertEqual((status["compile"]["tiles"]["requested"], status["compile"]["tiles"]["resolved"],
                          status["compile"]["PTX"]["tiles"]), ("auto", tiles, tiles))
        qualification = load(run / "initial_qualification.json")
        self.assertTrue(qualification["passed"])
        self.assertEqual((qualification["tiles"], qualification["fast_stats"]["quartic"]["occupancy"]["tiles"]), (tiles, tiles))
        options = ["--std=c++17"] + ([f"--maxrregcount={cap}"] if cap else [])
        self.assertEqual((status["compile"]["register_cap"], status["compile"]["compiler_options"],
                          qualification["register_cap"], qualification["fast_stats"]["quartic"]["occupancy"]["compiler_options"]),
                         (cap, options, cap, options))
        self.assertEqual([row["register_cap"] for row in status["compile"]["register_caps_tried"]],
                         [None, 128, 96, 64][:[None, 128, 96, 64].index(cap) + 1])
        checkpoint = load(run / "optimization.json")
        self.assertEqual(checkpoint["controls"]["quartic_backend"], "persistent_trajectory")
        self.assertEqual(checkpoint["controls"]["trajectory_controls"]["tiles"], tiles)
        self.assertEqual(checkpoint["runs"][-1]["controls"]["trajectory_controls"]["tiles"], tiles)
        self.assertEqual(checkpoint["runs"][-1]["controls"]["trajectory_controls"]["register_cap"], cap)
        self.assertEqual(checkpoint["runs"][-1]["source_sha256"], plan["runtime_source_sha256"])
        self.assertEqual(len(plan["runtime_source_sha256"]), 22)
        validation = load(run / "final_validation.json")
        self.assertTrue(validation["passed"])
        self.assertEqual((validation["trajectory_method"]["tiles"], validation["trajectory_method"]["register_cap"]),
                         (tiles, cap))
        self.assertEqual(validation["source_sha256"], plan["runtime_source_sha256"])
        state = io.read_portable(run / "optimized_state")
        self.assertEqual((state.origin, state.D, state.model, state.validation_passed), ("portable", 8, "phi4", True))
        self.assertEqual(state.energy, validation["evaluations"][-1]["energy"])
        return plan, checkpoint

    def test_new_run_then_resume(self):
        first = self.root / "first"
        result = optimize(first, "--D", "8", "--max-iterations", "2")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(first, 2, "mma")
        self.assertIsNone(plan["initial"])
        self.assertEqual(Path(checkpoint["controls"]["reference"]), first / "initial")
        self.assertEqual(io.read_portable(first / "initial").source, "random Z2 state, seed 0")

        second = self.root / "second"
        # The resume runs on a GPU without FP64 tensor cores: the fork records the scalar tiles.
        result = optimize(second, "--D", "8", "--max-iterations", "1", "--resume-from", str(first / "optimization.json"),
                          capability="7.0", registers=100)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(second, 3, "scalar", 96)
        self.assertEqual(Path(plan["initial"]["path"]), first / "initial")   # a resume keeps its reference
        self.assertFalse((second / "initial").exists())
        self.assertEqual([run["starting_iteration"] for run in checkpoint["runs"]], [0, 2])
        self.assertEqual([run["controls"]["trajectory_controls"]["tiles"] for run in checkpoint["runs"]],
                         ["mma", "scalar"])

    def test_tensor_core_tiles_are_refused_below_compute_capability_8(self):
        refused = self.root / "refused"
        result = optimize(refused, "--D", "8", "--tiles", "mma", capability="7.5")
        self.assertNotEqual(result.returncode, 0)
        message = ("--tiles mma needs FP64 tensor-core instructions (mma.sync m8n8k4 f64, compute capability 8.0 "
                   "or later); this GPU is 7.5: use --tiles auto or scalar")
        self.assertIn(message, result.stderr)
        status = load(refused / "status.json")
        self.assertEqual((status["state"], status["error"]), ("failed", "ValueError: " + message))
        self.assertEqual(load(refused / "plan.json")["tiles"], "mma")
        self.assertEqual(sorted(path.name for path in refused.iterdir()), ["plan.json", "status.json"])
        # A size that no register cap makes resident is refused before anything else too.
        refused = self.root / "too_large"
        result = optimize(refused, "--D", "8", capability="7.0", registers=60)
        self.assertNotEqual(result.returncode, 0)
        status = load(refused / "status.json")
        self.assertEqual(status["state"], "failed")
        self.assertIn("RuntimeError: D=8 requires 11 resident blocks; this GPU can hold 0 at most (register caps "
                      "[None, 128, 96, 64] tried)", status["error"])
        self.assertEqual(sorted(path.name for path in refused.iterdir()), ["plan.json", "status.json"])


if __name__ == "__main__":
    unittest.main()
