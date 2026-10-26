"""The generic optimizer end to end on the CPU: rcmps-gpu-optimize --run-gpu at D = 8.

cli.main plans the run, worker.main compiles, qualifies the first evaluation against the
CuPy oracle, optimizes through trajectory_sweep_driver and validates the result through
validate_trajectory_sweep; then a second run resumes from the first checkpoint. Everything
runs as written except the leaves that need a GPU (a NumPy-backed cupy, the CPU oracle of
the trajectory kernel, the CPU kinetic solvers): see worker_on_cpu.py, which runs in a child
interpreter. This checks that the parts of the runtime the worker reaches fit together (for
example the backend label it passes and the labels the drivers accept), not the GPU kernels.
About 15 s per run on one thread.

States without Z2 symmetry: a run at D = 5 with --symmetry none from a random state, on the
unpacked channels (the trajectory oracle, the CuPy oracle of the qualification and the ODE of
<phi> in the validation all on _ReducedQuarticRHS or the batched channels), whose history,
validation and optimized_state/ record <phi>; a Z2 run refuses its state (test_generic.py checks
the other refusals of the planner).

Line search: a D = 5 run with --line-search wolfe (the Hager-Zhang search of the robust
controller, Armijo backtracking when it fails), whose checkpoint records the choice, its
parameters and every search, each started at step 1; a resume keeps it and refuses the other.

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

    def check_run(self, run, iterations, tiles, cap=None, D=8, symmetry="z2"):
        plan, status = load(run / "plan.json"), load(run / "status.json")
        self.assertEqual((status["state"], status["final_consistency_passed"]), ("completed", True), status)
        self.assertEqual(status["iteration"], iterations)
        self.assertEqual((status["compile"]["numerical_launch_performed"], status["compile"]["actual_GPU_qualified"]),
                         (False, False))
        self.assertEqual((plan["schema_version"], plan["tiles"], plan["layout"]["tiles_requested"]), (6, "auto", "auto"))
        self.assertEqual((plan["symmetry"], status["compile"]["symmetry"], status["compile"]["PTX"]["unpacked"]),
                         (symmetry, symmetry, symmetry == "none"))
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
        self.assertEqual((checkpoint["controls"]["packed_z2"], checkpoint["controls"]["reduced"]),
                         (symmetry == "z2", symmetry == "none"))
        self.assertTrue(all("phi" in row for row in checkpoint["history"]))
        self.assertEqual(checkpoint["controls"]["trajectory_controls"]["tiles"], tiles)
        self.assertEqual(checkpoint["runs"][-1]["controls"]["trajectory_controls"]["tiles"], tiles)
        self.assertEqual(checkpoint["runs"][-1]["controls"]["trajectory_controls"]["register_cap"], cap)
        self.assertEqual(checkpoint["runs"][-1]["source_sha256"], plan["runtime_source_sha256"])
        # The Wolfe search starts at step 1, the Armijo backtracking at twice the last accepted step.
        self.assertEqual((checkpoint["controls"]["robust_controls"]["line_search"],
                          checkpoint["controls"]["robust_controls"]["wolfe_energy_atol"],
                          checkpoint["controls"]["warm_start_line_search"]),
                         (plan["line_search"], plan["wolfe_energy_atol"], plan["line_search"] == "armijo"))
        self.assertEqual(len(plan["runtime_source_sha256"]), 22)
        validation = load(run / "final_validation.json")
        self.assertTrue(validation["passed"])
        self.assertEqual((validation["trajectory_method"]["tiles"], validation["trajectory_method"]["register_cap"]),
                         (tiles, cap))
        self.assertEqual(validation["source_sha256"], plan["runtime_source_sha256"])
        state = io.read_portable(run / "optimized_state")
        self.assertEqual((state.origin, state.D, state.model, state.validation_passed), ("portable", D, "phi4", True))
        self.assertEqual(state.energy, validation["evaluations"][-1]["energy"])
        self.assertEqual((state.symmetry, validation["symmetry"]), ("none", "none") if symmetry == "none" else ("Z2", "Z2"))
        self.assertEqual(state.phi, validation["phi"])
        self.assertEqual(state.phi, validation["evaluations"][-1]["phi"])
        self.assertLess(validation["phi_ode_difference"], 1e-12)
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

    def test_a_state_without_symmetry_at_odd_D(self):
        run = self.root / "none"
        result = optimize(run, "--D", "5", "--symmetry", "none", "--seed", "4", "--coupling", "3.0",
                          "--max-iterations", "2")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(run, 2, "mma", D=5, symmetry="none")
        self.assertEqual((plan["layout"]["blocks"], plan["layout"]["layout"], plan["layout"]["symmetry"]),
                         (8, "unpacked", "none"))
        initial = io.read_portable(run / "initial")
        self.assertEqual((initial.symmetry, initial.source), ("none", "random state, seed 4"))
        self.assertTrue(initial.R.all() and initial.K.all())   # no zero blocks
        self.assertEqual(checkpoint["final_phi"], checkpoint["history"][-1]["phi"])
        self.assertNotEqual(checkpoint["final_phi"], 0.)
        # A Z2 run refuses its optimized state when planning (here already for its odd D).
        refused = self.root / "refused"
        result = optimize(refused, "--D", "5", "--initial", str(run / "optimized_state"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(refused.exists())
        self.assertIn("the packed Z2 implementation requires an even integer D >= 2", result.stderr)

    def test_the_Wolfe_line_search(self):
        run = self.root / "wolfe"
        result = optimize(run, "--D", "5", "--symmetry", "none", "--seed", "4", "--coupling", "3.0",
                          "--max-iterations", "4", "--line-search", "wolfe")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
        plan, checkpoint = self.check_run(run, 4, "mma", D=5, symmetry="none")
        self.assertEqual((plan["line_search"], plan["wolfe_energy_atol"]), ("wolfe", 1e-10))
        self.assertEqual(checkpoint["controls"]["robust_controls"]["hager_zhang"],
                         dict(c1=0.1, c2=0.9, theta=0.5, gamma=2 / 3, rho=5.0, max_iterations=10, max_evaluations=20))
        history, trials = checkpoint["history"], checkpoint["trial_history"]
        self.assertEqual({row["line_search"] for row in history}, {"wolfe"})
        self.assertTrue({row["wolfe_condition"] for row in history} & {"exact", "approximate"})
        self.assertEqual(len([row for row in trials if row.get("stage") == "result"]), len(history))
        self.assertTrue({"initial", "bracket"} <= {row.get("stage") for row in trials})
        # Each search starts at step 1, or at the step bound below it.
        first = [row["alpha"] for row in trials if row.get("stage") == "initial"]
        self.assertEqual(len(first), len(history))
        self.assertIn(1.0, first)
        self.assertLessEqual(max(first), 1.0)
        self.assertEqual(checkpoint["objective_evaluations"], sum("energy" in row for row in trials) + 1)
        # A resume keeps the line search; the other one is refused when planning.
        from rcmps_gpu.optimizer import cli
        resumed = cli.prepare(cli.parser().parse_args(["--D", "5", "--symmetry", "none", "--resume-from",
                                                       str(run / "optimization.json")]), cli.runtime_source_sha256())
        self.assertEqual((resumed["line_search"], resumed["wolfe_energy_atol"]), ("wolfe", 1e-10))
        refused = self.root / "refused"
        result = optimize(refused, "--D", "5", "--symmetry", "none", "--resume-from", str(run / "optimization.json"),
                          "--line-search", "armijo")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(refused.exists())
        self.assertIn("cannot resume an optimizer with another --line-search (the checkpoint's is wolfe)",
                      result.stderr)

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
