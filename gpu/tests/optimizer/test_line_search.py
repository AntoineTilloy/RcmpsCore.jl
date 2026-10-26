"""The opt-in Wolfe line search of the optimizer's robust controller (--line-search wolfe), on the CPU.

runtime/robust_optimizer.py hager_zhang is OptimKit 0.5.1's Hager-Zhang search as RcmpsCore's
SafeguardedLineSearch builds it: on ten one-dimensional functions it takes the trial steps of
Julia, step for step (GOLDEN, printed by qualification/optimizer/hager_zhang_references.jl).
The acceptance rule is SafeguardedLineSearch's: exact Wolfe, or approximate Wolfe (energy
within the tolerance, slope between c2 and 2 c1 - 1 times the initial one), else the Armijo
backtracking. On a synthetic objective whose energy carries an evaluation noise and whose
gradient is exact, the Armijo controller stops at numerical stagnation far above the target and
the Wolfe one reaches it. The choice is recorded and a resume cannot change it (planner and
drivers). No CuPy, no GPU; test_worker_cpu.py runs a Wolfe optimization end to end.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from rcmps_gpu.optimizer import cli
from rcmps_gpu.optimizer.runtime import robust_optimizer as ro
from rcmps_gpu.optimizer.runtime import trajectory_sweep_driver as trajectory
from rcmps_gpu.optimizer.runtime.numerics import stationary_state
from rcmps_gpu.optimizer.runtime.optimizer import OptimizationControls

# hager_zhang_references.jl with OptimKit 0.5.1 (2026-10-26): returned step, trial steps.
GOLDEN = {
    "quadratic_min_3": (1.0, [1.0]),
    "quadratic_min_20": (20.0, [1.0, 5.0, 25.0, 20.0]),
    "exponential": (1.8632664652169924, [10.0, 0.0018160796403875106, 3.996368940101441, 0.30090627092923133,
                                         3.4256266595047533, 1.8632664652169924]),
    "offset_energy": (1.0, [3.0, 1.0]),
    "energy_wall": (0.5, [1.0, 0.5, 0.75, 0.625, 0.5625, 0.53125, 0.515625, 0.5078125, 0.50390625, 0.501953125,
                          0.5009765625, 0.50048828125, 0.500244140625, 0.5001220703125, 0.50006103515625,
                          0.500030517578125, 0.5000152587890625, 0.5000076293945312, 0.5000038146972656,
                          0.5000019073486328, 0.5000009536743164]),
    "cosine": (2.9234789216178396, [1.0, 5.0, 2.9234789216178396]),
    "linear": (9.5367431640625e10, [0.001, 0.005, 0.025, 0.125, 0.625, 3.125, 15.625, 78.125, 390.625, 1953.125,
                                    9765.625, 48828.125, 244140.625, 1.220703125e6, 6.103515625e6, 3.0517578125e7,
                                    1.52587890625e8, 7.62939453125e8, 3.814697265625e9, 1.9073486328125e10,
                                    9.5367431640625e10]),
    "steep_wall": (0.7941306864828948, [8.0, 0.010958904109589041, 4.005479452054795, 0.02304344617341001,
                                        2.014261449114102, 0.03819429765948915, 1.0262278733867958,
                                        0.06848090577392925, 0.5473543895803625, 0.5620334995789937,
                                        0.7941306864828948]),
    "gaussian_bump": (0.021240040215458077, [4.0, 0.42236681228753803, 0.21118340614376901, 0.021240040215458077]),
    "cubic_wall": (0.19999980926513672, [4.0, 0.375, 0.1875, 0.28125, 0.234375, 0.2109375, 0.19921875, 0.205078125,
                                         0.2021484375, 0.20068359375, 0.199951171875, 0.2003173828125,
                                         0.20013427734375, 0.200042724609375, 0.1999969482421875,
                                         0.20001983642578125, 0.20000839233398438, 0.20000267028808594,
                                         0.19999980926513672, 0.20000123977661133, 0.20000052452087402,
                                         0.20000016689300537]),
}


def wall(x, lower, upper, height):
    return height if lower < x < upper else 0.0


def bump(a):
    return math.exp(-((a - 0.3) / 0.15) * ((a - 0.3) / 0.15))


# The functions of hager_zhang_references.jl (energy, slope) and the first trial step.
CASES = {
    "quadratic_min_3": (lambda a: ((a - 3) * (a - 3), 2 * (a - 3)), 1.0),
    "quadratic_min_20": (lambda a: ((a - 20) * (a - 20), 2 * (a - 20)), 1.0),
    "exponential": (lambda a: (math.exp(a) - 5 * a, math.exp(a) - 5), 10.0),
    "offset_energy": (lambda a: (1e-12 * (a - 1) * (a - 1) + (0.0 if a == 0 else 5e-12), 2e-12 * (a - 1)), 3.0),
    "energy_wall": (lambda a: (-a + (1e3 if a > 0.5 else 0.0), -1.0), 1.0),
    "cosine": (lambda a: (math.cos(0.3 + a) + 0.1 * (0.3 + a), -math.sin(0.3 + a) + 0.1), 1.0),
    "linear": (lambda a: (-a, -1.0), 1e-3),
    "steep_wall": (lambda a: (-a + 50 * max(0.0, a - 0.7) * max(0.0, a - 0.7), -1 + 100 * max(0.0, a - 0.7)), 8.0),
    "gaussian_bump": (lambda a: (-3 * a + a * a * a / 3 + 3 * bump(a),
                                 -3 + a * a - 2 * 3 * (a - 0.3) / (0.15 * 0.15) * bump(a)), 4.0),
    "cubic_wall": (lambda a: (-1.5 * a + a * a * a / 3 + wall(a, 0.2, 0.5, 3.0), -1.5 + a * a), 4.0),
}


def search(f, alpha, epsilon=1e-10):
    """hager_zhang on phi = f: the point it returns, the origin, the trial steps and stages."""
    steps, stages = [], []

    def take(step, stage):
        steps.append(step)
        stages.append(stage)
        return ro.LinePoint(step, *f(step))
        yield   # a generator, as the controller's
    origin = ro.LinePoint(0.0, *f(0.0))
    events = ro.hager_zhang(take, origin, alpha, epsilon)
    while True:
        try:
            next(events)
        except StopIteration as stop:
            return stop.value, origin, steps, stages


class NoisyQuadratic:
    """E = -0.4 + |R - target|_F^2 plus a deterministic evaluation noise of size sigma, with the
    exact metric gradient (R - target)(rho + 1e-3)^-1, whose slope along a step w is the
    derivative of the noiseless energy: the situation of the GPU objective near a stiff minimum,
    where energy differences drown in their resolution and the gradient remains accurate."""

    def __init__(self, D=3, seed=7, sigma=1e-11):
        rng = np.random.default_rng(seed)
        self.target = 0.4 * (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        self.r = self.target + 0.3 * (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        h = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
        self.k = 0.2 * (h + h.conj().T)
        self.sigma, self.D = sigma, D

    def __call__(self, r, k, rho):
        delta = r - self.target
        noise = self.sigma * math.sin(1e7 * float(np.trace(r).real) + 3e6 * float(np.trace(k).imag))
        energy = -0.4 + float(np.vdot(delta, delta).real) + noise
        return energy, delta @ np.linalg.inv(rho + 1e-3 * np.eye(self.D)), {}

    def optimize(self, line_search, **controls):
        q = -1j * self.k - self.r.conj().T @ self.r / 2
        rho, _ = stationary_state(q, self.r, tolerance=1e-13)
        energy, gradient, _ = self(self.r, self.k, rho)
        controls = OptimizationControls(**{**dict(max_iterations=300, gradient_tol=1e-10, energy_atol=1e-14,
                                                  warm_start_line_search=line_search == "armijo",
                                                  refresh_curvatures=True), **controls})
        robustness = ro.RobustnessControls(line_search=line_search,
                                           wolfe_energy_atol=ro.WOLFE_ENERGY_ATOL if line_search == "wolfe" else None)
        return list(ro.optimize(self.r, self.k, rho, energy, gradient, self, controls, robustness=robustness))


def fake_checkpoint(root, robust_controls):
    """A stopped trajectory checkpoint of this runtime at D = 10 (no run behind it), as
    test_generic.test_generic_checkpoint_no_reset_fork_and_resume_controls builds one, whose
    robust controls are robust_controls."""
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    ref = root / "initial"
    ref.mkdir()
    D = 10
    rng = np.random.default_rng(415)
    r = np.asarray(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)), dtype=np.complex128)
    k, rho, gradient = np.zeros_like(r), np.eye(D, dtype=np.complex128) / D, .01 * r
    for name, value in (("R", r), ("K", k), ("rho", rho), ("saved_gradient", gradient)):
        value.ravel(order="F").astype("<c16").tofile(ref / (name + ".bin"))
    (ref / "metadata.toml").write_text("D = 10\ncoupling = 2.7\n")
    args = trajectory.ORIGINAL_PARSE([str(ref), "--backend", "cupy", "--mode", "hybrid", "--packed-z2",
                                      "--quartic-backend", "persistent_trajectory", "--arnoldi", "gpu",
                                      "--projection", "hessenberg_gpu", "--window", "20"])
    args.trajectory_controls = dict(enabled=True, whole_trajectory_cooperative_kernel=True, kinetic="cooperative",
                                    max_device_seconds=300.)
    args.overlap_controls = dict(enabled=True, controller="robust")
    args.robust_controls = robust_controls
    hashes, controls = trajectory.source_hashes(args), vars(args)
    empty = np.empty((0, D, D), dtype=np.complex128)
    state = dict(R=r, K=k, rho=rho, gradient=gradient, energy=np.asarray(-.1), iteration=np.asarray(2),
                 lbfgs_steps=empty, lbfgs_changes=empty, lbfgs_inverse_curvatures=np.empty(0),
                 next_step=np.asarray(1.), precond_reg=np.asarray(1e-4), optimizer_seconds=np.asarray(17.),
                 optimizer_evaluations=np.asarray(3))
    old = root / "old.json"
    np.savez(old.with_suffix(".npz"), **state)
    report = dict(reference=dict(D=D, coupling=2.7), controls=controls, source_sha256=hashes,
                  runs=[dict(source_sha256=hashes, controls=controls)], history=[{}, {}], final_energy=-.1,
                  stop_reason="iteration limit", reference_sha256={p.name: sha(p) for p in ref.iterdir()})
    old.write_text(json.dumps(report))
    return old, ref


class HagerZhang(unittest.TestCase):
    def test_trial_steps_are_those_of_OptimKit(self):
        self.assertEqual(set(CASES), set(GOLDEN))
        reached = set()
        for name, (f, alpha) in CASES.items():
            point, _, steps, stages = search(f, alpha)
            self.assertEqual(steps, GOLDEN[name][1], name)
            self.assertEqual(point.alpha, GOLDEN[name][0], name)
            reached.update(zip(stages, stages[1:]))
        # Every branch of the search: acceptance of the first step, expansion, bisection from
        # the bracket and from an update, the secant updates.
        self.assertLessEqual({("initial", "bracket"), ("bracket", "bracket"), ("bracket", "update"),
                              ("initial", "bisect"), ("bisect", "bisect"), ("update", "bisect"),
                              ("initial", "update"), ("update", "update")}, reached)

    def test_the_parameters_are_those_of_RcmpsCore(self):
        hz = ro.HAGER_ZHANG
        self.assertEqual((hz.c1, hz.c2, hz.theta, hz.gamma, hz.rho, hz.max_iterations, hz.max_evaluations),
                         (0.1, 0.9, 0.5, 2 / 3, 5.0, 10, 20))
        self.assertEqual(ro.WOLFE_ENERGY_ATOL, 1e-10)
        self.assertEqual((cli.LINE_SEARCHES, cli.WOLFE_ENERGY_ATOL), (ro.LINE_SEARCHES, ro.WOLFE_ENERGY_ATOL))

    def test_acceptance(self):
        # An energy evaluated 5e-12 high everywhere but at the origin, above any Armijo bound:
        # the slope (0 at the minimum) accepts the step as approximately Wolfe.
        f = CASES["offset_energy"][0]
        point, origin, _, _ = search(f, 3.0)
        self.assertEqual(ro.wolfe_conditions(point, origin, 1e-10), (False, True))
        self.assertEqual(ro.wolfe_acceptance(point, origin, 1e-10), "approximate")
        self.assertGreater(point.energy, origin.energy + 1e-4 * point.alpha * origin.slope)
        self.assertIsNone(ro.wolfe_acceptance(point, origin, 1e-12))   # beyond the energy tolerance
        # The quadratic minimum is exactly Wolfe.
        point, origin, _, _ = search(CASES["quadratic_min_20"][0], 1.0)
        self.assertEqual(ro.wolfe_acceptance(point, origin, 1e-10), "exact")
        # A search stopped by its limits returns a point without either condition (its slope
        # is the initial one): the Armijo backtracking then runs.
        point, origin, steps, _ = search(CASES["energy_wall"][0], 1.0)
        self.assertEqual((point.alpha, len(steps)), (0.5, 21))
        self.assertEqual(ro.wolfe_conditions(point, origin, 1e-10), (False, False))
        self.assertIsNone(ro.wolfe_acceptance(point, origin, 1e-10))
        self.assertIsNone(ro.wolfe_acceptance(origin, origin, 1e-10))
        # Slopes beyond (2 c1 - 1) slope0 = 0.8 |slope0| overshoot; below c2 slope0 the step is short.
        origin = ro.LinePoint(0.0, 0.0, -1.0)
        for slope, condition in ((0.81, None), (0.79, "approximate"), (-0.91, None), (-0.89, "approximate")):
            self.assertEqual(ro.wolfe_acceptance(ro.LinePoint(1.0, 1e-11, slope), origin, 1e-10), condition, slope)
        self.assertEqual(ro.wolfe_acceptance(ro.LinePoint(1.0, -0.5, -0.5), origin, 0.0), "exact")

    def test_controls(self):
        self.assertEqual((ro.RobustnessControls().line_search, ro.RobustnessControls().wolfe_energy_atol),
                         ("armijo", None))
        for kwargs in (dict(line_search="wolfe"), dict(line_search="armijo", wolfe_energy_atol=1e-10),
                       dict(line_search="wolfe", wolfe_energy_atol=-1.0),
                       dict(line_search="wolfe", wolfe_energy_atol=math.inf), dict(line_search="hager_zhang")):
            with self.assertRaises(ValueError, msg=kwargs):
                ro.RobustnessControls(**kwargs)
        ro.RobustnessControls(line_search="wolfe", wolfe_energy_atol=0.0)


class Controller(unittest.TestCase):
    def test_the_Wolfe_search_converges_where_energy_differences_stagnate(self):
        problem = NoisyQuadratic()
        armijo = problem.optimize("armijo")
        stop = armijo[-1]
        self.assertEqual((stop["type"], stop["converged"]), ("stopped", False))
        self.assertTrue(stop["reason"].startswith("Numerical stagnation"), stop["reason"])
        self.assertGreater(stop["state"]["gradient_norm"], 1e-7)
        self.assertFalse(any("line_search" in event for event in armijo))   # the Armijo events are unchanged

        wolfe = problem.optimize("wolfe")
        stop = wolfe[-1]
        self.assertEqual((stop["reason"], stop["converged"]), ("gradient tolerance reached", True))
        self.assertLessEqual(stop["state"]["gradient_norm"], 1e-10)
        accepted = [event for event in wolfe if event["type"] == "accepted"]
        trials = [event for event in wolfe if event["type"] == "trial"]
        self.assertEqual({event["line_search"] for event in accepted}, {"wolfe"})
        self.assertEqual({event["wolfe_condition"] for event in accepted}, {"exact", "approximate", None})
        # The first step (the preconditioned gradient of norm 0.01) is expanded to the step bound,
        # where the search ends and the Armijo backtracking accepts step 1, as in RcmpsCore.
        self.assertEqual((accepted[0]["wolfe_condition"], accepted[0]["armijo_fallback"], accepted[0]["alpha"]),
                         (None, True, 1.0))
        self.assertIn("exceeds the step bound", accepted[0]["wolfe_failure"])
        self.assertEqual([event["line_search"] for event in trials if event["iteration"] == 1][-1], "armijo")
        # Accepted approximate steps raise the energy by less than the tolerance; exact ones lower it.
        for event in accepted[1:]:
            if event["wolfe_condition"] == "approximate":
                self.assertLess(-event["actual_energy_decrease"], ro.WOLFE_ENERGY_ATOL)
        # Each iteration ends its search with one result row, and every evaluation is a trial row.
        results = [event for event in trials if event.get("stage") == "result"]
        self.assertEqual(len(results), len(accepted))
        evaluated = [event for event in trials if "energy" in event]
        self.assertEqual(len(evaluated), stop["state"]["evaluations"])
        self.assertTrue(all(isinstance(event[key], bool) for event in evaluated if event.get("line_search") == "wolfe"
                            for key in ("exact_wolfe", "approximate_wolfe", "accepted")))
        json.dumps([{key: value for key, value in event.items() if key not in ("state", "stats", "stationary")}
                    for event in wolfe])   # the checkpoint rows encode


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, *arguments):
        return cli.prepare(cli.parser().parse_args(["--D", "10", *arguments]), cli.runtime_source_sha256())

    def test_planner(self):
        plan = self.plan()
        self.assertEqual((plan["schema_version"], plan["line_search"], plan["wolfe_energy_atol"]), (6, "armijo", None))
        plan = self.plan("--line-search", "wolfe")
        self.assertEqual((plan["line_search"], plan["wolfe_energy_atol"]), ("wolfe", 1e-10))
        self.assertEqual(self.plan("--line-search", "wolfe", "--wolfe-energy-atol", "1e-9")["wolfe_energy_atol"], 1e-9)
        with self.assertRaisesRegex(ValueError, "--wolfe-energy-atol requires --line-search wolfe"):
            self.plan("--wolfe-energy-atol", "1e-9")
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            self.plan("--line-search", "wolfe", "--wolfe-energy-atol", "-1")

    def test_a_resume_keeps_the_line_search_of_its_checkpoint(self):
        old, _ = fake_checkpoint(self.root, dict(line_search="wolfe", wolfe_energy_atol=1e-9))
        plan = self.plan("--resume-from", str(old))
        self.assertEqual((plan["line_search"], plan["wolfe_energy_atol"]), ("wolfe", 1e-9))
        self.assertEqual(self.plan("--resume-from", str(old), "--wolfe-energy-atol", "1e-10")["wolfe_energy_atol"], 1e-10)
        with self.assertRaisesRegex(ValueError, r"another --line-search \(the checkpoint's is wolfe\)"):
            self.plan("--resume-from", str(old), "--line-search", "armijo")

    def test_a_checkpoint_without_the_choice_ran_the_Armijo_search(self):
        old, ref = fake_checkpoint(self.root, dict(secant_gradient_floor=1e-10, max_recoveries=2))
        plan = self.plan("--resume-from", str(old), "--line-search", "armijo")
        self.assertEqual((plan["line_search"], plan["wolfe_energy_atol"]), ("armijo", None))
        with self.assertRaisesRegex(ValueError, r"another --line-search \(the checkpoint's is armijo\)"):
            self.plan("--resume-from", str(old), "--line-search", "wolfe")
        # The drivers refuse it too, before the fork writes anything.
        new = self.root / "new.json"
        base = [str(ref), "--backend", "cupy", "--mode", "hybrid", "--packed-z2", "--quartic-backend",
                "persistent_trajectory", "--arnoldi", "gpu", "--projection", "hessenberg_gpu", "--window", "20",
                "--resume-from", str(old), "--output", str(new)]
        with self.assertRaisesRegex(ValueError, r"resume cannot change the line search \(the checkpoint's is armijo\)"):
            trajectory.main([*base, "--controller", "robust", "--line-search", "wolfe"])
        with self.assertRaisesRegex(ValueError, "--wolfe-energy-atol requires --line-search wolfe"):
            trajectory.main([*base, "--controller", "robust", "--wolfe-energy-atol", "1e-9"])
        with self.assertRaisesRegex(ValueError, "robust controls require --controller robust"):
            trajectory.main([*base, "--line-search", "wolfe"])
        self.assertFalse(new.exists() or new.with_suffix(".npz").exists())
        self.assertNotIn("cupy", sys.modules)


if __name__ == "__main__":
    unittest.main()
