"""Integration tests of the validation path of eigsolve.py against the dense oracle.

excitation_eigsolve(validate=True) joins the two matrix-free routes: the Davidson search
and the fresh actions at the standard and refined settings (hamiltonian.py), and the
independent bilinear energies at both settings (bilinear.py), with the acceptance rule
and the refinement steps of solve_levels (src/excitation_eigsolve.jl:302-401). Every
number of the returned report is compared with dense_reference.eigsolve, the literal
port of the Julia driver on dense factors; the cited Julia tests are those of
test/excitation_eigsolve.jl.

States and settings are those of test_eigsolve.py: random_state(4, seed=11, scaling=0.5,
z2=True) (z2_test_state, cond(rho) = 14, sector dimension 8) and random_state(3, seed=5,
scaling=0.5) without parity, window max_point = 14, n_quad = 32, atol = 1e-13,
rtol = 1e-12, 32 kinetic nodes; refined settings are refined_integration (window 24,
n_quad 64, tolerances / 10) and 64 nodes. `scaling`: 'none' (S = 1), 'density'
(S = sqrt(diag rho)) and SKEW = exp(linspace(-2.5, 1.5, D)).

Both drivers run the same deterministic Davidson iteration from the same start, so the
operation counts and flags are equal and the numbers agree far inside the asserted 1e-8
(measured: 7e-13 on the validation energies and 7e-12 on the Davidson history at D <= 4,
4e-12 on the validation energies at D = 6).
RCMPS_EXCITATION_REPORT=1 prints the largest measured difference of every comparison.
The complementary momenta of the two sectors and the D = 6 state need
RCMPS_EXCITATION_SLOW=1 (about two minutes more).
"""
from __future__ import annotations

import functools
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bilinear
import dense_reference as ref
import eigsolve as es
import models
import ode
import potential
from davidson import DavidsonOptions
from states import SolverFailure, make_state, random_state

slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (more than about a minute)")
FREE, PHI4 = models.phi4_terms(0.0), models.phi4_terms(1.0)
NODES = 32
TOL = 1e-8
WINDOW = dict(max_point=14, n_quad=32, atol=1e-13, rtol=1e-12)
FOUR = ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")
NUMBERS = FOUR + ("standard_residual", "refined_residual", "energy_spread", "max_imaginary")
FLAGS = ("residual_target_met", "consistent", "accepted")
SAME = ("nev converged stop_reason refined_converged accepted numops action_calls start momentum sector gauge "
        "dimension metric_rank kinetic_nodes refined_kinetic_nodes validation_tol acceptance_residual davidson "
        "integration refined_integration").split()
REPORT = {}


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for what, error in REPORT.items():
            print(f"{what:46s} largest difference = {error:.2e}", file=sys.stderr)


def tuned(state, **controls):
    for name, value in controls.items():
        setattr(state.integrator, name, value)
    return state


def z2_state():
    s = random_state(4, seed=11, scaling=0.5, z2=True)
    s.coupling = 1.0
    return tuned(s, **WINDOW)


def skew(d):
    return np.exp(np.linspace(-2.5, 1.5, d))


def label(scaling):
    return scaling if isinstance(scaling, str) else "skew"


Z2 = z2_state()
PLAIN = tuned(random_state(3, seed=5, scaling=0.5), **WINDOW)


@functools.lru_cache(maxsize=None)
def oracle(sector, p, nev):
    """dense_reference.eigsolve of the Z2 state with validation, in the density gauge."""
    return ref.eigsolve(Z2, p, PHI4, nev=nev, sector=sector, kinetic_nodes=NODES)


class Case(unittest.TestCase):
    def assertClose(self, x, y, tol, what):
        """max |x - y| <= tol with nan matching nan; the largest distance is kept for the report."""
        x, y = np.asarray(x, dtype=np.complex128), np.asarray(y, dtype=np.complex128)
        self.assertEqual(x.shape, y.shape, what)
        self.assertTrue(np.array_equal(np.isnan(x), np.isnan(y)), what)
        finite = ~np.isnan(x)
        error = float(np.abs(x[finite] - y[finite]).max()) if finite.any() else 0.0
        REPORT[what] = max(REPORT.get(what, 0.0), error)
        self.assertTrue(error <= tol, f"{what}: max|x-y| = {error:.3e} exceeds {tol:.1e}")

    def assertSameReport(self, r, dense, state, tol=TOL):
        """Every entry of the oracle's report: identical bookkeeping and flags, numbers within
        `tol`, the same vectors up to a phase."""
        self.assertEqual(sorted(set(dense) - set(r)), [])
        for key in SAME:
            self.assertEqual(r[key], dense[key], key)
        self.assertEqual(r["parity"] is None, dense["parity"] is None)
        if dense["parity"] is not None:
            self.assertTrue(np.array_equal(r["parity"], dense["parity"]))
        for key in ("parity_error", "discarded_metric_weight", "metric_defect"):
            self.assertClose(r[key], dense[key], 1e-12, key)
        self.assertClose(r["density_condition"] / dense["density_condition"], 1.0, 1e-10, "density_condition")
        for key in ("energies", "residuals"):
            self.assertClose(r[key], dense[key], tol, key)
        self.assertEqual(len(r["history"]), len(dense["history"]))
        for row, expected in zip(r["history"], dense["history"]):
            self.assertEqual([row[k] for k in ("count", "cycle", "dimension")],
                             [expected[k] for k in ("count", "cycle", "dimension")])
            self.assertClose(row["energies"], expected["energies"], tol, "history energies")
            self.assertClose(row["residuals"], expected["residuals"], tol, "history residuals")
        self.assertEqual(r["preconditioner"] is None, dense["preconditioner"] is None)
        if dense["preconditioner"] is not None:
            pre, expected = r["preconditioner"], dense["preconditioner"]
            self.assertEqual((pre["nodes"], pre["floor"]), (expected["nodes"], expected["floor"]))
            self.assertClose(pre["minimum"], expected["minimum"], tol, "preconditioner minimum")
            self.assertLess(pre["skew"], TOL)
        self.assertEqual(len(r["corrections"]), len(dense["corrections"]))
        for row, expected in zip(r["corrections"], dense["corrections"]):
            self.assertEqual((row["step"], row["accepted"]), (expected["step"], expected["accepted"]))
            self.assertClose([row["energy"], row["residual"]], [expected["energy"], expected["residual"]], tol,
                             "corrections")
        self.assertEqual(r["validation"] is None, dense["validation"] is None)
        if dense["validation"] is None:
            self.assertIsNone(r["validated_energies"])
        else:
            self.assertEqual(len(r["validation"]), len(dense["validation"]))
            for v, expected in zip(r["validation"], dense["validation"]):
                self.assertEqual(sorted(v), sorted(expected))
                for name in NUMBERS:
                    self.assertClose(v[name], expected[name], tol, name)
                self.assertEqual([v[name] for name in FLAGS], [expected[name] for name in FLAGS])
            self.assertClose(r["validated_energies"], dense["validated_energies"], tol, "validated_energies")
        levels = len(dense["vectors"])
        self.assertEqual((len(r["vectors"]), len(r["coefficients"])), (levels, levels))
        unit = np.eye(levels)
        self.assertClose([[abs(np.vdot(x, y)) for y in dense["coefficients"]] for x in r["coefficients"]], unit,
                         1e-6, "coefficient overlaps")
        self.assertClose([[abs(ref.tangent_metric(state, w, u)) for u in dense["vectors"]] for w in r["vectors"]],
                         unit, 1e-6, "vector overlaps")
        self.assertClose([[ref.tangent_metric(state, w, u) for u in r["vectors"]] for w in r["vectors"]], unit,
                         1e-10, "metric of the vectors")
        self.assertEqual(sorted(r["timings"]), ["context", "operator", "preconditioner", "solve", "validation"])

    def assertValidated(self, r, tol=TOL):
        """The definitions of src/excitation_eigsolve.jl:360-378 applied to the reported
        numbers and, unless tol is None (synthetic levels), the loop of
        test/excitation_eigsolve.jl:185-190: each bilinear energy equals the fresh one at
        its settings and the imaginary parts vanish."""
        self.assertEqual(len(r["validation"]), r["nev"])
        for k, v in enumerate(r["validation"]):
            four = [v[name] for name in FOUR]
            residual = max(v["standard_residual"], v["refined_residual"])
            self.assertEqual(v["energy_spread"], max(four) - min(four))
            self.assertEqual(v["consistent"], v["energy_spread"] <= r["validation_tol"]
                             and v["max_imaginary"] <= r["validation_tol"])
            self.assertEqual(v["residual_target_met"], residual <= r["validation_tol"])
            self.assertEqual(v["accepted"], v["consistent"] and residual <= r["acceptance_residual"])
            if tol is None:
                continue
            self.assertLess(abs(v["bilinear_energy"] - v["standard_energy"]), tol)
            self.assertLess(abs(v["refined_bilinear_energy"] - v["refined_energy"]), tol)
            self.assertLess(v["max_imaginary"], tol)
            if r["refined_converged"] is None:
                self.assertLess(abs(v["standard_energy"] - r["energies"][k]), tol)
        self.assertEqual(r["accepted"], all(v["accepted"] for v in r["validation"]))
        self.assertEqual(r["validated_energies"], [v["refined_bilinear_energy"] for v in r["validation"]])
        self.assertIs(es.reported_energies(r), r["validated_energies"])
        if r["accepted"]:
            self.assertIs(es.accepted_energies(r), r["validated_energies"])
        else:
            with self.assertRaises(SolverFailure):
                es.accepted_energies(r)


class ValidatedLevels(Case):
    """excitation_eigsolve with validation in both parity sectors, at p = 0 and p = 0.5, for
    S = 1 and S = sqrt(rho): loop of test/excitation_eigsolve.jl:177-195 with nev = 2."""

    def check(self, sector, p, nev, scalings=("none", "density")):
        dense = oracle(sector, p, nev)
        self.assertTrue(dense["converged"] and dense["accepted"])
        for scaling in scalings:
            with self.subTest(sector=sector, p=p, nev=nev, scaling=label(scaling)):
                lines = []
                r = es.excitation_eigsolve(Z2, p, PHI4, nev=nev, sector=sector, scaling=scaling, kinetic_nodes=NODES,
                                           log=lines.append)
                self.assertSameReport(r, dense, Z2)
                self.assertValidated(r)
                self.assertTrue(r["converged"] and r["accepted"])
                self.assertTrue(all(v["accepted"] and v["consistent"] and v["residual_target_met"]
                                    for v in r["validation"]))
                self.assertEqual((r["scaling"], r["start"]), (label(scaling).replace("skew", "vector"),
                                                              "field" if sector == "odd" else "kinetic"))
                self.assertEqual(r["action_calls"], r["numops"] + nev)   # one standard fresh action per level
                self.assertEqual((r["integration"].max_point, r["integration"].n_quad), (14, 32))
                fine = r["refined_integration"]
                self.assertEqual((fine.max_point, fine.n_quad, fine.atol, fine.rtol, r["refined_kinetic_nodes"]),
                                 (24, 64, 1e-13 / 10, 1e-12 / 10, 2 * NODES))
                self.assertEqual(sum(line.startswith("validation ") for line in lines), nev)
                self.assertEqual(sum(line.startswith("refined operator ") for line in lines), 1)
                self.assertGreater(r["timings"]["validation"], 0.0)
        return dense

    def test_odd_sector(self):
        """Odd sector (start: field insertion) at p = 0.5, nev = 2: four validation energies
        per level (1e-8), accepted flags and validated energies are the oracle's, also for
        the arbitrary scaling SKEW."""
        self.check("odd", 0.5, 2, ("none", "density", skew(4)))

    def test_even_sector(self):
        """Even sector (start: coarse kinetic modes) at p = 0, nev = 2."""
        self.check("even", 0.0, 2)

    def test_four_levels(self):
        """test/excitation_eigsolve.jl:241-245: nev = 4 with the default options (maxspace 10)
        in the odd sector at p = 0; every level is accepted and consistent."""
        dense = self.check("odd", 0.0, 4, ("density",))
        self.assertEqual(dense["davidson"].maxspace, 10)

    @slow
    def test_other_momenta(self):
        """The remaining sector and momentum pairs: odd at p = 0 and even at p = 0.5."""
        self.check("odd", 0.0, 2)
        self.check("even", 0.5, 2)

    def test_free_dispersion(self):
        """test/excitation_eigsolve.jl:213-218 on the D = 1 coherent state R = 0.3 + 0.1i
        instead of the vacuum (Q = 0 has no decaying mode: ScaledState rejects it): the free
        model has the exact dispersion sqrt(1 + p^2) in the one-dimensional tangent space
        (1e-11), for the Davidson level and the four validation energies, with the library
        defaults (256 and 512 kinetic nodes)."""
        coherent = make_state([[0.3 + 0.1j]], [[0.0]])
        for p in (0.0, 0.7):
            r = es.excitation_eigsolve(coherent, p, FREE)
            self.assertEqual((r["sector"], r["dimension"], r["start"], r["numops"]), ("all", 1, "field", 1))
            self.assertTrue(r["converged"] and r["accepted"])
            self.assertValidated(r, 1e-11)
            exact = math.sqrt(1 + p * p)
            self.assertClose(r["energies"], [exact], 1e-11, "free dispersion")
            self.assertClose([r["validation"][0][name] for name in FOUR], [exact] * 4, 1e-11, "free dispersion")
            self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"]), (256, 512))
        with self.assertRaises(SolverFailure):
            es.excitation_eigsolve(make_state([[0.0]], [[0.0]]), 0.0, FREE)

    def test_validated_energy_of_the_returned_vector(self):
        """The validated energy is the oracle's bilinear energy, at the refined settings, of
        the returned tangent W, evaluated on the state in its input gauge (1e-8):
        excitation_matrix_element of test/excitation_eigsolve.jl:194 at the refined settings."""
        p = 0.5
        dense = oracle("odd", p, 2)
        r = es.excitation_eigsolve(Z2, p, PHI4, kinetic_nodes=NODES, start=dense["coefficients"][0])
        self.assertEqual((r["start"], r["numops"], r["accepted"]), ("supplied", 1, True))
        w = r["vectors"][0]
        ctx = ref.make_context(Z2, p)
        for name, context, nodes in (("bilinear_energy", ctx, NODES),
                                     ("refined_bilinear_energy", ref.retarget(ctx, integrator=r["refined_integration"]),
                                      2 * NODES)):
            value = (ref.matrix_element(ref.make_pair(context, w, w), PHI4, kinetic_nodes=nodes)
                     / ref.tangent_metric(Z2, w, w))
            self.assertClose(r["validation"][0][name], value.real, TOL, f"{name} of the returned vector")
            self.assertLess(abs(value.imag), TOL)
        self.assertClose(r["validated_energies"], dense["validated_energies"][:1], TOL, "validated_energies")

    def test_no_parity(self):
        """A state without parity at p = 0.5 (full tangent space of dimension 9 > maxspace:
        restarts) with validation, S = sqrt(rho). On this state the window 14 -> 24 moves
        the level by 2.2e-7: the converged level is not consistent at validation_tol = 1e-7
        and not accepted, as in the oracle, while each bilinear energy equals the fresh one
        at its settings; the same vector is accepted at validation_tol = 1e-6."""
        dense = ref.eigsolve(PLAIN, 0.5, PHI4, kinetic_nodes=NODES)
        r = es.excitation_eigsolve(PLAIN, 0.5, PHI4, kinetic_nodes=NODES)
        self.assertSameReport(r, dense, PLAIN)
        self.assertValidated(r)
        self.assertEqual((r["sector"], r["dimension"], r["start"], r["scaling"]), ("all", 9, "field", "density"))
        self.assertTrue(r["converged"] and any(row["cycle"] > 1 for row in r["history"]))
        v, = r["validation"]
        self.assertTrue(1e-7 < v["energy_spread"] < 1e-6 and 1e-7 < v["refined_residual"] < 1e-6)
        self.assertFalse(v["consistent"] or v["residual_target_met"] or v["accepted"] or r["accepted"])
        wide = es.excitation_eigsolve(PLAIN, 0.5, PHI4, kinetic_nodes=NODES, validation_tol=1e-6,
                                      start=r["coefficients"][0])
        self.assertEqual((wide["numops"], wide["converged"]), (1, True))
        self.assertValidated(wide)
        self.assertClose([wide["validation"][0][name] for name in FOUR], [v[name] for name in FOUR], 1e-10,
                         "warm start")
        self.assertTrue(wide["accepted"] and wide["validation"][0]["consistent"]
                        and wide["validation"][0]["residual_target_met"])

    @slow
    def test_restarted_search(self):
        """The 18-dimensional odd sector of random_state(6, seed=21, scaling=0.5, z2=True) at
        p = 0.5: the default maxspace = 8 restarts and the search stops at the residual
        target instead of spanning the sector; validation against the oracle."""
        s6 = tuned(random_state(6, seed=21, scaling=0.5, z2=True), **WINDOW)
        dense = ref.eigsolve(s6, 0.5, PHI4, kinetic_nodes=NODES)
        r = es.excitation_eigsolve(s6, 0.5, PHI4, kinetic_nodes=NODES)
        self.assertSameReport(r, dense, s6)
        self.assertValidated(r)
        self.assertEqual(r["dimension"], 18)
        self.assertTrue(r["converged"] and r["accepted"] and any(row["cycle"] > 1 for row in r["history"]))


class Refinement(Case):
    """refine_steps: preconditioned corrections under the refined action
    (src/excitation_eigsolve.jl:332-353)."""
    LOOSE = DavidsonOptions(tol=1e-2, maxops=3, stall_window=0)

    def test_refine_steps(self):
        """test/excitation_eigsolve.jl:246-252: a loose search (tol 1e-2, 3 operations, random
        start) and refine_steps = 3. Corrections, the refined vector's energy and residual,
        refined_converged, the validation of the refined vector and the acceptance are the
        oracle's (1e-8) for S = 1 and S = sqrt(rho). The three corrections are accepted
        (fresh residuals 0.37, 0.057, 0.016, 0.0036 <= tol: refined_converged); the four
        energies of the refined vector agree (spread 1.4e-8) but its residual exceeds
        acceptance_residual: consistent, not accepted."""
        request = dict(kinetic_nodes=NODES, refine_steps=3, start="random", davidson_options=self.LOOSE)
        dense = ref.eigsolve(Z2, 0.0, PHI4, **request)
        for scaling in ("none", "density"):
            with self.subTest(scaling=scaling):
                r = es.excitation_eigsolve(Z2, 0.0, PHI4, scaling=scaling, **request)
                self.assertSameReport(r, dense, Z2)
                self.assertValidated(r)
                corrections, v = r["corrections"], r["validation"][0]
                self.assertEqual([(row["step"], row["accepted"]) for row in corrections],
                                 [(step, True) for step in range(4)])
                residuals = [row["residual"] for row in corrections]
                self.assertEqual(residuals, sorted(residuals, reverse=True))
                self.assertLess(abs(r["energies"][0] - v["refined_energy"]), 1e-10)
                self.assertLess(abs(r["residuals"][0] - v["refined_residual"]), 1e-10)
                self.assertEqual((corrections[-1]["energy"], corrections[-1]["residual"]),
                                 (r["energies"][0], r["residuals"][0]))
                self.assertTrue(r["refined_converged"] and r["residuals"][0] <= self.LOOSE.tol)
                self.assertEqual((r["start"], r["numops"], r["stop_reason"], r["converged"]),
                                 ("random", 3, "maxops", False))
                # search, one standard action per correction step, one for the validation
                self.assertEqual(r["action_calls"], 3 + (len(corrections) - 1) + 1)
                self.assertTrue(v["consistent"] and not v["residual_target_met"])
                self.assertFalse(v["accepted"] or r["accepted"])

    def test_rejected_correction(self):
        """A correction that does not lower the fresh residual is recorded, not applied, and
        ends the refinement (src/excitation_eigsolve.jl:346-351). The line search makes a
        genuine rejection a roundoff event, so in both drivers the residual of the second
        candidate is replaced by twice that of the first: step 2 is rejected and the result
        is that of step 1."""
        request = dict(kinetic_nodes=NODES, refine_steps=3, start="random", davidson_options=self.LOOSE,
                       validate=False)

        def solve(module, function):
            real, vectors, residuals = module.fresh_level, [], []

            def fresh(op, x):
                level = real(op, x)
                if len(vectors) == 2:   # after step 0 and candidate 1: candidate 2
                    level["residual"] = 2 * residuals[1]
                vectors.append(x)
                residuals.append(level["residual"])
                return level

            with mock.patch.object(module, "fresh_level", fresh):
                return function(Z2, 0.0, PHI4, **request), vectors

        (r, vectors), (dense, _) = solve(es, es.excitation_eigsolve), solve(ref, ref.eigsolve)
        self.assertSameReport(r, dense, Z2)
        corrections = r["corrections"]
        self.assertEqual([(row["step"], row["accepted"]) for row in corrections], [(0, True), (1, True), (2, False)])
        self.assertEqual(corrections[2]["residual"], 2 * corrections[1]["residual"])
        self.assertEqual((r["energies"][0], r["residuals"][0]), (corrections[1]["energy"], corrections[1]["residual"]))
        self.assertEqual(len(vectors), 3)
        self.assertTrue(np.array_equal(r["coefficients"][0], vectors[1]))
        self.assertGreater(np.linalg.norm(vectors[2] - vectors[1]), 1e-3)
        self.assertEqual((r["refined_converged"], r["accepted"], r["action_calls"]), (False, False, 3 + 2))

    def test_refinement_without_validation(self):
        """Without validation a level is accepted when the search and the refinement both
        converged (src/excitation_eigsolve.jl:377-378). A converged search at tol = 1e-2
        needs no correction (step 0 only); the loose one is not accepted."""
        cases = ((dict(davidson_options=DavidsonOptions(tol=1e-2, stall_window=0)), True),
                 (dict(davidson_options=self.LOOSE, start="random"), False))
        for request, accepted in cases:
            request = dict(request, kinetic_nodes=NODES, refine_steps=2, validate=False)
            dense = ref.eigsolve(Z2, 0.0, PHI4, **request)
            r = es.excitation_eigsolve(Z2, 0.0, PHI4, **request)
            with self.subTest(accepted=accepted):
                self.assertSameReport(r, dense, Z2)
                self.assertEqual((r["validation"], r["validated_energies"]), (None, None))
                self.assertEqual((r["refined_integration"].max_point, r["refined_kinetic_nodes"]), (24, 2 * NODES))
                self.assertEqual(r["accepted"], accepted)
                self.assertEqual(r["accepted"], bool(r["converged"] and r["refined_converged"]))
                self.assertEqual(r["action_calls"], r["numops"] + len(r["corrections"]) - 1)
                self.assertIs(es.reported_energies(r), r["energies"])
                if accepted:
                    self.assertEqual([row["step"] for row in r["corrections"]], [0])
                    self.assertIs(es.accepted_energies(r), r["energies"])
                else:
                    with self.assertRaises(SolverFailure):
                        es.accepted_energies(r)


class Acceptance(Case):
    """The acceptance rule (src/excitation_eigsolve.jl:356-378) on synthetic levels: the
    fresh and bilinear levels of both drivers are replaced by prescribed numbers, so that
    every criterion and its boundary (<=) is reached. The search is one free action."""
    BASE = ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.125, 0.125))   # energies, imaginary parts, residuals
    LIMITS = dict(validation_tol=0.25, acceptance_residual=0.5)

    def both(self, levels, **request):
        """Production and oracle reports for `levels`, one (energies, imaginary parts,
        residuals) per vector in the order (standard, refined, bilinear, refined bilinear)."""
        nev = len(levels)
        fresh = [dict(energy=e[k], imaginary=i[k], residual=res[k], action=None) for e, i, res in levels for k in (0, 1)]
        pairs = [dict(energy=e[k], imaginary=i[k]) for e, i, _ in levels for k in (2, 3)]
        request = dict(dict(kinetic_nodes=8, preconditioner="none", nev=nev, start=list(np.eye(8)[:nev]),
                            davidson_options=DavidsonOptions(maxops=nev, stall_window=0)), **self.LIMITS, **request)
        reports = []
        for module, solve in ((es, es.excitation_eigsolve), (ref, ref.eigsolve)):
            with mock.patch.object(module, "fresh_level", side_effect=[dict(x) for x in fresh]) as actions, \
                    mock.patch.object(module, "bilinear_level", side_effect=[dict(x) for x in pairs]) as elements:
                reports.append(solve(Z2, 0.5, FREE, **request))
            self.assertEqual((actions.call_count, elements.call_count), (2 * nev, 2 * nev))
        return reports

    def perturbed(self, part, k, value):
        level = [list(x) for x in self.BASE]
        level[part][k] = value
        return tuple(tuple(x) for x in level)

    def test_criteria(self):
        """One quantity at a time leaves the base level (spread 0, no imaginary part,
        residuals 1/8; limits 1/4 and 1/2): flags (residual_target_met, consistent, accepted)."""
        cases = [(self.BASE, (True, True, True))]
        for k in range(4):
            cases += [(self.perturbed(0, k, 1.3), (True, False, False)),      # spread above the tolerance
                      (self.perturbed(0, k, 0.75), (True, True, True)),       # spread equal to it
                      (self.perturbed(1, k, -0.3), (True, False, False)),     # imaginary part above
                      (self.perturbed(1, k, -0.25), (True, True, True))]
        for k in range(2):
            cases += [(self.perturbed(2, k, 0.25), (True, True, True)),       # residual equal to validation_tol
                      (self.perturbed(2, k, 0.3), (False, True, True)),       # between the two limits
                      (self.perturbed(2, k, 0.5), (False, True, True)),       # equal to acceptance_residual
                      (self.perturbed(2, k, 0.6), (False, True, False))]
        for level, flags in cases:
            with self.subTest(level=level):
                r, dense = self.both([level])
                v, = r["validation"]
                self.assertEqual(r["validation"], dense["validation"])
                self.assertEqual(tuple(v[name] for name in FLAGS), flags)
                energies, imaginary, residuals = level
                self.assertEqual([v[name] for name in FOUR], list(energies))
                self.assertEqual((v["standard_residual"], v["refined_residual"]), residuals)
                self.assertEqual((v["energy_spread"], v["max_imaginary"]),
                                 (max(energies) - min(energies), max(abs(x) for x in imaginary)))
                self.assertEqual((r["accepted"], dense["accepted"]), (flags[2], flags[2]))
                self.assertEqual((r["validated_energies"], dense["validated_energies"]), ([energies[3]],) * 2)
                self.assertValidated(r, tol=None)
                # The search itself stopped unconverged after one action: only the validation counts.
                self.assertEqual((r["converged"], r["stop_reason"]), (False, "maxops"))

    def test_every_level_must_be_accepted(self):
        """nev = 2: the result is accepted when both levels are; accepted_energies names the
        rejected ones (zero based)."""
        good, bad = self.BASE, self.perturbed(2, 1, 0.6)
        for levels, rejected in (([good, good], []), ([good, bad], [1]), ([bad, good], [0]), ([bad, bad], [0, 1])):
            with self.subTest(rejected=rejected):
                r, dense = self.both(levels)
                self.assertEqual(r["validation"], dense["validation"])
                self.assertEqual([not v["accepted"] for v in r["validation"]], [k in rejected for k in (0, 1)])
                self.assertEqual((r["accepted"], dense["accepted"]), (not rejected,) * 2)
                self.assertEqual(r["validated_energies"], [1.0, 1.0])
                if rejected:
                    with self.assertRaises(SolverFailure) as failure:
                        es.accepted_energies(r)
                    self.assertIn(f"levels {rejected} rejected: max fresh residual 0.6 (limit 0.5), energy spread 0.0 "
                                  "and imaginary part 0.0 (limit 0.25 each)", str(failure.exception))

    def test_unvalidated_search(self):
        """The same unconverged one-action search without validation is not accepted."""
        request = dict(kinetic_nodes=8, preconditioner="none", start=np.eye(8)[0], validate=False,
                       davidson_options=DavidsonOptions(maxops=1, stall_window=0))
        r, dense = es.excitation_eigsolve(Z2, 0.5, FREE, **request), ref.eigsolve(Z2, 0.5, FREE, **request)
        self.assertSameReport(r, dense, Z2)
        self.assertEqual((r["accepted"], r["converged"], r["validation"], r["refined_integration"]),
                         (False, False, None, None))


class Dispersion(Case):
    """iterative_dispersion with validation (src/excitation_eigsolve.jl:483-501)."""

    def test_validated_dispersion(self):
        """test/excitation_eigsolve.jl:204-212: momenta (0, 0.5) on shared coordinates and one
        ScaledState; the second momentum starts from the first eigenvector. Solver levels
        and validated energies are the oracle's at each momentum (1e-8)."""
        momenta = [0.0, 0.5]
        results = es.iterative_dispersion(Z2, momenta, PHI4, kinetic_nodes=NODES)
        self.assertEqual([r["start"] for r in results], ["field", "supplied"])
        self.assertIs(results[0]["basis"], results[1]["basis"])
        for r, dense in zip(results, (oracle("odd", 0.0, 4), oracle("odd", 0.5, 2))):   # their lowest level
            self.assertTrue(r["converged"] and r["accepted"])
            self.assertValidated(r)
            self.assertClose(r["energies"], dense["energies"][:1], TOL, "dispersion energies")
            self.assertClose(r["validated_energies"], dense["validated_energies"][:1], TOL, "dispersion validated")
            self.assertClose([r["validation"][0][name] for name in FOUR],
                             [dense["validation"][0][name] for name in FOUR], TOL, "dispersion four energies")
            self.assertEqual(es.accepted_energies(r), r["validated_energies"])

    def test_strict(self):
        """test/excitation_eigsolve.jl:281-286: with validation_tol = 1e-16 no level is
        consistent (free model: 32 against 64 kinetic nodes); strict=False keeps every
        momentum, strict stops at the first one."""
        request = dict(kinetic_nodes=NODES, validation_tol=1e-16)
        lax = es.iterative_dispersion(Z2, [0.0, 0.5], FREE, strict=False, **request)
        self.assertEqual([(r["accepted"], r["converged"], r["start"]) for r in lax],
                         [(False, True, "field"), (False, True, "supplied")])
        self.assertFalse(any(r["validation"][0]["consistent"] for r in lax))
        self.assertTrue(all(r["validation"][0]["energy_spread"] < TOL for r in lax))
        self.assertEqual([es.reported_energies(r) for r in lax], [r["validated_energies"] for r in lax])
        solved = []
        with self.assertRaises(SolverFailure) as failure:
            es.iterative_dispersion(Z2, [0.0, 0.5], FREE, log=solved.append, **request)
        self.assertIn("p=0.0 not accepted (stop: converged, levels [0] rejected", str(failure.exception))
        self.assertEqual(sum(line.startswith("operator ") for line in solved), 1)


class Wiring(Case):
    """How solve_levels builds its two bilinear elements: `bilinear_element`, by default
    the class bilinear.BilinearElement."""

    def recorded(self, keyword, solve=es.excitation_eigsolve, state_arguments=None, **request):
        """Result and the elements built, with a recording subclass given as `bilinear_element`
        (keyword) or put in place of bilinear.BilinearElement (the default route)."""
        made = []

        class Recorded(bilinear.BilinearElement):
            def __init__(self, ss, momentum, terms, **options):
                super().__init__(ss, momentum, terms, **options)
                made.append((self, ss, momentum, terms, options))

        arguments = (Z2, 0.5, PHI4) if state_arguments is None else state_arguments
        request.setdefault("kinetic_nodes", NODES)
        if "start" not in request:   # one operation: the oracle's eigenvector
            request["start"] = oracle("odd", 0.5, 2)["coefficients"][0]
        if keyword:
            return solve(*arguments, bilinear_element=Recorded, **request), made
        with mock.patch.object(bilinear, "BilinearElement", Recorded):
            return solve(*arguments, **request), made

    def test_bilinear_elements(self):
        """One element at the standard and one at the refined settings, on the solver's
        ScaledState, at solver_tol, with ode.integrate_host whatever integrates the actions;
        each is evaluated once per level, and the reported energy is its `energy` on the hat
        tangent of the returned vector. `bilinear_integrate` selects another integrator
        (bilinear.integrate_scipy: same energies, 1e-10), `bilinear_element` another class."""
        r, made = self.recorded(False, solver_tol=1e-13, integrate=potential.integrate_scipy,
                                system=potential.ActionSystem)
        basis = r["basis"]
        self.assertEqual(len(made), 2)
        settings = ((r["integration"], NODES, "bilinear_energy"),
                    (r["refined_integration"], 2 * NODES, "refined_bilinear_energy"))
        for (element, ss, momentum, terms, options), (integrator, nodes, name) in zip(made, settings):
            self.assertIs(ss, basis.ss)
            self.assertEqual((momentum, terms), (0.5, PHI4))
            self.assertEqual(sorted(options), ["integrate", "integrator", "kinetic_nodes", "tol"])
            self.assertIs(options["integrate"], ode.integrate_host)
            self.assertEqual((options["integrator"], options["kinetic_nodes"], options["tol"]),
                             (integrator, nodes, 1e-13))
            self.assertEqual((element.elements, element.trajectories), (1, 1))
            value = element.energy(basis.hat(r["vectors"][0]))
            self.assertClose(r["validation"][0][name], value.real, 1e-10, f"{name} recomputed")
        self.assertTrue(r["accepted"])
        other, made = self.recorded(True, bilinear_integrate=bilinear.integrate_scipy)
        self.assertEqual([(options["integrator"], options["kinetic_nodes"]) for *_, options in made],
                         [(other["integration"], NODES), (other["refined_integration"], 2 * NODES)])
        self.assertTrue(all(options["integrate"] is bilinear.integrate_scipy and options["tol"] == 1e-14
                            for *_, options in made))   # 1e-14: the default solver_tol
        for name in FOUR:
            self.assertClose(other["validation"][0][name], r["validation"][0][name], 1e-10, f"{name} with SciPy")

    def test_no_bilinear_without_validation(self):
        """validate=False builds no bilinear element and no refined operator."""
        for keyword in (False, True):
            r, made = self.recorded(keyword, validate=False)
            self.assertEqual((made, r["validation"], r["refined_integration"], r["action_calls"]),
                             ([], None, None, 1))

    def test_dispersion(self):
        """iterative_dispersion forwards `bilinear_element`: two elements per momentum."""
        momenta = [0.0, 0.5]
        results, made = self.recorded(True, es.iterative_dispersion, (Z2, momenta, FREE), start="auto")
        self.assertEqual([momentum for _, _, momentum, _, _ in made], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual([element.elements for element, *_ in made], [1] * 4)
        self.assertTrue(all(r["accepted"] for r in results))


if __name__ == "__main__":
    unittest.main()
