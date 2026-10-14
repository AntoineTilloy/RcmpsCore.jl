"""Tests of hamiltonian.py: the complete matrix-free Hamiltonian action F(W).

The oracle is dense_reference.hamiltonian_action on the same state, integration settings
and fixed theta grid. Hat variables: Wh = W S, F = Fh S (per column).

States: states.random_state(2) and (3) as they are (rho not diagonal, so that
S = sqrt(diag(rho)) is just another scaling), and the Z2 state random_state(4, z2=True)
rotated to the density gauge (rho diagonal: S = sqrt(diag(rho)) gives rh = 1, the
production layout). p in (0, 0.7); S = 1, sqrt(diag(rho)), exp(linspace(-2.5, 1.5, D)).

Oracle      phi4_terms(g), single tangents and batches, to 1e-11; other term lists.
            RCMPS_EXCITATION_SLOW=1 runs both inputs in every cell (about a minute more).
Packed      system=PackedActionSystem with integrate=integrate_host, both sectors, to 1e-10.
Routes      kinetic specs and the quadratic monomial through the ODE against the resolvent
            quadrature with 256 nodes, to 2e-8 (the Julia test tolerance).
Interface   calls counter, shapes, rejections.

ODE tolerances atol 1e-14, rtol 1e-13 (as in test_packed.py): at atol 1e-13, rtol 1e-12 the
largest deviation from the oracle over the grid is 5.5e-12, here 1.2e-12 (D = 2, p = 0.7).
RCMPS_EXCITATION_REPORT=1 prints the largest measured errors at the end.
"""
from __future__ import annotations

import functools
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coordinates as tc
import dense_reference as ref
from hamiltonian import HamiltonianAction
from kinetic import KineticAction, TangentFrame, finish_boundary
from models import kinetic_spec, monomial_spec, phi4_terms
from ode import integrate_host
from packed import PackedActionSystem
from potential import ActionSystem, PotentialAction
from scaled import ScaledState, stack
from states import IntegratorParameters, random_state

norm = np.linalg.norm
INTEGRATOR = IntegratorParameters(n_quad=24, max_point=12.0, atol=1e-14, rtol=1e-13)
STATES = ((2, False), (3, False), (4, True))
MOMENTA = (0.0, 0.7)
SCALINGS = ("none", "density", "explicit")
G, NODES, TOL = 1.3, 16, 1e-13
SLOW = os.environ.get("RCMPS_EXCITATION_SLOW") == "1"
WORST = {}


def record(name, value):
    WORST[name] = max(WORST.get(name, 0.0), float(value))
    return float(value)


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for name, value in WORST.items():
            print(f"{name}: {value:.3e}", file=sys.stderr)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


@functools.lru_cache(maxsize=None)
def working_state(dim, z2):
    state = random_state(dim, z2=z2)
    if z2:
        state = tc.make_coordinates(state, sector="all", parity="auto", gauge="density").state
    return state.with_integrator(INTEGRATOR)


@functools.lru_cache(maxsize=None)
def scaled(dim, z2, scaling):
    explicit = np.exp(np.linspace(-2.5, 1.5, dim))
    return ScaledState(working_state(dim, z2), explicit if scaling == "explicit" else scaling)


@functools.lru_cache(maxsize=None)
def context(dim, z2, p):
    return ref.make_context(working_state(dim, z2), p, tol=1e-10)


def action(ss, p, terms, **kw):
    return HamiltonianAction(ss, p, terms, integrator=INTEGRATOR, kinetic_nodes=NODES, tol=TOL, **kw)


def hats(ss, w):
    """Batch of hat tangents Wh = W S."""
    return stack([m * ss.s_host for m in w])


def worst(f, expected):
    return max(norm(f[:, t, :] - e) / norm(e) for t, e in enumerate(expected))


class Oracle(unittest.TestCase):
    def test_phi4(self):
        """HamiltonianAction on phi4_terms(g) is dense_reference.hamiltonian_action to 1e-11 for
        every state, momentum and scaling. Cells alternate between a batch of two tangents and
        a single (D, D) tangent (RCMPS_EXCITATION_SLOW=1: both in every cell); calls counts tangents."""
        terms = phi4_terms(G)
        for dim, z2 in STATES:
            w = [gaussian(10, dim, dim), gaussian(11, dim, dim)]
            for j, p in enumerate(MOMENTA):
                ctx = context(dim, z2, p)
                expected = [ref.hamiltonian_action(ctx, m, terms, kinetic_nodes=NODES) for m in w]
                for k, scaling in enumerate(SCALINGS):
                    with self.subTest(dim=dim, z2=z2, p=p, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        s = ss.s_host
                        ham = action(ss, p, terms)
                        self.assertEqual((ham.calls, len(ham.potentials), len(ham.kinetic.terms)), (0, 1, 2))
                        if SLOW or (j + k) % 2 == 0:
                            batch = ham(hats(ss, w))
                            self.assertEqual((batch.shape, ham.calls), ((dim, 2, dim), 2))
                            self.assertLessEqual(record(f"phi4 batch {scaling}", worst(batch * s, expected)), 1e-11)
                        if SLOW or (j + k) % 2 == 1:
                            before = ham.calls
                            single = ham(w[1] * s)
                            self.assertEqual((single.shape, ham.calls - before), ((dim, dim), 1))
                            error = norm(single * s - expected[1]) / norm(expected[1])
                            self.assertLessEqual(record(f"phi4 single {scaling}", error), 1e-11)

    def test_density_gauge_fixed_point(self):
        """The Z2 test state is in the density gauge: with S = sqrt(diag(rho)), rh = 1."""
        ss = scaled(4, True, "density")
        self.assertLessEqual(norm(ss.rh - np.eye(4)), 1e-12)

    def test_routing(self):
        """Kinetic specs and quadratic monomials go to the resolvent, other monomials to the
        ODE, each with its coefficient; the sum is the oracle's action (D = 2, explicit S)."""
        terms = phi4_terms(G) + [(0.5, monomial_spec(2)), (-0.3, monomial_spec(3))]
        ss, ctx = scaled(2, False, "explicit"), context(2, False, 0.7)
        w = [gaussian(10, 2, 2)]
        ham = action(ss, 0.7, terms)
        self.assertEqual([c for c, _, _ in ham.kinetic.terms], [1.0, 1.0, 0.5])
        self.assertEqual([(c, p.spec) for c, p in ham.potentials], [(G, monomial_spec(4)), (-0.3, monomial_spec(3))])
        expected = [ref.hamiltonian_action(ctx, w[0], terms, kinetic_nodes=NODES)]
        self.assertLessEqual(record("routing", worst(ham(hats(ss, w)) * ss.s_host, expected)), 1e-11)

    def test_one_route_alone(self):
        """Without ODE term (g = 0) the action is KineticAction.action; without resolvent term
        (a cubic monomial alone) there is no kinetic part. Both agree with the oracle."""
        ss, ctx = scaled(3, False, "density"), context(3, False, 0.7)
        w = [gaussian(10, 3, 3)]
        free = action(ss, 0.7, phi4_terms(0.0))
        self.assertEqual(free.potentials, [])
        f = free(hats(ss, w))
        kinetic = KineticAction(ss, 0.7, phi4_terms(0.0), NODES, tol=TOL)
        direct = kinetic.action(TangentFrame(ss, 0.7, hats(ss, w), tol=TOL))
        self.assertLessEqual(norm(f - direct), 1e-12 * norm(direct))
        expected = [ref.hamiltonian_action(ctx, w[0], phi4_terms(0.0), kinetic_nodes=NODES)]
        self.assertLessEqual(record("kinetic alone", worst(f * ss.s_host, expected)), 1e-11)
        cubic = [(0.7, monomial_spec(3))]
        potential = action(ss, 0.7, cubic)
        self.assertIsNone(potential.kinetic)
        expected = [ref.hamiltonian_action(ctx, w[0], cubic, kinetic_nodes=NODES)]
        self.assertLessEqual(record("potential alone", worst(potential(hats(ss, w)) * ss.s_host, expected)), 1e-11)


class Packed(unittest.TestCase):
    def test_both_sectors(self):
        """With the Z2-packed equations (system=PackedActionSystem) integrated by
        ode.integrate_host, the action of a block-diagonal ('odd') and of a block-off-diagonal
        ('even') tangent is the oracle's to 1e-10, and stays in its sector."""
        dim, terms = 4, phi4_terms(G)
        mask = np.zeros((dim, dim), dtype=bool)
        mask[:2, :2] = mask[2:, 2:] = True
        raw = gaussian(12, dim, dim)
        w = [raw * mask, raw * ~mask]
        for p, scaling in ((0.0, "density"), (0.7, "explicit")):
            with self.subTest(p=p, scaling=scaling):
                ss, ctx = scaled(dim, True, scaling), context(dim, True, p)
                expected = [ref.hamiltonian_action(ctx, m, terms, kinetic_nodes=NODES) for m in w]
                ham = action(ss, p, terms, integrate=integrate_host, system=PackedActionSystem)
                potential = ham.potentials[0][1]
                self.assertTrue(potential.system is PackedActionSystem and potential.integrate is integrate_host)
                f = ham(hats(ss, w)) * ss.s_host
                self.assertLessEqual(record("packed", worst(f, expected)), 1e-10)
                self.assertLessEqual(norm(f[:, 0, :][~mask]), 1e-13 * norm(f[:, 0, :]))
                self.assertLessEqual(norm(f[:, 1, :][mask]), 1e-13 * norm(f[:, 1, :]))
                with self.assertRaises(ValueError):
                    ham(raw * ss.s_host)        # no definite Z2 class: the packed system refuses it


class Routes(unittest.TestCase):
    def test_ode_against_resolvent(self):
        """The kinetic specs and the quadratic monomial through the ODE route (PotentialAction)
        agree with the 256-node resolvent quadrature to 2e-8 (D = 2, max_point 24, n_quad 192)."""
        integrator = IntegratorParameters(n_quad=192, max_point=24.0, atol=1e-13, rtol=1e-12)
        state = random_state(2).with_integrator(integrator)
        w = [gaussian(10, 2, 2)]
        specs = {"kinetic": kinetic_spec(False), "commutator": kinetic_spec(True), "quadratic": monomial_spec(2)}
        for p, scaling in ((0.0, "none"), (0.7, "density")):
            ss = ScaledState(state, scaling)
            frame = TangentFrame(ss, p, hats(ss, w), tol=TOL)
            for name, spec in specs.items():
                with self.subTest(p=p, scaling=scaling, spec=name):
                    resolvent = KineticAction(ss, p, [(1.0, spec)], 256, tol=TOL).action(frame)
                    ode = PotentialAction(ss, p, spec, integrator, tol=TOL)
                    f, a0 = ode.bulk(frame)
                    boundary, _ = finish_boundary(frame, ode.boundary, a0, tol=TOL)
                    error = norm(f + boundary - resolvent) / norm(resolvent)
                    self.assertLessEqual(record(f"ODE - resolvent, {name}", error), 2e-8)


class Interface(unittest.TestCase):
    def test_rejections(self):
        """An empty term list and a nonfinite tangent (single or in a batch) raise ValueError
        and leave the calls counter alone."""
        ss = scaled(2, False, "none")
        with self.assertRaises(ValueError):
            action(ss, 0.0, [])
        ham = action(ss, 0.0, phi4_terms(0.0))
        for bad in (np.nan, np.inf):
            w = gaussian(10, 2, 2)
            w[0, 1] = bad
            with self.assertRaises(ValueError):
                ham(w)
            with self.assertRaises(ValueError):
                ham(stack([gaussian(11, 2, 2), w]))
        self.assertEqual(ham.calls, 0)

    def test_frame_and_defaults(self):
        """frame() accepts one (D, D) tangent or a batch; the default equations are
        potential.ActionSystem; the input is not modified; a single tangent gives the same
        (D, D) result as its column of a batch, and calls adds the number of tangents."""
        ss = scaled(2, False, "density")
        ham = action(ss, 0.7, phi4_terms(0.0) + [(0.2, monomial_spec(3))])
        self.assertIs(ham.potentials[0][1].system, ActionSystem)
        w = gaussian(10, 2, 2)
        self.assertEqual(ham.frame(w).count, 1)
        self.assertEqual(ham.frame(stack([w, 2 * w, 3 * w])).count, 3)
        self.assertEqual((ham.p, ham.kinetic_nodes), (0.7, NODES))
        before = w.copy()
        single = ham(w)
        self.assertTrue(np.array_equal(w, before))
        self.assertEqual((single.shape, ham.calls), ((2, 2), 1))
        batch = ham(stack([2 * w, w]))
        self.assertEqual((batch.shape, ham.calls), ((2, 2, 2), 3))
        self.assertLessEqual(norm(batch[:, 1, :] - single), 1e-11 * norm(single))
        self.assertLessEqual(norm(batch[:, 0, :] - 2 * single), 1e-10 * norm(single))


if __name__ == "__main__":
    unittest.main()
