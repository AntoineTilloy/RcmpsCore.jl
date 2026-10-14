"""Tests of kinetic.py: the resolvent (theta quadrature) part of the Hamiltonian action.

The oracle is dense_reference.kinetic_action_sum with v = -R'w and rw from
dense_reference.excitation_poisson, on the same fixed theta grid. Physical and hat
variables are converted here: Wh = W S, Vh = V S, rw = rwh S, F = Fh S (right
multiplication by diag(S), i.e. per column), x = S xh S, a0 = a0t S^-1.

Grid: states.random_state(3, 5) and Z2 states (4, 6); p in (0, 0.6); S = 1, sqrt(diag(rho))
and exp(linspace(-2.5, 1.5, D)); kinetic specs alone, together, and with a quadratic
monomial (all four source orders); batches of 3 complex tangents.

Oracle      action against the oracle at S = 1 (1e-11) and scaled (1e-10), linearity, statistics.
Structure   Hermitian white-coordinate form and Z2 selection rule in the density gauge.
Pieces      TangentFrame, finish_boundary, Boundary, node sums, routing and rejections.

RCMPS_EXCITATION_REPORT=1 prints the largest measured errors at the end.
"""
from __future__ import annotations

import functools
import math
import os
import sys
import unittest

import numpy as np

from rcmps_gpu.excitations import coordinates as tc
import dense_reference as ref
from rcmps_gpu.excitations.kinetic import (Boundary, KineticAction, TangentFrame, batch_adjoint, contract, finish_boundary,
                     kinetic_terms, node_sum)
from rcmps_gpu.excitations.models import kinetic_spec, monomial_spec, phi4_terms
from rcmps_gpu.excitations.scaled import ScaledState, stack
from rcmps_gpu.excitations.states import adj, random_state

norm = np.linalg.norm
STATES = ((3, False), (4, True), (5, False), (6, True))
MOMENTA = (0.0, 0.6)
SCALINGS = ("none", "density", "explicit")
KINETIC = [(1.0, kinetic_spec(False)), (1.0, kinetic_spec(True))]
TERMS = {"plain": [(1.3, kinetic_spec(False))], "commutator": [(0.7, kinetic_spec(True))],
         "quadratic": [(-0.4, monomial_spec(2))], "kinetic": KINETIC,
         "all": KINETIC + [(0.37, monomial_spec(2))]}
NODES, TOL = 8, 1e-12
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
def scaled(dim, z2, scaling):
    state = random_state(dim, z2=z2)
    return ScaledState(state, np.exp(np.linspace(-2.5, 1.5, dim)) if scaling == "explicit" else scaling)


@functools.lru_cache(maxsize=None)
def context(dim, z2, p):
    return ref.make_context(random_state(dim, z2=z2), p, tol=1e-10)


def tangents(dim, count=3, seed=10):
    return [gaussian(seed + t, dim, dim) for t in range(count)]


def environment(ctx, w):
    """(v, rw) of the oracle for a physical tangent w."""
    s = ctx.state
    v = -adj(s.R) @ w
    return v, ref.excitation_poisson(ctx, ref.ket_insertion(s, v, w, s.rho), shift=1)


def oracle(ctx, w, terms, nodes=NODES):
    return ref.kinetic_action_sum(ctx, w, *environment(ctx, w), terms, nodes)


def frame_of(ss, p, w, tol=TOL):
    """TangentFrame of physical tangents: Wh = W S."""
    return TangentFrame(ss, p, stack([m * ss.s_host[None, :] for m in w]), tol=tol)


def action(ss, p, terms, w, nodes=NODES, tol=TOL):
    """Physical actions F = Fh S of a list of physical tangents, and the KineticAction."""
    kinetic = KineticAction(ss, p, terms, nodes, tol=tol)
    return kinetic.action(frame_of(ss, p, w, tol)) * ss.s_host[None, None, :], kinetic


def worst(f, expected):
    return max(norm(f[:, t, :] - e) / norm(e) for t, e in enumerate(expected))


class Oracle(unittest.TestCase):
    def compare(self, scalings, bound, label):
        for dim, z2 in STATES:
            w = tangents(dim)
            for p in MOMENTA:
                ctx = context(dim, z2, p)
                for name, terms in TERMS.items():
                    expected = [oracle(ctx, m, terms) for m in w]
                    for scaling in scalings:
                        with self.subTest(dim=dim, z2=z2, p=p, terms=name, scaling=scaling):
                            f, _ = action(scaled(dim, z2, scaling), p, terms, w)
                            self.assertEqual(f.shape, (dim, 3, dim))
                            self.assertLessEqual(record(f"{label} {scaling}", worst(f, expected)), bound)

    def test_unit_scaling(self):
        """At S = 1 (Wh = W, F = Fh) the action of a batch of 3 complex tangents is the oracle's
        kinetic_action_sum to 1e-11: each kinetic spec alone, both, a quadratic monomial alone
        (four source orders), and all together, for p = 0 and 0.6, Z2 and not."""
        self.compare(("none",), 1e-11, "oracle")

    def test_scaled(self):
        """The same with Wh = W S and F = Fh S for S = sqrt(diag(rho)) and an explicit S, to 1e-10."""
        self.compare(("density", "explicit"), 1e-10, "oracle")

    def test_more_nodes(self):
        """A 32-node grid, D = 4 and 5: the node sums and scalars kappa scale with the grid."""
        for dim, z2 in ((4, True), (5, False)):
            ctx, w = context(dim, z2, 0.6), tangents(dim, 2)
            expected = [oracle(ctx, m, TERMS["all"], 32) for m in w]
            for scaling, bound in (("none", 1e-11), ("explicit", 1e-10)):
                f, kinetic = action(scaled(dim, z2, scaling), 0.6, TERMS["all"], w, nodes=32)
                self.assertEqual(kinetic.nodes, 32)
                self.assertLessEqual(record(f"oracle 32 nodes {scaling}", worst(f, expected)), bound)

    def test_equal_coefficients(self):
        """A kinetic spec and a quadratic monomial with the same coefficient and source operator
        (they share their resolvent families and scalars kappa) still add up to the oracle."""
        terms = [(1.0, kinetic_spec(False)), (1.0, monomial_spec(2))]
        for scaling, bound in (("none", 1e-11), ("explicit", 1e-10)):
            ctx, w = context(3, False, 0.6), tangents(3)
            f, _ = action(scaled(3, False, scaling), 0.6, terms, w)
            self.assertLessEqual(worst(f, [oracle(ctx, m, terms) for m in w]), bound)

    def test_linearity_and_batch_independence(self):
        """F(a W1 + b W2) = a F(W1) + b F(W2) for complex a, b, and a batch equals its tangents
        taken one at a time (fixed nodes: linear up to the solver tolerance)."""
        a, b = 0.7 - 0.4j, -1.1 + 0.3j
        for dim, z2 in STATES:
            for scaling in SCALINGS:
                with self.subTest(dim=dim, z2=z2, scaling=scaling):
                    ss, w = scaled(dim, z2, scaling), tangents(dim, 2)
                    f, kinetic = action(ss, 0.6, TERMS["all"], w + [a * w[0] + b * w[1]])
                    combined = a * f[:, 0, :] + b * f[:, 1, :]
                    self.assertLessEqual(record("linearity", norm(f[:, 2, :] - combined) / norm(combined)), 1e-10)
                    single = kinetic.action(frame_of(ss, 0.6, w[1:])) * ss.s_host[None, None, :]
                    self.assertLessEqual(record("batch", norm(single[:, 0, :] - f[:, 1, :]) / norm(f[:, 1, :])), 1e-10)

    def test_statistics(self):
        """stats counts the systems solved: 2 resolvent families of sources x nodes and 2 Poisson
        solves at construction; per action 2 families of tangents x sources x 2 x nodes and one
        boundary solve per tangent. Residuals stay below the tolerance."""
        ss = scaled(4, True, "density")
        for name, sources in (("plain", 1), ("commutator", 1), ("quadratic", 1), ("all", 2)):
            kinetic = KineticAction(ss, 0.6, TERMS[name], NODES, tol=TOL)
            self.assertEqual(kinetic.sources, [False, True] if sources == 2 else [name == "commutator"])
            self.assertEqual(kinetic.stats["solves"], 2 * sources * NODES + 2)
            kinetic.action(frame_of(ss, 0.6, tangents(4)))
            self.assertEqual(kinetic.stats["solves"], 2 * sources * NODES + 2 + 2 * 3 * sources * 2 * NODES + 3)
            self.assertGreater(kinetic.stats["iterations"], 0)
            self.assertLessEqual(kinetic.stats["max_residual"], TOL)


def white_form(ss, p, terms, c):
    """Matrix h[i, j] = <white tangent i, F(white tangent j)> from the coordinates `c`."""
    w = [tc.tangent_matrix(c, unit) for unit in np.eye(len(c))]
    f, _ = action(ss, p, terms, w)
    return np.column_stack([tc.covector_coordinates(c, f[:, j, :]) for j in range(len(c))])


class Structure(unittest.TestCase):
    def test_hermitian_white_form(self):
        """For a Z2 state in the density gauge the kinetic form in white coordinates
        (W = E_ij diag(rho)^-1/2) is Hermitian at fixed nodes, for every term set and both p.
        With S = sqrt(diag(rho)) the white tangents are the units Wh = E_ij and the matrix is
        Fh itself; it agrees with the S = 1 evaluation."""
        state = random_state(4, z2=True)
        c = tc.make_coordinates(state, sector="all", parity="auto", gauge="density")
        rho = np.diag(c.state.rho).real
        self.assertLessEqual(norm(c.whitening - np.diag(rho ** -0.5)), 1e-13)
        plain, density = ScaledState(c.state, "none"), ScaledState(c.state, "density")
        units = np.zeros((len(c), 4, 4), dtype=complex)
        units[np.arange(len(c)), c.rows, c.cols] = 1
        for p in MOMENTA:
            for name, terms in TERMS.items():
                with self.subTest(p=p, terms=name):
                    h = white_form(plain, p, terms, c)
                    self.assertLessEqual(record("hermiticity", norm(h - adj(h)) / norm(h)), 1e-12)
                    fh = KineticAction(density, p, terms, NODES, tol=TOL).action(
                        TangentFrame(density, p, stack(list(units)), tol=TOL))
                    direct = fh[c.rows, :, c.cols]
                    self.assertLessEqual(record("white form, density scaling", norm(direct - h) / norm(h)), 1e-11)
                    self.assertLessEqual(norm(direct - adj(direct)) / norm(direct), 1e-12)

    def test_parity_selection_rule(self):
        """For a Z2 state F maps block-diagonal tangents ('odd' sector) to block-diagonal
        covectors and off-diagonal ('even') to off-diagonal ones: no cross-sector element."""
        for dim in (4, 6):
            half = dim // 2
            mask = np.zeros((dim, dim), dtype=bool)
            mask[:half, :half] = mask[half:, half:] = True
            raw = gaussian(20, dim, dim)
            for scaling in SCALINGS:
                for p in MOMENTA:
                    with self.subTest(dim=dim, scaling=scaling, p=p):
                        f, _ = action(scaled(dim, True, scaling), p, TERMS["all"], [raw * mask, raw * ~mask])
                        self.assertLessEqual(record("cross sector", norm(f[:, 0, :][~mask]) / norm(f[:, 0, :])), 1e-13)
                        self.assertLessEqual(record("cross sector", norm(f[:, 1, :][mask]) / norm(f[:, 1, :])), 1e-13)


class Pieces(unittest.TestCase):
    def test_frame(self):
        """TangentFrame: Vh = -R' Wh, rw S is the oracle's ket Poisson environment, and ket /
        ket_trace_dual are the oracle's ket_insertion / ket_trace_dual on converted variables."""
        for dim, z2 in STATES:
            w = tangents(dim)
            for p in MOMENTA:
                ctx = context(dim, z2, p)
                state = ctx.state
                for scaling in SCALINGS:
                    with self.subTest(dim=dim, z2=z2, p=p, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        s = ss.s_host
                        frame = frame_of(ss, p, w)
                        self.assertEqual((frame.count, frame.p), (3, p))
                        x, a = gaussian(30, dim, 2, dim), gaussian(31, dim, 2, dim)    # hat HH, dual LL
                        for t, m in enumerate(w):
                            v, rw = environment(ctx, m)
                            self.assertLessEqual(norm(frame.vh[:, t, :] / s - v), 1e-13 * norm(v))
                            error = norm(frame.rw[:, t, :] * s - rw) / norm(rw)
                            self.assertLessEqual(record(f"rw {scaling}", error), 1e-10)
                            ket, dual = frame.ket(x, t), frame.ket_trace_dual(a, t)
                            for n in range(2):
                                big = s[:, None] * x[:, n, :] * s[None, :]
                                expected = ref.ket_insertion(state, v, m, big)
                                self.assertLessEqual(norm(ket[:, n, :] * s - expected), 1e-12 * norm(expected))
                                expected = ref.ket_trace_dual(state, v, m, a[:, n, :])
                                self.assertLessEqual(norm(dual[:, n, :] / s - expected), 1e-12 * norm(expected))

    def test_finish_boundary(self):
        """finish_boundary with d = dual Poisson(a1), g = Poisson(xout) is the oracle's
        action_boundaries(w, v, rw, a0, a1, xout) for random a1 (dual LL), xout = S xh S and
        a0 = a0t S^-1, per tangent; the input a0 batch is left untouched."""
        for dim, z2 in STATES:
            w = tangents(dim)
            a1, xout, a0 = gaussian(40, dim, dim), gaussian(41, dim, dim), tangents(dim, seed=42)
            for p in MOMENTA:
                ctx = context(dim, z2, p)
                expected = [ref.action_boundaries(ctx, m, *environment(ctx, m), a, a1, xout) for m, a in zip(w, a0)]
                for scaling in SCALINGS:
                    with self.subTest(dim=dim, z2=z2, p=p, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        s = ss.s_host
                        d, _ = ss.poisson(stack([a1]), "L", "L", dual=True, tol=TOL)
                        g, _ = ss.poisson(stack([xout / np.outer(s, s)]), "H", "H", tol=TOL)
                        a0t = stack([a * s[None, :] for a in a0])
                        before = a0t.copy()
                        out, info = finish_boundary(frame_of(ss, p, w), Boundary(d[:, 0, :], g[:, 0, :]), a0t, tol=TOL)
                        self.assertTrue(np.array_equal(a0t, before))
                        self.assertEqual(info["residuals"].shape, (3,))
                        bound = 1e-9 if scaling == "explicit" else 1e-10
                        error = worst(out * s[None, None, :], expected)
                        self.assertLessEqual(record(f"boundary {scaling}", error), bound)

    def test_boundary_algebra(self):
        """Boundary data add and scale componentwise (boundaries are linear in the term)."""
        one, two = Boundary(gaussian(50, 3, 3), gaussian(51, 3, 3)), Boundary(gaussian(52, 3, 3), gaussian(53, 3, 3))
        total, scaled_one = one + two, one.scaled(-0.3)
        self.assertTrue(np.array_equal(total.d, one.d + two.d) and np.array_equal(total.g, one.g + two.g))
        self.assertTrue(np.array_equal(scaled_one.d, -0.3 * one.d) and np.array_equal(scaled_one.g, -0.3 * one.g))

    def test_boundary_of_the_action(self):
        """KineticAction.boundary: d = sum of weighted dual Poisson(a1 c or c' a1) and g = Poisson of
        the summed second-source outputs; both have zero pairing, and the boundary is linear in
        the coefficients of the terms."""
        ss = scaled(4, False, "explicit")
        one = KineticAction(ss, 0.6, TERMS["all"], NODES, tol=TOL).boundary
        double = KineticAction(ss, 0.6, [(2 * c, spec) for c, spec in TERMS["all"]], NODES, tol=TOL).boundary
        for x, y in ((one.d, double.d), (one.g, double.g)):
            self.assertLessEqual(norm(2 * x - y), 1e-11 * norm(y))
        self.assertLessEqual(abs(ss.pairing(stack([one.d]), "L", "L", dual=True)[0]), 1e-12 * norm(one.d))
        self.assertLessEqual(abs(ss.pairing(stack([one.g]), "H", "H")[0]), 1e-12 * norm(one.g) * norm(ss.s_host) ** 2)

    def test_node_sums(self):
        """node_sum, contract and batch_adjoint against loops over the nodes."""
        d, n = 3, 5
        a, b, weights = gaussian(60, d, n, d), gaussian(61, d, n, d), np.linspace(0.1, 0.9, n)
        self.assertLessEqual(norm(node_sum(a, weights) - sum(weights[i] * a[:, i, :] for i in range(n))), 1e-14)
        expected = sum(weights[i] * a[:, i, :] @ b[:, i, :] for i in range(n))
        self.assertLessEqual(norm(contract(a, b, weights, np) - expected), 1e-14 * norm(expected))
        flipped = batch_adjoint(a, np)
        self.assertTrue(flipped.flags.c_contiguous)
        for i in range(n):
            self.assertTrue(np.array_equal(flipped[:, i, :], adj(a[:, i, :])))

    def test_rejections(self):
        """Only kinetic specs and quadratic monomials use the resolvent: other monomials and an
        empty term list raise ValueError."""
        ss = scaled(3, False, "none")
        for terms in ([(1.0, monomial_spec(4))], [(1.0, monomial_spec(3))], [(1.0, monomial_spec(1))],
                      KINETIC + [(0.5, monomial_spec(4))], []):
            with self.assertRaises(ValueError):
                KineticAction(ss, 0.0, terms, NODES)
        with self.assertRaises(ValueError):
            KineticAction(ss, 0.0, KINETIC, 1)      # kinetic_grid needs at least 2 nodes

    def test_routing(self):
        """kinetic_terms keeps the kinetic specs and the quadratic monomials, in order."""
        quadratic, cubic = (0.3, monomial_spec(2)), (0.2, monomial_spec(3))
        terms = phi4_terms(1.5)
        self.assertEqual(kinetic_terms(terms), terms[:2])
        self.assertEqual(kinetic_terms([cubic, quadratic] + terms), [quadratic] + terms[:2])
        self.assertEqual(kinetic_terms([cubic, terms[2]]), [])


if __name__ == "__main__":
    unittest.main()
