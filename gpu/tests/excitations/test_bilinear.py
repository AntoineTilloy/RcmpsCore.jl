"""Tests of bilinear.py, the matrix-free bilinear element in scaled variables.

References: the dense oracle dense_reference.py (make_pair, matrix_element, kinetic_sum,
observable_element and its unit maps), which bilinear.py never imports, and the action
hamiltonian.py as the independent reverse contraction. Ported Julia tests are cited in
the docstrings (test/excitations.jl, test/excitation_action.jl) with their tolerances;
`assertApprox` is Julia's isapprox, |x-y| <= max(atol, rtol max(|x|, |y|)).

Hat tangents (uh, wh) are drawn once per bond dimension; the oracle receives the physical
ones u = uh diag(1/S), w = wh diag(1/S). Scalings: "none" (S = 1, every hat quantity is the
CPU one), "explicit" (S = exp(linspace(-2.5, 1.5, D))) and "density" (S = sqrt(diag rho)).
With an explicit S the physical tangents and elements are up to ~150 times larger, so
absolute Julia tolerances are applied at S = 1 and relative ones otherwise.

What "any positive S" means here (scaled.py). The formulas do not depend on S; the numbers
do in two ways, each with its own test. (i) GMRES controls residuals relative to the hat
right-hand sides, so a result carries about tol (Smax/Smin)^2 unless S is aligned with
rho: the 1e-11 comparisons use the 1.7 decades of "explicit", and "wide" (S =
exp(linspace(-9, 0, D)), 4 decades) is held to that bound only (WideScaling). (ii) rh is
re-solved, which for S = sqrt(rho) corrects the stored density by about eps cond(rho):
on an ill-conditioned state the reference is the oracle on rho = S rh S (RefinedDensity).
The states of the 1e-11 comparisons have cond(rho) <= 59, where the two coincide; their
"density" scaling is used outside the density gauge, a test of the formulas only.

States (NumPy draws, random_state(D, seed, scaling=0.5)): D = 2 seed 3 (transfer gap 0.99,
cond rho 7.0), D = 3 seed 5 (gap 0.16, cond rho 2.7), Z2 D = 4 seed 11 (gap 0.20, cond rho
14); D = 6 seed 7 (gap 0.37, cond rho 18) for restarted GMRES; Z2 D = 12 and D = 16, seed 11
(gaps 0.50 and 0.91, cond rho 16 and 59) under RCMPS_EXCITATION_SLOW=1. Default window:
max_point 14, atol 1e-13, rtol 1e-12, 16 theta nodes.
RCMPS_EXCITATION_REPORT=1 prints the largest measured error of every check.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
import os
import sys
import unittest

import numpy as np

from rcmps_gpu.excitations import coordinates as tc
import dense_reference as ref
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations.bilinear import BilinearElement, Pair, PairSystem, integrate_scipy
from rcmps_gpu.excitations.hamiltonian import HamiltonianAction
from rcmps_gpu.excitations.kinetic import TangentFrame
from rcmps_gpu.excitations.scaled import ScaledState
from rcmps_gpu.excitations.states import IntegratorParameters, SolverFailure, adj, make_state, random_state

slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (bond dimensions above 6)")
norm = np.linalg.norm
ROUTINE = ("D2", "D3", "Z2")
STATES = {"D2": (2, 3, False), "D3": (3, 5, False), "Z2": (4, 11, True), "D6": (6, 7, False),
          "Z12": (12, 11, True), "D16": (16, 11, False)}
MOMENTA = (0.0, 0.6)
NODES = 16
PHI4 = models.phi4_terms(1.0)
MIXED = [(0.37, models.monomial_spec(2)), (0.21, models.monomial_spec(3))]
TERMS = (("phi4", PHI4), ("phi2+phi3", MIXED))
REPORT = {}


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for (test, tolerance), fraction in REPORT.items():
            print(f"{test:52s} {tolerance:22s} worst error / bound = {fraction:.2e}", file=sys.stderr)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def window(max_point=14.0, n_quad=32):
    return IntegratorParameters(n_quad=n_quad, max_point=max_point, atol=1e-13, rtol=1e-12)


@lru_cache(maxsize=None)
def state(name):
    if name == "D1":   # State([.3+.1im], 0) of test/excitations.jl:53
        return make_state([[0.3 + 0.1j]], [[0.0]])
    d, seed, z2 = STATES[name]
    return random_state(d, seed=seed, scaling=0.5, z2=z2)


@lru_cache(maxsize=None)
def scaled(name, scaling):
    s = state(name)
    if scaling == "explicit":
        return ScaledState(s, np.exp(np.linspace(-2.5, 1.5, s.D)))
    if scaling == "wide":
        return ScaledState(s, np.exp(np.linspace(-9.0, 0.0, s.D)))
    return ScaledState(s, scaling)


@lru_cache(maxsize=None)
def context(name, p, max_point=14.0):
    return ref.make_context(state(name).with_integrator(window(max_point)), p)


def tangents(ss):
    """Hat tangents (uh, wh) and the physical ones (u, w) = (uh, wh) diag(1/S)."""
    uh, wh = gaussian(1, ss.D, ss.D), gaussian(2, ss.D, ss.D)
    return uh, wh, uh / ss.s_host, wh / ss.s_host


def bilinear(name, scaling, p, terms, *, max_point=14.0, nodes=NODES, **kw):
    return BilinearElement(scaled(name, scaling), p, terms, integrator=window(max_point),
                           kinetic_nodes=nodes, **kw)


def to_physical(xh, s):
    """Four hat blocks (4, D, n, D) of types (HH, LH, HL, LL) -> physical (n, 4, D, D)."""
    row, col = s[:, None, None], s[None, None, :]
    blocks = (row * xh[0] * col, xh[1] * col, row * xh[2], xh[3])
    return np.stack([np.moveaxis(b, 1, 0) for b in blocks], axis=1)


def to_hat(x, s):
    row, col = s[:, None, None], s[None, None, :]
    b = [np.moveaxis(x[:, k], 0, 1) for k in range(4)]
    return np.stack((b[0] / row / col, b[1] / col, b[2] / row, b[3]))


class Case(unittest.TestCase):
    def assertApprox(self, x, y, atol=0.0, rtol=0.0, msg="", unit=1.0):
        """Julia's isapprox(x, y; atol, rtol), with the absolute tolerance in units of `unit`."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        error, bound = float(norm(x - y)), max(atol * unit, rtol * max(norm(x), norm(y)))
        key = (self.id().split(".", 1)[1], f"atol={atol:g} rtol={rtol:g}")
        REPORT[key] = max(REPORT.get(key, 0.0), error / bound if bound > 0 else float(error > 0))
        self.assertTrue(error <= bound, f"{msg}: |x-y| = {error:.3e} exceeds {bound:.3e}")

    def cases(self, scalings=("none", "explicit"), names=ROUTINE, momenta=MOMENTA):
        for scaling in scalings:
            for name in names:
                for p in momenta:
                    yield scaling, name, p, f"{name} p={p} S={scaling}"


class AgainstOracle(Case):
    """Items 1-3: element(uh, wh) = matrix_element(make_pair(ctx, u, w), terms) to 1e-11
    relative, for non-Z2 D = 2, 3 and Z2 D = 4, p = 0 and 0.6, phi^4 and a quadratic plus a
    cubic monomial. Both sides integrate the same ODE with SciPy's DOP853 at the same
    tolerances, so the comparison is not limited by the integration error."""

    def compare(self, scaling):
        for _, name, p, label in self.cases((scaling,)):
            ss = scaled(name, scaling)
            uh, wh, u, w = tangents(ss)
            pair = ref.make_pair(context(name, p), u, w)
            for title, terms in TERMS:
                exact = ref.matrix_element(pair, terms, kinetic_nodes=NODES)
                value = bilinear(name, scaling, p, terms).element(uh, wh)
                self.assertIsInstance(value, complex)
                self.assertApprox(value, exact, rtol=1e-11, msg=f"{label} {title}")

    def test_unscaled(self):
        self.compare("none")

    def test_explicit_scaling(self):
        self.compare("explicit")

    def test_density_scaling(self):
        """S = sqrt(diag(rho)), the production scaling, on states that are not in the
        density gauge: any positive S is admissible."""
        for _, name, p, label in self.cases(("density",), momenta=(0.6,)):
            ss = scaled(name, "density")
            uh, wh, u, w = tangents(ss)
            exact = ref.matrix_element(ref.make_pair(context(name, p), u, w), PHI4, kinetic_nodes=NODES)
            value = bilinear(name, "density", p, PHI4).element(uh, wh)
            self.assertApprox(value, exact, rtol=1e-11, msg=label)

    def test_bond_dimension_one(self):
        """D = 1 (coherent state, S = 0.7): batches (1, n, 1) and a vanishing [Q, R]."""
        ss = ScaledState(state("D1"), np.array([0.7]))
        uh, wh, u, w = tangents(ss)
        for p in MOMENTA:
            pair = ref.make_pair(context("D1", p), u, w)
            for title, terms in TERMS:
                exact = ref.matrix_element(pair, terms, kinetic_nodes=NODES)
                b = BilinearElement(ss, p, terms, integrator=window(), kinetic_nodes=NODES)
                self.assertApprox(b.element(uh, wh), exact, rtol=1e-11, msg=f"p={p} {title}")

    def test_energy(self):
        """bilinear_level (src/excitation_eigsolve.jl:248-253): element(w, w) / tr(w rho w'),
        real up to roundoff, against the oracle's matrix_element / tangent_metric."""
        for scaling, name, p, label in self.cases(momenta=(0.6,)):
            ss = scaled(name, scaling)
            _, wh, _, w = tangents(ss)
            ctx = context(name, p)
            exact = (ref.matrix_element(ref.make_pair(ctx, w, w), PHI4, kinetic_nodes=NODES)
                     / ref.tangent_metric(ctx.state, w, w))
            b = bilinear(name, scaling, p, PHI4)
            value = b.energy(wh)
            self.assertIsInstance(value, complex)
            self.assertApprox(value, exact, rtol=1e-11, msg=label)
            self.assertApprox(value.imag, 0.0, atol=1e-11, unit=abs(value), msg=label)
            metric = np.trace(wh @ ss.rh @ adj(wh))
            self.assertApprox(metric, ref.tangent_metric(ctx.state, w, w), rtol=1e-12, msg=label)
            self.assertApprox(value, b.element(wh, wh) / metric, rtol=1e-11, msg=label)

    def level(self, name, sector, p=0.6, **kw):
        """The production wiring of bilinear_level: white coordinates x of one sector in the
        density gauge (coordinates.py), W = tangent_matrix(c, x), S = sqrt(rho) so that
        rh = 1, Wh = W S and tr(Wh rh Wh') = |x|^2; against the oracle's bilinear_level."""
        c = tc.make_coordinates(state(name).with_integrator(window()), sector=sector)
        ss = ScaledState(c.state, "density")
        self.assertApprox(ss.rh, np.eye(ss.D), atol=1e-12, msg=name)
        x = gaussian(5, len(c))
        wh = tc.tangent_matrix(c, x) * ss.s_host
        ctx = ref.make_context(c.state, p)
        exact = ref.bilinear_level(ctx, ref.excitation_operator(ctx, c, PHI4, kinetic_nodes=NODES), x,
                                   kinetic_nodes=NODES)
        b = BilinearElement(ss, p, PHI4, integrator=window(), kinetic_nodes=NODES, **kw)
        self.assertApprox(b.energy(wh), exact["energy"] + 1j * exact["imaginary"], rtol=1e-11,
                          msg=f"{name} {sector}")
        self.assertApprox(b.pair(wh, wh).overlap, np.vdot(x, x), rtol=1e-12, msg=f"{name} {sector}")
        return b

    def test_density_gauge_level(self):
        for sector in ("odd", "even"):
            self.level("Z2", sector)

    def test_restarted_gmres(self):
        """D = 6: with restart = 10 < D^2 the solves go through GMRES restarts. S = 1 and
        the density gauge; with the ill-scaled explicit S the 1e-12 solver tolerance alone
        leaves up to 9e-12 at this D, which is not a test of the formulas."""
        ss = scaled("D6", "none")
        uh, wh, u, w = tangents(ss)
        exact = ref.matrix_element(ref.make_pair(context("D6", 0.6), u, w), PHI4, kinetic_nodes=NODES)
        for controls in ({}, dict(restart=10)):
            b = bilinear("D6", "none", 0.6, PHI4, **controls)
            self.assertApprox(b.element(uh, wh), exact, rtol=1e-11, msg=str(controls))
        self.assertGreater(b.stats["iterations"], 10 * b.stats["solves"])
        self.level("D6", "all", restart=10)

    @slow
    def test_larger_bond_dimensions(self):
        """D^2 above the default restart length: Z2 D = 12 (odd sector) and D = 16."""
        self.level("Z12", "odd")
        self.level("D16", "all")

    def test_kinetic_sum(self):
        """excitation_kinetic_sum alone (src/excitations.jl:64-86): both kinetic specs, one
        of them, a repeated source operator with other coefficients, on 4 and 16 nodes."""
        lists = (PHI4[:2], PHI4[:1], [(0.3, models.kinetic_spec(True))],
                 [(0.3, models.kinetic_spec(True)), (1.7, models.kinetic_spec(False)),
                  (-0.4, models.kinetic_spec(True))])
        for scaling, name, p, label in self.cases():
            ss = scaled(name, scaling)
            uh, wh, u, w = tangents(ss)
            pair = ref.make_pair(context(name, p), u, w)
            for k, terms in enumerate(lists):
                for nodes in (4, 16):
                    b = bilinear(name, scaling, p, terms, nodes=nodes)
                    self.assertApprox(b.element(uh, wh), ref.kinetic_sum(pair, terms, nodes), rtol=1e-11,
                                      msg=f"{label} list {k} nodes={nodes}")
                    self.assertEqual(b.trajectories, 0)

    def test_observables(self):
        """excitation_observable term by term (src/excitations.jl:48-62): monomials of
        degree 1 to 4 and both kinetic source graphs through the ODE."""
        specs = ([models.monomial_spec(n) for n in (1, 2, 3, 4)]
                 + [models.kinetic_spec(c) for c in (False, True)])
        for scaling, name, p, label in self.cases(names=("D2", "Z2"), momenta=(0.6,)):
            ss = scaled(name, scaling)
            uh, wh, u, w = tangents(ss)
            exact = ref.make_pair(context(name, p), u, w)
            b = bilinear(name, scaling, p, [])
            pair = b.pair(uh, wh)
            for k, spec in enumerate(specs):
                self.assertApprox(b.observable(pair, spec), ref.observable_element(exact, spec),
                                  rtol=1e-11, msg=f"{label} spec {k}")
            self.assertEqual(b.trajectories, len(specs))

    def test_constant_density_cancels(self):
        """test/excitations.jl:103-104: :phi^0: cancels for complex tangents (1e-10)."""
        for scaling, name, p, label in self.cases():
            uh, wh, _, _ = tangents(scaled(name, scaling))
            b = bilinear(name, scaling, p, [])
            value = b.observable(b.pair(uh, wh), models.monomial_spec(0))
            self.assertIsInstance(value, complex)
            self.assertApprox(value, 0.0, atol=1e-10, msg=label)


class Environments(Case):
    """Item 6 and test/excitations.jl:83-96 in scaled variables."""

    def test_pair_blocks(self):
        """right = (S x1h S, y2h S, S b3h, x4), left00 = S^-1 hh S^-1, overlap and gauge
        parts against the oracle's ExcitationPair (1e-10 relative; Julia checks 1e-10)."""
        for scaling, name, p, label in self.cases(("none", "explicit", "density")):
            ss = scaled(name, scaling)
            s = ss.s_host
            uh, wh, u, w = tangents(ss)
            pair, exact = Pair(ss, p, uh, wh), ref.make_pair(context(name, p), u, w)
            right = to_physical(pair.right, s)[0]
            for k in range(4):
                self.assertApprox(right[k], exact.right[k], rtol=1e-10, msg=f"{label} block {k}")
            self.assertApprox(pair.left / np.outer(s, s), exact.left00, rtol=1e-10, msg=label)
            self.assertApprox(pair.overlap, exact.overlap, rtol=1e-12, msg=label)
            self.assertApprox(pair.vh / s, exact.v, rtol=1e-13, msg=label)
            self.assertApprox(pair.vbrah / s, exact.vbra, rtol=1e-13, msg=label)
            self.assertEqual(len(pair.infos), 4)

    def test_fixed_vectors(self):
        """test/excitations.jl:83-91: the right environments are a zero vector of the
        connected transfer (1e-10), the left functional vanishes on them (1e-11) and on the
        range of the transfer (1e-10); here relative to the size of the blocks."""
        for scaling, name, p, label in self.cases():
            ss = scaled(name, scaling)
            uh, wh, _, _ = tangents(ss)
            pair = Pair(ss, p, uh, wh)
            size = norm(pair.right)
            self.assertApprox(pair.transfer(pair.right), np.zeros_like(pair.right), atol=1e-10, unit=size,
                              msg=label)
            self.assertApprox(pair.contract(pair.right), np.zeros(1), atol=1e-11, unit=size, msg=label)
            x = np.ascontiguousarray(gaussian(86, 4, ss.D, 3, ss.D))
            moved = pair.transfer(x)
            self.assertApprox(pair.contract(moved), np.zeros(3), atol=1e-10, unit=norm(moved), msg=label)

    def test_bra_is_adjoint_ket(self):
        """The bra environment of u is the adjoint of the ket environment rw of u, which the
        action computes (kinetic.TangentFrame): S b3h = (y2h S)'."""
        for scaling, name, p, label in self.cases():
            ss = scaled(name, scaling)
            uh, wh, _, _ = tangents(ss)
            pair = Pair(ss, p, uh, wh)
            frame = TangentFrame(ss, p, np.ascontiguousarray(np.stack([wh, uh], axis=1)))
            self.assertApprox(pair.right[1, :, 0, :], frame.rw[:, 0, :], rtol=1e-10, msg=label)
            self.assertApprox(pair.right[2, :, 0, :], adj(frame.rw[:, 1, :]), rtol=1e-10, msg=label)

    def test_unit_maps(self):
        """transfer, source, operators and contract on random blocks against the oracle's
        excitation_transfer, excitation_source_add, excitation_operators and
        excitation_contract after the change of variables (1e-12 relative; the oracle pair
        supplies overlap and tail, so its own solves do not enter the maps). No solve
        enters, so this holds for the 4-decade S "wide" as well: the formulas are
        S-independent whatever the range of S."""
        for scaling, name, p, label in self.cases(("none", "explicit", "density", "wide")):
            ss = scaled(name, scaling)
            s = ss.s_host
            uh, wh, u, w = tangents(ss)
            pair, exact = Pair(ss, p, uh, wh), ref.make_pair(context(name, p), u, w)
            pair.overlap, pair.left = exact.overlap, s[:, None] * exact.left00 * s[None, :]
            xh = np.ascontiguousarray(gaussian(87, 4, ss.D, 3, ss.D))
            x = to_physical(xh, s)
            self.assertApprox(to_physical(pair.transfer(xh), s), ref.excitation_transfer(exact, x), rtol=1e-12,
                              msg=label)
            contracted = [ref.excitation_contract(exact, x[i]) for i in range(3)]
            self.assertApprox(pair.contract(xh), contracted, rtol=1e-12, msg=label)
            for commutator in (False, True):
                operators = pair.operators(commutator)
                c, cw, cu = ref.excitation_operators(exact, commutator)
                self.assertApprox(operators[True][0], c, rtol=1e-13, msg=label)
                self.assertApprox(operators[True][1], c * s[None, :] / s[:, None], rtol=1e-13, msg=label)
                self.assertApprox(operators[True][2], cw * s, rtol=1e-12, msg=label)
                self.assertApprox(operators[False][2], adj(cu * s), rtol=1e-12, msg=label)
                for left in (False, True):
                    out = np.stack([ref.excitation_source_add(np.zeros_like(x[i]), x[i], (c, cw, cu), left, 1)
                                    for i in range(3)])
                    self.assertApprox(to_physical(pair.source(xh, operators, left), s), out, rtol=1e-12,
                                      msg=f"{label} commutator={commutator} left={left}")
            self.assertApprox(to_hat(x, s), xh, rtol=1e-14, msg="change of variables")


class WideScaling(Case):
    """An S spanning 4 decades and not aligned with rho, S = exp(linspace(-9, 0, D)):
    (Smax/Smin)^2 = 6.6e7. The kinetic sum then differs from the oracle by far more than
    1e-11 at the default tol = 1e-12 (measured 7e-8 for D = 3 and Z2 D = 4, 1e-15 for D = 2;
    at tol = 1e-14 still 9e-9 for Z2 D = 4, with residuals of 8e-15), because GMRES stops
    on residuals relative to the hat right-hand sides, whose small entries the contraction
    weights back up. The error stays below the attained residual times (Smax/Smin)^2
    (measured 30 to 60 times below where it exceeds 1e-11). This is a limit of the solves,
    not of the formulas (Environments.test_unit_maps), and it is absent for S = sqrt(rho)
    in the density gauge (AgainstOracle.test_density_gauge_level)."""

    def test_kinetic_sum_bound(self):
        for _, name, p, label in self.cases(("wide",)):
            ss = scaled(name, "wide")
            uh, wh, u, w = tangents(ss)
            exact = ref.kinetic_sum(ref.make_pair(context(name, p), u, w), PHI4[:2], NODES)
            ratio = (ss.s_host.max() / ss.s_host.min()) ** 2
            for tol in (1e-12, 1e-14):
                b = bilinear(name, "wide", p, PHI4[:2], tol=tol)
                value = b.element(uh, wh)
                self.assertLessEqual(b.stats["max_residual"], 20 * tol, label)
                bound = max(1e-11, b.stats["max_residual"] * ratio)
                self.assertApprox(value, exact, rtol=bound, msg=f"{label} tol={tol}")


class RefinedDensity(Case):
    """The one place where the result differs from the CPU control by more than roundoff:
    Julia evaluates every formula on the stored density (src/excitation_environments.jl:39,
    :111), bilinear.py and the action on rho = S rh S with rh re-solved (scaled.py).

    State: R = 0.8 (a + 0.01 a'), K = diag(0, 0.7, 1.4, 2.1), a the D = 4 lowering operator,
    in the density gauge; cond(rho) = 1.0e10, and a 50-digit stationary solve puts the
    hat-relative error of the stored rho at 2.5e-7 and that of S rh S at 4e-15. One fixed
    physical tangent W, phi^4 at g = 1.3, p = 0.6, 12 theta nodes. Measured: the energy
    with S = sqrt(rho) is 3.3e-7 above the oracle on the stored rho and 9e-15 from the
    oracle on S rh S; S = 1 and the explicit S keep the stored rho (no refinement
    iteration) and reproduce its oracle to 2.5e-12 and 1.2e-13."""
    P, TERMS, THETA = 0.6, models.phi4_terms(1.3), 12

    @classmethod
    def setUpClass(cls):
        lower = np.diag(np.sqrt(np.arange(1.0, 4.0)), 1).astype(complex)
        state = make_state(0.8 * (lower + 0.01 * adj(lower)), np.diag(0.7 * np.arange(4.0)), integrator=window())
        cls.chart = tc.make_coordinates(state)
        cls.state = cls.chart.state
        cls.w = tc.tangent_matrix(cls.chart, gaussian(2, len(cls.chart)))

    def oracle_energy(self, state):
        ctx = ref.make_context(state, self.P)
        return (ref.matrix_element(ref.make_pair(ctx, self.w, self.w), self.TERMS, kinetic_nodes=self.THETA)
                / ref.tangent_metric(state, self.w, self.w))

    def energy(self, ss):
        b = BilinearElement(ss, self.P, self.TERMS, integrator=window(), kinetic_nodes=self.THETA, tol=1e-14)
        return b.energy(self.w * ss.s_host)

    def test_density_scaling_uses_the_refined_density(self):
        t, d = self.state, self.state.D
        self.assertGreater(self.chart.density_condition, 5e9)
        stored = self.oracle_energy(t)
        ss = ScaledState(t, "density")
        self.assertGreater(int(ss.fixed_point["iterations"][0]), 0)
        rho = ss.s_host[:, None] * ss.rh * ss.s_host[None, :]
        self.assertApprox(np.trace(rho), 1.0, rtol=1e-14)
        refined = replace(t, rho=rho)
        value = self.energy(ss)
        self.assertApprox(value, self.oracle_energy(refined), rtol=1e-12, msg="oracle on S rh S")
        # The stored rho is off by about eps cond(rho) in its small block; measured ratio 0.02.
        limit = np.finfo(float).eps * self.chart.density_condition
        self.assertApprox(value, stored, rtol=limit, msg="oracle on the stored rho")
        for scaling in ("none", np.exp(np.linspace(-2.5, 1.5, d))):
            other = ScaledState(t, scaling)
            self.assertEqual(int(other.fixed_point["iterations"][0]), 0)
            self.assertApprox(self.energy(other), stored, rtol=1e-11, msg="stored rho is kept")

    def test_action_on_the_refined_density(self):
        """The action of the same W: F = Fh S against the oracle's action on rho = S rh S
        (1e-10; measured 5e-13, the ODE tolerance), and its Rayleigh quotient against the
        bilinear energy (1e-10; measured 3e-13). F itself hardly depends on the small
        eigenvalues of rho (1.1e-12 from the oracle on the stored rho): the 3.3e-7 of the
        energy is the norm tr(W rho W')."""
        t = self.state
        ss = ScaledState(t, "density")
        refined = replace(t, rho=ss.s_host[:, None] * ss.rh * ss.s_host[None, :])
        wh = self.w * ss.s_host
        action = HamiltonianAction(ss, self.P, self.TERMS, integrator=window(), kinetic_nodes=self.THETA,
                                   tol=1e-14, integrate=ode.integrate_host)(wh)
        exact = ref.hamiltonian_action(ref.make_context(refined, self.P), self.w, self.TERMS,
                                       kinetic_nodes=self.THETA)
        self.assertApprox(action * ss.s_host, exact, rtol=1e-10, msg="action")
        quotient = np.vdot(wh, action) / np.trace(wh @ ss.rh @ adj(wh))
        self.assertApprox(quotient, self.energy(ss), rtol=1e-10, msg="Rayleigh quotient")


class AgainstAction(Case):
    """Item 4: vdot(uh, F(wh)) = element(uh, wh) with the Julia tolerance of
    test/excitation_action.jl:11-13 (atol 2e-8, rtol 2e-9); S = 1 and explicit S."""

    def test_phi4(self):
        """Both routes use the same theta grid and the same window; the action adds its
        pullback quadrature, converged with n_quad = 48 far below the Julia tolerance (the
        second assertion, 1e-10 relative, is a regression bound and not a Julia test)."""
        for scaling, name, p, label in self.cases():
            ss = scaled(name, scaling)
            uh, wh, _, _ = tangents(ss)
            value = bilinear(name, scaling, p, PHI4).element(uh, wh)
            action = HamiltonianAction(ss, p, PHI4, integrator=window(n_quad=48), kinetic_nodes=NODES)(wh)
            self.assertApprox(np.vdot(uh, action), value, atol=2e-8, rtol=2e-9, msg=label)
            self.assertApprox(np.vdot(uh, action), value, rtol=1e-10, msg=label)

    def test_quadratic_and_cubic(self):
        """The action sends :phi^2: to the infinite-line resolvent, the bilinear route to
        the window ODE (src/excitations.jl:117-120): they agree once the window is resolved
        (max_point = 24 as in the Julia test). The action's pullback of the cubic term needs
        n_quad = 64 there (n_quad = 32 leaves 2e-7 relative; Julia uses 192); 64 theta nodes."""
        for scaling, name, p, label in self.cases(names=("D2", "Z2")):
            ss = scaled(name, scaling)
            uh, wh, _, _ = tangents(ss)
            value = bilinear(name, scaling, p, MIXED, max_point=24.0).element(uh, wh)
            action = HamiltonianAction(ss, p, MIXED, integrator=window(24.0, n_quad=64),
                                       kinetic_nodes=64)(wh)
            self.assertApprox(np.vdot(uh, action), value, atol=2e-8, rtol=2e-9, msg=label)


class Algebra(Case):
    """Item 5 and test/excitations.jl:109-122 for phi^4 at p = 0.7."""

    def test_sesquilinear_and_hermitian(self):
        """f(u,w) = conj(f(w,u)) (1e-9), antilinear in u and linear in w (2e-8): absolute
        at S = 1 as in Julia, in units of |f| for the explicit S; D = 2 and Z2 D = 4."""
        for scaling, name, p, label in self.cases(names=("D2", "Z2"), momenta=(0.7,)):
            ss = scaled(name, scaling)
            uh, wh, _, _ = tangents(ss)
            b = bilinear(name, scaling, p, PHI4)
            a = b.element(uh, wh)
            unit = 1.0 if scaling == "none" else abs(a)
            self.assertApprox(a, np.conj(b.element(wh, uh)), atol=1e-9, unit=unit, msg=label)
            self.assertApprox(b.element((2 + 1j) * uh, (0.3 - 0.7j) * wh), (2 - 1j) * (0.3 - 0.7j) * a,
                              atol=2e-8, unit=unit, msg=label)
            self.assertApprox(b.element(uh, wh + uh), a + b.element(uh, uh), atol=2e-8, unit=unit, msg=label)

    def test_kinetic_ode_route(self):
        """test/excitations.jl:121-122 (kinetic_backend=:ode): the two kinetic specs through
        the source ODE agree with the resolvent sum once window (max_point 24) and theta
        grid (256 nodes) are resolved; atol 2e-8, for f(u,w) and f(w,w)."""
        ss = scaled("D2", "none")
        uh, wh, _, _ = tangents(ss)
        b = bilinear("D2", "none", 0.7, PHI4[:2], max_point=24.0, nodes=256)
        for bra in (uh, wh):
            pair = b.pair(bra, wh)
            through_ode = sum(c * b.observable(pair, spec) for c, spec in PHI4[:2])
            self.assertApprox(through_ode, b.kinetic_sum(pair), atol=2e-8)


class Integrators(Case):
    def test_integrate_host(self):
        """Item 7: integrate=ode.integrate_host gives the SciPy value to 1e-10 relative."""
        for scaling, name, p, label in self.cases(momenta=(0.6,)):
            uh, wh, _, _ = tangents(scaled(name, scaling))
            for title, terms in TERMS:
                value = bilinear(name, scaling, p, terms).element(uh, wh)
                host = bilinear(name, scaling, p, terms, integrate=ode.integrate_host)
                self.assertApprox(host.element(uh, wh), value, rtol=1e-10, msg=f"{label} {title}")
                self.assertEqual(host.trajectories, sum(not spec.kinetic for _, spec in terms))

    def test_sample_points(self):
        """integrate_scipy has the calling convention of ode.integrate_host: samples at
        interior points and at the endpoint (dense output of DOP853; 1e-9 relative)."""
        ss = scaled("D2", "explicit")
        uh, wh, _, _ = tangents(ss)
        system = PairSystem(Pair(ss, 0.6, uh, wh), models.monomial_spec(3))
        points, _ = models.quadrature(window(n_quad=8))
        a = math.sqrt(14.0)
        samples = integrate_scipy(system, points, a, window())
        self.assertEqual(samples.shape, (17, 4, 2, 4, 2))
        self.assertApprox(samples, ode.integrate_host(system, points, a, window()), rtol=1e-9)
        self.assertApprox(samples[-1], integrate_scipy(system, (), a, window())[-1], rtol=1e-11)
        with self.assertRaises(ValueError):
            integrate_scipy(system, (0.0,), a, window())

    def test_failures(self):
        """quadrature_check (ArgumentError -> ValueError) and the step budget maxiters
        (SolverFailure, both integrators); kinetic-only elements never read the integrator."""
        ss = scaled("D2", "none")
        uh, wh, _, _ = tangents(ss)
        for bad in (dict(max_point=0.0), dict(atol=0.0), dict(rtol=0.0), dict(maxiters=0)):
            integrator = replace(window(), **bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                BilinearElement(ss, 0.0, MIXED, integrator=integrator).element(uh, wh)
            BilinearElement(ss, 0.0, PHI4[:2], integrator=integrator, kinetic_nodes=4).element(uh, wh)
        short = replace(window(), maxiters=3)
        for integrate in (None, ode.integrate_host):
            with self.assertRaises(SolverFailure):
                BilinearElement(ss, 0.0, MIXED, integrator=short, integrate=integrate).element(uh, wh)


class Bookkeeping(Case):
    def test_counters(self):
        """Solves per element: four Poisson solves, then one batch of 2 orders x n nodes per
        source operator and block; the ground block is solved once per BilinearElement."""
        ss = scaled("Z2", "explicit")
        uh, wh, _, _ = tangents(ss)
        n = 8
        b = bilinear("Z2", "explicit", 0.6, PHI4, nodes=n)
        self.assertEqual((b.stats["solves"], b.trajectories, b.elements), (0, 0, 0))
        first = b.element(uh, wh)
        self.assertEqual((b.stats["solves"], b.trajectories, b.elements), (4 + 4 * 4 * n, 1, 1))
        b.energy(wh)
        self.assertEqual((b.stats["solves"], b.trajectories, b.elements), (8 + 7 * 4 * n, 2, 2))
        self.assertLessEqual(b.stats["max_residual"], 20 * b.tol)
        self.assertGreater(b.stats["iterations"], b.stats["solves"])
        self.assertApprox(b.element(uh, wh), first, rtol=1e-11, msg="cached ground block")
        single = bilinear("Z2", "explicit", 0.6, [(0.3, models.kinetic_spec(True))], nodes=n)
        single.element(uh, wh)
        self.assertEqual(single.stats["solves"], 4 + 4 * 2 * n)

    def test_solver_controls(self):
        """`tol` and GMRES controls reach every solve: a loose tolerance costs fewer
        iterations and moves the value by about that tolerance."""
        ss = scaled("Z2", "none")
        uh, wh, _, _ = tangents(ss)
        tight = bilinear("Z2", "none", 0.6, PHI4[:2])
        loose = bilinear("Z2", "none", 0.6, PHI4[:2], tol=1e-5, restart=8)
        a, b = tight.element(uh, wh), loose.element(uh, wh)
        self.assertLess(loose.stats["iterations"], tight.stats["iterations"])
        self.assertGreater(loose.stats["max_residual"], 1e-9)
        self.assertGreater(abs(a - b), 1e-11 * abs(a))
        self.assertApprox(b, a, rtol=1e-3)

    def test_arguments(self):
        """ExcitationPair's DimensionMismatch and ArgumentError (ValueError here), invalid
        momentum and kinetic nodes; an empty model is zero as in excitation_element."""
        ss = scaled("D2", "none")
        uh, wh, _, _ = tangents(ss)
        b = bilinear("D2", "none", 0.0, PHI4)
        for u, w in ((np.ones((1, 1)), wh), (uh, np.ones((3, 3))), (uh, wh[:, :1])):
            with self.assertRaises(ValueError):
                b.element(u, w)
        for bad in (math.nan, math.inf):
            broken = wh.copy()
            broken[1, 0] = bad
            for u, w in ((uh, broken), (broken, wh)):
                with self.assertRaises(ValueError):
                    b.element(u, w)
            with self.assertRaises(ValueError):
                b.energy(broken)
            with self.assertRaises(ValueError):
                bilinear("D2", "none", bad, PHI4)
        for nodes in (1, 2.5, None):
            with self.assertRaises(ValueError, msg=str(nodes)):
                bilinear("D2", "none", 0.0, PHI4, nodes=nodes)
        self.assertEqual(bilinear("D2", "none", 0.0, MIXED, nodes=None).kinetics, [])
        self.assertEqual(bilinear("D2", "none", 0.0, []).element(uh, wh), 0j)
        self.assertEqual(b.elements, 0)

    def test_zero_tangent(self):
        """A zero tangent: the element vanishes and bilinear_level is 0/0, which is
        NaN + NaN im in Julia (src/excitation_eigsolve.jl:248-253; the level is then not
        accepted) and must not be Python's ZeroDivisionError."""
        ss = scaled("D2", "explicit")
        _, wh, _, _ = tangents(ss)
        zero = np.zeros_like(wh)
        b = bilinear("D2", "explicit", 0.6, PHI4)
        self.assertEqual((b.element(zero, wh), b.element(wh, zero)), (0j, 0j))
        value = b.energy(zero)
        self.assertIsInstance(value, complex)
        self.assertTrue(math.isnan(value.real) and math.isnan(value.imag))

    def test_real_and_integer_tangents(self):
        """Tangents are converted to complex128 (Matrix{ComplexF64}(u) in ExcitationPair)."""
        uh, wh = np.eye(2), np.array([[1, 2], [3, 4]])
        b = bilinear("D2", "none", 0.6, PHI4[:2])
        exact = ref.kinetic_sum(ref.make_pair(context("D2", 0.6), uh, wh), PHI4[:2], NODES)
        self.assertApprox(b.element(uh, wh), exact, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
