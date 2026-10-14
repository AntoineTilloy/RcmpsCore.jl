"""Tests of the matrix-free production driver eigsolve.py against the dense oracle.

The oracle is dense_reference.py (literal port of the Julia solver); the cited Julia
tests are those of test/excitation_eigsolve.jl. States: random_state(4, seed=11,
scaling=0.5, z2=True) (z2_test_state of the Julia tests; cond(rho) = 14) and
random_state(3, seed=5, scaling=0.5) without parity, on the window max_point = 14,
n_quad = 32, atol = 1e-13, rtol = 1e-12, with 32 kinetic nodes. The production
operator integrates both trajectories of an action as one system (ode.integrate_host),
the oracle separately with SciPy: they agree at the integration tolerance (about
1e-12 here), far inside the tolerances 1e-9 (matrices) and 1e-8 (energies) asserted.

`scaling`: 'none' is S = 1, 'density' is S = sqrt(diag rho), and SKEW is the arbitrary
positive vector exp(linspace(-2.5, 1.5, D)); physical results must not depend on it.
That holds to the tolerances below because cond(rho) is small and SKEW spans 1.7 decades:
see IllConditioned for the density that 'density' refines, and scaled.py for the range of S.
Other states: cat_state() (no unique fixed point) and the D = 3 state of IllConditioned.

Tests marked slow need RCMPS_EXCITATION_SLOW=1: the D = 6 restarts (about 2 minutes)
and the two D = 16 runs on the stored Julia state (about 12 and 7 minutes on one
thread), which print their progress and report.
"""
from __future__ import annotations

from dataclasses import replace
import functools
import math
import os
import sys
import time
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coordinates as tc
import dense_reference as ref
import eigsolve as es
import models
import states
from davidson import DavidsonOptions, check_davidson
from ode import integrate_host
from packed import PackedActionSystem
from potential import ActionSystem, integrate_scipy
from scaled import ScaledState
from states import IntegratorParameters, SolverFailure, adj, make_state, random_state

slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (more than about a minute)")
norm = np.linalg.norm
FREE, PHI4 = models.phi4_terms(0.0), models.phi4_terms(1.0)
NODES = 32
WINDOW = dict(max_point=14, n_quad=32, atol=1e-13, rtol=1e-12)
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "test", "fixtures",
                       "phi4_g2_D16_z2_state.jld2")
RECORDED_MASS = 0.3523798021192911   # benchmark/excitation_mass_d16.jl


def tuned(state, **controls):
    for name, value in controls.items():
        setattr(state.integrator, name, value)
    return state


def z2_state():
    s = random_state(4, seed=11, scaling=0.5, z2=True)
    s.coupling = 1.0
    return tuned(s, **WINDOW)


def plain_state():
    return tuned(random_state(3, seed=5, scaling=0.5), **WINDOW)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def skew(d):
    return np.exp(np.linspace(-2.5, 1.5, d))


Z2, PLAIN = z2_state(), plain_state()


@functools.lru_cache(maxsize=None)
def oracle(sector, p, model="phi4", nodes=NODES, metric_tol=0.0):
    """The oracle's sector matrix in the density gauge, assembled from its actions."""
    return ref.dense_hamiltonian(Z2, p, PHI4 if model == "phi4" else FREE, sector=sector, gauge="density",
                                 metric_tol=metric_tol, assembly="action", kinetic_nodes=nodes)


@functools.lru_cache(maxsize=None)
def plain_spectrum():
    """excitation_spectrum of the D = 3 state at p = 0.5 (full tangent space, original gauge)."""
    return ref.spectrum(PLAIN, 0.5, PHI4, nev=2, assembly="action", kinetic_nodes=NODES)


def production(chart, scaling):
    ss = es.SolverState(chart.state, scaling)
    return ss, es.WhiteBasis(chart, ss)


def operator_matrix(op):
    return np.column_stack([op(unit) for unit in np.eye(len(op.basis))])


def oracle_matrix(chart, p, terms, nodes):
    """The oracle's operator in the white coordinates of a given chart, column by column."""
    op = ref.excitation_operator(ref.make_context(chart.state, p), chart, terms, kinetic_nodes=nodes)
    return np.column_stack([op(unit) for unit in np.eye(len(chart))])


def cat_state():
    """Two copies of a D = 2 state with jumps R0 and -R0 (a Z2 'cat'), in the balanced
    layout: R = [[0, R0], [R0, 0]], K = K0 + K0, rho = (rho0 + rho0) / 2. L has the second
    zero mode [[0, rho0], [rho0, 0]]: the fixed point is not unique."""
    a, zero = random_state(2, seed=3, scaling=0.5), np.zeros((2, 2))
    s = make_state(np.block([[zero, a.R], [a.R, zero]]), np.block([[a.K, zero], [zero, a.K]]),
                   np.block([[a.rho, zero], [zero, a.rho]]) / 2)
    s.coupling = 0.5
    return tuned(s, **WINDOW)


DROP = 0.0828   # between the two smallest eigenvalues of rho(Z2), 0.0696 and 0.0961 of the largest


class Case(unittest.TestCase):
    def assertClose(self, x, y, tol, msg=""):
        """max |x - y| <= tol, elementwise, reporting the measured distance."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        error = float(np.abs(x - y).max()) if x.size else 0.0
        self.assertTrue(error <= tol, f"{msg} max|x-y| = {error:.3e} exceeds {tol:.1e}")


class WhiteBasisTest(Case):
    """tangent_matrix, tangent_coordinates, covector_coordinates and original_tangent
    (src/excitation_eigsolve.jl:123-140) in hat variables."""

    def test_maps_are_the_cpu_ones(self):
        """In every scaling the maps reproduce the CPU coordinates of the same vector:
        physical W, projection, covector pairing and metric (1e-13)."""
        s = Z2
        for sector in ("odd", "even", "all"):
            c = tc.make_coordinates(s, sector=sector, parity="auto" if sector != "all" else None)
            n, d = len(c), s.D
            x, y = gaussian(1, n), gaussian(2, n)
            w = tc.tangent_matrix(c, x)
            f = gaussian(3, d, d)
            for scaling in ("none", "density", skew(d)):
                with self.subTest(sector=sector, scaling=scaling if isinstance(scaling, str) else "skew"):
                    ss, basis = production(c, scaling)
                    scale = ss.s_host[None, :]
                    self.assertEqual((len(basis), basis.D), (n, d))
                    self.assertLess(basis.defect, 1e-13)
                    self.assertClose(basis.bh, ss.s_host[:, None] * c.whitening, 1e-13)   # Bh = S B
                    wh = basis.tangent(x)
                    self.assertClose(wh / scale, w, 1e-13)                                # W = Wh S^-1
                    self.assertClose(basis.physical(x), tc.original_tangent(c, w), 1e-13)
                    self.assertClose(basis.physical_tangent(wh), tc.original_tangent(c, w), 1e-13)
                    self.assertClose(basis.hat(tc.original_tangent(c, w)), wh, 1e-13)
                    self.assertClose(basis.coordinates(wh), x, 1e-13)
                    self.assertClose(basis.coordinates(gaussian(4, d, d) * scale),
                                     tc.tangent_coordinates(c, gaussian(4, d, d)), 1e-13)
                    self.assertClose(basis.covector(f / scale), tc.covector_coordinates(c, f), 1e-13)  # F = Fh S
                    self.assertClose(np.vdot(y, basis.covector(f / scale)), np.vdot(tc.tangent_matrix(c, y), f), 1e-13)
                    self.assertClose(np.trace(wh @ ss.rh @ adj(wh)).real, norm(x) ** 2, 1e-13)
                    self.assertClose(ref.tangent_metric(s, basis.physical(x), basis.physical(y)), np.vdot(x, y), 1e-13)
                    batch = basis.tangents(np.array([x, y]))
                    self.assertEqual(batch.shape, (d, 2, d))
                    self.assertClose(batch[:, 1, :], basis.tangent(y), 1e-14)
                    self.assertClose(basis.coordinates_batch(batch), np.array([x, y]), 1e-13)
                    self.assertClose(basis.covectors(batch), np.array([basis.covector(batch[:, k, :]) for k in (0, 1)]), 1e-14)

    def test_labels_and_requirements(self):
        """S = 1 keeps chart.whitening unchanged; scaled variables need the density gauge
        and a ScaledState built on chart.state."""
        c = tc.make_coordinates(Z2)
        ss, basis = production(c, "none")
        self.assertClose(basis.bh, c.whitening, 0.0)
        self.assertEqual((basis.scaling, es.WhiteBasis(c, ss, "none").scaling), ("none", "none"))
        self.assertEqual(production(c, skew(4))[1].scaling, "vector")
        self.assertEqual(es.WhiteBasis(c, es.SolverState(c.state, "density"), "density").scaling, "density")
        original = tc.make_coordinates(Z2, gauge="original")
        self.assertClose(production(original, "none")[1].bh, original.whitening, 0.0)
        with self.assertRaises(ValueError):
            production(original, "density")
        with self.assertRaises(ValueError):
            es.WhiteBasis(c, ScaledState(original.state, "none"))
        for bad in (np.zeros(7), np.zeros((2, 8)), np.zeros(9)):
            with self.assertRaises(ValueError):
                basis.tangent(bad)
        with self.assertRaises(ValueError):
            basis.tangents(np.zeros(8))
        with self.assertRaises(ValueError):
            basis.hat(np.zeros((3, 3)))

    def test_dropped_metric_directions(self):
        """metric_tol = 0.2 drops the two smallest eigenvalues of rho (0.047, 0.065 of
        0.68): Bh = S B on the kept columns and the free sector matrix is the oracle's
        elementwise (1e-11) in every scaling. Each parity block keeps one eigenvector
        here, which LAPACK returns real and positive: the phases that distinguish the
        chart's eigen-whitening from a Cholesky factor are in test_eigen_whitened_charts."""
        for sector in ("odd", "even"):
            dense = oracle(sector, 0.5, "free", 8, 0.2)
            c = tc.make_coordinates(Z2, sector=sector, metric_tol=0.2)
            self.assertEqual((c.metric_rank, len(c), sorted(set(c.cols.tolist()))), (2, 4, [1, 3]))
            self.assertClose(c.whitening, dense["coordinates"].whitening, 0.0)
            for scaling in ("none", "density", skew(4)):
                ss, basis = production(c, scaling)
                self.assertEqual(basis.kept.tolist(), [False, True, False, True])
                self.assertLess(basis.defect, 1e-13)
                self.assertClose(basis.bh, ss.s_host[:, None] * c.whitening, 1e-13)
                op = es.ExcitationOperator(basis, 0.5, FREE, integrator=c.state.integrator, kinetic_nodes=8)
                self.assertClose(operator_matrix(op), dense["hamiltonian"], 1e-11)

    def test_eigen_whitened_charts(self):
        """src/excitation_eigsolve.jl:102-110: once a column is dropped the chart whitens
        every block by the eigenvectors of the rotated rho, whose phases are fixed by
        roundoff. metric_tol = DROP drops one column (rank 3, columns [1, 2, 3]); on this
        machine diag(B)[2] = 2.6952 + 2.8502i. The hat whitening must be S B with those
        phases (1e-13), so that the same coefficient vector is the same physical tangent
        and the free sector matrix is the oracle's elementwise (1e-11) in every scaling.
        (The positive Cholesky factor of rh gave |Bh - S B| = 0.79 and matrices 0.45 and
        0.59 from the oracle's, with equal spectra.) So that the test does not rest on the
        phase LAPACK happens to return, the same is asserted for charts with imposed
        column phases exp(0.7i j): the one-column-dropped chart and the Cholesky chart."""
        phases = np.exp(0.7j * np.arange(4))
        for sector in ("odd", "even"):
            dense = oracle(sector, 0.5, "free", 8, DROP)
            c = tc.make_coordinates(Z2, sector=sector, metric_tol=DROP)
            self.assertEqual((c.metric_rank, len(c), sorted(set(c.cols.tolist()))), (3, 6, [1, 2, 3]))
            self.assertClose(c.whitening, dense["coordinates"].whitening, 0.0)
            full = tc.make_coordinates(Z2, sector=sector)
            charts = [("dropped", c, dense["hamiltonian"])]
            for name, chart in (("dropped, phased", c), ("cholesky, phased", full)):
                phased = replace(chart, whitening=chart.whitening * phases[None, :])
                charts.append((name, phased, oracle_matrix(phased, 0.5, FREE, 8)))
            for name, chart, want in charts:
                x = gaussian(1, len(chart))
                w = tc.tangent_matrix(chart, x)
                for scaling in ("none", "density", skew(4)):
                    msg = f"{sector} {name} {scaling if isinstance(scaling, str) else 'skew'}"
                    ss, basis = production(chart, scaling)
                    self.assertEqual(basis.kept.tolist(), [name == "cholesky, phased", True, True, True])
                    self.assertLess(basis.defect, 1e-13, msg)
                    self.assertClose(basis.bh, ss.s_host[:, None] * chart.whitening, 1e-13, msg)
                    self.assertClose(basis.physical(x), tc.original_tangent(chart, w), 1e-13, msg)
                    self.assertClose(basis.coordinates(basis.tangent(x)), x, 1e-13, msg)
                    op = es.ExcitationOperator(basis, 0.5, FREE, integrator=chart.state.integrator, kinetic_nodes=8)
                    self.assertClose(operator_matrix(op), want, 1e-11, msg)

    def test_dropped_direction_with_ode_terms(self):
        """The same one-column-dropped chart with the phi^4 term, which test_dropped_metric_
        directions (free model) never integrates, on potential.ActionSystem and on the
        Z2-packed system (tangents M Bh' keep exact block zeros): operator matrix against
        the oracle's elementwise (1e-9, as OperatorTest) and packed against unpacked (1e-10)."""
        for sector in ("odd", "even"):
            dense = oracle(sector, 0.5, "phi4", NODES, DROP)
            c = tc.make_coordinates(Z2, sector=sector, metric_tol=DROP)
            for scaling in ("density", skew(4)):
                _, basis = production(c, scaling)
                matrices = [operator_matrix(es.ExcitationOperator(
                    basis, 0.5, PHI4, integrator=c.state.integrator, kinetic_nodes=NODES, integrate=integrate_host,
                    system=system)) for system in (None, PackedActionSystem)]
                for h in matrices:
                    self.assertClose(h, dense["hamiltonian"], 1e-9, sector)
                self.assertClose(matrices[1], matrices[0], 1e-10, sector)


class OperatorTest(Case):
    """ExcitationOperator, kinetic_preconditioner and field_seed
    (src/excitation_eigsolve.jl:142-209) against the oracle on the same white basis."""

    def check_sector(self, sector):
        for p in (0.0, 0.5):
            dense = oracle(sector, p)
            c = tc.make_coordinates(Z2, sector=sector)
            self.assertClose(c.whitening, dense["coordinates"].whitening, 0.0)
            ctx = ref.make_context(c.state, p)
            pre0, seed0 = ref.kinetic_preconditioner(ctx, c, PHI4, nodes=8), ref.field_seed(ctx, c)
            for scaling in ("none", "density"):
                with self.subTest(sector=sector, p=p, scaling=scaling):
                    ss, basis = production(c, scaling)
                    op = es.ExcitationOperator(basis, p, PHI4, integrator=c.state.integrator, kinetic_nodes=NODES)
                    h = operator_matrix(op)
                    self.assertEqual(op.calls, len(c))
                    self.assertClose(h, dense["hamiltonian"], 1e-9)
                    self.assertClose(h, adj(h), 1e-9)
                    pre = es.kinetic_preconditioner(basis, ss, p, PHI4, nodes=8, tol=1e-12)
                    self.assertClose(pre["matrix"], pre0["matrix"], 1e-9)
                    self.assertClose(pre["values"], pre0["values"], 1e-9)
                    self.assertEqual((pre["floor"], pre["nodes"]), (1.0, 8))
                    self.assertLess(pre["skew"], 1e-10)
                    r = gaussian(9, len(c))
                    self.assertClose(pre["apply"](r), pre0["apply"](r), 1e-9)
                    seed = es.field_seed(basis, ss, p)
                    self.assertClose(seed, seed0, 1e-11)
                    self.assertEqual(norm(seed) > 1e-6, sector == "odd")   # the field insertion is odd

    def test_odd_sector(self):
        """Operator matrix on all unit coefficient vectors = the oracle's action-assembled
        sector matrix, elementwise (1e-9) and Hermitian (1e-9), for S = 1 and
        S = sqrt(rho), p in (0, 0.5); preconditioner matrix, spectrum, apply and floor
        (test/excitation_eigsolve.jl:239-243); field seed (:229-236)."""
        self.check_sector("odd")

    def test_even_sector(self):
        self.check_sector("even")

    def test_scaling_independence(self):
        """Without ODE terms nothing but roundoff depends on S: free operator matrix,
        coarse kinetic matrix and field seed for S = 1, sqrt(rho) and the arbitrary SKEW
        agree to 1e-11, at p = 0.5 in both sectors; the coarse matrix is the kinetic-only
        sector matrix of the oracle on 8 nodes (1e-10) whatever the model."""
        p = 0.5
        for sector in ("odd", "even"):
            c = tc.make_coordinates(Z2, sector=sector)
            results = []
            for scaling in ("none", "density", skew(4)):
                ss, basis = production(c, scaling)
                op = es.ExcitationOperator(basis, p, FREE, integrator=c.state.integrator, kinetic_nodes=NODES)
                pre = es.kinetic_preconditioner(basis, ss, p, PHI4, nodes=8, tol=1e-12)
                results.append((operator_matrix(op), pre["matrix"], es.field_seed(basis, ss, p)))
                self.assertClose(es.kinetic_preconditioner(basis, ss, p, FREE, nodes=8, tol=1e-12)["matrix"],
                                 pre["matrix"], 1e-12)
            for other in results[1:]:
                for a, b in zip(results[0], other):
                    self.assertClose(a, b, 1e-11)
            self.assertClose(results[0][0], oracle(sector, p, "free")["hamiltonian"], 1e-11)
            k = results[1][1]
            self.assertClose((k + adj(k)) / 2, oracle(sector, p, "free", 8)["hamiltonian"], 1e-10)

    def test_operator_and_settings(self):
        """calls, with_action (with_context: new settings, calls = 0, same basis), the ODE
        backend and equations keywords, and vdot(x, op(y)) = vdot(W(x), F(W(y))) against the
        oracle's action (1e-10)."""
        p = 0.5
        c = tc.make_coordinates(Z2)
        ss, basis = production(c, "density")
        op = es.ExcitationOperator(basis, p, PHI4, integrator=c.state.integrator, kinetic_nodes=NODES)
        x, y = gaussian(8, len(c)), gaussian(9, len(c))
        self.assertEqual(op.calls, 0)
        hy = op(y)
        self.assertEqual((op.calls, op.action.calls), (1, 1))
        f = ref.hamiltonian_action(ref.make_context(c.state, p), tc.tangent_matrix(c, y), PHI4, kinetic_nodes=NODES)
        self.assertClose(hy, tc.covector_coordinates(c, f), 1e-10)
        self.assertClose(np.vdot(x, hy), np.vdot(tc.tangent_matrix(c, x), f), 1e-10)
        scipy = es.ExcitationOperator(basis, p, PHI4, integrator=c.state.integrator, kinetic_nodes=NODES,
                                      integrate=integrate_scipy, system=ActionSystem)
        self.assertClose(scipy(y), hy, 1e-10)
        for scaling in ("none", skew(4)):   # with the ODE term: S-independent at the integration tolerance
            other = es.ExcitationOperator(production(c, scaling)[1], p, PHI4, integrator=c.state.integrator,
                                          kinetic_nodes=NODES)
            self.assertClose(other(y), hy, 1e-10)
        fine = es.refined_integration(op.integrator)
        self.assertEqual((fine.n_quad, fine.max_point, fine.maxiters), (64, 24, op.integrator.maxiters))
        self.assertClose([fine.atol, fine.rtol], [1e-14, 1e-13], 1e-25)
        reference = op.with_action(integrator=fine, kinetic_nodes=2 * NODES)
        self.assertEqual((reference.calls, reference.kinetic_nodes, reference.integrator.n_quad), (0, 64, 64))
        self.assertIs(reference.basis, basis)
        self.assertIs(reference.settings["integrate"], op.settings["integrate"])
        self.assertEqual((op.kinetic_nodes, op.integrator.n_quad, op.calls), (NODES, 32, 1))
        same = op.with_action()
        self.assertEqual((same.kinetic_nodes, same.integrator, same.calls), (NODES, op.integrator, 0))
        self.assertLess(norm(reference(y) - hy), 1e-6)   # the window 14 -> 24 changes the action
        self.assertIsNot(op.integrator, c.state.integrator)
        with self.assertRaises(ValueError):
            es.ExcitationOperator(basis, p, PHI4, integrator=c.state.integrator, kinetic_nodes=1)

    def test_solver_statistics(self):
        """SolverState counts every batched solve; solver_statistics also reports the fixed
        point and the eigenbases, with or without counters."""
        c = tc.make_coordinates(Z2)
        plain = es.solver_statistics(ScaledState(c.state, "density"))
        self.assertEqual(sorted(plain), ["eigenbasis_condition", "fixed_point_residual", "scale_range"])
        ss = es.SolverState(c.state, "density")
        self.assertEqual(ss.stats, dict(solves=0, iterations=0, steps=0, max_residual=0.0))
        rhs = np.ascontiguousarray(gaussian(5, 4, 3, 4))
        _, info = ss.poisson(rhs, "L", "H", 0.3j)
        _, more = ss.resolvent(rhs, "H", "H", 0.7)
        report = es.solver_statistics(ss)
        self.assertEqual(report["solves"], 6)
        self.assertEqual(report["iterations"], int(info["iterations"].sum() + more["iterations"].sum()))
        self.assertEqual(report["steps"], info["steps"] + more["steps"])
        self.assertEqual(report["max_residual"], max(info["residuals"].max(), more["residuals"].max()))
        self.assertLess(report["fixed_point_residual"], 1e-13)
        self.assertEqual(report["scale_range"], (ss.s_host.min(), ss.s_host.max()))

    def test_preconditioner_details(self):
        """Chunked assembly, the eigen-floor rule r -> sum_k v_k <v_k, r> / max(floor, kappa_k),
        a supplied floor, and the argument errors of kinetic_preconditioner
        (src/excitation_eigsolve.jl:181-201)."""
        p = 0.5
        c = tc.make_coordinates(Z2)
        ss, basis = production(c, "density")
        before = dict(ss.stats)
        pre = es.kinetic_preconditioner(basis, ss, p, PHI4, tol=1e-12)
        self.assertGreater(ss.stats["solves"], before["solves"])
        self.assertLessEqual(ss.stats["max_residual"], 2e-11)
        self.assertEqual(sorted(pre), ["apply", "floor", "matrix", "nodes", "skew", "values", "vectors"])
        k = pre["matrix"]
        for chunk in (1, 3, 8):
            self.assertClose(es.kinetic_preconditioner(basis, ss, p, PHI4, tol=1e-12, chunk=chunk)["matrix"], k, 1e-12)
        self.assertClose(es.kinetic_preconditioner(basis, ss, p, PHI4)["matrix"], k, 1e-8)   # default tol 1e-10
        values, vectors = np.linalg.eigh((k + adj(k)) / 2)
        self.assertClose(pre["values"], values, 0.0)
        self.assertLess(values.min(), 1.0)   # the floor acts on this state
        self.assertClose(pre["skew"], norm(k - adj(k)) / max(1, norm(k)), 0.0)
        r = gaussian(9, len(c))
        self.assertClose(pre["apply"](r), vectors @ ((adj(vectors) @ r) / np.maximum(1.0, values)), 1e-14)
        wide = es.kinetic_preconditioner(basis, ss, p, PHI4, tol=1e-12, floor=2.5)
        self.assertEqual(wide["floor"], 2.5)
        self.assertClose(wide["apply"](r), vectors @ ((adj(vectors) @ r) / np.maximum(2.5, values)), 1e-14)
        doubled = [(2.0, spec) for _, spec in FREE]
        self.assertEqual(es.kinetic_preconditioner(basis, ss, p, doubled, tol=1e-12)["floor"], 2.0)
        for request in (dict(nodes=1), dict(floor=0.0), dict(floor=math.inf)):
            with self.assertRaises(ValueError):
                es.kinetic_preconditioner(basis, ss, p, PHI4, **request)
        with self.assertRaises(ValueError):
            es.kinetic_preconditioner(basis, ss, p, [(1.0, models.monomial_spec(4))])


class EigsolveTest(Case):
    """excitation_eigsolve and solve_levels against the oracle's dense spectra."""
    KEYS = ("energies vectors coefficients residuals converged stop_reason refined_converged accepted nev "
            "validation_tol acceptance_residual numops action_calls history momentum sector parity parity_error "
            "gauge dimension metric_rank discarded_metric_weight metric_defect density_condition start davidson "
            "preconditioner corrections validation validated_energies integration refined_integration "
            "kinetic_nodes refined_kinetic_nodes timings coordinates scaling solver basis").split()

    def check_levels(self, sector, scaling, nev, p=0.5):
        """Loop body of test/excitation_eigsolve.jl:177-195 without validation: lowest
        levels (1e-8), start kind, preconditioner, orthonormal W in the gauge of the input
        state, and the bookkeeping of solve_levels."""
        s = Z2
        reference = np.linalg.eigvalsh(oracle(sector, p)["hamiltonian"])
        lines, rows = [], []
        r = es.excitation_eigsolve(s, p, PHI4, nev=nev, sector=sector, scaling=scaling, kinetic_nodes=NODES,
                                   validate=False, log=lines.append, callback=lambda row, u, au: rows.append(row))
        self.assertEqual(sorted(r), sorted(self.KEYS))
        self.assertTrue(r["converged"] and r["accepted"], f"stop_reason {r['stop_reason']}, residuals {r['residuals']}")
        self.assertClose(r["energies"], reference[:nev], 1e-8)
        self.assertEqual(r["start"], "field" if sector == "odd" else "kinetic")
        self.assertLess(r["preconditioner"]["skew"], 1e-8)
        self.assertEqual((r["preconditioner"]["nodes"], r["preconditioner"]["floor"]), (8, 1.0))
        self.assertEqual((r["sector"], r["gauge"], r["scaling"], r["dimension"], r["metric_rank"]),
                         (sector, "density", scaling, 8, 4))
        self.assertEqual((r["momentum"], r["nev"], r["kinetic_nodes"], r["refined_kinetic_nodes"]), (p, nev, NODES, None))
        self.assertEqual((r["stop_reason"], len(r["history"]), r["action_calls"]), ("converged", r["numops"], r["numops"]))
        self.assertEqual((r["validation"], r["validated_energies"], r["refined_converged"], r["refined_integration"]),
                         (None, None, None, None))
        self.assertEqual((len(r["corrections"]), r["davidson"].maxspace, r["integration"].n_quad), (0, 8, 32))
        self.assertEqual(rows, r["history"])
        self.assertEqual(sum(line.startswith("davidson  ") for line in lines), r["numops"])
        self.assertTrue(lines[0].startswith("context ") and any(line.startswith("preconditioner ") for line in lines))
        c, basis = r["coordinates"], r["basis"]
        self.assertIs(basis.chart, c)
        for k in range(nev):
            for l in range(nev):
                self.assertClose(ref.tangent_metric(s, r["vectors"][k], r["vectors"][l]), float(k == l), 1e-10)
            w = tc.original_tangent(c, tc.tangent_matrix(c, r["coefficients"][k]))
            self.assertClose(r["vectors"][k], w, 1e-12)
        pair = ref.make_pair(ref.make_context(s, p), r["vectors"][0], r["vectors"][0])
        self.assertClose(ref.matrix_element(pair, PHI4, kinetic_nodes=NODES), r["energies"][0], 1e-8)
        solver = r["solver"]
        self.assertTrue(solver["solves"] > 0 and solver["iterations"] > 0 and solver["steps"] > 0)
        self.assertLess(solver["max_residual"], 1e-9)
        self.assertLess(solver["fixed_point_residual"], 1e-12)
        self.assertLess(solver["white_defect"], 1e-12)
        self.assertEqual(sorted(solver["eigenbasis_condition"]), ["H", "L"])
        self.assertEqual(sorted(r["timings"]), ["context", "operator", "preconditioner", "solve", "validation"])
        return r

    def test_odd_levels(self):
        """Odd sector, nev in (1, 2), S = 1 and S = sqrt(rho): the lowest oracle levels (1e-8)."""
        for scaling in ("none", "density"):
            for nev in (1, 2):
                with self.subTest(scaling=scaling, nev=nev):
                    self.check_levels("odd", scaling, nev)

    def test_even_levels(self):
        for scaling in ("none", "density"):
            for nev in (1, 2):
                with self.subTest(scaling=scaling, nev=nev):
                    self.check_levels("even", scaling, nev)

    def test_scaling_vector_and_original_gauge(self):
        """The arbitrary scaling SKEW and the unscaled original gauge give the same levels."""
        reference = np.linalg.eigvalsh(oracle("odd", 0.5)["hamiltonian"])
        r = es.excitation_eigsolve(Z2, 0.5, PHI4, scaling=skew(4), kinetic_nodes=NODES, validate=False)
        self.assertEqual((r["scaling"], r["gauge"], r["converged"]), ("vector", "density", True))
        self.assertClose(r["energies"], reference[:1], 1e-8)
        r = es.excitation_eigsolve(Z2, 0.5, PHI4, gauge="original", scaling="none", kinetic_nodes=NODES, validate=False)
        self.assertEqual((r["scaling"], r["gauge"], r["converged"]), ("none", "original", True))
        self.assertClose(r["energies"], reference[:1], 1e-8)
        self.assertClose(ref.tangent_metric(Z2, r["vectors"][0], r["vectors"][0]), 1.0, 1e-10)

    def test_default_terms(self):
        """terms=None is phi4_terms(state.coupling), here g = 1."""
        reference = np.linalg.eigvalsh(oracle("odd", 0.5)["hamiltonian"])
        r = es.excitation_eigsolve(Z2, 0.5, kinetic_nodes=NODES, validate=False)
        self.assertClose(r["energies"], reference[:1], 1e-8)
        with self.assertRaises(ValueError):
            es.excitation_eigsolve(PLAIN, 0.5, kinetic_nodes=NODES, validate=False)   # coupling is nan

    def test_default_terms_need_the_phi4_model(self):
        """src/excitation_eigsolve.jl:468 and src/excitations.jl:12-28: the default model
        dispatches on the stored model name. Only phi4 is ported: another name (which
        states.load_jld2 carries over from a checkpoint) must not receive phi^4 terms at
        its coupling. Both entry points and the oracle refuse it before any solve;
        explicit terms are accepted whatever the name."""
        other = replace(Z2, model="shg")
        self.assertEqual(models.default_terms(Z2), models.phi4_terms(1.0))
        with mock.patch.object(es, "SolverState", side_effect=AssertionError("a context was built")):
            for call in (lambda: es.excitation_eigsolve(other, 0.5, kinetic_nodes=NODES, validate=False),
                         lambda: es.iterative_dispersion(other, [0.5], kinetic_nodes=NODES, validate=False),
                         lambda: models.default_terms(other)):
                with self.assertRaises(ValueError) as failure:
                    call()
                self.assertIn("only the phi4 model has default terms", str(failure.exception))
        for call in (lambda: ref.eigsolve(other, 0.5, kinetic_nodes=NODES, validate=False),
                     lambda: ref.dense_hamiltonian(other, 0.5, None, kinetic_nodes=8)):
            with self.assertRaises(ValueError):
                call()
        r = es.excitation_eigsolve(other, 0.5, PHI4, kinetic_nodes=NODES, validate=False,
                                   davidson_options=DavidsonOptions(maxops=1, stall_window=0))
        self.assertEqual(r["numops"], 1)

    def test_one_dropped_direction(self):
        """The driver on the one-column-dropped chart (metric_tol = DROP, odd sector,
        dimension 6, nev = 2). The oracle's white coefficients are the same vectors in
        every scaling: supplied as `start` they converge at once (2 operations, one per
        vector; with a Cholesky hat whitening the scaled runs started 0.1 and 0.25 above
        the levels and took 6), and the returned coefficients and tangents are the
        oracle's up to a phase (1e-6). A cold validated run is accepted on the same
        energy (1e-8), its four energies within 1e-8 (the window 14 -> 24)."""
        request = dict(nev=2, sector="odd", metric_tol=DROP, kinetic_nodes=NODES, validate=False)
        dense = ref.eigsolve(Z2, 0.5, PHI4, **request)
        self.assertTrue(dense["converged"])
        self.assertEqual((dense["metric_rank"], dense["dimension"]), (3, 6))
        for scaling in ("none", "density", skew(4)):
            msg = scaling if isinstance(scaling, str) else "skew"
            r = es.excitation_eigsolve(Z2, 0.5, PHI4, scaling=scaling, start=dense["coefficients"], **request)
            self.assertEqual((r["start"], r["converged"], r["numops"]), ("supplied", True, 2), msg)
            self.assertClose(r["energies"], dense["energies"], 1e-8, msg)
            for x, y, w, v in zip(r["coefficients"], dense["coefficients"], r["vectors"], dense["vectors"]):
                self.assertClose(abs(np.vdot(y, x)), 1.0, 1e-6, msg)
                self.assertClose(w * np.vdot(w, v) / abs(np.vdot(w, v)), v, 1e-6, msg)
        cold = es.excitation_eigsolve(Z2, 0.5, PHI4, sector="odd", metric_tol=DROP, kinetic_nodes=NODES)
        self.assertTrue(cold["converged"] and cold["accepted"])
        self.assertEqual((cold["metric_rank"], cold["dimension"], cold["scaling"]), (3, 6, "density"))
        self.assertClose(cold["energies"], dense["energies"][:1], 1e-8)
        self.assertLess(cold["validation"][0]["energy_spread"], 1e-8)

    def test_nan_energies_are_rejected(self):
        """src/excitation_eigsolve.jl:360-370: maximum and minimum propagate NaN and
        NaN <= tol is false, so a level with a NaN energy is not accepted. Python's max and
        min ignore a NaN that is not their first argument: a plug-in `bilinear_element`
        returning NaN was accepted with validated energy NaN. Also the zero-norm case of
        bilinear_level, NaN + NaN im in Julia."""
        class Broken:
            def __init__(self, *args, **kw):
                pass

            def energy(self, wh):
                return complex(math.nan, 0.0)

        r = es.excitation_eigsolve(Z2, 0.5, PHI4, kinetic_nodes=NODES, bilinear_element=Broken)
        v, = r["validation"]
        self.assertTrue(r["converged"] and v["residual_target_met"])
        self.assertLess(abs(v["standard_energy"] - v["refined_energy"]), 1e-8)   # the two fresh actions alone agree
        self.assertTrue(math.isnan(v["energy_spread"]) and math.isnan(r["validated_energies"][0]))
        self.assertFalse(v["consistent"] or v["accepted"] or r["accepted"])
        with self.assertRaises(SolverFailure):
            es.accepted_energies(r)
        with self.assertRaises(SolverFailure):
            es.iterative_dispersion(Z2, [0.5], PHI4, kinetic_nodes=NODES, bilinear_element=Broken)
        c = tc.make_coordinates(Z2)
        level = es.bilinear_level(Broken(), production(c, "density")[1], np.eye(len(c))[0])
        self.assertTrue(math.isnan(level["energy"]) and level["imaginary"] == 0.0)

    def test_unique_fixed_point_is_required(self):
        """src/excitation_environments.jl:24-31, reached from excitation_eigsolve.jl:475 and
        :490: a state without a unique, mixing fixed point is refused before any solve, with
        Julia's message. The Z2 cat state (two zero modes) passes every solve of an action
        at p != 0: before check_mixing the driver returned energies for it (-1.6193 odd,
        -3.7901 even at p = 0.3). check_mixing passes the test states without touching the
        solver statistics."""
        cat = cat_state()
        with self.assertRaises(SolverFailure) as failure:
            ref.make_context(tc.make_coordinates(cat).state, 0.3)
        self.assertEqual(str(failure.exception), es.MIXING_ERROR)
        with mock.patch.object(es, "WhiteBasis", side_effect=AssertionError("a basis was built")):
            for scaling in ("none", "density"):
                for call in (lambda: es.excitation_eigsolve(cat, 0.3, scaling=scaling, kinetic_nodes=NODES),
                             lambda: es.excitation_eigsolve(cat, 0.0, sector="even", scaling=scaling,
                                                            kinetic_nodes=NODES, validate=False),
                             lambda: es.iterative_dispersion(cat, [0.3], scaling=scaling, kinetic_nodes=NODES)):
                    with self.assertRaises(SolverFailure) as failure:
                        call()
                    self.assertTrue(str(failure.exception).startswith(es.MIXING_ERROR), str(failure.exception))
        for state in (Z2, PLAIN):
            for scaling in ("none", "density"):
                ss = es.SolverState(tc.make_coordinates(state).state, scaling)
                es.check_mixing(ss)
                self.assertEqual(ss.stats, dict(solves=0, iterations=0, steps=0, max_residual=0.0))

    def test_no_parity(self):
        """test/excitation_eigsolve.jl:272-283: a state without parity at p = 0.5, full
        tangent space of dimension 9 > maxspace (restarts), nev = 2 (1e-8), and a warm
        start from the dense eigenvector supplied as a tangent W in the gauge of the state
        (at most 2 operations), for S = sqrt(rho); S = 1 with nev = 1."""
        s, dense = PLAIN, plain_spectrum()
        r = es.excitation_eigsolve(s, 0.5, PHI4, nev=2, kinetic_nodes=NODES, validate=False)
        self.assertEqual((r["sector"], r["dimension"], r["converged"], r["start"], r["scaling"]),
                         ("all", 9, True, "field", "density"))
        self.assertIsNone(r["parity"])
        self.assertTrue(math.isnan(r["parity_error"]))
        self.assertClose(r["energies"], dense["energies"], 1e-8)
        self.assertTrue(any(row["cycle"] > 1 for row in r["history"]))
        self.assertEqual((r["action_calls"], r["accepted"]), (r["numops"], True))
        for scaling in ("density", "none"):
            warm = es.excitation_eigsolve(s, 0.5, PHI4, kinetic_nodes=NODES, validate=False, scaling=scaling,
                                          start=dense["vectors"][0])
            self.assertEqual(warm["start"], "supplied")
            self.assertLessEqual(warm["numops"], 2)
            self.assertClose(warm["energies"][0], dense["energies"][0], 1e-8)
        cold = es.excitation_eigsolve(s, 0.5, PHI4, kinetic_nodes=NODES, validate=False, scaling="none")
        self.assertTrue(cold["converged"])
        self.assertClose(cold["energies"], dense["energies"][:1], 1e-8)
        self.assertClose(abs(np.vdot(cold["coefficients"][0], warm["coefficients"][0])), 1.0, 1e-6)

    def test_plain_search_and_function_preconditioner(self):
        """preconditioner 'none' with start 'random', a function preconditioner, coordinate
        starts (one vector, a list) and start 'kinetic'."""
        s, dense = PLAIN, plain_spectrum()
        r = es.excitation_eigsolve(s, 0.5, PHI4, kinetic_nodes=NODES, validate=False, preconditioner="none",
                                   start="random")
        self.assertEqual((r["start"], r["preconditioner"], r["converged"]), ("random", None, True))
        self.assertClose(r["energies"], dense["energies"][:1], 1e-8)
        x = r["coefficients"][0]
        calls = []
        for start in (x, [x], [x, gaussian(3, 9)]):
            warm = es.excitation_eigsolve(s, 0.5, PHI4, kinetic_nodes=NODES, validate=False, start=start,
                                          preconditioner=lambda v: calls.append(1) or v)
            self.assertEqual((warm["start"], warm["preconditioner"], warm["converged"]), ("supplied", None, True))
            self.assertLessEqual(warm["numops"], 3)
            self.assertClose(warm["energies"], dense["energies"][:1], 1e-8)
        kinetic = es.excitation_eigsolve(s, 0.5, PHI4, kinetic_nodes=NODES, validate=False, start="kinetic",
                                         davidson_options=DavidsonOptions(maxops=1, stall_window=0))
        self.assertEqual((kinetic["start"], kinetic["stop_reason"], kinetic["accepted"], kinetic["numops"]),
                         ("kinetic", "maxops", False, 1))

    def test_validation(self):
        """test/excitation_eigsolve.jl:181-190 and 245-250: with validation the four energies
        (fresh standard and refined actions, standard and refined bilinear) agree to 1e-8
        (measured spread 3.9e-9, the window 14 -> 24), each bilinear energy equals the fresh
        one at its settings (1e-9), the level is accepted, and the four energies are the
        oracle's (1e-8). Refined settings follow refined_integration."""
        s, p = Z2, 0.5
        lines = []
        r = es.excitation_eigsolve(s, p, PHI4, kinetic_nodes=NODES, log=lines.append)
        self.assertTrue(r["converged"] and r["accepted"])
        self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"], r["action_calls"]), (32, 64, r["numops"] + 1))
        fine = r["refined_integration"]
        self.assertEqual((fine.max_point, fine.n_quad), (24, 64))
        self.assertClose([fine.atol, fine.rtol], [1e-14, 1e-13], 1e-25)
        v, = r["validation"]
        names = ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")
        four = [v[name] for name in names]
        self.assertLess(max(four) - min(four), 1e-8)
        self.assertClose(v["energy_spread"], max(four) - min(four), 0.0)
        self.assertLess(abs(v["bilinear_energy"] - v["standard_energy"]), 1e-9)
        self.assertLess(abs(v["refined_bilinear_energy"] - v["refined_energy"]), 1e-9)
        self.assertLess(v["max_imaginary"], 1e-9)
        self.assertLess(max(v["standard_residual"], v["refined_residual"]), 1e-7)
        self.assertTrue(v["accepted"] and v["consistent"] and v["residual_target_met"])
        self.assertClose(v["standard_energy"], r["energies"][0], 1e-9)
        self.assertEqual(r["validated_energies"], [v["refined_bilinear_energy"]])
        self.assertEqual(es.reported_energies(r), r["validated_energies"])
        self.assertEqual(es.accepted_energies(r), r["validated_energies"])
        self.assertTrue(any(line.startswith("validation 0 ") for line in lines))
        self.assertGreater(r["timings"]["validation"], 0.0)
        dense = ref.eigsolve(s, p, PHI4, kinetic_nodes=NODES)
        self.assertClose(four, [dense["validation"][0][name] for name in names], 1e-8)
        self.assertEqual(sorted(set(dense) - set(r)), [])
        strict = es.excitation_eigsolve(s, p, PHI4, kinetic_nodes=NODES, validation_tol=1e-16)
        self.assertFalse(strict["accepted"] or strict["validation"][0]["consistent"])
        self.assertTrue(strict["converged"])
        with self.assertRaises(SolverFailure):
            es.accepted_energies(strict)
        self.assertEqual(es.reported_energies(strict), strict["validated_energies"])

    def test_refine_steps(self):
        """test/excitation_eigsolve.jl:251-257: a loose search (tol 1e-2, 3 operations, random
        start) followed by refine_steps = 3 corrections under the refined action. The
        corrections are the oracle's (same random start; 1e-8)."""
        options = DavidsonOptions(tol=1e-2, maxops=3, stall_window=0)
        request = dict(kinetic_nodes=NODES, refine_steps=3, start="random", davidson_options=options)
        loose = es.excitation_eigsolve(Z2, 0.0, PHI4, **request)
        corrections, validation = loose["corrections"], loose["validation"][0]
        self.assertTrue(len(corrections) > 1 and corrections[0]["step"] == 0)
        self.assertClose(loose["energies"][0], validation["refined_energy"], 1e-10)
        self.assertClose(loose["residuals"][0], validation["refined_residual"], 1e-10)
        self.assertLessEqual(loose["residuals"][0], corrections[0]["residual"])
        self.assertEqual(loose["converged"], loose["stop_reason"] == "converged")
        self.assertIsInstance(loose["refined_converged"], (bool, np.bool_))
        self.assertEqual(loose["refined_converged"], loose["residuals"][0] <= options.tol)
        self.assertEqual((loose["start"], loose["numops"]), ("random", 3))
        accepted = [row for row in corrections if row["accepted"]]
        self.assertEqual(loose["action_calls"], 3 + (len(corrections) - 1) + 1)   # search, corrections, validation
        self.assertEqual(accepted[-1]["residual"], loose["residuals"][0])
        dense = ref.eigsolve(Z2, 0.0, PHI4, **request)
        self.assertEqual(len(corrections), len(dense["corrections"]))
        for row, expected in zip(corrections, dense["corrections"]):
            self.assertEqual((row["step"], row["accepted"]), (expected["step"], expected["accepted"]))
            self.assertClose([row["energy"], row["residual"]], [expected["energy"], expected["residual"]], 1e-8)

    def test_request_validation(self):
        """test/excitation_eigsolve.jl:290-300 and check_levels_request: cheap requests fail
        before any solve. ArgumentError and DimensionMismatch are ValueError; an unknown
        keyword is a TypeError in Python (a ValueError through iterative_dispersion);
        max_bond has no counterpart. Added: gauge 'original' needs scaling 'none'."""
        s = Z2
        requests = [dict(start=[gaussian(k, 8) for k in range(8)]),       # at most maxspace-1 = 7 starts
                    dict(start=gaussian(1, 7)),                           # wrong coordinate length
                    dict(start=gaussian(1, 3, 3)),                        # wrong tangent size
                    dict(refined_kinetic_nodes=1),
                    dict(validate=1),
                    dict(nev=2, davidson_options=DavidsonOptions(maxspace=2)),
                    dict(nev=9),                                          # sector dimension is 8
                    dict(nev=0),
                    dict(refine_steps=1, nev=2),
                    dict(refine_steps=-1),
                    dict(start="kinetic", preconditioner="none"),
                    dict(start="invalid"),
                    dict(preconditioner="invalid"),
                    dict(preconditioner=3),
                    dict(kinetic_nodes=1),
                    dict(kinetic_nodes=256.0),                            # None is the D default (DefaultsTest)
                    dict(preconditioner_nodes=1),
                    dict(refined=dict(n_quad=64)),
                    dict(refined=IntegratorParameters(n_quad=4)),
                    dict(validation_tol=0.0),
                    dict(acceptance_residual=math.inf),
                    dict(solver_tol=0.0),
                    dict(preconditioner_tol=-1.0),
                    dict(chunk=0),
                    dict(gauge="original"),                               # with the default scaling 'density'
                    dict(gauge="original", scaling=np.ones(4)),
                    dict(gauge="invalid", scaling="none"),
                    dict(sector="invalid"),
                    dict(metric_tol=1.0),
                    dict(parity=[1, -1, 1, -1])]
        late = (dict(scaling="invalid"), dict(scaling=np.ones(3)))       # rejected by ScaledState itself
        with mock.patch.object(es, "SolverState", side_effect=AssertionError("a context was built")):
            for k, request in enumerate(requests):
                with self.assertRaises(ValueError, msg=f"request {k}: {sorted(request)}"):
                    es.excitation_eigsolve(s, 0.0, PHI4, **request)
        for request in late:
            with self.assertRaises(ValueError):
                es.excitation_eigsolve(s, 0.0, PHI4, **request)
        with self.assertRaises(ValueError):
            es.excitation_eigsolve(s, math.nan, PHI4)
        with self.assertRaises(ValueError):
            es.excitation_eigsolve(s, 0.0, PHI4, sector="even", start="field")   # the field insertion is odd
        with self.assertRaises(ValueError):
            es.excitation_eigsolve(PLAIN, 0.0, PHI4, sector="odd")               # no parity
        with self.assertRaises(TypeError):
            es.excitation_eigsolve(s, 0.0, PHI4, kinetic_node=3)
        with self.assertRaises(ValueError):
            es.iterative_dispersion(s, [0.0], PHI4, kinetic_node=3)
        with self.assertRaises(ValueError):
            es.iterative_dispersion(s, [0.0], PHI4, start=gaussian(1, 7))
        with self.assertRaises(ValueError):
            es.iterative_dispersion(s, [math.inf], PHI4, validate=False)
        c = tc.make_coordinates(s)
        self.assertEqual(es.check_levels_request(c, nev=4).maxspace, 10)
        self.assertEqual(es.check_levels_request(c).maxspace, 8)
        options = DavidsonOptions(maxspace=5)
        self.assertIs(es.check_levels_request(c, davidson_options=options, callback=None, log=print), options)

    @slow
    def test_restarts(self):
        """test/excitation_eigsolve.jl:260-271: the 18-dimensional odd sector of a D = 6 Z2
        state (random_state(6, seed=21, scaling=0.5, z2=True), cond(rho) = 40) forces
        restarts; fresh and thick restarts agree with the oracle matrix (1e-8)."""
        s6 = tuned(random_state(6, seed=21, scaling=0.5, z2=True), **WINDOW)
        odd6 = np.linalg.eigvalsh(ref.dense_hamiltonian(s6, 0.0, PHI4, sector="odd", assembly="action",
                                                        kinetic_nodes=NODES)["hamiltonian"])
        for options in (DavidsonOptions(tol=1e-8, maxspace=6, stall_window=0),
                        DavidsonOptions(tol=1e-8, maxspace=8, retain=3, refresh=False, stall_window=0)):
            r = es.excitation_eigsolve(s6, 0.0, PHI4, nev=2, kinetic_nodes=NODES, davidson_options=options,
                                       validate=False)
            self.assertTrue(r["converged"], f"stop_reason {r['stop_reason']}, residuals {r['residuals']}")
            self.assertEqual(r["dimension"], 18)
            self.assertTrue(any(row["cycle"] > 1 for row in r["history"]))
            self.assertClose(r["energies"], odd6[:2], 1e-8)


class IllConditioned(Case):
    """Production against the control for an ill-conditioned rho. The states above have
    cond(rho) = 14, where it does not show: Julia's context snapshots the stored rho
    (src/excitation_environments.jl:18-35), while ScaledState re-solves the fixed point in
    hat variables, which for scaling 'density' corrects the relative error of the small
    eigenvalues of the stored rho (scaled.py). So production with 'density' reproduces the
    oracle on rho = S rh S, production with 'none' the oracle on the stored rho, and the
    two differ by about eps cond(rho) |H|: agreement with Julia numbers cannot be better."""

    def test_refined_density(self):
        """R = 0.8 (a + 0.005 a'), K = diag(0, 0.7, 1.4), a the D = 3 lowering operator:
        cond(rho) = 1.6e9, transfer gap 0.32; a 50-digit stationary solve puts the
        hat-relative error of the stored rho at 1.4e-8 and that of S rh S at 5e-15. Whole
        tangent space (dimension 9), phi^4 at g = 1, p = 0.45, 16 theta nodes. Spectrum of
        the operator matrix, measured: 'density' is 1.5e-11 from the oracle on S rh S and
        9.3e-8 from the oracle on the stored rho; 'none' is 1.5e-11 from the latter."""
        lower = np.diag(np.sqrt(np.arange(1.0, 3.0)), 1).astype(complex)
        s = tuned(make_state(0.8 * (lower + 0.005 * adj(lower)), np.diag(0.7 * np.arange(3.0))), **WINDOW)
        p, nodes = 0.45, 16

        def dense_spectrum(state):
            return np.linalg.eigvalsh(ref.dense_hamiltonian(state, p, PHI4, gauge="density", metric_tol=0.0,
                                                            kinetic_nodes=nodes)["hamiltonian"])

        def spectrum(basis):
            op = es.ExcitationOperator(basis, p, PHI4, integrator=c.state.integrator, kinetic_nodes=nodes, tol=1e-14)
            h = operator_matrix(op)
            return np.linalg.eigvalsh((h + adj(h)) / 2)

        c = tc.make_coordinates(s)
        self.assertEqual((c.sector, len(c)), ("all", 9))
        self.assertGreater(c.density_condition, 1e9)
        stored = dense_spectrum(s)
        ss, basis = production(c, "density")
        self.assertGreater(int(ss.fixed_point["iterations"][0]), 0)
        refined = make_state(c.state.R, c.state.K, ss.s_host[:, None] * ss.rh * ss.s_host[None, :],
                             integrator=replace(c.state.integrator))
        values = spectrum(basis)
        self.assertClose(values, dense_spectrum(refined), 1e-10, "oracle on S rh S")
        self.assertClose(values, stored, np.finfo(float).eps * c.density_condition * np.abs(stored).max(),
                         "oracle on the stored rho")
        plain, basis = production(c, "none")
        self.assertEqual(int(plain.fixed_point["iterations"][0]), 0)
        self.assertClose(spectrum(basis), stored, 1e-10, "stored rho is kept")


class DispersionTest(Case):
    """iterative_dispersion (src/excitation_eigsolve.jl:483-502) and the acceptance helpers."""

    def test_warm_start(self):
        """test/excitation_eigsolve.jl:210-218: shared coordinates and ScaledState; each
        momentum after the first starts from the previous coefficients: one operation when
        the momentum is repeated, fewer than a cold search for a nearby one. Energies are
        the oracle's (1e-8)."""
        s, dense = PLAIN, plain_spectrum()
        momenta = [0.5, 0.5, 0.52]
        lines = []
        results = es.iterative_dispersion(s, momenta, PHI4, kinetic_nodes=NODES, validate=False, log=lines.append)
        self.assertEqual([r["momentum"] for r in results], momenta)
        self.assertEqual([r["start"] for r in results], ["field", "supplied", "supplied"])
        self.assertTrue(all(r["converged"] and r["accepted"] for r in results))
        self.assertEqual(results[1]["numops"], 1)
        self.assertLess(results[2]["numops"], results[0]["numops"])
        self.assertClose([results[0]["energies"][0], results[1]["energies"][0]], [dense["energies"][0]] * 2, 1e-8)
        moved = ref.spectrum(s, 0.52, PHI4, assembly="action", kinetic_nodes=NODES)
        self.assertClose(results[2]["energies"], moved["energies"], 1e-8)
        self.assertIs(results[0]["basis"], results[2]["basis"])
        self.assertIs(results[0]["coordinates"], results[2]["coordinates"])
        self.assertGreater(results[0]["timings"]["context"], 0.0)
        self.assertEqual([r["timings"]["context"] for r in results[1:]], [0.0, 0.0])
        self.assertEqual([es.accepted_energies(r) for r in results], [r["energies"] for r in results])
        self.assertEqual(sum(line.startswith("context ") for line in lines), 1)
        self.assertEqual(sum(line.startswith("operator ") for line in lines), 3)
        # A supplied start is used at every momentum.
        x = results[0]["coefficients"][0]
        supplied = es.iterative_dispersion(s, [0.5, 0.5], PHI4, kinetic_nodes=NODES, validate=False, start=x)
        self.assertEqual([(r["start"], r["numops"]) for r in supplied], [("supplied", 1), ("supplied", 1)])

    def test_strict(self):
        """strict fails at the first momentum whose levels are not accepted (here an
        unconverged two-operation search without validation); strict=False keeps every
        momentum and reports acceptance (test/excitation_eigsolve.jl:284-289)."""
        request = dict(kinetic_nodes=NODES, validate=False, davidson_options=DavidsonOptions(maxops=2, stall_window=0))
        with self.assertRaises(SolverFailure) as failure:
            es.iterative_dispersion(PLAIN, [0.5, 0.6], PHI4, **request)
        self.assertIn("p=0.5 not accepted (stop: maxops", str(failure.exception))
        lax = es.iterative_dispersion(PLAIN, [0.5, 0.6], PHI4, strict=False, **request)
        self.assertEqual([(r["accepted"], r["stop_reason"], r["start"], r["numops"]) for r in lax],
                         [(False, "maxops", "field", 2), (False, "maxops", "supplied", 2)])
        self.assertEqual(es.reported_energies(lax[1]), lax[1]["energies"])
        with self.assertRaises(SolverFailure):
            es.accepted_energies(lax[1])
        self.assertEqual(es.iterative_dispersion(PLAIN, [], PHI4, **request), [])


class DefaultsTest(Case):
    """default_kinetic_nodes, its resolution by every public entry point, the two Davidson
    schedules and sector_dimension."""

    def test_default_kinetic_nodes(self):
        """256 up to D = 64, 512 beyond, not extrapolated past the measured D = 128."""
        for d, nodes in ((1, 256), (16, 256), (64, 256), (np.int64(64), 256), (65, 512), (96, 512), (128, 512),
                         (129, 512), (256, 512), (1024, 512)):
            self.assertEqual(es.default_kinetic_nodes(d), nodes, f"D = {d}")
        for bad in (0, -4, 64.0, True, None):
            with self.assertRaises(ValueError):
                es.default_kinetic_nodes(bad)
        self.assertEqual((es.resolve_kinetic_nodes(None, 64), es.resolve_kinetic_nodes(None, 65),
                          es.resolve_kinetic_nodes(16, 128)), (256, 512, 16))

    def test_entry_points_resolve_none(self):
        """excitation_eigsolve and iterative_dispersion hand the integer default to the
        request check (D = 16: 256; D = 66: 512, from a state that is never solved); an
        explicit integer is passed unchanged. check_levels_request, solve_levels and
        ExcitationOperator resolve None the same way, each on its own (D = 66: 512, seen
        through the node count each one checks or passes on; nothing is solved there)."""
        seen = []

        def capture(state, request, **controls):
            seen.append(request["kinetic_nodes"])
            raise RuntimeError("request captured")

        small, large = (random_state(d, seed=3, scaling=0.5, z2=True) for d in (16, 66))
        with mock.patch.object(es, "make_basis", side_effect=capture):
            for call in (lambda s, **kw: es.excitation_eigsolve(s, 0.0, PHI4, **kw),
                         lambda s, **kw: es.iterative_dispersion(s, [0.0], PHI4, **kw)):
                for state, request in ((small, {}), (large, {}), (small, dict(kinetic_nodes=None)),
                                       (large, dict(kinetic_nodes=None)), (large, dict(kinetic_nodes=300))):
                    with self.assertRaisesRegex(RuntimeError, "request captured"):
                        call(state, **request)
        self.assertEqual(seen, [256, 512, 256, 512, 300] * 2)
        self.assertTrue(all(type(nodes) is int for nodes in seen))
        es.check_levels_request(tc.make_coordinates(large))                     # None accepted at any D
        with self.assertRaises(ValueError):
            es.check_levels_request(tc.make_coordinates(large), kinetic_nodes=1)
        c = tc.make_coordinates(Z2)
        basis = production(c, "density")[1]
        self.assertEqual(es.ExcitationOperator(basis, 0.5, PHI4, integrator=c.state.integrator).kinetic_nodes, 256)

        # check_levels_request checks the D-dependent default, not a constant.
        charts = {16: tc.make_coordinates(small), 66: tc.make_coordinates(large)}
        for d, request, nodes in ((16, {}, 256), (66, {}, 512), (66, dict(kinetic_nodes=None), 512),
                                  (66, dict(kinetic_nodes=300), 300)):
            with mock.patch.object(es, "check_kinetic_nodes", wraps=es.check_kinetic_nodes) as checked:
                es.check_levels_request(charts[d], preconditioner="none", **request)
            self.assertEqual([call.args[0] for call in checked.call_args_list], [nodes], f"D = {d}, {request}")

        # solve_levels hands the integer default to the request check (stub basis of D = 66).
        passed = []

        def captured(chart, **request):
            passed.append(request["kinetic_nodes"])
            raise RuntimeError("request captured")

        for d, request, nodes in ((16, {}, 256), (66, {}, 512), (66, dict(kinetic_nodes=None), 512),
                                  (66, dict(kinetic_nodes=300), 300)):
            stub = types.SimpleNamespace(chart=charts[d], ss=None, D=d)
            with mock.patch.object(es, "check_levels_request", side_effect=captured), \
                    self.assertRaisesRegex(RuntimeError, "request captured"):
                es.solve_levels(stub, 0.0, PHI4, **request)
            self.assertEqual((passed[-1], type(passed[-1])), (nodes, int), f"D = {d}, {request}")

        # ExcitationOperator builds its action with the D-dependent default (stub basis).
        built = []

        def action(ss, momentum, terms, **controls):
            built.append(controls["kinetic_nodes"])
            return object()

        integrator = IntegratorParameters(**WINDOW)
        with mock.patch.object(es, "HamiltonianAction", side_effect=action):
            for d, request, nodes in ((16, {}, 256), (66, {}, 512), (66, dict(kinetic_nodes=None), 512),
                                      (66, dict(kinetic_nodes=300), 300)):
                op = es.ExcitationOperator(types.SimpleNamespace(D=d, ss=None), 0.0, PHI4, integrator=integrator,
                                           **request)
                self.assertEqual((op.kinetic_nodes, built[-1]), (nodes, nodes), f"D = {d}, {request}")
                self.assertEqual(op.with_action().kinetic_nodes, nodes)        # with_action keeps the count
                self.assertEqual(op.with_action(kinetic_nodes=600).kinetic_nodes, 600)

    def test_none_is_bitwise_256(self):
        """On the D = 4 state, kinetic_nodes=None and kinetic_nodes=256 give the same result
        to the last bit: two Davidson operations, then validation (refined: 512 nodes)."""
        options = DavidsonOptions(maxops=2, stall_window=0)
        runs = [es.excitation_eigsolve(Z2, 0.5, PHI4, davidson_options=options, **request)
                for request in ({}, dict(kinetic_nodes=None), dict(kinetic_nodes=256))]
        for r in runs:
            self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"], r["numops"]), (256, 512, 2))
        for r in runs[:2]:
            for key in ("energies", "residuals", "coefficients", "vectors"):
                self.assertTrue(np.array_equal(np.asarray(r[key]), np.asarray(runs[2][key])), key)
            self.assertEqual(r["history"], runs[2]["history"])
            self.assertEqual(r["validation"], runs[2]["validation"])
            self.assertEqual(r["validated_energies"], runs[2]["validated_energies"])
        dispersion = [es.iterative_dispersion(Z2, [0.5], PHI4, validate=False, davidson_options=options, strict=False,
                                              **request)[0] for request in ({}, dict(kinetic_nodes=256))]
        self.assertEqual(dispersion[0]["kinetic_nodes"], 256)
        self.assertTrue(np.array_equal(dispersion[0]["coefficients"], dispersion[1]["coefficients"]))
        self.assertEqual(dispersion[0]["history"], dispersion[1]["history"])

    def test_davidson_schedules(self):
        """research_davidson_options: maxspace = min(160, n), retain = min(max(12, nev),
        maxspace - 1), thick restarts; ValueError when nev >= maxspace; equal to Julia's
        research DavidsonOptions wherever Julia's check_davidson accepts them. The library
        schedule is the default of excitation_eigsolve."""
        def fields(o):
            return (o.maxspace, o.retain, o.refresh)

        for (n, nev), expected in (((128, 1), (128, 12, False)), ((2048, 1), (160, 12, False)),
                                   ((8, 1), (8, 7, False)), ((16, 13), (16, 13, False)), ((16, 15), (16, 15, False)),
                                   ((2048, 13), (160, 13, False)), ((2048, 159), (160, 159, False)),
                                   ((np.int64(12), 1), (12, 11, False))):
            self.assertEqual(fields(es.research_davidson_options(n, nev)), expected, f"n = {n}, nev = {nev}")
        self.assertEqual(fields(es.research_davidson_options(2048, 1, maxspace=40)), (40, 12, False))
        self.assertEqual(fields(es.research_davidson_options(2048, 1, maxspace=10)), (10, 9, False))
        self.assertEqual(fields(es.research_davidson_options(2048, 2, retain=3)), (160, 3, False))
        controlled = es.research_davidson_options(2048, 1, tol=1e-9, maxops=50, stall_energy_atol=1e-10)
        self.assertEqual((controlled.tol, controlled.maxops, controlled.stall_energy_atol,
                          controlled.stall_residual_limit), (1e-9, 50, 1e-10, None))
        self.assertEqual(es.research_davidson_options(2048, 1),
                         DavidsonOptions(maxspace=160, retain=12, refresh=False))
        for n, nev, kw in ((8, 8, {}), (16, 16, {}), (1, 1, {}), (2048, 13, dict(maxspace=13)),
                           (2048, 1, dict(retain=160)), (2048, 3, dict(retain=2)), (0, 1, {}), (8.0, 1, {})):
            with self.assertRaises(ValueError, msg=f"n = {n}, nev = {nev}, {kw}"):
                es.research_davidson_options(n, nev, **kw)
        # Julia's research configuration DavidsonOptions(maxspace=min(160,n), retain=12, refresh=false)
        # wherever Julia's check_davidson accepts it (nev <= 12, n >= 13); elsewhere retain is nev
        # (nev > 12) or maxspace - 1 (n <= 12).
        for n in (2, 8, 12, 13, 14, 128, 160, 161, 2048):
            for nev in (1, 2, 7, 12, 13, 40, 159):
                if nev >= min(160, n):
                    continue
                julia = DavidsonOptions(maxspace=min(160, n), retain=12, refresh=False)
                try:
                    check_davidson(julia, nev)
                    accepted = True
                except ValueError:
                    accepted = False
                self.assertEqual(accepted, nev <= 12 and n >= 13, f"n = {n}, nev = {nev}")
                options = es.research_davidson_options(n, nev)
                retain = 12 if accepted else nev if nev > 12 else options.maxspace - 1
                self.assertEqual(fields(options), (min(160, n), retain, False), f"n = {n}, nev = {nev}")
                if accepted:
                    self.assertEqual(options, julia)
        c = tc.make_coordinates(Z2)
        for nev in (1, 4):
            self.assertEqual(es.library_davidson_options(nev), es.check_levels_request(c, nev=nev))
            self.assertEqual(es.library_davidson_options(nev), DavidsonOptions(maxspace=max(8, 2 * nev + 2)))
        self.assertEqual(fields(es.library_davidson_options(13)), (28, None, True))
        self.assertEqual(fields(es.library_davidson_options(2, maxspace=20, retain=3, tol=1e-9)), (20, 3, True))

    def test_sector_dimension(self):
        """sector_dimension is the dimension the solver builds, with the same sector
        resolution and metric cutoff."""
        for state, request, n in ((Z2, {}, 8), (Z2, dict(sector="even"), 8), (Z2, dict(sector="all"), 16),
                                  (Z2, dict(metric_tol=DROP), 6), (Z2, dict(sector="all", metric_tol=DROP), 12),
                                  (Z2, dict(parity=None), 16), (PLAIN, {}, 9)):
            self.assertEqual(es.sector_dimension(state, **request), n, str(request))
            self.assertEqual(len(tc.make_coordinates(state, **request)), n)
        r = es.excitation_eigsolve(Z2, 0.5, PHI4, metric_tol=DROP, kinetic_nodes=NODES, validate=False,
                                   davidson_options=DavidsonOptions(maxops=1))
        self.assertEqual(r["dimension"], es.sector_dimension(Z2, metric_tol=DROP))
        with self.assertRaises(ValueError):
            es.sector_dimension(PLAIN, sector="odd")


class AcceptanceTest(Case):
    """The whole production pipeline on the stored Julia state."""

    @slow
    def test_d16_mass(self):
        """benchmark/excitation_mass_d16.jl: D = 16 phi4 g = 2 Z2 ground state, window 30,
        n_quad 512, atol 1e-13, rtol 1e-12, library defaults (odd sector, density gauge, 256
        kinetic nodes, 8 preconditioner nodes), refined = (n_quad 768, window 40, atol 1e-15,
        rtol 1e-14) with 512 kinetic nodes. The validated mass must equal the recorded Julia
        value 0.3523798021192911 within 1e-9 (the benchmark's gate) with S = sqrt(rho).
        Measured: difference 1.95e-11, 23 Davidson operations (21 expansions and the two
        refresh evaluations of the default maxspace = 8; the research run's 16 used a space
        that never restarts, see test_d16_research_schedule)."""
        s = tuned(states.load_jld2(FIXTURE), max_point=30.0, n_quad=512, atol=1e-13, rtol=1e-12)
        refined = IntegratorParameters(max_point=40.0, n_quad=768, atol=1e-15, rtol=1e-14)
        tick = time.time()
        r = es.excitation_eigsolve(s, 0.0, models.phi4_terms(2.0), refined=refined, refined_kinetic_nodes=512,
                                   gauge="density", scaling="density",
                                   log=lambda line: print(f"[{time.time() - tick:7.1f} s] {line}", flush=True))
        v = r["validation"][0]
        mass = r["validated_energies"][0]
        print(f"converged={r['converged']} accepted={r['accepted']} stop={r['stop_reason']} numops={r['numops']} "
              f"action_calls={r['action_calls']} solver_residual={r['residuals'][0]!r} seconds={time.time() - tick:.1f}")
        print(f"sector={r['sector']} dimension={r['dimension']} start={r['start']} "
              f"density_condition={r['density_condition']!r} preconditioner={r['preconditioner']}")
        print(f"validation={v}")
        print(f"timings={r['timings']}")
        print(f"solver={r['solver']}")
        print(f"validated mass = {mass!r}\nrecorded mass  = {RECORDED_MASS!r}\ndifference     = {mass - RECORDED_MASS!r}",
              flush=True)
        self.assertEqual((s.D, r["sector"], r["dimension"], r["gauge"], r["scaling"], r["start"]),
                         (16, "odd", 128, "density", "density", "field"))
        self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"], r["preconditioner"]["nodes"]), (256, 512, 8))
        self.assertTrue(r["converged"] and r["accepted"])
        self.assertLessEqual(abs(mass - RECORDED_MASS), 1e-9)
        self.assertLessEqual(r["numops"], 25)
        self.assertLess(v["energy_spread"], 1e-7)
        self.assertLess(max(v["standard_residual"], v["refined_residual"]), 1e-5)

    @slow
    def test_d16_research_schedule(self):
        """The research mass scan (refactor/research/mass_scan/run_mass.jl, masses.csv row
        D = 16, g = 2) used the same operator, field seed and 8-node preconditioner but a
        Davidson space that never restarts (maxspace = min(160, n), retain = 12): it took
        16 operations to a fresh residual 3.80e-8. The same schedule here must take at most
        17 (measured: 16, residual 3.76e-8; the library default, maxspace = 8 with fresh
        restarts, spends more: see test_d16_mass). No validation; the standard-setting
        energy is within 1e-9 of the recorded mass (measured 1.1e-10)."""
        s = tuned(states.load_jld2(FIXTURE), max_point=30.0, n_quad=512, atol=1e-13, rtol=1e-12)
        tick = time.time()
        r = es.excitation_eigsolve(s, 0.0, models.phi4_terms(2.0), validate=False,
                                   davidson_options=DavidsonOptions(maxspace=128, retain=12, refresh=False),
                                   log=lambda line: print(f"[{time.time() - tick:7.1f} s] {line}", flush=True))
        print(f"converged={r['converged']} stop={r['stop_reason']} numops={r['numops']} "
              f"energy={r['energies'][0]!r} residual={r['residuals'][0]!r} "
              f"difference={r['energies'][0] - RECORDED_MASS!r} seconds={time.time() - tick:.1f}", flush=True)
        self.assertTrue(r["converged"] and r["accepted"])
        self.assertTrue(all(row["cycle"] == 1 for row in r["history"]))
        self.assertLessEqual(r["numops"], 17)
        self.assertLessEqual(abs(r["energies"][0] - RECORDED_MASS), 1e-9)


if __name__ == "__main__":
    unittest.main()
