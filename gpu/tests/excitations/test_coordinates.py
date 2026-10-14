"""White tangent coordinates on the CPU, D <= 8.

Ports the coordinate assertions of the testset "Parity sectors and white coordinates"
of test/excitation_eigsolve.jl (same tolerances), with states.random_state(z2=True) in
place of RandomStateZ2, and adds index-order, duality and gauge-equivalence checks.
"""
import math
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_array_equal

from rcmps_gpu.excitations.coordinates import (TangentCoordinates, coordinate_tangent, covector_coordinates,
                         make_coordinates, original_tangent, tangent_basis,
                         tangent_coordinates, tangent_matrix)
from rcmps_gpu.excitations.states import (IntegratorParameters, SolverFailure, adj, excitation_parity, lindblad,
                    lindblad_adjoint, make_state, parity_error, random_state, transfer_matrix,
                    unvec, vec)

GAUGES = ("density", "original")
Z2_STATES = ((4, 11), (6, 21), (8, 5))  # (D, seed)


def norm(x):
    return float(np.linalg.norm(x))


def metric(s, u, w):
    """tangent_metric(s, u, w) = tr(w rho u') of src/excitation_environments.jl."""
    return np.trace(w @ s.rho @ adj(u))


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def z2_test_state(d=4, seed=11):
    """z2_test_state() of test/excitation_eigsolve.jl, with NumPy draws."""
    s = random_state(d, scaling=.5, seed=seed, z2=True)
    s.integrator = IntegratorParameters(n_quad=48, max_point=14., atol=1e-11, rtol=1e-10)
    s.model, s.coupling, s.e0 = "phi4", 1., -.25
    return s


def julia_indices(d, columns, sector, z):
    """Julia's [CartesianIndex(i,j) for j in columns for i in 1:d if allowed(i,j)], with
    one-based loops, returned as zero-based (row, column) pairs."""
    def allowed(i, j):
        return sector == "all" or (sector == "odd") == (z[i - 1] == z[j - 1])
    return [(i - 1, j - 1) for j in range(1, d + 1) if j - 1 in columns
            for i in range(1, d + 1) if allowed(i, j)]


def kept_columns(c):
    return [j for j in range(c.state.D) if np.any(c.whitening[:, j] != 0)]


def dark_state(g=None):
    r = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    return make_state(r if g is None else g.T @ r @ g, np.zeros((2, 2)))


def rotation2(seed):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((2, 2)))[0]


class ParityAndWhiteCoordinates(unittest.TestCase):
    def test_parity_detection(self):
        s = z2_test_state()
        assert_array_equal(excitation_parity(s), [1, 1, -1, -1])
        self.assertIsNone(excitation_parity(random_state(4, seed=3)))
        for z in ([1, -1, 1, -1], [1, 1, 1, 1]):
            with self.assertRaises(ValueError):
                excitation_parity(s, z)

    def test_sectors_and_round_trips(self):
        # Julia loop over gauge and sector at D=4; the same assertions at D=6, 8.
        for d, seed in Z2_STATES:
            s = z2_test_state(d, seed)
            z = excitation_parity(s)
            Z = np.diag(z)
            for gauge in GAUGES:
                for sector, dimension in (("odd", d * d // 2), ("even", d * d // 2), ("all", d * d)):
                    with self.subTest(D=d, gauge=gauge, sector=sector):
                        c = make_coordinates(s, sector=sector, gauge=gauge)
                        self.assertIsInstance(c, TangentCoordinates)
                        self.assertEqual((len(c), c.sector, c.gauge), (dimension, sector, gauge))
                        self.assertLess(c.metric_defect, 1e-12)
                        x, y = gaussian(2, dimension), gaussian(3, dimension)
                        wx, wy = tangent_matrix(c, x), tangent_matrix(c, y)
                        self.assertLessEqual(abs(metric(c.state, wy, wx) - np.vdot(y, x)), 1e-12)
                        self.assertLessEqual(norm(tangent_coordinates(c, wx) - x), 1e-12)
                        wo = original_tangent(c, wx)
                        self.assertLessEqual(
                            abs(metric(s, original_tangent(c, wy), wo) - np.vdot(y, x)), 1e-12)
                        if sector == "odd":
                            self.assertLess(norm(Z @ wo @ Z - wo), 1e-14)
                        if sector == "even":
                            self.assertLess(norm(Z @ wo @ Z + wo), 1e-14)
                        self.assertLessEqual(norm(coordinate_tangent(c, wo) - wx), 1e-12)
                        # tr(W rho W') = |x|^2 and the reported diagnostics.
                        self.assertLessEqual(abs(metric(c.state, wx, wx) - norm(x) ** 2), 1e-12)
                        lam = np.linalg.eigvalsh(s.rho)
                        self.assertEqual(c.metric_rank, d)
                        self.assertEqual(c.discarded_metric_weight, 0.)
                        self.assertAlmostEqual(c.density_condition / (lam[-1] / lam[0]), 1., delta=1e-12)
                        assert_array_equal(c.parity, z)
                        self.assertEqual(c.parity_error, parity_error(s, z))
                        self.assertLessEqual(c.parity_error, 1e-12)

    def test_defaults_and_argument_errors(self):
        s, generic = z2_test_state(), random_state(3, seed=4)
        self.assertEqual(make_coordinates(s).sector, "odd")
        self.assertEqual(make_coordinates(s).gauge, "density")
        c = make_coordinates(generic)
        self.assertEqual((c.sector, len(c)), ("all", 9))
        self.assertTrue(c.parity is None and math.isnan(c.parity_error))
        c = make_coordinates(s, parity=None)
        self.assertTrue(c.sector == "all" and c.parity is None and math.isnan(c.parity_error))
        for state, kw in ((generic, dict(sector="odd")), (generic, dict(sector="even")),
                          (s, dict(gauge="invalid")), (s, dict(metric_tol=1.)),
                          (s, dict(metric_tol=-1e-3)), (s, dict(metric_tol=math.nan)),
                          (s, dict(metric_tol=math.inf)), (s, dict(sector="invalid")),
                          (s, dict(sector="odd", parity=None)), (s, dict(parity="invalid")),
                          (s, dict(parity=[1, -1, 1, -1])), (s, dict(parity=[1, 1, 1, 1])),
                          (s, dict(parity=[1, 1, -1]))):
            with self.subTest(**kw), self.assertRaises(ValueError):
                make_coordinates(state, **kw)
        c = make_coordinates(s)
        # Also lengths that NumPy would silently broadcast into M.
        for x in (np.zeros(len(c) + 1), np.zeros(len(c) - 1), np.zeros((len(c), 1)), np.zeros(1), 1.):
            with self.assertRaises(ValueError):
                tangent_matrix(c, x)

    def test_dark_state(self):
        # Rank-deficient density: unresolved directions are dropped, never regularized.
        dark = dark_state()
        for gauge in GAUGES:
            c = make_coordinates(dark, sector="all", gauge=gauge, metric_tol=1e-12)
            self.assertEqual((c.metric_rank, len(c)), (1, 2))
            # R = |0><1| is Z2 odd with z = (1, -1), which parity='auto' finds; rho = |0><0|.
            assert_array_equal(c.parity, [1, -1])
            assert_array_equal((c.rows, c.cols), ([0, 1], [0, 0]))
            self.assertLessEqual(norm(c.whitening - np.diag([1., 0.])), 1e-14)
            self.assertTrue(np.all(c.whitening[:, 1] == 0) and np.all(c.whitening[1, :] == 0))
            self.assertLess(c.discarded_metric_weight, 1e-14)
            self.assertEqual(c.density_condition, 1.)
            self.assertEqual(len(make_coordinates(dark, gauge=gauge, metric_tol=1e-12)), 1)

    def check_rank_one(self, s, seed):
        both = []
        for gauge in GAUGES:
            with self.subTest(seed=seed, gauge=gauge):
                c = make_coordinates(s, sector="all", gauge=gauge)
                self.assertEqual((c.metric_rank, len(c)), (1, 2))
                x = gaussian(seed, 2)
                w = tangent_matrix(c, x)
                self.assertLessEqual(abs(metric(c.state, w, w).real - norm(x) ** 2), 1e-12)
                self.assertLessEqual(norm(tangent_coordinates(c, w) - x), 1e-12)
                self.assertIsNone(c.parity)
                self.assertLess(c.metric_defect, 1e-12)
                self.assertAlmostEqual(c.density_condition, 1., delta=1e-14)
                columns = kept_columns(c)
                self.assertEqual(len(columns), 1)
                self.assertEqual(list(zip(c.rows.tolist(), c.cols.tolist())),
                                 julia_indices(2, columns, "all", None))
                both.append(c)
        self.assert_same_tangent_space(*both)
        return both

    def test_rotated_dark_states(self):
        # Rotated dark states have a roundoff eigenvalue of either sign. The Float64 floor
        # drops it even at the default metric_tol=0, in both gauges.
        for seed in range(1, 7):
            for c in self.check_rank_one(dark_state(rotation2(seed)), seed):
                self.assertTrue(0 <= c.discarded_metric_weight < 1e-14)
        # The same with an injected eigenvalue of known sign below the cutoff 20 eps:
        # the discarded weight is sum(abs, lambda[.!keep]).
        g = rotation2(1)
        base = dark_state(g)
        for value in (2e-15, -2e-15):
            rho = g.T @ np.diag([1., value]) @ g
            s = replace(base, rho=(rho + rho.T) / (2 * np.trace(rho)))
            self.assertEqual(np.sign(np.linalg.eigvalsh(s.rho)[0]), np.sign(value))
            for c in self.check_rank_one(s, 7):
                self.assertAlmostEqual(c.discarded_metric_weight, 2e-15, delta=5e-16)
        # The floor itself, 10 D eps max(lambda) = 4.44e-15 at D = 2
        # (src/excitation_eigsolve.jl:86-88): 4e-15 is dropped (10 eps alone would keep it) and
        # 5e-15 is kept, with Cholesky whitening and density_condition = lambda_max/lambda_min.
        for value, rank in ((4e-15, 1), (5e-15, 2)):
            s = replace(dark_state(), rho=np.diag([1., value]) / (1 + value))
            for gauge in GAUGES:
                with self.subTest(value=value, gauge=gauge):
                    c = make_coordinates(s, sector="all", parity=None, gauge=gauge)
                    self.assertEqual((c.metric_rank, len(c)), (rank, 2 * rank))
                    self.assertLess(c.metric_defect, 1e-12)
                    if rank == 1:
                        self.assertAlmostEqual(c.discarded_metric_weight, value, delta=1e-17)
                        self.assertAlmostEqual(c.density_condition, 1., delta=1e-14)
                    else:
                        self.assertEqual(c.discarded_metric_weight, 0.)
                        self.assertAlmostEqual(c.density_condition * value, 1., delta=1e-12)

    def test_parity_tolerance(self):
        # parity_tol is the tolerance of excitation_parity (src/excitation_eigsolve.jl:73), for
        # the detection and for a supplied parity: a Z2 state whose K breaks the symmetry at
        # 1e-9 has no parity at the default 1e-12.
        base = z2_test_state()
        k = base.K.copy()
        k[0, 2] += 1e-9
        k[2, 0] += 1e-9
        s, z = make_state(base.R, k), np.array([1, 1, -1, -1])
        error = parity_error(s, z)
        self.assertTrue(1e-9 < error < 1e-8)
        self.assertIsNone(excitation_parity(s))
        for gauge in GAUGES:
            with self.subTest(gauge=gauge):
                c = make_coordinates(s, gauge=gauge)
                self.assertTrue(c.sector == "all" and c.parity is None and len(c) == 16)
                with self.assertRaises(ValueError):
                    make_coordinates(s, gauge=gauge, parity=z)
                for parity in ("auto", z):
                    c = make_coordinates(s, gauge=gauge, parity=parity, parity_tol=1e-6)
                    self.assertEqual((c.sector, len(c)), ("odd", 8))
                    assert_array_equal(c.parity, z)
                    self.assertEqual(c.parity_error, error)
                    self.assertLess(c.metric_defect, 1e-7)

    def assert_same_tangent_space(self, cd, co):
        """The density-gauge basis, mapped back, is a unitary recombination of the
        original-gauge basis: same sector subspace, same metric."""
        t = np.stack([tangent_coordinates(co, original_tangent(cd, w)) for w in tangent_basis(cd)], axis=1)
        self.assertEqual(t.shape, (len(co), len(cd)))
        self.assertLessEqual(norm(adj(t) @ t - np.eye(len(cd))), 1e-12)
        self.assertLessEqual(norm(t @ adj(t) - np.eye(len(co))), 1e-12)

    def test_density_versus_original_gauge(self):
        for d, seed in Z2_STATES:
            s = z2_test_state(d, seed)
            for sector in ("odd", "even", "all"):
                with self.subTest(D=d, sector=sector):
                    cd, co = (make_coordinates(s, sector=sector, gauge=gauge) for gauge in GAUGES)
                    self.assert_same_tangent_space(cd, co)
        s = random_state(5, scaling=.5, seed=7)
        self.assert_same_tangent_space(*(make_coordinates(s, gauge=gauge) for gauge in GAUGES))

    def test_index_enumeration(self):
        # Explicit zero-based images of Julia's one-based comprehension at D=4, z=(1,1,-1,-1).
        s = z2_test_state()
        expected = {"odd": ([0, 1, 0, 1, 2, 3, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]),
                    "even": ([2, 3, 2, 3, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2, 3, 3]),
                    "all": (np.tile(np.arange(4), 4), np.repeat(np.arange(4), 4))}
        for gauge in GAUGES:
            for sector, (rows, cols) in expected.items():
                c = make_coordinates(s, sector=sector, gauge=gauge)
                assert_array_equal(c.rows, rows)
                assert_array_equal(c.cols, cols)
                self.assertTrue(np.issubdtype(c.rows.dtype, np.integer)
                                and np.issubdtype(c.cols.dtype, np.integer))
        for d, seed in Z2_STATES:
            s = z2_test_state(d, seed)
            z = excitation_parity(s)
            for gauge in GAUGES:
                for sector in ("odd", "even", "all"):
                    c = make_coordinates(s, sector=sector, gauge=gauge)
                    self.assertEqual(kept_columns(c), list(range(d)))
                    pairs = list(zip(c.rows.tolist(), c.cols.tolist()))
                    self.assertEqual(pairs, julia_indices(d, range(d), sector, z))
                    self.assertEqual(len(set(pairs)), len(c))
                    # x fills M in this order: M[rows[k], cols[k]] = x[k], W = M B'.
                    x = gaussian(4, len(c))
                    m = np.zeros((d, d), dtype=np.complex128)
                    for k, (i, j) in enumerate(pairs):
                        m[i, j] = x[k]
                    assert_array_equal(tangent_matrix(c, x), m @ adj(c.whitening))
                    if sector == "all":  # column-major, as states.vec
                        f = gaussian(5, d, d)
                        b, rho = c.whitening, c.state.rho
                        self.assertLessEqual(norm(tangent_matrix(c, x) - unvec(x, d) @ adj(b)), 1e-13)
                        self.assertLessEqual(norm(covector_coordinates(c, f) - vec(f @ b)), 1e-13)
                        self.assertLessEqual(norm(tangent_coordinates(c, f) - vec(f @ rho @ b)), 1e-13)

    def test_supplied_parity(self):
        # Interleaved blocks z = (1,-1,1,-1): the Z2 test state with indices 1 and 2 swapped.
        base = z2_test_state()
        p = [0, 2, 1, 3]
        interleaved = make_state(base.R[np.ix_(p, p)], base.K[np.ix_(p, p)])
        # Unbalanced blocks z = (1,1,-1).
        r, k = .5 * gaussian(8, 3, 3), .5 * gaussian(9, 3, 3)
        r[:2, :2] = r[2, 2] = k[:2, 2] = k[2, :2] = 0
        unbalanced = make_state(r, k + adj(k))
        for s, z, sizes in ((interleaved, np.array([1, -1, 1, -1]), dict(odd=8, even=8, all=16)),
                            (unbalanced, np.array([1, 1, -1]), dict(odd=5, even=4, all=9))):
            d = s.D
            Z = np.diag(z)
            offblock = z[:, None] != z[None, :]
            self.assertIsNone(excitation_parity(s))
            self.assertEqual(make_coordinates(s).sector, "all")
            self.assertEqual(make_coordinates(s, parity=z).sector, "odd")
            for gauge in GAUGES:
                for sector, dimension in sizes.items():
                    with self.subTest(z=z, gauge=gauge, sector=sector):
                        c = make_coordinates(s, sector=sector, parity=z, gauge=gauge)
                        self.assertEqual(len(c), dimension)
                        self.assertLess(c.metric_defect, 1e-12)
                        self.assertEqual(list(zip(c.rows.tolist(), c.cols.tolist())),
                                         julia_indices(d, range(d), sector, z))
                        # Column j of B, and of the rotation, lies in the parity block of j.
                        self.assertTrue(np.all(c.whitening[offblock] == 0))
                        self.assertTrue(np.all(c.rotation[offblock] == 0))
                        x = gaussian(6, dimension)
                        w = tangent_matrix(c, x)
                        self.assertLessEqual(abs(metric(c.state, w, w) - norm(x) ** 2), 1e-12)
                        self.assertLessEqual(norm(tangent_coordinates(c, w) - x), 1e-12)
                        wo = original_tangent(c, w)
                        sign = {"odd": -1, "even": 1}.get(sector)
                        if sign:
                            self.assertLess(norm(Z @ wo @ Z + sign * wo), 1e-14)
                        if gauge == "density":
                            lam = np.diag(c.state.rho).real
                            self.assertLessEqual(norm(c.state.rho - np.diag(lam)), 1e-13)
                            for value in (1, -1):  # ascending within each parity block
                                self.assertTrue(np.all(np.diff(lam[z == value]) >= 0))
            # The odd and even projections resolve the identity of the full tangent space.
            co, ce = (make_coordinates(s, sector=sector, parity=z) for sector in ("odd", "even"))
            w = gaussian(7, d, d)
            parts = [tangent_matrix(c, tangent_coordinates(c, w)) for c in (co, ce)]
            self.assertLessEqual(norm(parts[0] + parts[1] - w), 1e-12)
            self.assertLessEqual(abs(metric(co.state, parts[0], parts[1])), 1e-12)

    def test_covector_duality(self):
        # vdot(x, covector_coordinates(F)) == vdot(tangent_matrix(x), F) = tr(W' F).
        cases = [(z2_test_state(d, seed), dict(sector=sector))
                 for d, seed in Z2_STATES for sector in ("odd", "even", "all")]
        cases += [(random_state(5, scaling=.5, seed=7), {}), (dark_state(rotation2(3)), {})]
        for s, kw in cases:
            for gauge in GAUGES:
                with self.subTest(D=s.D, gauge=gauge, **kw):
                    c = make_coordinates(s, gauge=gauge, **kw)
                    x, f = gaussian(10, len(c)), gaussian(11, s.D, s.D)
                    w = tangent_matrix(c, x)
                    y = covector_coordinates(c, f)
                    self.assertEqual(y.shape, (len(c),))
                    self.assertLessEqual(abs(np.vdot(x, y) - np.vdot(w, f)), 1e-12)
                    self.assertLessEqual(abs(np.vdot(x, y) - np.trace(adj(w) @ f)), 1e-12)
                    # F = W rho is the covector of the metric: its coordinates are x.
                    self.assertLessEqual(norm(covector_coordinates(c, w @ c.state.rho) - x), 1e-12)

    def test_gauge_maps_are_inverse(self):
        for s, kw in ((z2_test_state(), dict(sector="all")), (z2_test_state(8, 5), {}),
                      (random_state(5, scaling=.5, seed=7), {})):
            d = s.D
            w = gaussian(12, d, d)  # any matrix, not only a sector tangent
            c = make_coordinates(s, gauge="density", **kw)
            u = c.rotation
            self.assertLessEqual(norm(adj(u) @ u - np.eye(d)), 1e-13)
            self.assertLessEqual(norm(coordinate_tangent(c, original_tangent(c, w)) - w), 1e-12)
            self.assertLessEqual(norm(original_tangent(c, coordinate_tangent(c, w)) - w), 1e-12)
            assert_array_equal(original_tangent(c, w), u @ w @ adj(u))
            assert_array_equal(coordinate_tangent(c, w), adj(u) @ w @ u)
            # The metric is gauge covariant: tr(W rho U') is the same in both gauges.
            v = gaussian(13, d, d)
            self.assertLessEqual(abs(metric(c.state, coordinate_tangent(c, v), coordinate_tangent(c, w))
                                     - metric(s, v, w)), 1e-12)
            c = make_coordinates(s, gauge="original", **kw)
            assert_array_equal(c.rotation, np.eye(d))
            assert_array_equal(original_tangent(c, w), w)
            assert_array_equal(coordinate_tangent(c, w), w)

    def test_density_gauge_state(self):
        for s in (z2_test_state(), z2_test_state(6, 21), z2_test_state(8, 5),
                  random_state(5, scaling=.5, seed=7)):
            s.integrator = IntegratorParameters(n_quad=48, max_point=14., atol=1e-11, rtol=1e-10)
            s.model, s.coupling, s.e0, s.source = "phi4", 2.5, -.75, "somewhere"
            saved = [x.copy() for x in (s.R, s.K, s.Q, s.rho)]
            d = s.D
            with self.subTest(D=d):
                c = make_coordinates(s)
                t, u = c.state, c.rotation
                # Unitary gauge change of R, K, Q and rho.
                self.assertLessEqual(norm(adj(u) @ u - np.eye(d)), 1e-13)
                for name in ("R", "K", "Q", "rho"):
                    self.assertLessEqual(norm(getattr(t, name) - adj(u) @ getattr(s, name) @ u), 1e-13, name)
                # rho is diagonal, ascending within each parity block, with the same spectrum.
                lam = np.diag(t.rho).real
                self.assertLessEqual(norm(t.rho - np.diag(lam)), 1e-13)
                blocks = [np.arange(d)] if c.parity is None else [c.parity == 1, c.parity == -1]
                for blk in blocks:
                    self.assertTrue(np.all(np.diff(lam[blk]) > 0))
                self.assertLessEqual(norm(np.sort(lam) - np.linalg.eigvalsh(s.rho)), 1e-13)
                self.assertLessEqual(norm(np.linalg.eigvalsh(t.rho) - np.linalg.eigvalsh(s.rho)), 1e-13)
                self.assertLessEqual(abs(np.trace(t.rho) - 1), 1e-13)
                assert_array_equal(t.rho, adj(t.rho))
                # Still a left-canonical stationary state, with the same parity.
                self.assertLessEqual(norm(lindblad_adjoint(t.Q, t.R, np.eye(d))), 1e-13)
                residual = norm(lindblad(t.Q, t.R, t.rho))
                self.assertLessEqual(residual, 1e-12)
                self.assertLessEqual(abs(residual - norm(lindblad(s.Q, s.R, s.rho))), 1e-13)
                if c.parity is not None:
                    self.assertLessEqual(parity_error(t, c.parity), 1e-13)
                    assert_array_equal(excitation_parity(t), c.parity)
                # Lindblad invariants: l_t = S l_s S' with the unitary S = u^T (x) u'
                # (column-major vec), hence equal traces of powers and equal spectra.
                ls, lt = transfer_matrix(s.Q, s.R), transfer_matrix(t.Q, t.R)
                similarity = np.kron(u.T, adj(u))
                self.assertLessEqual(norm(lt - similarity @ ls @ adj(similarity)), 1e-12)
                ps, pt = np.eye(d * d), np.eye(d * d)
                for power in range(1, 5):
                    ps, pt = ps @ ls, pt @ lt
                    self.assertLessEqual(abs(np.trace(pt) - np.trace(ps)),
                                         1e-11 * max(1, abs(np.trace(ps))), power)
                es, et = np.linalg.eigvals(ls), np.linalg.eigvals(lt)
                distance = abs(es[:, None] - et[None, :])
                self.assertLessEqual(max(distance.min(axis=0).max(), distance.min(axis=1).max()), 1e-10)
                self.assertLessEqual(abs(es).min(), 1e-12)
                self.assertLessEqual(abs(et).min(), 1e-12)
                # Gauge-invariant expectation values.
                for a in (lambda q: adj(q.R) @ q.R, lambda q: q.R + adj(q.R), lambda q: q.K,
                          lambda q: adj(q.R) @ adj(q.R) @ q.R @ q.R):
                    self.assertLessEqual(abs(np.trace(t.rho @ a(t)) - np.trace(s.rho @ a(s))), 1e-13)
                # Parameters are copied (copy_param! and e0), the input state is untouched.
                self.assertEqual((t.integrator, t.model, t.coupling, t.e0, t.source),
                                 (s.integrator, "phi4", 2.5, -.75, "somewhere"))
                self.assertIsNot(t.integrator, s.integrator)
                self.assertIsNot(t, s)
                for x, y in zip((s.R, s.K, s.Q, s.rho), saved):
                    assert_array_equal(x, y)
                self.assertIs(make_coordinates(s, gauge="original").state, s)

    def test_whitening_structure(self):
        for s in (z2_test_state(), z2_test_state(8, 5), random_state(5, scaling=.5, seed=7)):
            d = s.D
            z = excitation_parity(s)
            blocks = [np.arange(d)] if z is None else [np.flatnonzero(z == 1), np.flatnonzero(z == -1)]
            offblock = np.zeros((d, d), dtype=bool) if z is None else z[:, None] != z[None, :]
            for gauge in GAUGES:
                with self.subTest(D=d, gauge=gauge):
                    c = make_coordinates(s, gauge=gauge)
                    b, rho = c.whitening, c.state.rho
                    self.assertLess(c.metric_defect, 1e-12)
                    self.assertAlmostEqual(c.metric_defect, norm(adj(b) @ rho @ b - np.eye(d)), delta=1e-15)
                    self.assertTrue(np.all(b[offblock] == 0) and np.all(c.rotation[offblock] == 0))
                    # Cholesky whitening: B = inv(U) on each block, rho = U'U, U upper
                    # triangular with a positive diagonal.
                    assert_array_equal(np.tril(b, -1), 0)
                    self.assertTrue(np.all(np.diag(b).real > 0))
                    self.assertLessEqual(np.abs(np.diag(b).imag).max(), 1e-14)
                    for blk in blocks:
                        ix = np.ix_(blk, blk)
                        factor = np.linalg.inv(b[ix])
                        self.assertLessEqual(norm(adj(factor) @ factor - rho[ix]), 1e-13)
                        self.assertLessEqual(norm(factor - np.linalg.cholesky(rho[ix], upper=True)), 1e-13)
                    if gauge == "density":  # diagonal rho: B = rho^(-1/2) up to roundoff
                        exact = np.diag(1 / np.sqrt(np.diag(rho).real))
                        self.assertLessEqual(norm(b - exact), 1e-12 * norm(exact))

    def test_metric_tol_drops_directions(self):
        s = z2_test_state()
        d, z = s.D, excitation_parity(s)
        lam = np.linalg.eigvalsh(s.rho)
        # Between the two smallest eigenvalues: exactly one direction, in one block, is dropped.
        tol = math.sqrt(lam[0] * lam[1]) / lam[-1]
        blocks = [np.flatnonzero(z == 1), np.flatnonzero(z == -1)]
        spectra = [np.linalg.eigvalsh(s.rho[np.ix_(blk, blk)]) for blk in blocks]
        full = int(np.argmax([values[0] for values in spectra]))  # the block that loses nothing
        for gauge in GAUGES:
            for sector, dimension in (("odd", 6), ("even", 6), ("all", 12)):
                with self.subTest(gauge=gauge, sector=sector):
                    c = make_coordinates(s, sector=sector, gauge=gauge, metric_tol=tol)
                    b = c.whitening
                    self.assertEqual((c.metric_rank, len(c)), (3, dimension))
                    self.assertAlmostEqual(c.discarded_metric_weight, lam[0], delta=1e-14)
                    self.assertAlmostEqual(c.density_condition / (lam[-1] / lam[1]), 1., delta=1e-12)
                    self.assertLess(c.metric_defect, 1e-12)
                    columns = kept_columns(c)
                    self.assertEqual(len(columns), 3)
                    self.assertEqual(list(zip(c.rows.tolist(), c.cols.tolist())),
                                     julia_indices(d, columns, sector, z))
                    self.assertAlmostEqual(
                        c.metric_defect,
                        norm(adj(b) @ c.state.rho @ b - np.diag(np.isin(np.arange(d), columns) * 1.)),
                        delta=1e-15)
                    self.assertTrue(np.all(b[z[:, None] != z[None, :]] == 0))
                    # Eigen-whitening replaces Cholesky in EVERY block once a direction is
                    # dropped: B'B = diag(mask/lambda), also on the block that keeps all.
                    gram = adj(b) @ b
                    self.assertLessEqual(norm(gram - np.diag(np.diag(gram))), 1e-12 * norm(gram))
                    kept = np.sort(1 / np.diag(gram).real[columns])
                    self.assertLessEqual(norm(kept - lam[1:]), 1e-13)
                    ix = np.ix_(blocks[full], blocks[full])
                    self.assertLessEqual(norm(np.sort(1 / np.diag(gram[ix]).real) - spectra[full]), 1e-13)
                    if gauge == "original":  # not a triangular (Cholesky) factor
                        self.assertGreater(norm(np.tril(b[ix], -1)), 1e-3)
                    x, y = gaussian(2, dimension), gaussian(3, dimension)
                    wx, wy = tangent_matrix(c, x), tangent_matrix(c, y)
                    self.assertLessEqual(abs(metric(c.state, wy, wx) - np.vdot(y, x)), 1e-12)
                    self.assertLessEqual(norm(tangent_coordinates(c, wx) - x), 1e-12)
                    wo = original_tangent(c, wx)
                    sign = {"odd": -1, "even": 1}.get(sector)
                    if sign:
                        self.assertLess(norm(np.diag(z) @ wo @ np.diag(z) + sign * wo), 1e-14)
            self.assert_same_tangent_space(*(make_coordinates(s, sector="all", gauge=g, metric_tol=tol)
                                             for g in GAUGES))
        # A cutoff between the two largest eigenvalues keeps one direction; tol -> 1 keeps the top.
        for tol, rank in ((math.sqrt(lam[-2] * lam[-1]) / lam[-1], 1), (1 - 1e-9, 1),
                          (math.sqrt(lam[1] * lam[2]) / lam[-1], 2)):
            c = make_coordinates(s, sector="all", metric_tol=tol)
            self.assertEqual((c.metric_rank, len(c)), (rank, rank * d))
            self.assertAlmostEqual(c.discarded_metric_weight, lam[:d - rank].sum(), delta=1e-14)
            self.assertLess(c.metric_defect, 1e-12)

    def test_basis_and_projection(self):
        cases = [(z2_test_state(d, seed), dict(sector=sector))
                 for d, seed in Z2_STATES[:2] for sector in ("odd", "even", "all")]
        cases += [(random_state(5, scaling=.5, seed=7), {}), (dark_state(rotation2(2)), {})]
        for s, kw in cases:
            for gauge in GAUGES:
                with self.subTest(D=s.D, gauge=gauge, **kw):
                    c = make_coordinates(s, gauge=gauge, **kw)
                    n, d = len(c), s.D
                    basis = tangent_basis(c)
                    self.assertEqual(len(basis), n)
                    for k, w in enumerate(basis):
                        assert_array_equal(w, tangent_matrix(c, np.arange(n) == k))
                    gram = np.array([[metric(c.state, u, w) for w in basis] for u in basis])
                    self.assertLessEqual(norm(gram - np.eye(n)), 1e-12)
                    # tangent_coordinates is the metric-orthogonal projection onto the basis,
                    # and W -> W(x(W)) is a projector (the identity for sector 'all' at full rank).
                    w = gaussian(14, d, d)
                    x = tangent_coordinates(c, w)
                    self.assertLessEqual(norm(x - np.array([metric(c.state, u, w) for u in basis])), 1e-12)
                    pw = tangent_matrix(c, x)
                    self.assertLessEqual(norm(tangent_coordinates(c, pw) - x), 1e-12)
                    self.assertLessEqual(norm(pw - sum(xk * u for xk, u in zip(x, basis))), 1e-12)
                    if n == d * d:
                        self.assertLessEqual(norm(pw - w), 1e-12)

    def test_vacuum(self):
        # The one-dimensional vacuum tangent space of the free-dispersion test.
        for gauge in GAUGES:
            c = make_coordinates(make_state(np.zeros((1, 1)), np.zeros((1, 1))), gauge=gauge)
            self.assertEqual((c.sector, len(c), c.metric_rank, c.density_condition), ("all", 1, 1, 1.))
            assert_array_equal((c.rows, c.cols), ([0], [0]))
            assert_array_equal(c.whitening, [[1.]])
            assert_array_equal(tangent_matrix(c, [2j]), [[2j]])

    def test_solver_failures(self):
        s = random_state(4, scaling=.5, seed=7)
        # No resolved eigenvalue (a density that could not come from make_state).
        for gauge in GAUGES:
            with self.assertRaisesRegex(SolverFailure, "empty tangent metric support"):
                make_coordinates(replace(s, rho=np.zeros_like(s.rho)), gauge=gauge)
        # Hermitian(rho) reads the upper triangle (Julia's default uplo): a perturbed lower
        # triangle leaves the whitening unchanged and is caught by the metric defect.
        c = make_coordinates(s, gauge="original")
        broken = replace(s, rho=s.rho + .05j * np.tril(np.ones((4, 4)), -1))
        with self.assertRaisesRegex(SolverFailure, "tangent whitening is unresolved"):
            make_coordinates(broken, gauge="original")
        loose = replace(s, rho=s.rho + 1e-8j * np.tril(np.ones((4, 4)), -1))
        d = make_coordinates(loose, gauge="original")
        assert_array_equal(d.whitening, c.whitening)
        self.assertTrue(1e-9 < d.metric_defect <= 1e-6)
        # The acceptance threshold is defect <= 1e-6 (src/excitation_eigsolve.jl:116-117); the
        # defect is linear in this perturbation (16.6 times its size).
        inside = make_coordinates(replace(s, rho=s.rho + 4e-8j * np.tril(np.ones((4, 4)), -1)), gauge="original")
        self.assertTrue(5e-7 < inside.metric_defect <= 1e-6)
        with self.assertRaisesRegex(SolverFailure, "tangent whitening is unresolved: metric defect 1.6[0-9]*e-06"):
            make_coordinates(replace(s, rho=s.rho + 1e-7j * np.tril(np.ones((4, 4)), -1)), gauge="original")
        # The same holds for eigen-whitening (one direction dropped by metric_tol).
        lam = np.linalg.eigvalsh(s.rho)
        tol = math.sqrt(lam[0] * lam[1]) / lam[-1]
        c, d = (make_coordinates(q, gauge="original", metric_tol=tol) for q in (s, loose))
        self.assertEqual((c.metric_rank, d.metric_rank, len(d)), (3, 3, 12))
        assert_array_equal(d.whitening, c.whitening)
        self.assertTrue(1e-9 < d.metric_defect <= 1e-6)


if __name__ == "__main__":
    unittest.main()
