"""Port of the testset "Restarted Hermitian Davidson" of test/excitation_eigsolve.jl.

Julia's MersenneTwister matrices are replaced by NumPy default_rng draws with the same
seeds, spectra and scales; every tolerance is the Julia one. `x ≈ y atol=a` on vectors
is norm(x - y) <= a, as Julia's isapprox with an explicit atol.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from rcmps_gpu.excitations.davidson import DavidsonOptions, check_davidson, davidson, davidson_stalled, orthogonalize, stall_limit
from rcmps_gpu.excitations.states import SolverFailure
from _paths import FIXTURES

FIXTURE = FIXTURES / "g2725_D48_first85.csv"
norm = np.linalg.norm


def gaussian(seed, *shape):
    """Complex normal with unit variance, as Julia's randn(ComplexF64, ...)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def unitary(seed, n):
    return np.linalg.qr(gaussian(seed, n, n))[0]


def hermitian(a):
    """Julia's Hermitian(a): the upper triangle and the real part of the diagonal."""
    upper = np.triu(a, 1)
    return upper + upper.conj().T + np.diag(a.diagonal().real)


def stallrows(energies, residuals):
    return [dict(energies=[e], residual=r) for e, r in zip(energies, residuals)]


class DavidsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = n = 40
        u = unitary(7, n)
        cls.values = values = np.r_[[.3, .5, .9], np.linspace(2, 30, n - 3)]
        cls.A = A = hermitian(u @ np.diag(values) @ u.conj().T)
        cls.x0 = gaussian(1, n)
        # Diagonal preconditioner from a perturbed operator, as for the coarse kinetic matrix.
        cls.P = hermitian(u @ np.diag(values * (1 + .2 * np.random.default_rng(2).random(n))) @ u.conj().T)
        cls.op = staticmethod(lambda x: A @ x)

    def test_fresh_and_thick_restarts(self):
        A, values = self.A, self.values
        for refresh, maxspace in ((True, 8), (False, 16)):
            for nev in (1, 3):
                for name, precondition in (("identity", None), ("P", lambda r: np.linalg.solve(self.P, r))):
                    with self.subTest(refresh=refresh, nev=nev, precondition=name):
                        o = DavidsonOptions(tol=1e-9, maxops=400, maxspace=maxspace,
                                            retain=nev if refresh else nev + 3, refresh=refresh, stall_window=0)
                        r = davidson(self.op, self.x0, nev=nev, precondition=precondition, options=o)
                        self.assertTrue(r["converged"] and r["stop_reason"] == "converged")
                        self.assertLessEqual(norm(np.array(r["energies"]) - values[:nev]), 1e-10)
                        self.assertEqual(len(r["vectors"]), nev)
                        for k in range(nev):
                            v = r["vectors"][k]
                            self.assertLessEqual(abs(norm(A @ v - r["energies"][k] * v) - r["residuals"][k]), 1e-12)
                            self.assertLessEqual(norm(r["actions"][k] - A @ v), 1e-12)
                        self.assertEqual(len(r["history"]), r["numops"])
                        # Not in Julia: a restart at dimension maxspace starts the next cycle, from
                        # dimension 1 (fresh rebuild) or retain+1 (thick restart).
                        dims, cycles = [1], [1]
                        for _ in range(r["numops"] - 1):
                            restart = dims[-1] == maxspace
                            dims.append((1 if refresh else o.retain + 1) if restart else dims[-1] + 1)
                            cycles.append(cycles[-1] + restart)
                        self.assertEqual([row["dimension"] for row in r["history"]], dims)
                        self.assertEqual([row["cycle"] for row in r["history"]], cycles)
                        if precondition is None:  # both restart kinds are exercised
                            self.assertGreater(cycles[-1], 1)

    def test_fresh_restart_recomputes_actions(self):
        # Fresh restarts recompute the retained action and record how much it changed.
        calls = [0]

        def op(x):
            calls[0] += 1
            return self.A @ x

        r = davidson(op, self.x0, options=DavidsonOptions(tol=1e-9, maxops=400, stall_window=0))
        self.assertEqual(calls[0], r["numops"])
        self.assertTrue(any(row["cycle"] > 1 for row in r["history"]))
        self.assertLess(max(row["refresh_change"] for row in r["history"]), 1e-12)
        # Not in Julia: with an operator that drifts by 1e-9 per call, exactly the `retain`
        # rebuilt rows of every later cycle record a change, of the size of the drift.
        calls[0] = 0

        def drifting(x):
            calls[0] += 1
            return self.A @ x + 1e-9 * calls[0] * x

        r = davidson(drifting, self.x0, options=DavidsonOptions(tol=1e-12, maxops=40, maxspace=6, retain=3,
                                                                stall_window=0))
        self.assertTrue(r["stop_reason"] == "maxops" and r["numops"] == calls[0] == 40)
        self.assertEqual([row["dimension"] for row in r["history"]][:13], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1])
        for row in r["history"]:
            if row["cycle"] > 1 and row["dimension"] <= 3:
                self.assertTrue(1e-10 < row["refresh_change"] < 1e-6, row)
            else:
                self.assertEqual(row["refresh_change"], 0.)

    def test_maxops_and_stalled_stops(self):
        limited = davidson(self.op, self.x0, options=DavidsonOptions(maxops=2, stall_window=0))
        self.assertTrue(limited["stop_reason"] == "maxops" and not limited["converged"])
        permissive = DavidsonOptions(tol=1e-14, maxops=60, maxspace=16, retain=4, refresh=False, stall_window=2,
                                     stall_minops=3, stall_rtol=.99, stall_residual_limit=100.,
                                     stall_energy_atol=100.)
        early = davidson(self.op, self.x0, options=permissive)
        self.assertTrue(early["stop_reason"] == "stalled" and not early["converged"])

    def test_failures_and_argument_checks(self):
        with self.assertRaisesRegex(SolverFailure, "not Hermitian"):
            davidson(lambda x: (1 + 1e-4j) * x, np.ones(3))
        with self.assertRaisesRegex(ValueError, "nev <= retain < maxspace"):
            davidson(lambda x: x, np.ones(3), options=DavidsonOptions(maxspace=1))
        with self.assertRaisesRegex(ValueError, "exceeds the dimension"):
            davidson(lambda x: x, np.ones(3), nev=4, options=DavidsonOptions(maxspace=8))
        with self.assertRaisesRegex(ValueError, "nonzero and independent"):
            davidson(lambda x: x, np.zeros(3))
        # At most maxspace-1 start vectors; loose targets need no explicit stall limit.
        with self.assertRaisesRegex(ValueError, "at most maxspace-1=3 start vectors"):
            davidson(self.op, [gaussian(k, self.n) for k in range(1, 5)], options=DavidsonOptions(maxspace=4))
        self.assertEqual(check_davidson(DavidsonOptions(tol=1e-4), 1), 1)
        self.assertEqual(check_davidson(DavidsonOptions(tol=1e-4, stall_window=0), 1), 1)
        self.assertEqual(stall_limit(DavidsonOptions(tol=1e-4)), 1e-3)
        with self.assertRaisesRegex(ValueError, "maxops must be at least nev"):
            davidson(self.op, self.x0, nev=3, options=DavidsonOptions(maxops=2))

    def test_breakdown_of_a_spanned_inexact_operator(self):
        # An inexact (slightly non-Hermitian) operator whose whole space is spanned returns
        # its Ritz pairs as 'breakdown' instead of throwing.
        H = hermitian(gaussian(8, 6, 6))
        S = gaussian(9, 6, 6)
        S = S - S.conj().T
        broken = davidson(lambda x: (H + 1.5e-7 * S) @ x, np.ones(6),
                          options=DavidsonOptions(tol=1e-12, hermitian_tol=1e-5, stall_window=0))
        self.assertTrue(broken["stop_reason"] == "breakdown" and not broken["converged"])
        self.assertLessEqual(abs(broken["energies"][0] - np.linalg.eigvalsh(H)[0]), 1e-5)

    def test_stall_rule_on_synthetic_histories(self):
        # Stall rule: synthetic histories (ported from the research solver tests).
        o = DavidsonOptions(stall_residual_limit=1e-6, stall_energy_atol=1e-10)
        k40, k80 = np.arange(1, 41), np.arange(1, 81)
        self.assertTrue(davidson_stalled(stallrows([.02] * 40, [3e-7] * 40), o))
        self.assertFalse(davidson_stalled(stallrows([.02] * 39, [3e-7] * 39), o))
        self.assertFalse(davidson_stalled(stallrows([.02] * 80, [1e-4] * 80), o))
        self.assertFalse(davidson_stalled(stallrows(.02 + 1e-8 * k40, [3e-7] * 40), o))
        self.assertFalse(davidson_stalled(stallrows([.02] * 80, 1e-6 * .97 ** k80), o))
        self.assertFalse(davidson_stalled(stallrows([.02] * 40, [3e-7] * 40), DavidsonOptions(stall_window=0)))
        self.assertFalse(davidson_stalled([], o))
        # Two roots after fresh restarts: rows rebuilding the space carry only the converged
        # first root and must not make a still-improving second root look stalled.
        mixed = []
        for k in range(1, 61):
            mixed.append(dict(energies=[.1], residual=1e-9))
            mixed.append(dict(energies=[.1, .3], residual=5e-7 * .9 ** k))
        self.assertFalse(any(davidson_stalled(mixed[:2 * k], o) for k in range(1, 61)))
        flat2 = [dict(energies=[.1, .3], residual=3e-7) for _ in range(40)]
        drift2 = [dict(energies=[.1, .3 + 1e-8 * k], residual=3e-7) for k in range(1, 41)]
        self.assertTrue(davidson_stalled(flat2, o) and not davidson_stalled(drift2, o))

    def test_maxops_never_truncates_the_targeted_roots(self):
        # A 'maxops' stop never truncates the targeted roots, even during a fresh rebuild.
        for maxops in range(2, 14):
            with self.subTest(maxops=maxops):
                r = davidson(self.op, self.x0, nev=2,
                             options=DavidsonOptions(tol=1e-14, maxspace=4, maxops=maxops, stall_window=0))
                self.assertTrue(len(r["energies"]) == 2 and r["numops"] <= maxops)
                # Not in Julia: a restart (every 4 calls) with one call left stops one call early.
                self.assertEqual(r["stop_reason"], "maxops")
                self.assertEqual(r["numops"], maxops - (maxops % 4 == 1))

    def test_small_start_and_callback_shapes(self):
        # Small start vectors are scaled, not rejected; the callback sees the documented shapes.
        shapes = []
        r = davidson(self.op, 1e-9 * self.x0, nev=2, options=DavidsonOptions(tol=1e-9, maxops=400, stall_window=0),
                     callback=lambda row, u, au: shapes.append((row["dimension"], u.shape, au.shape)))
        self.assertTrue(r["converged"])
        self.assertLessEqual(norm(np.array(r["energies"]) - self.values[:2]), 1e-10)
        self.assertEqual(len(shapes), r["numops"])
        self.assertTrue(all(q[0] <= 8 and q[1] == q[2] and q[1][0] == self.n and q[1][1] == min(2, q[0])
                            for q in shapes))

    def test_breakdown_only_when_no_unconverged_root_can_expand(self):
        # A floor-limited root 1 (anti-Hermitian part in its 3-dim block) must not stop root 2
        # from converging.
        U1, S1, U2 = unitary(11, 3), gaussian(12, 3, 3), unitary(13, 60)
        B1 = U1 @ np.diag([.1, 30., 40.]) @ U1.conj().T + 3e-6 * (S1 - S1.conj().T)
        B2 = U2 @ np.diag(np.r_[.5, np.linspace(1, 20, 59)]) @ U2.conj().T
        blockop = lambda x: np.r_[B1 @ x[:3], B2 @ x[3:]]
        # The floor-limited block is spanned from the start, so root 1 is present but cannot expand.
        starts = [np.r_[np.arange(3) == j, np.zeros(60)] for j in range(3)] + [np.r_[np.zeros(3), gaussian(15, 60)]]
        r = davidson(blockop, starts, nev=2,
                     options=DavidsonOptions(tol=1e-7, maxspace=80, retain=4, refresh=False, hermitian_tol=1e-5,
                                             stall_window=0, maxops=300))
        self.assertLessEqual(abs(r["energies"][0] - .1), 1e-5)
        self.assertLessEqual(abs(r["energies"][1] - .5), 1e-9)
        self.assertTrue(r["residuals"][0] > 1e-7 and r["residuals"][1] <= 1e-7 and r["stop_reason"] == "breakdown")

    def test_stall_rule_on_the_captured_d48_history(self):
        # The stall rule on a captured D=48 residual history stops at its plateau only.
        rows = [line.split(",") for line in FIXTURE.read_text().splitlines()[1:]]
        history = [dict(energies=[float(q[1])], residual=float(q[2])) for q in rows]
        self.assertEqual(len(history), 85)
        o = DavidsonOptions(stall_residual_limit=1e-6, stall_energy_atol=1e-10)
        # Julia's findfirst over history[1:k]: `first` is a one-based row count.
        first = next((k for k in range(1, len(history) + 1) if davidson_stalled(history[:k], o)), None)
        self.assertTrue(first is not None and 50 <= first <= 65, first)

    # The remaining tests have no Julia counterpart: they pin the Python contract.

    def test_result_and_history_contract(self):
        x0 = self.x0.copy()
        r = davidson(self.op, [x0.real.tolist(), x0], nev=2, options=DavidsonOptions(tol=1e-9, maxops=400))
        self.assertTrue(np.array_equal(x0, self.x0))
        self.assertEqual(set(r), {"energies", "vectors", "actions", "residuals", "converged", "stop_reason",
                                  "numops", "history"})
        self.assertTrue(type(r["converged"]) is bool and type(r["numops"]) is int and r["stop_reason"] == "converged")
        self.assertTrue(all(type(e) is float for e in r["energies"] + r["residuals"]))
        for k in range(2):
            self.assertTrue(r["vectors"][k].shape == r["actions"][k].shape == (self.n,))
            self.assertTrue(r["vectors"][k].dtype == r["actions"][k].dtype == np.complex128)
            self.assertLessEqual(abs(norm(r["vectors"][k]) - 1), 1e-14)
        keys = {"count", "cycle", "dimension", "energies", "residuals", "energy", "residual", "hermiticity",
                "orthogonality", "refresh_change"}
        for count, row in enumerate(r["history"], 1):
            self.assertEqual(set(row), keys)
            self.assertTrue(row["count"] == count and 1 <= row["dimension"] <= 8)
            self.assertTrue(row["energy"] == row["energies"][0] and row["residual"] == max(row["residuals"]))
            self.assertEqual(len(row["energies"]), min(2, row["dimension"]))
            self.assertLess(row["orthogonality"], 1e-12)
        self.assertEqual(r["history"][0]["cycle"], 1)
        self.assertEqual(r["history"][-1]["residuals"], r["residuals"])

    def test_expansion_direction_and_fallback(self):
        # The second search vector is -r/|r| for the first residual r (identity preconditioner).
        A, xs = self.A, []
        davidson(lambda x: xs.append(x.copy()) or A @ x, self.x0, options=DavidsonOptions(maxops=2, stall_window=0))
        x = self.x0 / norm(self.x0)
        r = A @ x - np.vdot(x, A @ x).real * x
        self.assertLessEqual(norm(xs[0] - x), 1e-14)
        self.assertLessEqual(norm(xs[1] + r / norm(r)), 1e-13)
        # A preconditioned residual inside the search space (below 1e-13) falls back to the
        # residual itself instead of declaring a breakdown.
        o = DavidsonOptions(tol=1e-9, maxops=400, maxspace=16, retain=4, refresh=False, stall_window=0)
        for precondition in (np.zeros_like, lambda r: self.x0 * np.vdot(self.x0, r)):
            r = davidson(self.op, self.x0, precondition=precondition, options=o)
            self.assertEqual(r["stop_reason"], "converged")
            self.assertLessEqual(abs(r["energies"][0] - self.values[0]), 1e-10)

    def test_partial_rows_never_stall(self):
        # nev=2 with an exact first root: the one-root rows that open each fresh rebuild
        # satisfy the stall rule on their own from the third cycle on, but a search only
        # stalls on a row that carries every targeted root.
        n, A = self.n, self.A
        o = DavidsonOptions(tol=1e-8, maxops=40, maxspace=4, retain=2, stall_window=2, stall_minops=3,
                            stall_residual_limit=100., stall_energy_atol=1e-13)
        r = davidson(lambda x: np.r_[.1 * x[:1], A @ x[1:]], [np.eye(n + 1)[0], np.r_[0, self.x0]], nev=2, options=o)
        history = r["history"]
        partial = [k for k in range(1, len(history) + 1)
                   if len(history[k - 1]["energies"]) < 2 and davidson_stalled(history[:k], o)]
        self.assertEqual(partial[:2], [9, 13])
        self.assertTrue(len(r["energies"]) == 2 and r["stop_reason"] == "maxops" and r["numops"] == 40)

    def test_orthogonalize_takes_two_passes(self):
        # One Gram-Schmidt pass leaves a relative overlap of order eps/1e-10 here.
        b = gaussian(3, 50)
        b /= norm(b)
        x = orthogonalize(b + 1e-10 * gaussian(4, 50), [b])
        self.assertLess(abs(np.vdot(b, x)) / norm(x), 1e-14)

    def test_stall_window_boundaries(self):
        # Julia slices (one based, n rows): old best over full[1:n-window], energies over
        # the window+1 rows full[n-window:n]. Here n=4 and window=2.
        o = DavidsonOptions(stall_window=2, stall_minops=3, stall_residual_limit=1., stall_energy_atol=1e-10)
        self.assertTrue(davidson_stalled(stallrows([1., 0., 0., 0.], [1e-3] * 4), o))
        self.assertFalse(davidson_stalled(stallrows([0., 1., 0., 0.], [1e-3] * 4), o))
        self.assertTrue(davidson_stalled(stallrows([0.] * 4, [1e-3, 1e-4, 1e-4, 1e-4]), o))
        self.assertFalse(davidson_stalled(stallrows([0.] * 4, [1e-3, 1e-3, 1e-4, 1e-4]), o))
        self.assertFalse(davidson_stalled(stallrows([0.] * 2, [1e-3] * 2), o))
        self.assertTrue(davidson_stalled(stallrows([0.] * 3, [1e-3] * 3), o))
        self.assertFalse(davidson_stalled(stallrows([0.] * 3, [1e-3, 1e-3, 2.]), o))

    def test_invariant_start_continues_with_a_deterministic_direction(self):
        # An exact eigenvector of a diagonal operator with nev=2: the residual is exactly zero
        # and m < nev, so the search continues along default_rng(count) (MersenneTwister(count)
        # in Julia), here count=1.
        values, n = self.values, self.n
        o = DavidsonOptions(tol=1e-9, maxops=400, stall_window=0)
        seen = []
        r = davidson(lambda x: values * x, np.eye(n)[0], nev=2, options=o,
                     callback=lambda row, u, au: seen.append(u))
        self.assertEqual(r["history"][0]["residual"], 0.)
        rng = np.random.default_rng(1)
        y = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
        y[0] = 0
        self.assertEqual(seen[1].shape, (n, 2))
        self.assertLessEqual(abs(norm(seen[1].conj().T @ y) / norm(y) - 1), 1e-12)
        self.assertTrue(r["converged"])
        self.assertLessEqual(norm(np.array(r["energies"]) - values[:2]), 1e-10)
        # A roundoff-level residual (below the 1e-13 fallback, above eps) is still a direction.
        r = davidson(self.op, np.linalg.eigh(self.A)[1][:, 0], nev=2, options=o)
        self.assertTrue(0 < r["history"][0]["residual"] < 1e-13 and r["converged"])
        self.assertLessEqual(norm(np.array(r["energies"]) - values[:2]), 1e-10)

    def test_invalid_actions_and_starts(self):
        with self.assertRaisesRegex(SolverFailure, "invalid operator action"):
            davidson(lambda x: np.full(3, np.nan), np.ones(3))
        with self.assertRaisesRegex(SolverFailure, "invalid operator action"):
            davidson(lambda x: np.ones(4), np.ones(3))
        with self.assertRaisesRegex(SolverFailure, "invalid preconditioned residual"):
            davidson(self.op, self.x0, precondition=lambda r: r[:-1])
        for bad in ([], np.zeros(0), [np.ones(3), np.ones(4)], np.eye(3), [np.inf, 1., 1.], [np.ones((2, 3))]):
            with self.assertRaises(ValueError):
                davidson(lambda x: x, bad)
        for o in (DavidsonOptions(tol=0.), DavidsonOptions(maxops=0), DavidsonOptions(hermitian_tol=math.inf),
                  DavidsonOptions(retain=8), DavidsonOptions(stall_window=-1), DavidsonOptions(stall_rtol=1.),
                  DavidsonOptions(tol=1e-3, stall_residual_limit=1e-4)):
            with self.assertRaises(ValueError):
                check_davidson(o, 1)
        for nev, o in ((0, DavidsonOptions()), (2, DavidsonOptions(maxspace=2))):
            with self.assertRaises(ValueError):
                check_davidson(o, nev)
        self.assertEqual(check_davidson(DavidsonOptions(retain=3), 2), 3)

    def inputs(self, starts, **options):
        """The vectors on which the operator is applied, in order."""
        xs = []
        davidson(lambda x: xs.append(x.copy()) or self.A @ x, starts,
                 options=DavidsonOptions(stall_window=0, **options))
        return xs

    def test_dependent_starts_are_dropped(self):
        # src/davidson.jl:113-119: a start x is kept when its part y orthogonal to the kept
        # starts has norm(y) > sqrt(eps()) * norm(x), the scale being that of x itself;
        # otherwise it is dropped, not rejected.
        x0, n, A = self.x0, self.n, self.A
        x = x0 / norm(x0)
        residual = A @ x - np.vdot(x, A @ x).real * x
        expansion = -residual / norm(residual)
        # Four dependent starts count as one: with maxspace=4 three kept starts are the limit.
        r = davidson(self.op, [x0, 2 * x0, -x0, 1j * x0], options=DavidsonOptions(maxspace=4, maxops=3, stall_window=0))
        self.assertEqual([row["dimension"] for row in r["history"]], [1, 2, 3])
        # The second search vector is then the expansion -r/|r|, not the roundoff of 2 x0.
        xs = self.inputs([x0, 2 * x0], maxops=2)
        self.assertLessEqual(norm(xs[1] - expansion), 1e-13)
        # The threshold itself, bracketed within a factor two on either side.
        u = orthogonalize(gaussian(5, n), [x])
        u /= norm(u)
        threshold = math.sqrt(np.finfo(float).eps) * norm(2 * x0)
        xs = self.inputs([x0, 2 * x0 + .5 * threshold * u], maxops=2)
        self.assertLessEqual(norm(xs[1] - expansion), 1e-13)
        xs = self.inputs([x0, 2 * x0 + 2 * threshold * u], maxops=2)
        self.assertLessEqual(norm(xs[1] - u), 1e-6)

    def test_stop_priority_on_the_last_call(self):
        # src/davidson.jl:162-164: converged, then stalled, then maxops. A search that
        # converges or stalls on its last allowed operator call is not a 'maxops' stop.
        def run(maxops, **options):
            r = davidson(self.op, self.x0, options=DavidsonOptions(maxops=maxops, **options))
            return r["stop_reason"], r["converged"], r["numops"]

        converging = dict(tol=1e-9, stall_window=0)
        n = run(400, **converging)[2]
        self.assertGreater(n, 8)
        self.assertEqual(run(n, **converging), ("converged", True, n))
        self.assertEqual(run(n + 1, **converging), ("converged", True, n))
        self.assertEqual(run(n - 1, **converging), ("maxops", False, n - 1))
        stalling = dict(tol=1e-14, maxspace=16, retain=4, refresh=False, stall_window=2, stall_minops=3,
                        stall_rtol=.99, stall_residual_limit=100., stall_energy_atol=100.)
        m = run(60, **stalling)[2]
        self.assertEqual(run(m, **stalling), ("stalled", False, m))
        self.assertEqual(run(m - 1, **stalling), ("maxops", False, m - 1))

    def test_stall_needs_more_rows_than_the_window(self):
        # src/davidson.jl:63: n >= max(stall_minops, window+1), here with stall_minops below
        # the window so that window+1 alone decides.
        o = DavidsonOptions(stall_window=2, stall_minops=1, stall_residual_limit=1.)
        self.assertFalse(davidson_stalled(stallrows([0.] * 2, [1e-3] * 2), o))
        self.assertTrue(davidson_stalled(stallrows([0.] * 3, [1e-3] * 3), o))

    def test_expansion_root_when_every_present_root_is_converged(self):
        # src/davidson.jl:190-191: with m < nev roots present and none above tol, the space is
        # expanded along the LAST present root (candidates = [m]). Diagonal operator, nev=3,
        # starts e0 + 1e-5 e5 and e1 + 1e-5 e7: residuals 3e-5 e5 and 5e-5 e7, both below tol,
        # so the third search vector is -e7 (residual of root 1), not -e5 (root 0).
        values, n, xs = self.values, self.n, []
        starts = [np.eye(n)[0] + 1e-5 * np.eye(n)[5], np.eye(n)[1] + 1e-5 * np.eye(n)[7]]
        r = davidson(lambda x: xs.append(x.copy()) or values * x, starts, nev=3,
                     options=DavidsonOptions(tol=1e-3, maxops=3, stall_window=0))
        self.assertEqual([len(row["energies"]) for row in r["history"]], [1, 2, 3])
        self.assertTrue(0 < max(r["history"][1]["residuals"]) < 1e-4)
        self.assertLessEqual(norm(xs[2] + np.eye(n)[7]), 1e-4)
        self.assertLessEqual(abs(xs[2][5]), 1e-9)

    def test_roundoff_residual_is_a_direction(self):
        # src/davidson.jl:185-188 with the start e0 + 1e-14 e5 of a diagonal operator and nev=2:
        # the residual is 3.3e-14 e5 exactly. Below 1e-13 the expansion is the residual itself
        # (+e5, not the preconditioned -e5), and above eps() it is a direction, not `nothing`
        # (which would continue along a random vector).
        values, n, xs = self.values, self.n, []
        r = davidson(lambda x: xs.append(x.copy()) or values * x, np.eye(n)[0] + 1e-14 * np.eye(n)[5], nev=2,
                     options=DavidsonOptions(maxops=2, stall_window=0))
        self.assertTrue(np.finfo(float).eps < r["history"][0]["residual"] < 1e-13)
        self.assertLessEqual(norm(xs[1] - np.eye(n)[5]), 1e-12)


if __name__ == "__main__":
    unittest.main()
