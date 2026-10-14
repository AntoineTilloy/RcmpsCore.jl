"""Tests of scaled.py: scaled variables, typed Lindblad maps and batched GMRES solves.

Every definition is rebuilt here from the state and S alone. With G_L = 1 and G_H = S,
a primal of type (a, b) is the physical X = G_a xh G_b and a dual (covector) of type
(a, b) is Y = G_a^-1 yh G_b^-1; Q_H = S^-1 Q S and R_H = S^-1 R S. Dense D^2 x D^2
references use numpy.kron on the column-major vec; at S = 1 the Poisson solves are also
compared with the oracle dense_reference.

Grid: states.random_state(D), D = 2..6, Z2 and not; S = 1 ("none"), sqrt(diag(rho))
("density") and exp(linspace(-2.5, 1.5, D)) ("explicit").

Batches     (1) lmul, rmul, stack, batch_norm against loops.
State       (2) generators in both gauges, eigenbases, the hat fixed point rh.
Maps        (3, 4) lmap against the definition, its physical form, duality, nojump_inverse.
ZeroModes   (5) zero_mode, pairing, project.
Resolvent   (6, 9) the four resolvent types of the algorithm, dense solves, scaling invariance.
Poisson     (7) the seven pinned Poisson types, dense pinned solves, the oracle at S = 1.
Gmres       (8) restarts, zero right-hand sides, chunking, tolerance floor, failures.

RCMPS_EXCITATION_REPORT=1 prints the largest measured errors at the end.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
import sys
import unittest

import numpy as np

import dense_reference as ref
from rcmps_gpu.excitations.scaled import RESTART, ScaledState, batch_norm, gmres, lmul, rmul, stack
from rcmps_gpu.excitations.states import SolverFailure, State, adj, lindblad, lindblad_adjoint, random_state

norm = np.linalg.norm
STATES = ((2, False), (3, False), (4, False), (4, True), (5, False), (6, False), (6, True))
SCALINGS = ("none", "density", "explicit")
TYPES = [(a, b, dual) for dual in (False, True) for a in "LH" for b in "LH"]
# (a, b, dual, sign): sigma = 1 + sign i z p for a resolvent, shift = sign i p for a Poisson solve.
RESOLVENTS = (("H", "H", False, 0), ("L", "H", False, 1), ("L", "L", True, 0), ("L", "H", True, -1))
POISSONS = (("H", "H", False, 0), ("L", "H", False, 1), ("H", "L", False, -1), ("L", "L", False, 0),
            ("L", "L", True, 0), ("L", "H", True, -1), ("H", "H", True, 0))
NODES = np.array([0.0, 0.15, 0.5, 0.8, 0.97, 1.0])      # z = cos(theta), with both ends
TOL = 1e-12
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


def explicit(dim):
    return np.exp(np.linspace(-2.5, 1.5, dim))


@functools.lru_cache(maxsize=None)
def scaled(dim, z2, scaling):
    return ScaledState(random_state(dim, z2=z2), explicit(dim) if scaling == "explicit" else scaling)


def grid(dims=(2, 3, 4, 5, 6)):
    return [(dim, z2, scaling) for dim, z2 in STATES if dim in dims for scaling in SCALINGS]


def generators(ss):
    """{gauge: (Q, R)} from the state and S alone."""
    s, state = ss.s_host, ss.state
    inverse, direct = np.diag(1 / s), np.diag(s)
    return {"L": (state.Q, state.R), "H": (inverse @ state.Q @ direct, inverse @ state.R @ direct)}


def gauge(ss, a, b, dual):
    """Diagonal weights (left, right) with physical = left hat right."""
    g = {"L": np.ones(ss.D), "H": ss.s_host}
    return (1 / g[a], 1 / g[b]) if dual else (g[a], g[b])


def physical(ss, x, a, b, dual=False):
    """Physical variable of a hat matrix (D, D) or batch (D, n, D)."""
    left, right = gauge(ss, a, b, dual)
    return left.reshape((-1,) + (1,) * (x.ndim - 1)) * x * right


def hat(ss, x, a, b, dual=False):
    left, right = gauge(ss, a, b, dual)
    return x / left.reshape((-1,) + (1,) * (x.ndim - 1)) / right


def reference_map(ss, x, a, b, dual):
    """L_ab or L*_ab on a batch, system by system, from explicit matrices."""
    (qa, ra), (qb, rb) = generators(ss)[a], generators(ss)[b]
    out = np.empty_like(x)
    for n in range(x.shape[1]):
        m = x[:, n, :]
        out[:, n, :] = (adj(qa) @ m + m @ qb + adj(ra) @ m @ rb if dual
                        else qa @ m + m @ adj(qb) + ra @ m @ adj(rb))
    return out


def dense_map(ss, a, b, dual):
    """D^2 x D^2 matrix of L_ab or L*_ab on the column-major vec: vec(AXB) = (B^T kron A) vec(X)."""
    (qa, ra), (qb, rb) = generators(ss)[a], generators(ss)[b]
    one = np.eye(ss.D)
    if dual:
        return np.kron(one, adj(qa)) + np.kron(qb.T, one) + np.kron(rb.T, adj(ra))
    return np.kron(one, qa) + np.kron(qb.conj(), one) + np.kron(rb.conj(), ra)


def vectors(x):
    """Batch (D, n, D) -> (n, D^2) column-major vecs."""
    return x.transpose(1, 2, 0).reshape(x.shape[1], -1)


def batch(v, d):
    return np.ascontiguousarray(v.reshape(-1, d, d).transpose(2, 0, 1))


def reference_zero(ss, a, b, dual):
    """Hat form of the physical zero mode: rho for a primal, 1 for a dual."""
    return hat(ss, np.eye(ss.D, dtype=complex) if dual else ss.state.rho, a, b, dual)


def reference_pairing(ss, x, a, b, dual):
    """tr(X) for a primal, tr(rho Y) for a dual, of the physical variables of a batch."""
    y = physical(ss, x, a, b, dual)
    return np.array([np.trace(ss.state.rho @ y[:, n, :] if dual else y[:, n, :]) for n in range(x.shape[1])])


def relative(x, y):
    """Largest per-system relative difference of two batches."""
    return float((batch_norm(x - y) / batch_norm(y)).max())


class Batches(unittest.TestCase):
    def test_products_against_loops(self):
        """lmul, rmul, stack and batch_norm are their per-system definitions on (D, n, D)."""
        d, n = 4, 5
        x, a = gaussian(1, d, n, d), gaussian(2, d, d)
        left, right = lmul(a, x), rmul(x, a)
        matrices = [gaussian(3 + s, d, d) for s in range(n)]
        stacked = stack(matrices)
        self.assertEqual(stacked.shape, (d, n, d))
        self.assertTrue(stacked.flags.c_contiguous)
        for s in range(n):
            self.assertLessEqual(norm(left[:, s, :] - a @ x[:, s, :]), 1e-14)
            self.assertLessEqual(norm(right[:, s, :] - x[:, s, :] @ a), 1e-14)
            self.assertTrue(np.array_equal(stacked[:, s, :], matrices[s]))
        self.assertTrue(np.allclose(batch_norm(x), [norm(x[:, s, :]) for s in range(n)], rtol=1e-14, atol=0))
        self.assertEqual(batch_norm(x).shape, (n,))


class StateTests(unittest.TestCase):
    def test_generators_and_eigenbases(self):
        """Q_L = Q, Q_H = S^-1 Q S (R likewise), Qd, Rd their adjoints, V lam V^-1 = Q_a."""
        for dim, z2, scaling in grid():
            with self.subTest(dim=dim, z2=z2, scaling=scaling):
                ss = scaled(dim, z2, scaling)
                s, rho = ss.s_host, ss.state.rho
                expected = {"none": np.ones(dim), "density": np.sqrt(np.diag(rho).real), "explicit": explicit(dim)}
                self.assertLessEqual(norm(s - expected[scaling]), 1e-15)
                self.assertTrue(np.array_equal(ss.Q["L"], ss.state.Q) and np.array_equal(ss.R["L"], ss.state.R))
                for g, (q, r) in generators(ss).items():
                    self.assertLessEqual(norm(ss.Q[g] - q), 1e-14 * norm(q))
                    self.assertLessEqual(norm(ss.R[g] - r), 1e-14 * norm(r))
                    self.assertTrue(np.array_equal(ss.Qd[g], adj(ss.Q[g])) and np.array_equal(ss.Rd[g], adj(ss.R[g])))
                    v, vi, lam = ss.V[g], ss.Vi[g], ss.lam[g]
                    self.assertLessEqual(record("eigenbasis", norm(v @ np.diag(lam) @ vi - q) / norm(q)), 1e-12)
                    self.assertLessEqual(norm(v @ vi - np.eye(dim)), 1e-12)
                    self.assertTrue(np.array_equal(ss.Vd[g], adj(v)) and np.array_equal(ss.Vid[g], adj(vi)))
                    self.assertLess(lam.real.max(), 0)

    def test_fixed_point(self):
        """rh is Hermitian, L_HH(rh) = 0, tr(S rh S) = 1 and S rh S = rho to 1e-11."""
        for dim, z2, scaling in grid():
            with self.subTest(dim=dim, z2=z2, scaling=scaling):
                ss = scaled(dim, z2, scaling)
                s, rh = ss.s_host, ss.rh
                q, r = generators(ss)["H"]
                residual = norm(q @ rh + rh @ adj(q) + r @ rh @ adj(r))
                scale = max(1, 2 * norm(q) + norm(r) ** 2)
                self.assertLessEqual(record("fixed point residual", residual / scale), 1e-12)
                self.assertAlmostEqual(ss.fixed_point["residual"], residual, delta=1e-14 * max(1, norm(q)))
                self.assertTrue(np.array_equal(rh, adj(rh)))
                self.assertLessEqual(abs(np.trace(s[:, None] * rh * s[None, :]) - 1), 1e-14)
                self.assertLessEqual(record("S rh S - rho", norm(s[:, None] * rh * s[None, :] - ss.state.rho)), 1e-11)

    def test_fixed_point_is_refined(self):
        """From a guess wrong by 1e-4 the GMRES refinement still returns a Hermitian rh with
        S rh S = rho to 1e-11."""
        for dim, z2, scaling in grid((3, 4, 6)):
            with self.subTest(dim=dim, z2=z2, scaling=scaling):
                state = random_state(dim, z2=z2)
                noise = gaussian(7, dim, dim)
                wrong = dataclasses.replace(state, rho=state.rho + 1e-4 * (noise + adj(noise)))
                ss = ScaledState(wrong, explicit(dim) if scaling == "explicit" else scaling)
                s = ss.s_host
                self.assertGreater(ss.fixed_point["steps"], 0)
                self.assertTrue(np.array_equal(ss.rh, adj(ss.rh)))
                error = norm(s[:, None] * ss.rh * s[None, :] - state.rho)
                self.assertLessEqual(record("refined S rh S - rho", error), 1e-11)

    def test_left_canonical(self):
        """L*_LL(1) = 0, and more generally L*_ab(G_a G_b) = 0: the left fixed point is 1."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            one = stack([np.eye(dim, dtype=complex)])
            self.assertLessEqual(norm(ss.lmap(one, "L", "L", dual=True)), 1e-13)
            self.assertLessEqual(norm(lindblad_adjoint(ss.state.Q, ss.state.R, np.eye(dim))), 1e-13)
            for a, b, dual in TYPES:
                if dual:
                    y = stack([reference_zero(ss, a, b, True)])
                    self.assertLessEqual(norm(ss.lmap(y, a, b, dual=True)) / norm(y), 1e-13, (dim, scaling, a, b))

    def test_rejections(self):
        """Unknown scalings, non-positive or mis-sized vectors raise ValueError; a Q without
        decay raises SolverFailure (the no-jump preconditioner would be singular)."""
        state = random_state(3)
        for scaling in ("sqrt", np.ones(4), np.array([1.0, 0.0, 1.0]), np.array([1.0, -1.0, 1.0]),
                        np.array([1.0, np.inf, 1.0]), np.array([1.0, np.nan, 1.0])):
            with self.assertRaises(ValueError):
                ScaledState(state, scaling)
        k = np.diag([1.0, 2.0, 3.0]).astype(complex)
        frozen = State(np.zeros((3, 3), dtype=complex), k, -1j * k, np.eye(3, dtype=complex) / 3)
        with self.assertRaises(SolverFailure):
            ScaledState(frozen, "none")


class Maps(unittest.TestCase):
    def test_lmap_definition(self):
        """lmap(x, a, b, dual) is Q_a x + x Q_b' + R_a x R_b' (dual: Q_a' y + y Q_b + R_a' y R_b)."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            x = gaussian(21, dim, 3, dim)
            for a, b, dual in TYPES:
                with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                    error = relative(ss.lmap(x, a, b, dual), reference_map(ss, x, a, b, dual))
                    self.assertLessEqual(record("lmap", error), 1e-13)

    def test_lmap_is_the_physical_map(self):
        """G_a L_ab(xh) G_b = L(G_a xh G_b), and G_a^-1 L*_ab(yh) G_b^-1 = L*(G_a^-1 yh G_b^-1)."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            q, r = ss.state.Q, ss.state.R
            x = gaussian(22, dim, 3, dim)
            for a, b, dual in TYPES:
                with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                    lifted = physical(ss, x, a, b, dual).transpose(1, 0, 2)
                    expected = (lindblad_adjoint if dual else lindblad)(q, r, lifted).transpose(1, 0, 2)
                    error = relative(physical(ss, ss.lmap(x, a, b, dual), a, b, dual), expected)
                    self.assertLessEqual(record("physical lmap", error), 1e-12)

    def test_duality(self):
        """<yh, L_ab(xh)> = <L*_ab(yh), xh> (Hilbert-Schmidt), both equal to the physical
        <Y, L(X)> with X = G_a xh G_b, Y = G_a^-1 yh G_b^-1; trace form tr(y L_ab(x)) = tr(L*_ba(y) x)."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            x, y = gaussian(23, dim, 1, dim), gaussian(24, dim, 1, dim)
            for a in "LH":
                for b in "LH":
                    with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b):
                        lx, ly = ss.lmap(x, a, b), ss.lmap(y, a, b, dual=True)
                        scale = norm(lx) * norm(y)
                        value = np.vdot(y, lx)
                        self.assertLessEqual(record("duality", abs(value - np.vdot(ly, x)) / scale), 1e-13)
                        big_x, big_y = physical(ss, x, a, b)[:, 0, :], physical(ss, y, a, b, True)[:, 0, :]
                        direct = np.vdot(big_y, lindblad(ss.state.Q, ss.state.R, big_x))
                        self.assertLessEqual(record("physical duality", abs(value - direct) / scale), 1e-12)
                        swapped = ss.lmap(y, b, a, dual=True)
                        traces = np.trace(y[:, 0, :] @ lx[:, 0, :]), np.trace(swapped[:, 0, :] @ x[:, 0, :])
                        self.assertLessEqual(abs(traces[0] - traces[1]) / scale, 1e-13)

    def test_nojump_inverse(self):
        """nojump_inverse solves sigma X - z (Q_a X + X Q_b') = C, and the dual
        sigma Y - z (Q_a' Y + Y Q_b) = C, for per-system sigma and z (Poisson: sigma = i p, z = 1)."""
        rng = np.random.default_rng(31)
        n = 5
        sigma = 1 + 0.5 * gaussian(32, n)
        z = rng.uniform(0, 1, n)
        cases = ((sigma, z), (np.full(n, 0.6j), np.ones(n)), (np.zeros(n, dtype=complex), np.ones(n)))
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            gens = generators(ss)
            c = gaussian(33, dim, n, dim)
            for a, b, dual in TYPES:
                (qa, _), (qb, _) = gens[a], gens[b]
                for sg, zz in cases:
                    x = ss.nojump_inverse(c, a, b, sg, zz, dual)
                    back = np.empty_like(x)
                    for i in range(n):
                        m = x[:, i, :]
                        nojump = adj(qa) @ m + m @ qb if dual else qa @ m + m @ adj(qb)
                        back[:, i, :] = sg[i] * m - zz[i] * nojump
                    with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                        bound = 1e-10 if scaling == "explicit" else 1e-12
                        self.assertLessEqual(record(f"nojump {scaling}", relative(back, c)), bound)


class ZeroModes(unittest.TestCase):
    def test_zero_modes(self):
        """zero_mode is the hat form of rho (dual: of 1), is annihilated by its map, has unit
        pairing; pairing is tr(X) (dual: tr(rho Y)); project is idempotent and removes the pairing."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            x = gaussian(41, dim, 4, dim)
            for a, b, dual in TYPES:
                with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                    zero = ss.zero_mode(a, b, dual)
                    expected = reference_zero(ss, a, b, dual)
                    self.assertLessEqual(norm(zero - expected) / norm(expected), 1e-11)
                    zb = stack([zero])
                    self.assertLessEqual(record("L(zero)", norm(reference_map(ss, zb, a, b, dual)) / norm(zero)), 1e-12)
                    self.assertLessEqual(abs(ss.pairing(zb, a, b, dual)[0] - 1), 1e-13)
                    pairing, expected = ss.pairing(x, a, b, dual), reference_pairing(ss, x, a, b, dual)
                    self.assertEqual(pairing.shape, (4,))
                    error = np.abs(pairing - expected).max() / np.abs(expected).max()
                    self.assertLessEqual(record("pairing", error), 1e-11)
                    once = ss.project(x, a, b, dual)
                    self.assertLessEqual(relative(ss.project(once, a, b, dual), once), 1e-13)
                    self.assertLessEqual(np.abs(ss.pairing(once, a, b, dual)).max() / np.abs(pairing).max(), 1e-12)
                    self.assertLessEqual(relative(once, x - expected[None, :, None] * zero[:, None, :]), 1e-11)


def solve_resolvent(ss, a, b, dual, sign, p, *, seed=51, **controls):
    """A resolvent of the algorithm on random hat right-hand sides at the nodes NODES."""
    rhs = gaussian(seed, ss.D, len(NODES), ss.D)
    sigma = 1 + sign * 1j * NODES * p
    x, info = ss.resolvent(rhs, a, b, NODES, sigma, dual=dual, tol=TOL, **controls)
    return rhs, sigma, x, info


class Resolvent(unittest.TestCase):
    def test_residual(self):
        """(sigma - z L_ab) X = rhs to the requested relative residual for the four types of the
        algorithm, per system with z = 0, ..., 1 (at z = 0 the solution is rhs / sigma)."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            for a, b, dual, sign in RESOLVENTS:
                for p in (0.0, 0.6):
                    with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual, p=p):
                        rhs, sigma, x, info = solve_resolvent(ss, a, b, dual, sign, p)
                        back = sigma[None, :, None] * x - NODES[None, :, None] * reference_map(ss, x, a, b, dual)
                        self.assertLessEqual(record("resolvent residual", relative(back, rhs)), 1.1 * TOL)
                        self.assertLessEqual(info["residuals"].max(), TOL)
                        self.assertEqual(info["iterations"].shape, (len(NODES),))
                        self.assertLessEqual(norm(x[:, 0, :] - rhs[:, 0, :]), 1e-14 * norm(rhs[:, 0, :]))

    def test_dense(self):
        """Agreement with numpy.linalg.solve on sigma - z kron(L_ab), D <= 4."""
        for dim, z2, scaling in grid((2, 3, 4)):
            ss = scaled(dim, z2, scaling)
            for a, b, dual, sign in RESOLVENTS:
                with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                    rhs, sigma, x, _ = solve_resolvent(ss, a, b, dual, sign, 0.6)
                    matrix, one = dense_map(ss, a, b, dual), np.eye(dim * dim)
                    expected = batch(np.array([np.linalg.solve(sg * one - z * matrix, v)
                                               for sg, z, v in zip(sigma, NODES, vectors(rhs))]), dim)
                    self.assertLessEqual(record(f"resolvent dense {scaling}", relative(x, expected)), 1e-10)

    def test_scalar_arguments(self):
        """Scalar z and sigma are broadcast over the batch."""
        ss = scaled(4, False, "density")
        rhs = gaussian(52, 4, 3, 4)
        x, _ = ss.resolvent(rhs, "L", "H", 0.7, 1 + 0.42j, tol=TOL)
        y, _ = ss.resolvent(rhs, "L", "H", np.full(3, 0.7), np.full(3, 1 + 0.42j), tol=TOL)
        self.assertTrue(np.array_equal(x, y))
        back = (1 + 0.42j) * x - 0.7 * reference_map(ss, x, "L", "H", False)
        self.assertLessEqual(relative(back, rhs), 1.1 * TOL)

    def test_scaling_invariance(self):
        """The physical solution of a resolvent (S xh S for type HH) does not depend on S, to
        1e-10 at tol = 1e-13: the hat residual of the explicit S is amplified by up to cond(S)^2
        = 3e3 (measured 2.4e-10 at tol = 1e-12, 1.5e-11 here; 6e-14 between S = 1 and density)."""
        for dim, z2 in STATES:
            for a, b, dual, sign in RESOLVENTS:
                with self.subTest(dim=dim, z2=z2, a=a, b=b, dual=dual):
                    rhs = gaussian(53, dim, len(NODES), dim)        # physical right-hand sides
                    sigma = 1 + sign * 0.6j * NODES
                    out = {}
                    for scaling in SCALINGS:
                        ss = scaled(dim, z2, scaling)
                        x, _ = ss.resolvent(hat(ss, rhs, a, b, dual), a, b, NODES, sigma, dual=dual, tol=1e-13)
                        out[scaling] = physical(ss, x, a, b, dual)
                    for scaling in SCALINGS[1:]:
                        error = relative(out[scaling], out["none"])
                        self.assertLessEqual(record(f"scaling invariance {scaling}", error), 1e-10)


def solve_poisson(ss, a, b, dual, sign, p, *, seed=61, count=3):
    rhs = gaussian(seed, ss.D, count, ss.D)
    x, info = ss.poisson(rhs, a, b, sign * 1j * p, dual=dual, tol=TOL)
    zero = reference_zero(ss, a, b, dual)
    target = rhs - zero[:, None, :] * reference_pairing(ss, rhs, a, b, dual)[None, :, None]
    return rhs, zero, target, x, info


class Poisson(unittest.TestCase):
    def test_residual_and_gauge(self):
        """(-L_ab + shift) X = rhs - pairing(rhs) zero with pairing(X) = 0, for the seven
        Poisson types of the algorithm at p = 0 and 0.6."""
        for dim, z2, scaling in grid():
            ss = scaled(dim, z2, scaling)
            for a, b, dual, sign in POISSONS:
                for p in (0.0, 0.6):
                    with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual, p=p):
                        _, _, target, x, info = solve_poisson(ss, a, b, dual, sign, p)
                        back = sign * 1j * p * x - reference_map(ss, x, a, b, dual)
                        self.assertLessEqual(record(f"poisson residual {scaling}", relative(back, target)),
                                             1e-9 if scaling == "explicit" else 1e-11)
                        self.assertLessEqual(info["residuals"].max(), TOL)
                        pairing = np.abs(reference_pairing(ss, x, a, b, dual)).max()
                        scale = np.abs(reference_pairing(ss, np.abs(x) + 0j, a, b, dual)).max()
                        self.assertLessEqual(record(f"poisson pairing {scaling}", pairing / scale), 1e-11)

    def test_dense(self):
        """Agreement with a dense solve of the pinned matrix shift - L_ab + |zero><pairing|, D <= 4."""
        for dim, z2, scaling in grid((2, 3, 4)):
            ss = scaled(dim, z2, scaling)
            for a, b, dual, sign in POISSONS:
                with self.subTest(dim=dim, z2=z2, scaling=scaling, a=a, b=b, dual=dual):
                    _, zero, target, x, _ = solve_poisson(ss, a, b, dual, sign, 0.6)
                    units = batch(np.eye(dim * dim, dtype=complex), dim)
                    functional = reference_pairing(ss, units, a, b, dual)       # pairing as a row vector
                    zero_vec = zero.reshape(-1, order="F")
                    pinned = (sign * 0.6j * np.eye(dim * dim) - dense_map(ss, a, b, dual)
                              + np.outer(zero_vec, functional))
                    solution = np.linalg.solve(pinned, vectors(target).T).T
                    solution = solution - np.outer(solution @ functional, zero_vec)
                    self.assertLessEqual(record(f"poisson dense {scaling}", relative(x, batch(solution, dim))),
                                         1e-8 if scaling == "explicit" else 1e-10)

    def test_oracle(self):
        """At S = 1 the primal solves are dense_reference.excitation_poisson (shift 0, +1, -1) and
        the dual ones action_dual_poisson (unshifted, and shifted for -i p), to 1e-10."""
        for dim, z2 in STATES:
            ss = scaled(dim, z2, "none")
            for p in (0.0, 0.6):
                ctx = ref.make_context(ss.state, p, tol=1e-10)
                for a, b, dual, sign in POISSONS:
                    with self.subTest(dim=dim, z2=z2, a=a, b=b, dual=dual, p=p):
                        rhs, _, _, x, _ = solve_poisson(ss, a, b, dual, sign, p)
                        for t in range(rhs.shape[1]):
                            expected = (ref.action_dual_poisson(ctx, rhs[:, t, :], shifted=sign != 0) if dual
                                        else ref.excitation_poisson(ctx, rhs[:, t, :], shift=sign))
                            error = norm(x[:, t, :] - expected) / norm(expected)
                            self.assertLessEqual(record("poisson oracle", error), 1e-10)


def toy(seed, d, n, strength):
    """Independent systems x_s -> (1 + strength A_s) vec(x_s), with the dense matrices."""
    matrices = np.eye(d * d) + strength * gaussian(seed, n, d * d, d * d) / d

    def operator(x):
        v = x.transpose(1, 0, 2).reshape(n, d * d)
        return np.einsum("nij,nj->ni", matrices, v).reshape(n, d, d).transpose(1, 0, 2)

    def solve(rhs):
        v = rhs.transpose(1, 0, 2).reshape(n, d * d)
        return np.array([np.linalg.solve(m, c) for m, c in zip(matrices, v)]).reshape(n, d, d).transpose(1, 0, 2)

    return operator, solve


def identity(x):
    return x


class Gmres(unittest.TestCase):
    def test_restart(self):
        """A restart length below the number of iterations needed still converges."""
        operator, solve = toy(71, 4, 3, 0.3)
        rhs = gaussian(72, 4, 3, 4)
        x, info = gmres(operator, identity, rhs, tol=TOL, restart=4)
        self.assertGreater(info["steps"], 8)
        self.assertGreater(info["iterations"].min(), 4)
        self.assertLessEqual(info["residuals"].max(), TOL)
        self.assertLessEqual(relative(operator(x), rhs), 1.1 * TOL)
        self.assertLessEqual(relative(x, solve(rhs)), 1e-11)

    def test_zero_right_hand_side(self):
        """A system with zero right-hand side returns exactly zero next to nonzero ones, and an
        all-zero batch returns zero without iterating."""
        operator, solve = toy(73, 3, 3, 0.3)
        rhs = gaussian(74, 3, 3, 3)
        rhs[:, 1, :] = 0
        x, info = gmres(operator, identity, rhs, tol=TOL)
        self.assertTrue((x[:, 1, :] == 0).all())
        self.assertEqual((info["iterations"][1], info["residuals"][1]), (0, 0.0))
        self.assertLessEqual(norm(x - solve(rhs)), 1e-11 * norm(x))
        x, info = gmres(operator, identity, np.zeros_like(rhs), tol=TOL)
        self.assertTrue((x == 0).all())
        self.assertEqual(info["steps"], 0)

    def test_initial_guess_and_absolute_scale(self):
        """x0 equal to the solution needs no iteration; relative_to replaces ||rhs|| as the scale."""
        operator, solve = toy(75, 3, 2, 0.3)
        rhs = gaussian(76, 3, 2, 3)
        exact = solve(rhs)
        x, info = gmres(operator, identity, rhs, tol=TOL, x0=exact)
        self.assertEqual(info["steps"], 0)
        self.assertTrue(np.array_equal(x, exact))
        x, info = gmres(operator, identity, rhs, tol=1e-6, relative_to=1e3)
        absolute = batch_norm(operator(x) - rhs)
        self.assertLessEqual(absolute.max(), 1e-3)
        self.assertGreater(absolute.max(), 1e-9)       # it stopped at the absolute target, not far below
        self.assertTrue(np.allclose(info["residuals"], absolute / 1e3, rtol=1e-6, atol=0))

    def test_chunking(self):
        """max_basis_bytes splits a batch into chunks solved one after the other: same solution,
        per-system iteration counts and residuals for the whole batch."""
        for dim, scaling in ((4, "density"), (6, "explicit")):
            whole = scaled(dim, False, scaling)
            per_system = (min(RESTART, dim * dim) + 1) * dim * dim * 16
            for a, b, dual, sign in RESOLVENTS:
                for systems in (2, 1):
                    with self.subTest(dim=dim, a=a, b=b, dual=dual, systems=systems):
                        chunked = ScaledState(whole.state, whole.s_host, max_basis_bytes=systems * per_system + 1)
                        rhs, sigma, x, info = solve_resolvent(whole, a, b, dual, sign, 0.6)
                        y, parts = chunked.resolvent(rhs, a, b, NODES, sigma, dual=dual, tol=TOL)
                        self.assertEqual(y.shape, x.shape)
                        self.assertTrue(y.flags.c_contiguous)
                        self.assertLessEqual(record("chunking", relative(y, x)), 1e-10)
                        self.assertEqual(parts["iterations"].shape, (len(NODES),))
                        self.assertEqual(parts["residuals"].shape, (len(NODES),))
                        self.assertLessEqual(parts["residuals"].max(), TOL)
                        self.assertGreater(parts["steps"], info["steps"])       # the chunks' cycles add up

    def test_acceptance(self):
        """With an unreachable tolerance (1e-17) the solve returns exactly when its final true
        residual is at most max(accept * tol, accept_floor) and raises SolverFailure otherwise;
        the default floor (1e-11) accepts the roundoff level that 20 * tol alone rejects."""
        ss = scaled(3, False, "none")
        rhs = gaussian(77, 3, 3, 3)
        z = np.array([0.3, 0.7, 1.0])
        solve = functools.partial(ss.resolvent, rhs, "H", "H", z, tol=1e-17)
        x, info = solve(accept=math.inf)
        floor = info["residuals"].max()
        self.assertTrue(20e-17 < floor < 1e-14)
        self.assertLessEqual(relative(x - z[None, :, None] * reference_map(ss, x, "H", "H", False), rhs), 1e-14)
        factor = floor / 1e-17
        cases = ((dict(accept=1.01 * factor, accept_floor=0.0), dict(accept=0.99 * factor, accept_floor=0.0)),
                 (dict(accept=1.0, accept_floor=1.01 * floor), dict(accept=1.0, accept_floor=0.99 * floor)),
                 ({}, dict(accept_floor=0.0)))
        for accepted, rejected in cases:
            y, _ = solve(**accepted)
            self.assertTrue(np.array_equal(x, y))
            with self.assertRaises(SolverFailure):
                solve(**rejected)

    def test_floor_ends_the_iteration(self):
        """With an unreachable tolerance the iteration stops at the roundoff floor (a cycle that
        ended on its estimates without halving a true residual), far below maxiter = 250:
        20 steps at D = 3 with S = 1, 89 at D = 6 with the explicit S, 33 for GMRES(4) on the toy."""
        z = np.array([0.0, 0.3, 0.7, 0.95, 1.0])
        for dim, scaling, steps, floor in ((3, "none", 60, 1e-14), (6, "explicit", 150, 1e-11)):
            with self.subTest(dim=dim, scaling=scaling):
                ss = scaled(dim, False, scaling)
                _, info = ss.resolvent(gaussian(1, dim, 5, dim), "H", "H", z, tol=1e-17, accept=math.inf)
                self.assertLess(record(f"steps to the floor {scaling}", info["steps"]), steps)
                self.assertLessEqual(info["residuals"].max(), floor)
        operator, solve = toy(71, 4, 3, 0.3)
        rhs = gaussian(72, 4, 3, 4)
        x, info = gmres(operator, identity, rhs, tol=1e-17, restart=4, accept=math.inf)
        self.assertLess(info["steps"], 80)
        self.assertLessEqual(relative(x, solve(rhs)), 1e-14)

    def test_failures(self):
        """Nonfinite input raises SolverFailure; so does running out of iterations."""
        ss = scaled(4, False, "density")
        for bad in (np.nan, np.inf):
            rhs = gaussian(78, 4, 3, 4)
            rhs[1, 2, 3] = bad
            with np.errstate(invalid="ignore"):
                with self.assertRaises(SolverFailure):
                    ss.resolvent(rhs, "L", "H", 0.5, tol=TOL)
                with self.assertRaises(SolverFailure):
                    ss.poisson(rhs, "L", "H", 0.6j, tol=TOL)
        operator, _ = toy(79, 4, 2, 0.3)
        with self.assertRaises(SolverFailure):
            gmres(operator, identity, gaussian(80, 4, 2, 4), tol=TOL, maxiter=3)
        x, info = gmres(operator, identity, gaussian(80, 4, 2, 4), tol=TOL, maxiter=3, accept=math.inf)
        self.assertEqual(info["steps"], 3)
        self.assertGreater(info["residuals"].max(), 20 * TOL)


if __name__ == "__main__":
    unittest.main()
