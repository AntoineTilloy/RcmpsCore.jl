"""Tests of potential.py (ODE part of the Hamiltonian action) and ode.py (host DOP853).

References: the formulas of action_rhs! and excitation_observable_action
(src/excitation_action.jl), written out here with explicit matrix products at S = 1; the
oracle dense_reference (observable_action, action_rhs); exact solutions of a toy system.
State layout (4, channels, D, D) = (x, y, a1, a0): Julia's forward blocks (1, 2) are (x, y)
and its adjoint blocks (1, 2) are (a0, a1). Physical and hat variables: x = S xh S,
y = yh S, a1 unscaled, a0 = a0t S^-1; W = Wh S^-1, V = Vh S^-1, rw = rwh S, F = Fh S.

RightHandSide   ActionSystem.rhs and initial data: Julia formulas, oracle, scaling covariance.
Pullback        ActionSystem.pullback on arbitrary samples: Julia formulas, chunking.
Action          PotentialAction (bulk + boundary) against the oracle with both integrators.
HostIntegrator  ode.integrate_host on a toy system: exact samples, no step across t = 0
                (steps replayed from the recorded right-hand-side calls), statistics, failures.

Integration: n_quad 24, max_point 12, atol 1e-13, rtol 1e-12 (about 1 s per trajectory).
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
from scipy.integrate import quad
from scipy.integrate._ivp import dop853_coefficients as dop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dense_reference as ref
from kinetic import TangentFrame, finish_boundary
from models import kinetic_spec, monomial_spec, quadrature, transformed_source
from ode import integrate_host
from potential import ActionSystem, PotentialAction, integrate_scipy
from scaled import ScaledState, stack
from states import IntegratorParameters, SolverFailure, adj, random_state

norm = np.linalg.norm
INTEGRATOR = IntegratorParameters(n_quad=24, max_point=12.0, atol=1e-13, rtol=1e-12)
SPECS = {"phi2": monomial_spec(2), "phi3": monomial_spec(3), "phi4": monomial_spec(4),
         "kinetic": kinetic_spec(False), "commutator": kinetic_spec(True)}
ORACLE_SPECS = ("phi4", "phi3", "commutator")
TOL = 1e-13
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
    state = random_state(dim, z2=z2).with_integrator(INTEGRATOR)
    return ScaledState(state, np.exp(np.linspace(-2.5, 1.5, dim)) if scaling == "explicit" else scaling)


def to_hat(ss, y):
    """Physical state (..., 4, C, D, D) -> hat variables."""
    s, out = ss.s_host, y.copy()
    out[..., 0, :, :, :] = y[..., 0, :, :, :] / np.outer(s, s)
    out[..., 1, :, :, :] = y[..., 1, :, :, :] / s
    out[..., 3, :, :, :] = y[..., 3, :, :, :] * s
    return out


def hat_tangent(ss, w, v, rw):
    s = ss.s_host
    return w * s, v * s, rw / s


def source_operators(state, p, spec, w, v):
    """action_operators: (C, C_W) = (R, W), or the commutator lift ([Q, R], [Q, W] + [V, R] - i p W)."""
    q, r = state.Q, state.R
    if not spec.commutator:
        return r, w
    return q @ r - r @ q, q @ w - w @ q + v @ r - r @ v - 1j * p * w


def julia_rhs(state, p, spec, w, v, y, jac, j):
    """action_rhs! (src/excitation_action.jl, lines 19-54) for both trajectories at once."""
    q, r = state.Q, state.R
    qd, rd = adj(q), adj(r)
    c, cw = source_operators(state, p, spec, w, v)
    out = np.zeros_like(y)
    for k in range(spec.channels):
        x, ket, a1, a0 = y[0, k], y[1, k], y[2, k], y[3, k]
        out[0, k] = jac * (q @ x + x @ qd + r @ x @ rd)
        out[1, k] = jac * (q @ ket + ket @ qd + r @ ket @ rd - 1j * p * ket + v @ x + w @ x @ rd)
        out[2, k] = jac * (qd @ a1 + a1 @ q + rd @ a1 @ r)
        out[3, k] = jac * (qd @ a0 + a0 @ q + rd @ a0 @ r + 1j * p * a0 + a1 @ v + rd @ a1 @ w)
    for edge in spec.edges:
        a, src, dst = edge.coefficient * j, edge.src, edge.dst
        if edge.left:       # forward src -> dst; adjoint dst -> src
            out[0, dst] += a * (c @ y[0, src])
            out[1, dst] += a * (c @ y[1, src] + cw @ y[0, src])
            out[2, src] += a * (y[2, dst] @ c)
            out[3, src] += a * (y[3, dst] @ c + y[2, dst] @ cw)
        else:
            out[0, dst] += a * (y[0, src] @ adj(c))
            out[1, dst] += a * (y[1, src] @ adj(c))
            out[2, src] += a * (adj(c) @ y[2, dst])
            out[3, src] += a * (adj(c) @ y[3, dst])
    return out


def julia_pullback(state, p, spec, w, v, samples, points, weights):
    """Quadrature loop of excitation_observable_action (lines 104-121) without the boundary
    terms, and its boundary inputs (a0, a1, xout) at t = +a."""
    q, r, rho = state.Q, state.R, state.rho
    n = len(points)
    out = np.zeros_like(rho)
    for i in range(n):
        forward, backward = samples[i], samples[n - 1 - i]
        dx, j = weights[i] * 2 * abs(points[i]), weights[i] * transformed_source(points[i])
        for k in range(spec.channels):
            x0, x1, a1, a0 = forward[0, k], forward[1, k], backward[2, k], backward[3, k]
            out += dx * ((a0 @ r - r @ a0) @ x0 + (a1 @ r - r @ a1) @ x1 + a1 @ w @ x0 - np.trace(a1 @ x0) * (w @ rho))
        for edge in spec.edges:
            if edge.left:
                continue
            b = backward[3, edge.dst] @ forward[0, edge.src] + backward[2, edge.dst] @ forward[1, edge.src]
            if spec.commutator:
                b = adj(q) @ b - b @ adj(q) - r @ b @ adj(r) + r @ adj(r) @ b + 1j * p * b
            out += j * edge.coefficient * b
    return out, samples[n][3, 0], samples[n][2, 0], samples[n][0, spec.channels - 1]


def environment(ctx, w):
    s = ctx.state
    v = -adj(s.R) @ w
    return v, ref.excitation_poisson(ctx, ref.ket_insertion(s, v, w, s.rho), shift=1)


class RightHandSide(unittest.TestCase):
    def test_julia_formulas(self):
        """At S = 1 ActionSystem.rhs is action_rhs! on a random state and tangent (V and rw
        independent of W, so that every term is seen), for every spec and p = 0, 0.6."""
        for dim, z2 in ((3, False), (4, True)):
            ss = scaled(dim, z2, "none")
            w, v, rw = gaussian(1, dim, dim), gaussian(2, dim, dim), gaussian(3, dim, dim)
            for name, spec in SPECS.items():
                for p in (0.0, 0.6):
                    with self.subTest(dim=dim, spec=name, p=p):
                        system = ActionSystem(ss, p, spec, (w, v, rw))
                        y = gaussian(4, *system.shape)
                        expected = julia_rhs(ss.state, p, spec, w, v, y, 0.83, 0.47)
                        result = system.rhs(y, 0.83, 0.47)
                        for block in range(4):
                            error = norm(result[block] - expected[block]) / norm(expected[block])
                            self.assertLessEqual(record("rhs, Julia formulas", error), 1e-13, block)

    def test_oracle(self):
        """At S = 1 the forward blocks are dense_reference.action_rhs(adjoint=False) and the
        (a0, a1) blocks its adjoint closure, at times on both sides of 0."""
        ss = scaled(3, False, "none")
        ctx = ref.make_context(ss.state, 0.6, tol=1e-10)
        w, v, rw = gaussian(1, 3, 3), gaussian(2, 3, 3), gaussian(3, 3, 3)
        for name, spec in SPECS.items():
            system = ActionSystem(ss, 0.6, spec, (w, v, rw))
            y = gaussian(5, *system.shape)
            c, cw = ref.action_operators(ctx, w, v, spec.commutator)
            for t in (-1.3, 0.0, 0.4):
                result = system.rhs(y, 2 * abs(t), transformed_source(t))
                forward = ref.action_rhs(ctx, w, v, spec, c, cw, False)(t, np.stack([y[0], y[1]], axis=1))
                backward = ref.action_rhs(ctx, w, v, spec, c, cw, True)(t, np.stack([y[3], y[2]], axis=1))
                expected = np.stack([forward[:, 0], forward[:, 1], backward[:, 1], backward[:, 0]])
                self.assertLessEqual(norm(result - expected), 1e-13 * norm(expected), (name, t))

    def test_scaling_covariance(self):
        """For S = sqrt(diag(rho)) and an explicit S the right-hand side of the converted state
        and tangent is the converted right-hand side at S = 1."""
        for dim, z2 in ((3, False), (4, True)):
            plain = scaled(dim, z2, "none")
            w, v, rw = gaussian(1, dim, dim), gaussian(2, dim, dim), gaussian(3, dim, dim)
            for name, spec in SPECS.items():
                y = gaussian(6, 4, spec.channels, dim, dim)
                reference = ActionSystem(plain, 0.6, spec, (w, v, rw)).rhs(y, 0.83, 0.47)
                for scaling in ("density", "explicit"):
                    with self.subTest(dim=dim, spec=name, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        system = ActionSystem(ss, 0.6, spec, hat_tangent(ss, w, v, rw))
                        result, expected = system.rhs(to_hat(ss, y), 0.83, 0.47), to_hat(ss, reference)
                        for block in range(4):
                            error = norm(result[block] - expected[block]) / norm(expected[block])
                            self.assertLessEqual(record(f"rhs covariance {scaling}", error), 1e-12, block)

    def test_initial_data(self):
        """Initial state: x_0 = rh, y_0 = rw, a1_{C-1} = 1, everything else zero; without a
        tangent W = V = rw = 0 and the tangent blocks of the right-hand side stay zero."""
        ss = scaled(3, False, "explicit")
        rw = gaussian(3, 3, 3)
        for name, spec in SPECS.items():
            system = ActionSystem(ss, 0.6, spec, (gaussian(1, 3, 3), gaussian(2, 3, 3), rw))
            expected = np.zeros((4, spec.channels, 3, 3), dtype=complex)
            expected[0, 0], expected[1, 0], expected[2, -1] = ss.rh, rw, np.eye(3)
            self.assertEqual(system.shape, expected.shape)
            self.assertTrue(np.array_equal(system.initial, expected), name)
            ground = ActionSystem(ss, 0.6, spec)
            y = gaussian(7, *ground.shape)
            y[1] = y[3] = 0
            out = ground.rhs(y, 0.83, 0.47)
            self.assertTrue(np.array_equal(ground.initial[1], np.zeros_like(expected[1])))
            self.assertEqual((norm(out[1]), norm(out[3])), (0.0, 0.0), name)


class Pullback(unittest.TestCase):
    def test_julia_formulas_and_chunks(self):
        """pullback on arbitrary samples is the quadrature loop of excitation_observable_action:
        bulk Fh S, boundary covector a0t S^-1, a1 and xout = S xh S at +a; and it does not
        depend on the chunk size (smaller than, equal to, and above the number of nodes)."""
        points, weights = quadrature(INTEGRATOR)
        n = len(points)
        for dim, z2 in ((3, False), (4, True)):
            w, v, rw = gaussian(1, dim, dim), gaussian(2, dim, dim), gaussian(3, dim, dim)
            for name, spec in SPECS.items():
                samples = gaussian(8, n + 1, 4, spec.channels, dim, dim)
                state = scaled(dim, z2, "none").state
                expected = julia_pullback(state, 0.6, spec, w, v, samples, points, weights)
                for scaling in ("none", "explicit"):
                    with self.subTest(dim=dim, spec=name, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        s = ss.s_host
                        system = ActionSystem(ss, 0.6, spec, hat_tangent(ss, w, v, rw))
                        hat = to_hat(ss, samples)
                        out, a0, (a1, xout) = system.pullback(hat, points, weights)
                        result = (out * s, a0 / s, a1, s[:, None] * xout * s)
                        for label, x, e in zip(("bulk", "a0", "a1", "xout"), result, expected):
                            error = norm(x - e) / norm(e)
                            self.assertLessEqual(record(f"pullback {label} {scaling}", error), 1e-11, label)
                        for chunk in (1, 5, n - 1, n, 3 * n):
                            other, b0, (b1, yout) = system.pullback(hat, points, weights, chunk=chunk)
                            self.assertLessEqual(record("pullback chunk", norm(other - out) / norm(out)), 1e-13, chunk)
                            for x, e in ((b0, a0), (b1, a1), (yout, xout)):
                                self.assertTrue(np.array_equal(x, e))


@functools.lru_cache(maxsize=None)
def potential(dim, z2, scaling, p, name, host):
    return PotentialAction(scaled(dim, z2, scaling), p, SPECS[name], INTEGRATOR,
                           integrate=integrate_host if host else integrate_scipy, tol=TOL)


def complete(pot, frame):
    """Physical F = Fh S of one term: bulk plus the boundary terms of kinetic.finish_boundary."""
    f, a0 = pot.bulk(frame)
    boundary, _ = finish_boundary(frame, pot.boundary, a0, tol=TOL)
    return (f + boundary) * pot.ss.s_host[None, None, :]


class Action(unittest.TestCase):
    def test_oracle(self):
        """bulk + finish_boundary is dense_reference.observable_action (no coefficient) for
        :phi^4:, :phi^3: and the commutator kinetic spec, with integrate_scipy and with
        ode.integrate_host: each within 1e-10 of the oracle, and of each other."""
        for dim, z2, p, scalings in ((3, False, 0.6, ("none", "explicit")), (4, True, 0.0, ("density",))):
            ctx = ref.make_context(scaled(dim, z2, "none").state, p, tol=1e-10)
            w = gaussian(10, dim, dim)
            for name in ORACLE_SPECS:
                expected = ref.observable_action(ctx, w, *environment(ctx, w), SPECS[name])
                for scaling in scalings:
                    with self.subTest(dim=dim, z2=z2, p=p, spec=name, scaling=scaling):
                        ss = scaled(dim, z2, scaling)
                        frame = TangentFrame(ss, p, stack([w * ss.s_host]), tol=TOL)
                        f = [complete(potential(dim, z2, scaling, p, name, host), frame)[:, 0, :]
                             for host in (False, True)]
                        for label, x in zip(("scipy", "host"), f):
                            self.assertLessEqual(record(f"oracle, {label}", norm(x - expected) / norm(expected)), 1e-10)
                        self.assertLessEqual(record("scipy - host", norm(f[0] - f[1]) / norm(expected)), 1e-10)

    def test_batch_and_counters(self):
        """A batch is its tangents one at a time, linear in W at integration accuracy; one
        trajectory at construction (W = 0, tangent-independent boundary data) and one per tangent."""
        dim, z2, p = 3, False, 0.6
        pot = potential(dim, z2, "density", p, "phi3", False)
        ss, before = pot.ss, pot.trajectories
        w = [gaussian(11, dim, dim), gaussian(12, dim, dim)]
        w.append(0.7j * w[0] - 0.4 * w[1])
        f = complete(pot, TangentFrame(ss, p, stack([m * ss.s_host for m in w]), tol=TOL))
        self.assertEqual(pot.trajectories - before, 3)
        combined = 0.7j * f[:, 0, :] - 0.4 * f[:, 1, :]
        self.assertLessEqual(record("linearity", norm(f[:, 2, :] - combined) / norm(combined)), 1e-10)
        single = complete(pot, TangentFrame(ss, p, stack([w[1] * ss.s_host]), tol=TOL))
        self.assertLessEqual(record("batch", norm(single[:, 0, :] - f[:, 1, :]) / norm(single)), 1e-11)
        fresh = PotentialAction(ss, p, SPECS["phi2"], dataclasses.replace(INTEGRATOR, max_point=4.0), tol=TOL)
        self.assertEqual(fresh.trajectories, 1)
        self.assertAlmostEqual(fresh.endpoint, 2.0)
        self.assertEqual(len(fresh.points), 2 * INTEGRATOR.n_quad)

    def test_trajectories_of_the_two_integrators(self):
        """integrate_scipy and integrate_host return the same samples (len(points) + 1, 4, C, D, D):
        the state at the quadrature points and at +a; the start is not among them."""
        ss = scaled(3, False, "explicit")
        w = gaussian(10, 3, 3)
        frame = TangentFrame(ss, 0.6, stack([w * ss.s_host]), tol=TOL)
        system = ActionSystem(ss, 0.6, SPECS["commutator"], (frame.wh[:, 0, :], frame.vh[:, 0, :], frame.rw[:, 0, :]))
        points, _ = quadrature(INTEGRATOR)
        endpoint = math.sqrt(INTEGRATOR.max_point)
        one = integrate_scipy(system, points, endpoint, INTEGRATOR)
        two = integrate_host(system, points, endpoint, INTEGRATOR)
        self.assertEqual(one.shape, (len(points) + 1,) + system.shape)
        self.assertEqual(two.shape, one.shape)
        scale = np.abs(one).max(axis=(0, 2, 3, 4), keepdims=True)
        self.assertLessEqual(record("samples, scipy - host", (np.abs(one - two) / scale).max()), 1e-9)
        # The constant channels: x_0 = rh and a1_{C-1} = 1 are stationary.
        self.assertLessEqual(np.abs(two[:, 0, 0] - ss.rh).max(), 1e-10 * norm(ss.rh))
        self.assertLessEqual(np.abs(two[:, 2, -1] - np.eye(3)).max(), 1e-10)

    def test_scipy_failures(self):
        """integrate_scipy is CPU only (ValueError) and raises SolverFailure beyond maxiters
        right-hand-side evaluations."""
        ss = scaled(3, False, "none")
        system = ActionSystem(ss, 0.0, SPECS["phi2"])
        points, _ = quadrature(INTEGRATOR)
        with self.assertRaises(SolverFailure):
            integrate_scipy(system, points, math.sqrt(12.0), dataclasses.replace(INTEGRATOR, maxiters=50))
        system.xp = object()
        with self.assertRaises(ValueError):
            integrate_scipy(system, points, math.sqrt(12.0), INTEGRATOR)


def source_integral(lower, upper):
    """Integral of 2|t| J(t^2) over (lower, upper), split at the cusp t = 0."""
    cuts = [lower] + ([0.0] if lower < 0 < upper else []) + [upper]
    return sum(quad(transformed_source, a, b, epsabs=1e-13, epsrel=1e-13)[0] for a, b in zip(cuts, cuts[1:]))


class Toy:
    """Scalar equations on a (2, 2) state, recording |t| = jac / 2 of every call:
    y00' = jac lam y00, y01' = jac, y10' = source, y11' = jac mu y11, from (1, 0, 0, 1) at -a.
    Exact: exp(lam u), u, int_{-a}^{t} 2|s| J(s^2) ds, exp(mu u) with u = t|t| + a^2."""
    xp = np

    def __init__(self, a, lam=-0.35 + 1.1j, mu=0.2 - 0.7j):
        self.a, self.lam, self.mu, self.calls = a, lam, mu, []
        self.initial = np.eye(2, dtype=complex)

    def rhs(self, y, jac, source):
        self.calls.append(jac / 2)
        return np.array([[jac * self.lam * y[0, 0], jac], [source, jac * self.mu * y[1, 1]]], dtype=complex)

    def exact(self, t):
        u = t * abs(t) + self.a ** 2
        return np.array([[np.exp(self.lam * u), u], [source_integral(-self.a, t), np.exp(self.mu * u)]])


def replay(calls, a):
    """Steps of integrate_host from the recorded |t|: one call at -a, then per attempt the
    stages t + C[s] h (s = 1..11) and t + h, then three dense stages after an accepted step
    that holds sample points. Signs are those of the half-interval of the step, so a stage
    beyond t = 0 does not fit and raises AssertionError. Returns accepted times and counts."""
    c, calls = dop.C, np.asarray(calls)

    def stages(block, t, sign):
        """Step length if sign * block are the 12 stage times of an attempt from t."""
        if len(block) < 12:
            return None
        times = sign * block
        h = times[11] - t
        return h if h > 0 and np.allclose(times[:11], t + c[1:12] * h, rtol=0, atol=1e-12) else None

    assert calls[0] == a
    t, i = -a, 1
    accepted, counts = [t], dict(attempts=0, accepted=0, rejected=0, dense_steps=0)
    while i < len(calls):
        sign = -1.0 if t < 0 else 1.0
        h = stages(calls[i:i + 12], t, sign)
        assert h is not None, f"attempt {counts['attempts']} from t = {t} leaves its half-interval"
        new = sign * calls[i + 11]
        i += 12
        counts["attempts"] += 1
        rest = calls[i:]
        if len(rest) >= 3 and np.allclose(sign * rest[:3], t + c[13:16] * h, rtol=0, atol=1e-12):
            counts["dense_steps"] += 1
            i += 3
        elif len(rest):
            retry, follow = stages(rest[:12], t, sign), stages(rest[:12], new, -1.0 if new < 0 else 1.0)
            assert (retry is None) != (follow is None), "ambiguous call sequence"
            if retry is not None:
                counts["rejected"] += 1
                continue
        counts["accepted"] += 1
        t = new + 0.0       # -0.0 -> 0.0
        accepted.append(t)
    return accepted, counts


class HostIntegrator(unittest.TestCase):
    def run_toy(self, integrator, points, **kw):
        toy, statistics = Toy(math.sqrt(integrator.max_point)), {}
        samples = integrate_host(toy, points, toy.a, integrator, statistics=statistics, **kw)
        return toy, samples, statistics

    def test_exact_samples(self):
        """Samples at the quadrature points and at +a reproduce the exact solutions (complex
        exponentials of t|t|, and the integral of the source with its cusp at 0)."""
        integrator = IntegratorParameters(n_quad=24, max_point=9.0, atol=1e-13, rtol=1e-12)
        points, _ = quadrature(integrator)
        toy, samples, _ = self.run_toy(integrator, points)
        self.assertEqual(samples.shape, (len(points) + 1, 2, 2))
        self.assertTrue(np.array_equal(toy.initial, np.eye(2)))
        for t, sample in zip(np.r_[points, toy.a], samples):
            exact = toy.exact(t)
            error = (np.abs(sample - exact) / np.maximum(1, np.abs(exact))).max()
            self.assertLessEqual(record("toy samples", error), 1e-10, t)

    def test_no_step_across_zero(self):
        """Replaying the steps from the recorded right-hand-side calls: every stage of every
        attempt stays in its half-interval, one accepted step ends exactly at t = 0 and the
        last exactly at +a, also with a first step as long as the half-interval. Hence the
        kink of 2|t| is integrated exactly (y01 = t|t| + a^2 to roundoff) at loose tolerances."""
        loose = IntegratorParameters(n_quad=8, max_point=9.0, atol=1e-5, rtol=1e-4)
        tight = IntegratorParameters(n_quad=8, max_point=9.0, atol=1e-13, rtol=1e-12)
        points, _ = quadrature(loose)
        for integrator, first in ((loose, 0.01), (loose, 6.0), (tight, 0.01), (tight, 6.0)):
            with self.subTest(rtol=integrator.rtol, first=first):
                toy, samples, statistics = self.run_toy(integrator, points, initial_step=first)
                accepted, counts = replay(toy.calls, toy.a)
                self.assertIn(0.0, accepted)
                self.assertEqual((accepted[0], accepted[-1]), (-toy.a, toy.a))
                self.assertTrue((np.diff(accepted) > 0).all())
                self.assertEqual(counts, {key: statistics[key] for key in counts})
                if first == 6.0:
                    self.assertGreater(counts["rejected"], 0)
                clock = np.r_[points, toy.a] * np.abs(np.r_[points, toy.a]) + toy.a ** 2
                self.assertLessEqual(record("toy clock", np.abs(samples[:, 0, 1] - clock).max()), 1e-13)

    def test_statistics(self):
        """statistics holds attempts = accepted + rejected, the steps with dense output, and
        rhs_calls = 1 + 12 attempts + 3 dense_steps, which is the number of calls made."""
        integrator = IntegratorParameters(n_quad=12, max_point=9.0, atol=1e-11, rtol=1e-10)
        toy, _, statistics = self.run_toy(integrator, quadrature(integrator)[0])
        self.assertEqual(set(statistics), {"attempts", "accepted", "rejected", "dense_steps", "rhs_calls"})
        self.assertEqual(statistics["attempts"], statistics["accepted"] + statistics["rejected"])
        self.assertEqual(statistics["rhs_calls"], 1 + 12 * statistics["attempts"] + 3 * statistics["dense_steps"])
        self.assertEqual(statistics["rhs_calls"], len(toy.calls))
        self.assertTrue(0 < statistics["dense_steps"] <= statistics["accepted"])

    def test_points_on_step_ends(self):
        """A sample point that is the end of a step (t = 0 always is) takes the step's result;
        no points at all returns the endpoint alone."""
        integrator = IntegratorParameters(n_quad=8, max_point=9.0, atol=1e-13, rtol=1e-12)
        toy, samples, _ = self.run_toy(integrator, np.array([-1.0, 0.0, 2.0]))
        for t, sample in zip((-1.0, 0.0, 2.0, toy.a), samples):
            self.assertLessEqual(np.abs(sample - toy.exact(t)).max() / np.abs(toy.exact(t)).max(), 1e-10, t)
        toy, alone, _ = self.run_toy(integrator, np.array([]))
        self.assertEqual(alone.shape, (1, 2, 2))
        self.assertLessEqual(np.abs(alone[0] - samples[-1]).max(), 1e-10)

    def test_failures(self):
        """maxiters bounds the attempted steps: one less than needed raises SolverFailure, the
        exact number passes; a point outside (-a, a) and a nonfinite right-hand side raise too."""
        integrator = IntegratorParameters(n_quad=8, max_point=9.0, atol=1e-9, rtol=1e-8)
        points, _ = quadrature(integrator)
        _, reference, statistics = self.run_toy(integrator, points)
        needed = statistics["attempts"]
        _, again, _ = self.run_toy(dataclasses.replace(integrator, maxiters=needed), points)
        self.assertTrue(np.array_equal(again, reference))
        for maxiters in (needed - 1, 3):
            with self.assertRaises(SolverFailure):
                self.run_toy(dataclasses.replace(integrator, maxiters=maxiters), points)
        with self.assertRaises(SolverFailure):
            self.run_toy(integrator, np.array([1.0, 3.5]))
        toy = Toy(3.0, lam=math.nan)
        with np.errstate(invalid="ignore"), self.assertRaises(SolverFailure):
            integrate_host(toy, points, toy.a, integrator)


if __name__ == "__main__":
    unittest.main()
