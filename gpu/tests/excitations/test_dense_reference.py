"""Independent tests of the dense excitation oracle dense_reference.py.

Written from the API contract and the Julia sources only (dense_reference.py was not
read), except the last class, IntegrationAdditions, added when the modules were put
together, and the final-check additions listed below. Each test ports a Julia test of
test/excitations.jl, test/excitation_action.jl or test/excitation_eigsolve.jl (cited in its
docstring), or checks a formula of docs/excitations.tex with explicit matrices coded here
(`Brute`, `finite_box_element`), which never call the oracle.

Final check (oracle open; each addition fails on a one-line mutation of the oracle that the
earlier tests let through): Environments.test_short_window_boundaries, test_residual_checks,
test_integrator_failures and the mixing clause of test_fixed_point_must_be_unique_and_mixing;
the 2-node routing lines of KineticBruteForce; in SectorsAndIterativeSolver the tolerance
of retarget, the weighted terms of test_field_seed_and_preconditioner,
test_validation_contexts_and_grids, the correction loop of test_refine_steps and the
tighter pair comparisons of check_levels; and six tests of IntegrationAdditions, from
test_refinement_stops_at_the_target to test_parity_tolerance_is_forwarded.

Out of scope, as in the task: the :shg/:sg/:schwinger vertex models, parallel=, the
kinetic_backend switch and adaptive kinetic node refinement (replaced by observable_* on
the kinetic specs and by node doubling), dispersion_relation, excitation_gap and
relativistic_diagnostics (no oracle counterpart).

Conventions: Julia's RandomState(...; seed) becomes states.random_state (NumPy draws, so
the states differ from the Julia ones); seeds were chosen for well-conditioned states and
the transfer gap and cond(rho) are quoted where a state is built. `assertApprox` is
Julia's isapprox, |x-y|_2 <= max(atol, rtol max(|x|_2, |y|_2)) with rtol = 0 whenever
atol > 0; tolerances are the Julia ones unless a docstring says otherwise. Channels and
indices are zero based, H(U,W) = vdot(U, F(W)) = tr(U' F(W)), and vec is column-major.

Tests expected to need more than about a minute require RCMPS_EXCITATION_SLOW=1.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
import sys
import types
import unittest
from unittest import mock
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, lu_solve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coordinates as tc
import dense_reference as ref
import models
import states
from davidson import DavidsonOptions
from states import IntegratorParameters, SolverFailure, adj, make_state, random_state, unvec, vec

slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (more than about a minute)")
norm = np.linalg.norm
FREE, PHI4 = models.phi4_terms(0.0), models.phi4_terms(1.0)


class Case(unittest.TestCase):
    def assertApprox(self, x, y, atol=0.0, rtol=None, msg=""):
        """Julia's isapprox(x, y; atol, rtol), reporting the measured distance."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        if rtol is None:
            rtol = 0.0 if atol > 0 else math.sqrt(np.finfo(float).eps)
        error, bound = norm(x - y), max(atol, rtol * max(norm(x), norm(y)))
        self.assertTrue(error <= bound, f"{msg} |x-y| = {error:.3e} exceeds {bound:.3e}")


def gaussian(seed, *shape):
    """Seeded complex normal draws (unit variance), in place of randn(MersenneTwister(seed),ComplexF64,...)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def tuned(state, **controls):
    for name, value in controls.items():
        setattr(state.integrator, name, value)
    return state


def vacuum():
    return make_state(np.zeros((1, 1)), np.zeros((1, 1)))


def coherent():
    """State([.3+.1im], 0) with max_point = 30 (test/excitations.jl:53-54); coupling g = 2."""
    return tuned(make_state([[0.3 + 0.1j]], [[0.0]], coupling=2.0), max_point=30)


def generic_state(**controls):
    """D = 2 state without symmetry: transfer gap 0.99, cond(rho) = 7.0."""
    return tuned(random_state(2, seed=3, scaling=0.5), **controls)


def z2_state():
    """z2_test_state() of test/excitation_eigsolve.jl:105-111: transfer gap 0.20, cond(rho) = 14."""
    s = random_state(4, seed=11, scaling=0.5, z2=True)
    s.coupling = 1.0
    return tuned(s, max_point=14, n_quad=48, atol=1e-11, rtol=1e-10)


def unit(d, i, j):
    e = np.zeros((d, d), dtype=np.complex128)
    e[i, j] = 1
    return e


def ket_environment(ctx, w):
    """(v, rw) as formed at the top of excitation_hamiltonian_action (src/excitation_action.jl:212-213)."""
    s = ctx.state
    v = -adj(s.R) @ w
    return v, ref.excitation_poisson(ctx, ref.ket_insertion(s, v, w, s.rho), shift=1)


def blocks(pair):
    """The four D x D blocks (00, ket, bra, ket+bra) of Pair.right. The contract only says
    "4 blocks": accept (4, D, D) (or a list) and Julia's (D, D, 4); block 00 is rho."""
    rho, right = pair.context.state.rho, np.asarray(pair.right)
    d = len(rho)
    if right.shape == (4, d, d) and np.allclose(right[0], rho, rtol=0, atol=1e-13):
        return [right[k] for k in range(4)]
    if right.shape == (d, d, 4) and np.allclose(right[..., 0], rho, rtol=0, atol=1e-13):
        return [right[..., k] for k in range(4)]
    raise AssertionError(f"Pair.right has shape {right.shape} or its first block is not rho")


def connected_transfer(pair, x):
    """excitation_transfer! (src/excitation_environments.jl:135-155) on four blocks, rewritten
    here from eq. (uncentered)/(connected) of docs/excitations.tex."""
    s, p = pair.context.state, pair.context.momentum
    ket = lambda y: pair.v @ y + pair.w @ y @ adj(s.R)
    bra = lambda y: y @ adj(pair.vbra) + s.R @ y @ adj(pair.u)
    out = [states.lindblad(s.Q, s.R, y) for y in x]
    out[1] = out[1] - 1j * p * x[1] + ket(x[0])
    out[2] = out[2] + 1j * p * x[2] + bra(x[0])
    out[3] = out[3] + ket(x[2]) + bra(x[1]) + pair.w @ x[0] @ adj(pair.u) - pair.overlap * x[0]
    return out


def contract(pair, x):
    """excitation_contract (src/excitation_environments.jl:157)."""
    return np.trace(x[3]) + np.trace(pair.left00 @ x[0])


# ---------------------------------------------------------------------------------------
# Independent references: explicit superoperator matrices, never the oracle's own solves.

def left(a):
    """vec(A X) = left(A) vec(X), column-major."""
    return np.kron(np.eye(len(a)), a)


def right(b):
    """vec(X B) = right(B) vec(X)."""
    return np.kron(b.T, np.eye(len(b)))


def sandwich(a, b):
    """vec(A X B) = sandwich(A, B) vec(X)."""
    return np.kron(b.T, a)


class Brute:
    """docs/excitations.tex for one pair (U, W) with explicit matrices: D^2 x D^2
    superoperators, the 4D^2 x 4D^2 connected generator of eq. (connected) in channel
    order (00, ket, bra, ket+bra), its right zero vector r of eqs. (singles), (double), the
    left zero functional l of eq. (left), and the lifted sources of eq. (sources)."""

    def __init__(self, s, p, u, w):
        d, n = s.D, s.D ** 2
        one, eye, zero = np.eye(d, dtype=complex), np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)
        R, Q, rho = s.R, s.Q, s.rho
        v, vu = -adj(R) @ w, -adj(R) @ u                           # left gauge, eq. (gauge)
        L = left(Q) + right(adj(Q)) + sandwich(R, adj(R))
        dual = left(adj(Q)) + right(Q) + sandwich(adj(R), R)       # L*(A) = Q'A + AQ + R'AR
        P = np.outer(vec(rho), vec(one).conj())                    # P X = rho tr X
        A = left(v) + sandwich(w, adj(R))                          # A_W(X) = V X + W X R'
        B = right(adj(vu)) + sandwich(R, adj(u))                   # B_U(X) = X V_U' + R X U'
        N = np.trace(w @ rho @ adj(u))                             # eq. (metric)
        r00 = vec(rho)
        r10 = np.linalg.solve(-L + 1j * p * eye + P, A @ r00)
        r01 = np.linalg.solve(-L - 1j * p * eye + P, B @ r00)
        F = A @ r01 + B @ r10 + vec(w @ rho @ adj(u))
        r11 = np.linalg.solve(-L + P, F - P @ F)                   # G F, eq. (G)
        h = np.linalg.solve(-dual + np.outer(vec(one), vec(rho).conj()), vec(adj(u) @ w - N * one))
        self.generator = np.block([[L, zero, zero, zero],
                                   [A, L - 1j * p * eye, zero, zero],
                                   [B, zero, L + 1j * p * eye, zero],
                                   [sandwich(w, adj(u)) - N * eye, B, A, L]])
        self.right = [unvec(x, d) for x in (r00, r10, r01, r11)]
        self.r = np.concatenate([r00, r10, r01, r11])
        self.h, self.overlap = unvec(h, d), N
        self.ell = np.concatenate([vec(self.h.T), np.zeros(2 * n), vec(one)])  # l(X) = ell @ X, no conjugation
        self.operators = {False: (R, w, u),                        # eq. (kinetic) for c = [Q,R]
                          True: (Q @ R - R @ Q, Q @ w - w @ Q + v @ R - R @ v - 1j * p * w,
                                 Q @ u - u @ Q + vu @ R - R @ vu - 1j * p * u)}
        self.zero = zero

    def sources(self, commutator):
        """(S_L, S_R) of eq. (sources)."""
        c, cw, cu = self.operators[commutator]
        z, lc, lw, rc, ru = self.zero, left(c), left(cw), right(adj(c)), right(adj(cu))
        return (np.block([[lc, z, z, z], [lw, lc, z, z], [z, z, lc, z], [z, z, lw, lc]]),
                np.block([[rc, z, z, z], [z, rc, z, z], [ru, z, rc, z], [z, ru, z, rc]]))

    def resolvent_element(self, terms, nodes):
        """Sum over terms of coefficient times eq. (kinetic-resolvent) (two source orders),
        or the infinite-line :phi^2: formula of the reverse-contraction section (four orders)
        for a non-kinetic spec, on kinetic_grid(nodes): the weights contain 2/pi, so each
        order has weight one half."""
        theta, weights = models.kinetic_grid(nodes)
        eye, total = np.eye(len(self.r)), 0j
        for z, weight in zip(np.cos(theta), weights):
            resolvent = np.linalg.inv(eye - z * self.generator)
            for coefficient, spec in terms:
                sl, sr = self.sources(spec.commutator)
                orders = ((sr, sl), (sl, sr)) if spec.kinetic else tuple((a, b) for a in (sl, sr) for b in (sl, sr))
                for first, second in orders:
                    total += coefficient * weight / 2 * (self.ell @ (second @ (resolvent @ (first @ self.r))))
        return total


def brute_resolvent_action(s, p, w, terms, nodes):
    """F with F[i,j] = H(E_ij, W): the action read off the bilinear form, one bra unit at a time."""
    d = s.D
    return np.array([[Brute(s, p, unit(d, i, j), w).resolvent_element(terms, nodes) for j in range(d)]
                     for i in range(d)])


def finite_box_element(s, p, u, w, spec, decay_lengths=32):
    """Port of finite_box_element (test/excitations.jl:1-30): ordinary doubled ket and bra
    matrices of eq. (lift) and their full Kronecker transfer on a finite box, with no
    connected transfer, no Poisson environment and no excitation source routine; the
    extensive part 2 B N_UW e_O is subtracted explicitly as in eq. (box). DOP853 at
    rtol = atol = 1e-12 replaces Vern7 at the same tolerances."""
    d = s.D
    z, one, shift = np.zeros((d, d)), np.eye(2 * d), 1j * p * np.eye(d)
    qk, rk = np.block([[s.Q, z], [-adj(s.R) @ w, s.Q - shift]]), np.block([[s.R, z], [w, s.R]])
    qb, rb = np.block([[s.Q, z], [-adj(s.R) @ u, s.Q - shift]]), np.block([[s.R, z], [u, s.R]])
    l = np.kron(one, qk) + np.kron(qb.conj(), one) + np.kron(rb.conj(), rk)
    ck = qk @ rk - rk @ qk if spec.commutator else rk
    cb = qb @ rb - rb @ qb if spec.commutator else rb
    sl, sr = np.kron(one, ck), np.kron(cb.conj(), one)
    gap = math.inf if d == 1 else np.sort(-np.linalg.eigvals(states.transfer_matrix(s.Q, s.R)).real)[1]
    padding = decay_lengths / gap                                  # transfer_gap(s), src/transfer.jl:162-166
    prop = expm(padding * l)
    x = np.zeros((2 * d, 2 * d), dtype=complex)
    x[:d, :d] = s.rho
    y = np.zeros((4 * d * d, spec.channels), dtype=complex)
    y[:, 0] = prop @ vec(x)

    def rhs(t, flat):
        y = flat.reshape(-1, spec.channels)
        dy, j = 2 * abs(t) * (l @ y), models.transformed_source(t)
        for edge in spec.edges:
            dy[:, edge.dst] += edge.coefficient * j * ((sl if edge.left else sr) @ y[:, edge.src])
        return dy.reshape(-1)

    a, flat = math.sqrt(s.integrator.max_point), y.reshape(-1)
    for span in ((-a, 0.0), (0.0, a)):                             # tstops=[0.]
        solution = solve_ivp(rhs, span, flat, method="DOP853", rtol=1e-12, atol=1e-12)
        assert solution.success, solution.message
        flat = solution.y[:, -1]
    final = unvec(prop @ flat.reshape(-1, spec.channels)[:, -1], 2 * d)
    return np.trace(final[d:, d:]) - 2 * (padding + a * a) * np.trace(w @ s.rho @ adj(u)) * np.trace(final[:d, :d])


# ---------------------------------------------------------------------------------------

class AnalyticReferences(Case):
    """Item 1: D = 1 states with exact tangent energies (docs/excitations.tex, Verification 1-2)."""

    def test_references_of_this_file(self):
        """The independent references themselves against the same exact formulas: the brute
        resolvent form on the vacuum, and the finite box on the coherent state, where
        :phi^4: contributes 12*0.3^2/omega and the two kinetic source graphs omega."""
        one = np.ones((1, 1), dtype=complex)
        for p in (0.0, 0.7, 2.0):
            omega = math.sqrt(1 + p * p)
            self.assertApprox(Brute(vacuum(), p, one, one).resolvent_element(FREE, 64), omega, atol=1e-12)
            self.assertApprox(Brute(coherent(), p, one, one).resolvent_element(FREE, 64), omega, atol=1e-12)
            box = lambda spec: finite_box_element(coherent(), p, one, one, spec)
            self.assertApprox(box(models.monomial_spec(4)), 12 * 0.3 ** 2 / omega, atol=1e-10)
            self.assertApprox(box(models.kinetic_spec(False)) + box(models.kinetic_spec(True)), omega, atol=1e-10)

    def test_free_vacuum_action(self):
        """test/excitation_action.jl:25-26: F(W) = sqrt(1+p^2) W on the vacuum (atol 1e-11);
        256 fixed kinetic nodes (the contract default) replace the adaptive rule."""
        w1 = np.ones((1, 1), dtype=complex)
        for p in (0.0, 0.7, 2.0):
            f = ref.hamiltonian_action(ref.make_context(vacuum(), p), w1, FREE)
            self.assertApprox(f, math.sqrt(1 + p * p) * w1, atol=1e-11)

    def test_free_vacuum_spectrum(self):
        """test/excitations.jl:33-52 (dispersion_relation and excitation_gap on the dense
        route): energies at p = -2, -.5, 0, .5, 2 (atol 1e-11 on the vector), unit rest gap
        (1e-12), unit-norm Ritz vectors, residuals below 1e-12; both assemblies."""
        ps = [-2.0, -0.5, 0.0, 0.5, 2.0]
        for assembly in ("action", "bilinear"):
            results = [ref.spectrum(vacuum(), p, FREE, assembly=assembly, kinetic_nodes=256) for p in ps]
            self.assertApprox([r["energies"][0] for r in results], np.sqrt(1 + np.square(ps)), atol=1e-11)
            self.assertApprox(results[2]["energies"][0], 1.0, atol=1e-12)
            for p, r in zip(ps, results):
                self.assertApprox(ref.tangent_metric(vacuum(), r["vectors"][0], r["vectors"][0]), 1.0)
                self.assertLess(max(r["residuals"]), 1e-12)
                self.assertEqual((r["momentum"], np.shape(r["hamiltonian"]), len(r["basis"])), (p, (1, 1), 1))

    def test_free_vacuum_eigsolve(self):
        """test/excitation_eigsolve.jl:222-228: exact free dispersion in the one-dimensional
        vacuum tangent space (atol 1e-11); the callback sees every operator call."""
        for p in (0.0, 0.7):
            rows = []
            r = ref.eigsolve(vacuum(), p, FREE, callback=lambda row, u, au: rows.append((row["count"], u.shape, au.shape)))
            self.assertEqual((r["sector"], r["dimension"], bool(r["converged"])), ("all", 1, True))
            self.assertEqual(len(r["energies"]), 1)
            self.assertApprox(r["energies"][0], math.sqrt(1 + p * p), atol=1e-11)
            self.assertEqual(rows, [(k + 1, (1, 1), (1, 1)) for k in range(r["numops"])])
            self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"]), (256, 512))
        r = ref.eigsolve(vacuum(), 0.7, FREE, kinetic_nodes=24, refined_kinetic_nodes=40)
        self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"]), (24, 40))
        self.assertApprox(r["validated_energies"][0], math.sqrt(1.49), atol=1e-11)

    def test_coherent_spectrum(self):
        """test/excitations.jl:53-60 with ExcitationModel(:phi4,2.): omega + 24*0.3^2/omega at
        p = 0, .7, 2 (atol 1e-8); p = 0 is the excitation_gap line 1 + 24*0.3^2. Julia's
        default assembly is :bilinear; the contract default 'action' is checked as well."""
        s, terms = coherent(), models.phi4_terms(2.0)
        for p in (0.0, 0.7, 2.0):
            omega = math.sqrt(1 + p * p)
            for assembly in ("bilinear", "action"):
                energy = ref.spectrum(s, p, terms, assembly=assembly)["energies"][0]
                self.assertApprox(energy, omega + 24 * 0.3 ** 2 / omega, atol=1e-8, msg=f"p={p} {assembly}")

    def test_coherent_action_and_eigsolve(self):
        """The same formula through hamiltonian_action (rho = 1, so F = E W) and through
        eigsolve with terms=None, i.e. models.phi4_terms(state.coupling = 2); the level is
        validated and accepted, and not accepted at validation_tol = 1e-16
        (test/excitation_eigsolve.jl:258-259, without excitation_gap)."""
        s, w1 = coherent(), np.ones((1, 1), dtype=complex)
        for p in (0.0, 0.7, 2.0):
            omega = math.sqrt(1 + p * p)
            exact = omega + 24 * 0.3 ** 2 / omega
            f = ref.hamiltonian_action(ref.make_context(s, p), w1, models.phi4_terms(2.0))
            self.assertApprox(f, exact * w1, atol=1e-8)
            r = ref.eigsolve(s, p)
            self.assertTrue(r["converged"] and r["accepted"])
            self.assertApprox(r["energies"][0], exact, atol=1e-8)
            self.assertApprox(r["validated_energies"][0], exact, atol=1e-8)
        strict = ref.eigsolve(s, 0.7, validation_tol=1e-16)
        self.assertFalse(strict["accepted"] or strict["validation"][0]["consistent"])
        self.assertApprox(strict["energies"][0], math.sqrt(1.49) + 24 * 0.3 ** 2 / math.sqrt(1.49), atol=1e-8)

    def test_relative_tolerance_is_clipped_silently(self):
        """Contract of the oracle integrator: rtol below SciPy's floor 100 eps is clipped to
        max(rtol, 2.3e-14) by the oracle itself, so no SciPy warning is emitted (the research
        refinements request rtol = 1e-14 and below); the coherent energy is unchanged."""
        s, w1 = tuned(coherent(), rtol=1e-15, atol=1e-13), np.ones((1, 1), dtype=complex)
        ctx = ref.make_context(s, 0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f = ref.hamiltonian_action(ctx, w1, models.phi4_terms(2.0))
            element = ref.matrix_element(ref.make_pair(ctx, w1, w1), models.phi4_terms(2.0))
        exact = math.sqrt(1.49) + 24 * 0.3 ** 2 / math.sqrt(1.49)
        self.assertApprox(f, exact * w1, atol=1e-8)
        self.assertApprox(element, exact, atol=1e-8)


class ElementaryMaps(Case):
    """Insertion maps of src/excitation_environments.jl:100-101 and
    src/excitation_action.jl:13-17, 74-80, with the trace dualities that define them."""

    @classmethod
    def setUpClass(cls):
        cls.s, cls.p = random_state(3, seed=5, scaling=0.5), 0.4
        cls.ctx = ref.make_context(cls.s, cls.p)
        cls.a, cls.x, cls.u, cls.w = (gaussian(40 + k, 3, 3) for k in range(4))

    def test_tangent_metric(self):
        """tangent_metric (src/excitation_environments.jl:85-88): tr(W rho U'), antilinear in U."""
        s, u, w = self.s, self.u, self.w
        n = ref.tangent_metric(s, u, w)
        self.assertApprox(n, np.trace(w @ s.rho @ adj(u)), atol=1e-14)
        self.assertApprox(ref.tangent_metric(s, 2j * u, (1 - 1j) * w), -2j * (1 - 1j) * n, atol=1e-14)
        self.assertGreater(ref.tangent_metric(s, w, w).real, 0)

    def test_insertions_and_trace_duals(self):
        """A_W, B_U and tr A_W(X) = tr B_U(X) = 0 in the left gauge V = -R'W (docs, after
        eq. (gauge)); ket_trace_dual and bra_transfer_pull are their trace duals:
        tr(a A_W(x)) = tr(A_W^d(a) x) and tr(a B_U(x)) = vdot(U, (aR - Ra) x)."""
        s, a, x, u, w = self.s, self.a, self.x, self.u, self.w
        R = s.R
        v, vu = -adj(R) @ w, -adj(R) @ u
        ket, bra = ref.ket_insertion(s, v, w, x), ref.bra_insertion(s, vu, u, x)
        self.assertApprox(ket, v @ x + w @ x @ adj(R), atol=1e-13)
        self.assertApprox(bra, x @ adj(vu) + R @ x @ adj(u), atol=1e-13)
        self.assertLess(abs(np.trace(ket)) + abs(np.trace(bra)), 1e-13)
        dual, pull = ref.ket_trace_dual(s, v, w, a), ref.bra_transfer_pull(s, a, x)
        self.assertApprox(dual, a @ v + adj(R) @ a @ w, atol=1e-13)
        self.assertApprox(pull, (a @ R - R @ a) @ x, atol=1e-13)
        self.assertApprox(np.trace(dual @ x), np.trace(a @ ket), atol=1e-13)
        self.assertApprox(np.vdot(u, pull), np.trace(a @ bra), atol=1e-13)

    def test_commutator_operators(self):
        """action_operators: (R, W), or ([Q,R], [Q,W] + [V,R] - ipW) of eq. (kinetic); a pure
        gauge pair (V, W) = ([X,Q] + ipX, [X,R]) gives c_W = [X,[Q,R]] (docs, covariance
        check). bra_source_pull moves U' to the front of tr(P c_U'):
        tr(P c_U') = vdot(U, Q'P - PQ' - RPR' + RR'P + ipP)."""
        s, ctx, p, a, x, u, w = self.s, self.ctx, self.p, self.a, self.x, self.u, self.w
        Q, R = s.Q, s.R
        v, vu = -adj(R) @ w, -adj(R) @ u
        c, cw = ref.action_operators(ctx, w, v, False)
        self.assertApprox(c, R, atol=1e-14)
        self.assertApprox(cw, w, atol=1e-14)
        c, cw = ref.action_operators(ctx, w, v, True)
        self.assertApprox(c, Q @ R - R @ Q, atol=1e-13)
        self.assertApprox(cw, Q @ w - w @ Q + v @ R - R @ v - 1j * p * w, atol=1e-13)
        gauge = ref.action_operators(ctx, x @ R - R @ x, x @ Q - Q @ x + 1j * p * x, True)[1]
        self.assertApprox(gauge, x @ c - c @ x, atol=1e-12)
        self.assertApprox(ref.bra_source_pull(ctx, a, False), a, atol=1e-15)
        pull = ref.bra_source_pull(ctx, a, True)
        self.assertApprox(pull, adj(Q) @ a - a @ adj(Q) - R @ a @ adj(R) + R @ adj(R) @ a + 1j * p * a, atol=1e-13)
        cu = ref.action_operators(ctx, u, vu, True)[1]
        self.assertApprox(np.vdot(u, pull), np.trace(a @ adj(cu)), atol=1e-12)


class Environments(Case):
    """Item 2: port of "Tangent environments and independent finite-box contractions"
    (test/excitations.jl:77-106). RandomState(2;seed=82,scaling=.7) becomes
    random_state(2, seed=82, scaling=0.7): transfer gap 1.62, cond(rho) = 6.3."""

    @classmethod
    def setUpClass(cls):
        cls.s = tuned(random_state(2, seed=82, scaling=0.7), max_point=18, atol=1e-12, rtol=1e-12)
        cls.u, cls.w = gaussian(84, 2, 2), gaussian(85, 2, 2)

    def test_context_factors(self):
        """ExcitationContext/transfer_factors (src/excitation_environments.jl:18-43): the
        stored generator and the pinned LU factors of -l+P, -l+ip+P, -l-ip+P with
        P = vec(rho) vec(1)', which are one object at p = 0."""
        s, eye = self.s, np.eye(4)
        l = states.transfer_matrix(s.Q, s.R)
        projector, y = np.outer(vec(s.rho), vec(np.eye(2)).conj()), gaussian(7, 4)
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            self.assertEqual((ctx.momentum, ctx.tol), (p, 1e-10))
            self.assertApprox(ctx.transfer, l, atol=1e-14)
            self.assertApprox(ctx.state.rho, s.rho, atol=1e-15)
            for factor, shift in ((ctx.poisson, 0), (ctx.ketfactor, 1), (ctx.brafactor, -1)):
                exact = np.linalg.solve(-l + 1j * shift * p * eye + projector, y)
                self.assertApprox(lu_solve(factor, y), exact, atol=1e-12, msg=f"p={p} shift={shift}")
        ctx = ref.make_context(s, 0.0, tol=1e-9)
        self.assertIs(ctx.ketfactor, ctx.poisson)
        self.assertIs(ctx.brafactor, ctx.poisson)
        self.assertEqual(ctx.tol, 1e-9)

    def test_poisson_solves(self):
        """excitation_poisson (src/excitation_environments.jl:73-82): tr x = 0 and
        -L(x) + i shift p x = b - tr(b) rho for shift = 0, +1, -1, with b not traceless."""
        s, b = self.s, gaussian(21, 2, 2)
        rhs = b - np.trace(b) * s.rho
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            for shift in (0, 1, -1):
                x = ref.excitation_poisson(ctx, b, shift=shift)
                self.assertLess(abs(np.trace(x)), 1e-10)
                self.assertLess(norm(-states.lindblad(s.Q, s.R, x) + 1j * shift * p * x - rhs), 1e-10)
            self.assertApprox(ref.excitation_poisson(ctx, b), ref.excitation_poisson(ctx, b, shift=0), atol=1e-15)

    def test_dual_poisson_solves(self):
        """action_dual_poisson (src/excitation_action.jl:81-89): tr(rho x) = 0 and
        -L*(x) - [i p x] = b - tr(rho b) 1. They are the trace duals of the unshifted and of
        the BRA (shift = -1) solves: tr(a G_shift(b)) = tr(G^d(a) b)."""
        s, a, b = self.s, gaussian(22, 2, 2), gaussian(21, 2, 2)
        rhs = a - np.trace(s.rho @ a) * np.eye(2)
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            for shifted in (False, True):
                x = ref.action_dual_poisson(ctx, a, shifted=shifted)
                self.assertLess(abs(np.trace(s.rho @ x)), 1e-10)
                self.assertLess(norm(-states.lindblad_adjoint(s.Q, s.R, x) - 1j * p * shifted * x - rhs), 1e-10)
                primal = ref.excitation_poisson(ctx, b, shift=-1 if shifted else 0)
                self.assertApprox(np.trace(a @ primal), np.trace(x @ b), atol=1e-10)
            self.assertApprox(ref.action_dual_poisson(ctx, a), ref.action_dual_poisson(ctx, a, shifted=False), atol=1e-15)

    def test_pair_blocks(self):
        """test/excitations.jl:83-96: the right blocks are a zero vector of the connected
        transfer (1e-10), the left functional annihilates them (1e-11) and the range of the
        transfer (1e-10); overlap = tr(W rho U'); the tail solves the dense
        boundary_environment equation for U'W (1e-10); ket, bra and double are traceless
        (1e-10). The blocks are also compared with the explicit solves of `Brute`."""
        s, u, w = self.s, self.u, self.w
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            pair = ref.make_pair(ctx, u, w)
            r = blocks(pair)
            self.assertLess(norm(np.array(connected_transfer(pair, r))), 1e-10)
            self.assertLess(abs(contract(pair, r)), 1e-11)
            x = list(gaussian(86, 4, 2, 2))
            self.assertLess(abs(contract(pair, connected_transfer(pair, x))), 1e-10)
            self.assertApprox(pair.overlap, np.trace(w @ s.rho @ adj(u)))
            self.assertApprox(pair.overlap, ref.tangent_metric(s, u, w))
            self.assertLess(sum(abs(np.trace(r[k])) for k in (1, 2, 3)), 1e-10)
            brute = Brute(s, p, u, w)
            self.assertLess(norm(pair.left00 - brute.h), 1e-10)
            self.assertLess(abs(np.trace(s.rho @ pair.left00)), 1e-10)
            for k in range(4):
                self.assertApprox(r[k], brute.right[k], atol=1e-10, msg=f"p={p} block {k}")
            self.assertApprox(pair.v, -adj(s.R) @ w, atol=1e-14)
            self.assertApprox(pair.vbra, -adj(s.R) @ u, atol=1e-14)
            self.assertApprox(pair.w, w, atol=0.0, rtol=0.0)
            self.assertApprox(pair.u, u, atol=0.0, rtol=0.0)
            self.assertIs(pair.context, ctx)

    def test_finite_box_contractions(self):
        """test/excitations.jl:97-102: observable_element against the independent finite-box
        contraction for :phi^2:, :phi^4: and the commutator kinetic source graph
        (atol 3e-7, rtol 3e-8), at p = 0 and 0.6. kinetic_spec(False) is added."""
        s, u, w = self.s, self.u, self.w
        specs = (models.monomial_spec(2), models.monomial_spec(4), models.kinetic_spec(True), models.kinetic_spec(False))
        for p in (0.0, 0.6):
            pair = ref.make_pair(ref.make_context(s, p), u, w)
            for k, spec in enumerate(specs):
                exact = finite_box_element(s, p, u, w, spec)
                self.assertApprox(ref.observable_element(pair, spec), exact, atol=3e-7, rtol=3e-8, msg=f"p={p} spec {k}")

    def test_finite_box_action(self):
        """The same independent reference for the reverse contraction: F[i,j] is the
        finite-box element with the bra unit E_ij (H(U,W) = tr(U'F)), for :phi^4: at p = 0.6,
        with the tolerance of the finite-box test (atol 3e-7, rtol 3e-8)."""
        s, w, p, spec = self.s, self.w, 0.6, models.monomial_spec(4)
        ctx = ref.make_context(s, p)
        v, rw = ket_environment(ctx, w)
        exact = np.array([[finite_box_element(s, p, unit(2, i, j), w, spec) for j in range(2)] for i in range(2)])
        self.assertApprox(ref.observable_action(ctx, w, v, rw, spec), exact, atol=3e-7, rtol=3e-8)

    def test_constant_density_cancels(self):
        """test/excitations.jl:103-104: :phi^0: cancels exactly for off-diagonal complex
        tangents (1e-10); its reverse contraction vanishes as well."""
        s, u, w = self.s, self.u, self.w
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            self.assertLess(abs(ref.observable_element(ref.make_pair(ctx, u, w), models.monomial_spec(0))), 1e-10)
            v, rw = ket_environment(ctx, w)
            self.assertLess(norm(ref.observable_action(ctx, w, v, rw, models.monomial_spec(0))), 1e-10)

    def test_short_window_boundaries(self):
        """The boundary pieces of the reverse contraction (src/excitation_action.jl:90-96 and
        120-121: action_boundaries with the adjoint covectors at channel 1, the forward state
        of the last channel at t = a, and the rw start block of lines 61-62) are below 1e-9 on
        the resolved windows of the other tests. On a short window (max_point = 1.5) they are
        of order 1e-2 to 1e-1, and the finite box, which has no environment at all, is still
        an exact reference: measured |E - box| <= 2e-12 and |F - F_box| <= 2e-12 (atol 1e-10)."""
        s = tuned(random_state(2, seed=82, scaling=0.7), max_point=1.5, n_quad=64, atol=1e-13, rtol=1e-12)
        p, u, w = 0.6, self.u, self.w
        ctx = ref.make_context(s, p)
        pair = ref.make_pair(ctx, u, w)
        v, rw = ket_environment(ctx, w)
        for k, spec in enumerate((models.monomial_spec(4), models.kinetic_spec(True))):
            self.assertApprox(ref.observable_element(pair, spec), finite_box_element(s, p, u, w, spec), atol=1e-10,
                              msg=f"element, spec {k}")
            exact = np.array([[finite_box_element(s, p, unit(2, i, j), w, spec) for j in range(2)] for i in range(2)])
            action = ref.observable_action(ctx, w, v, rw, spec)
            self.assertApprox(action, exact, atol=1e-10, msg=f"action, spec {k}")
            self.assertGreater(norm(action - ref.observable_action(ctx, w, v, np.zeros_like(rw), spec)), 1e-3)

    def test_residual_checks(self):
        """The residual checks norm(residual) <= 20 tol max(1, norm(rhs)) of excitation_poisson,
        of the ExcitationPair tail and of action_dual_poisson (src/excitation_environments.jl:
        79-80, 120-121; src/excitation_action.jl:86-87): each solve passes with 20 tol a factor
        1.5 above its measured relative residual and is a SolverFailure a factor 1.5 below.
        The tolerance is replaced in a built context (make_context would reject it at the
        mixing check); the tail check comes after three Poisson solves, which keep theirs."""
        s, u, w, p = self.s, self.u, self.w, 0.6
        ctx, one, b = ref.make_context(s, p), np.eye(2), gaussian(21, 2, 2)

        def bracket(solve, residual, rhs, message):
            relative = norm(residual) / max(1, norm(rhs))
            self.assertTrue(0 < relative < 1e-13, message)
            solve(replace(ctx, tol=1.5 * relative / 20))
            with self.assertRaisesRegex(SolverFailure, message):
                solve(replace(ctx, tol=relative / 1.5 / 20))

        for shift in (0, 1, -1):
            x, rhs = ref.excitation_poisson(ctx, b, shift=shift), b - np.trace(b) * s.rho
            bracket(lambda c: ref.excitation_poisson(c, b, shift=shift),
                    -states.lindblad(s.Q, s.R, x) + 1j * shift * p * x - rhs, rhs, "tangent boundary solve residual")
        for shifted in (False, True):
            x, rhs = ref.action_dual_poisson(ctx, b, shifted=shifted), b - np.trace(s.rho @ b) * one
            bracket(lambda c: ref.action_dual_poisson(c, b, shifted=shifted),
                    -states.lindblad_adjoint(s.Q, s.R, x) - 1j * p * shifted * x - rhs, rhs,
                    "excitation action boundary residual")
        with self.assertRaisesRegex(SolverFailure, "tangent boundary solve residual"):
            ref.make_pair(replace(ctx, tol=1e-19), u, w)
        rhs, poisson = adj(u) @ w - ref.tangent_metric(s, u, w) * one, ref.excitation_poisson
        with mock.patch.object(ref, "excitation_poisson", lambda c, x, shift=0: poisson(ctx, x, shift=shift)):
            bracket(lambda c: ref.make_pair(c, u, w),
                    -states.lindblad_adjoint(s.Q, s.R, ref.make_pair(ctx, u, w).left00) - rhs, rhs,
                    "tangent right-tail solve residual")

    def test_integrator_failures(self):
        """successful_retcode and all(isfinite, ...) of both ODE routes (src/excitations.jl:57-60,
        src/excitation_action.jl:66-70) and test/excitations.jl:165-166: with maxiters = 1 the
        vacuum kinetic source ODEs (Julia: excitation_gap with kinetic_backend=:ode) are a
        SolverFailure. The oracle counts right-hand sides: N of them mean between (N-4)/15 and
        (N-4)/12 attempted DOP853 steps, so a budget of (N-4)/12 steps must pass and one below
        (N-4)/15 must fail. maxiters = 0 is an ArgumentError (quadrature_check)."""
        one = np.ones((1, 1), dtype=complex)

        def routes(maxiters, kinetic=models.kinetic_spec(False)):
            state = vacuum()
            state.integrator.maxiters = maxiters
            ctx = ref.make_context(state, 0.0)
            v, rw = ket_environment(ctx, one)
            return (lambda: ref.observable_element(ref.make_pair(ctx, one, one), kinetic),
                    lambda: ref.observable_action(ctx, one, v, rw, kinetic))

        for _, spec in FREE:
            for route, label in zip(routes(1, spec), ("excitation ODE failed: MaxIters", "excitation action ODE failed: MaxIters")):
                with self.assertRaisesRegex(SolverFailure, label):
                    route()
        for route in routes(0):
            with self.assertRaises(ValueError):
                route()
        integrator = ref.solve_ivp
        for k, label in enumerate(("excitation ODE failed: MaxIters", "excitation action ODE failed: MaxIters")):
            halves = []                                        # right-hand sides of each half-window solve

            def counting(*arguments, **controls):
                solution = integrator(*arguments, **controls)
                halves.append(solution.nfev)
                return solution

            with mock.patch.object(ref, "solve_ivp", counting):
                routes(1_000_000)[k]()
            self.assertEqual(len(halves), 2 * (k + 1))         # the action integrates two trajectories
            evaluations = max(halves[0] + halves[1], halves[-2] + halves[-1])
            self.assertGreater(evaluations, 100)
            routes(math.ceil((evaluations - 4) / 12))[k]()
            with self.assertRaisesRegex(SolverFailure, label):
                routes((evaluations - 4) // 15 - 1)[k]()
        # A solver failure and a nonfinite final state, on the generic state.
        s, u, w = self.s, self.u, self.w
        ctx = ref.make_context(s, 0.6)
        pair = ref.make_pair(ctx, u, w)
        v, rw = ket_environment(ctx, w)
        huge = models.Observable(1, (models.Edge(0, 0, True, 1e200), models.Edge(0, 0, False, 1e200)))
        element, action = lambda spec: ref.observable_element(pair, spec), lambda spec: ref.observable_action(ctx, w, v, rw, spec)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(SolverFailure, "excitation ODE failed"):
                element(huge)
            with self.assertRaisesRegex(SolverFailure, "excitation action ODE failed"):
                action(huge)
        nan = lambda fun, span, y, t_eval, **kw: types.SimpleNamespace(
            success=True, message="", y=np.full((len(y), len(t_eval)), np.nan + 0j))
        with mock.patch.object(ref, "solve_ivp", nan):
            with self.assertRaisesRegex(SolverFailure, "nonfinite excitation trajectory"):
                element(models.monomial_spec(2))
            with self.assertRaisesRegex(SolverFailure, "nonfinite excitation action trajectory"):
                action(models.monomial_spec(2))

    def test_fixed_point_must_be_unique_and_mixing(self):
        """ExcitationContext (src/excitation_environments.jl:24-31): R = 0, K = diag(0,1) has
        transfer spectrum {0, 0, i, -i} (two fixed points and peripheral modes): SolverFailure.
        Invalid momentum or tolerance is an ArgumentError (ValueError here)."""
        idle = make_state(np.zeros((2, 2)), np.diag([0.0, 1.0]), rho=np.eye(2) / 2)
        with self.assertRaises(SolverFailure):
            ref.make_context(idle, 0.0)
        # The second clause alone (lines 29-30): besides the single zero mode every eigenvalue
        # must have Re z < -tol*scale with scale = max(1, |l|_F). R = sqrt(1e-3)|0><1|,
        # K = diag(1,-1) has the spectrum {0, -1e-3, -5e-4 +- 2i} and |l|_F = 2.83: accepted at
        # the default tolerance, rejected for a band tol*scale = 7.5e-4 (one zero mode still).
        slow = make_state(math.sqrt(1e-3) * np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]))
        l = states.transfer_matrix(slow.Q, slow.R)
        self.assertApprox(np.sort_complex(np.linalg.eigvals(l)), [-1e-3, -5e-4 - 2j, -5e-4 + 2j, 0], atol=1e-12)
        self.assertGreater(norm(l), 2.8)
        self.assertEqual(ref.make_context(slow, 0.0).tol, 1e-10)
        with self.assertRaises(SolverFailure):
            ref.make_context(slow, 0.0, tol=0.75e-3 / norm(l))
        ref.make_context(slow, 0.0, tol=0.25e-3 / norm(l))
        for bad in (dict(p=math.nan), dict(p=0.0, tol=0.0), dict(p=0.0, tol=math.inf)):
            with self.assertRaises(ValueError, msg=str(bad)):
                ref.make_context(self.s, **bad)


class KineticBruteForce(Case):
    """Item 6: the kinetic resolvent sums against `Brute`, i.e. eq. (kinetic-resolvent) and
    the :phi^2: resolvent formula of docs/excitations.tex evaluated with an explicit
    4D^2 x 4D^2 connected generator and explicit inverses; none of the oracle's Poisson or
    resolvent routines enters the reference. Both sides are finite sums on the same
    kinetic_grid, so they agree to roundoff (asserted at 1e-10; not a Julia test).
    State: random_state(2, seed=2, scaling=0.5), transfer gap 0.16, cond(rho) = 1.4."""

    @classmethod
    def setUpClass(cls):
        cls.s = tuned(random_state(2, seed=2, scaling=0.5), max_point=24, n_quad=192, atol=1e-12, rtol=1e-12)
        cls.u, cls.w = gaussian(31, 2, 2), gaussian(32, 2, 2)
        cls.quadratic = [(0.37, models.monomial_spec(2))]

    def test_reference_is_consistent(self):
        """Checks of the reference itself: generator r = 0, l generator = 0, l(r) = 0
        (eqs. (connected), (left)), and Hermiticity of the brute kinetic form."""
        s, u, w = self.s, self.u, self.w
        for p in (0.0, 0.6):
            b = Brute(s, p, u, w)
            self.assertLess(norm(b.generator @ b.r), 1e-12)
            self.assertLess(norm(b.ell @ b.generator), 1e-12)
            self.assertLess(abs(b.ell @ b.r), 1e-12)
            for terms in (FREE, self.quadratic):
                swapped = Brute(s, p, w, u).resolvent_element(terms, 8)
                self.assertApprox(b.resolvent_element(terms, 8), np.conj(swapped), atol=1e-12)

    def test_kinetic_sum(self):
        """excitation_kinetic_sum (src/excitations.jl:64-86) against eq. (kinetic-resolvent)."""
        s, u, w = self.s, self.u, self.w
        for p in (0.0, 0.6):
            pair, brute = ref.make_pair(ref.make_context(s, p), u, w), Brute(s, p, u, w)
            for nodes in (4, 16):
                for terms in (FREE, FREE[:1], [(0.3, models.kinetic_spec(True))]):
                    self.assertApprox(ref.kinetic_sum(pair, terms, nodes), brute.resolvent_element(terms, nodes),
                                      atol=1e-10, msg=f"p={p} nodes={nodes}")
            self.assertApprox(ref.matrix_element(pair, FREE, kinetic_nodes=16), brute.resolvent_element(FREE, 16),
                              atol=1e-10)

    def test_kinetic_action_sum(self):
        """excitation_kinetic_action_sum (src/excitation_action.jl:132-179) against the brute
        form on the bra units E_ij: kinetic terms (two source orders), a quadratic monomial
        (four orders) and their sum; rw is the explicit r_10 of eq. (singles)."""
        s, w = self.s, self.w
        v = -adj(s.R) @ w
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            rw = Brute(s, p, w, w).right[1]
            for k, terms in enumerate((FREE, self.quadratic, FREE + self.quadratic)):
                for nodes in (4, 16):
                    exact = brute_resolvent_action(s, p, w, terms, nodes)
                    self.assertApprox(ref.kinetic_action_sum(ctx, w, v, rw, terms, nodes), exact, atol=1e-10,
                                      msg=f"p={p} terms {k} nodes={nodes}")
            mixed = FREE + self.quadratic
            self.assertApprox(ref.hamiltonian_action(ctx, w, mixed, kinetic_nodes=16),
                              brute_resolvent_action(s, p, w, mixed, 16), atol=1e-10, msg="quadratic routing")
            # The routing itself (src/excitation_action.jl:214-218) shows on a 2-node grid only:
            # there the resolvent sum is 7e-3 away from the windowed source ODE, which does not
            # depend on the grid, while 16 nodes agree with the ODE to 3e-11.
            coarse = ref.hamiltonian_action(ctx, w, self.quadratic, kinetic_nodes=2)
            self.assertApprox(coarse, brute_resolvent_action(s, p, w, self.quadratic, 2), atol=1e-10, msg="coarse routing")
            self.assertGreater(norm(coarse - ref.hamiltonian_action(ctx, w, self.quadratic, kinetic_nodes=16)), 1e-3)
            # Julia compares Vector{Edge}: a spec built from a list of edges is the same monomial.
            listed = models.Observable(3, list(models.monomial_spec(2).edges))
            self.assertTrue(models.is_quadratic(listed) and not models.is_quadratic(models.monomial_spec(4)))
            self.assertApprox(ref.hamiltonian_action(ctx, w, [(0.37, listed)], kinetic_nodes=2), coarse, atol=0.0, rtol=0.0)

    def test_action_contracts_to_bilinear_sum(self):
        """On a common fixed grid the kinetic action is the exact reverse contraction of the
        bilinear sum: vdot(U, F) = excitation_kinetic_sum, to roundoff (1e-11)."""
        s, u, w = self.s, self.u, self.w
        for p in (0.0, 0.6):
            ctx = ref.make_context(s, p)
            v, rw = ket_environment(ctx, w)
            for nodes in (2, 32):
                f = ref.kinetic_action_sum(ctx, w, v, rw, FREE, nodes)
                self.assertApprox(np.vdot(u, f), ref.kinetic_sum(ref.make_pair(ctx, u, w), FREE, nodes), atol=1e-11)

    def test_quadratic_monomial_routes(self):
        """excitation_hamiltonian_action sends :phi^2: to the infinite-line resolvent
        (src/excitation_action.jl:214-218) while excitation_element integrates its source
        ODE on the window (src/excitations.jl:117-120): both agree with each other and with
        the brute resolvent form once the window is resolved (max_point = 24; atol 2e-8,
        rtol 2e-9 as for the action tests, where Julia covers this route with :schwinger)."""
        s, u, w, terms = self.s, self.u, self.w, self.quadratic
        for p in (0.0, 0.7):
            ctx = ref.make_context(s, p)
            pair = ref.make_pair(ctx, u, w)
            windowed = ref.matrix_element(pair, terms)
            self.assertApprox(np.vdot(u, ref.hamiltonian_action(ctx, w, terms)), windowed, atol=2e-8, rtol=2e-9)
            self.assertApprox(Brute(s, p, u, w).resolvent_element(terms, 64), windowed, atol=2e-8, rtol=2e-9)
            self.assertApprox(0.37 * ref.observable_element(pair, models.monomial_spec(2)), windowed, atol=1e-12)
            # The bilinear route never reads the kinetic grid for a monomial: the same number.
            self.assertEqual(ref.matrix_element(pair, terms, kinetic_nodes=2), windowed)


class ActionVersusBilinear(Case):
    """Item 3: ports of "Reverse-contracted excitation Hamiltonian action"
    (test/excitation_action.jl:1-15, 23-24) and of the sesquilinearity block of
    test/excitations.jl:108-122, for ExcitationModel(:phi4,1.) on generic_state()."""

    @classmethod
    def setUpClass(cls):
        cls.u, cls.w = gaussian(1, 2, 2), gaussian(2, 2, 2)

    def test_action_matches_bilinear_and_is_linear(self):
        """test/excitation_action.jl:2-15 (max_point 24, n_quad 192, tolerances 1e-12):
        vdot(U, F(W)) = H(U,W) (atol 2e-8, rtol 2e-9) and F((1+i)W) = (1+i)F(W) (atol 2e-8)
        at p = 0 and 0.7. Both sides use the same fixed kinetic grid (256 nodes)."""
        s = generic_state(max_point=24, n_quad=192, atol=1e-12, rtol=1e-12)
        u, w = self.u, self.w
        for p in (0.0, 0.7):
            ctx = ref.make_context(s, p)
            a = ref.hamiltonian_action(ctx, w, PHI4)
            exact = ref.matrix_element(ref.make_pair(ctx, u, w), PHI4)
            self.assertApprox(np.vdot(u, a), exact, atol=2e-8, rtol=2e-9, msg=f"p={p}")
            self.assertApprox(ref.hamiltonian_action(ctx, (1 + 1j) * w, PHI4), (1 + 1j) * a, atol=2e-8, msg=f"p={p}")

    def test_kinetic_source_ode_matches_resolvent(self):
        """test/excitation_action.jl:23-24 and test/excitations.jl:121-122: the
        kinetic_backend=:ode route is the generic source ODE applied to the two kinetic
        specs; it agrees with the infinite-line resolvent sum once the window is resolved
        (atol 2e-8), for the action and for the bilinear elements f(u,w) and f(w,w)."""
        s = generic_state(max_point=24, n_quad=192, atol=1e-12, rtol=1e-12)
        u, w = self.u, self.w
        ctx = ref.make_context(s, 0.7)
        v, rw = ket_environment(ctx, w)
        ode = sum(c * ref.observable_action(ctx, w, v, rw, spec) for c, spec in FREE)
        self.assertApprox(ode, ref.kinetic_action_sum(ctx, w, v, rw, FREE, 256), atol=2e-8)
        for bra in (u, w):
            pair = ref.make_pair(ctx, bra, w)
            ode = sum(c * ref.observable_element(pair, spec) for c, spec in FREE)
            self.assertApprox(ode, ref.kinetic_sum(pair, FREE, 256), atol=2e-8)

    def test_sesquilinear_and_hermitian(self):
        """test/excitations.jl:109-120 (max_point 30, tolerances 1e-12, p = 0.7):
        f(u,w) = conj(f(w,u)) (1e-9), antilinear in u and linear in w (2e-8)."""
        s = generic_state(max_point=30, atol=1e-12, rtol=1e-12)
        u, w = self.u, self.w
        ctx = ref.make_context(s, 0.7)
        f = lambda u, w: ref.matrix_element(ref.make_pair(ctx, u, w), PHI4)
        a = f(u, w)
        self.assertApprox(a, np.conj(f(w, u)), atol=1e-9)
        self.assertApprox(f((2 + 1j) * u, (0.3 - 0.7j) * w), (2 - 1j) * (0.3 - 0.7j) * a, atol=2e-8)
        self.assertApprox(f(u, w + u), a + f(u, u), atol=2e-8)


class DenseHamiltonian(Case):
    """Item 4: excitation_hamiltonian and excitation_spectrum (src/excitations.jl:158-209)."""

    def test_assembly_action_versus_bilinear(self):
        """test/excitation_action.jl:29-32: the :action and :bilinear assemblies give the
        same matrix and levels (atol 2e-8, Frobenius) at p = 0.7; the returned basis is
        orthonormal in the tangent metric (1e-10) and reproduces the Ritz vectors."""
        s = generic_state(max_point=24, n_quad=192, atol=1e-12, rtol=1e-12)
        direct = ref.spectrum(s, 0.7, PHI4, nev=2, assembly="action")
        bilinear = ref.spectrum(s, 0.7, PHI4, nev=2, assembly="bilinear")
        self.assertApprox(direct["hamiltonian"], bilinear["hamiltonian"], atol=2e-8)
        self.assertApprox(direct["energies"], bilinear["energies"], atol=2e-8)
        for result in (direct, bilinear):
            h, basis = np.asarray(result["hamiltonian"]), result["basis"]
            self.assertEqual((h.shape, len(basis), len(result["energies"]), len(result["vectors"])), ((4, 4), 4, 2, 2))
            self.assertApprox(h, adj(h), atol=1e-14)
            self.assertLess(result["hermiticity_error"], 1e-7)       # the contract's hermitian_tol
            gram = np.array([[ref.tangent_metric(s, basis[i], basis[j]) for j in range(4)] for i in range(4)])
            self.assertApprox(gram, np.eye(4), atol=1e-10)
            coefficients = np.asarray(result["coefficients"])        # Julia: n x nev matrix of columns
            if coefficients.shape == (2, 4):                         # also accepted: a list of nev vectors
                coefficients = coefficients.T
            self.assertEqual(coefficients.shape, (4, 2))
            for k in range(2):
                c = coefficients[:, k]
                self.assertApprox(result["vectors"][k], sum(c[i] * basis[i] for i in range(4)), atol=1e-12)
                self.assertApprox(norm(h @ c - result["energies"][k] * c), result["residuals"][k], atol=1e-12)
        self.assertLess(bilinear["hermiticity_error"], 1e-9)         # as test/excitations.jl:124

    def test_spectrum_and_gauge_invariance(self):
        """test/excitations.jl:123-135 (bilinear assembly, Julia's default; max_point 30,
        tolerances 1e-12, p = 0.7, nev = 4): Hermiticity defect < 1e-9, residuals < 1e-10,
        sorted levels, metric-orthonormal Ritz vectors (1e-10), f(v1,v1) = E1 (1e-8), and
        the spectrum is invariant under a unitary auxiliary change of basis (2e-8)."""
        s, p = generic_state(max_point=30, atol=1e-12, rtol=1e-12), 0.7
        result = ref.spectrum(s, p, PHI4, nev=4, assembly="bilinear")
        energies, vectors = np.asarray(result["energies"]), result["vectors"]
        self.assertLess(result["hermiticity_error"], 1e-9)
        self.assertLess(max(result["residuals"]), 1e-10)
        self.assertTrue(np.all(np.diff(energies) >= 0))
        gram = np.array([[ref.tangent_metric(s, vectors[i], vectors[j]) for j in range(4)] for i in range(4)])
        self.assertApprox(gram, np.eye(4), atol=1e-10)
        pair = ref.make_pair(ref.make_context(s, p), vectors[0], vectors[0])
        self.assertApprox(ref.matrix_element(pair, PHI4), energies[0], atol=1e-8)
        g = np.linalg.qr(gaussian(3, 2, 2))[0]
        transformed = make_state(adj(g) @ s.R @ g, adj(g) @ s.K @ g, integrator=replace(s.integrator))
        rotated = ref.spectrum(transformed, p, PHI4, nev=4, assembly="bilinear")
        self.assertApprox(rotated["energies"], energies, atol=2e-8)

    def test_rank_deficient_density(self):
        """test/excitations.jl:136-140: for the dark state R = |0><1| the null metric
        direction is discarded, not regularized: rank 1 and a 2 x 2 matrix."""
        dark = make_state([[0, 1], [0, 0]], np.zeros((2, 2)))
        h = ref.dense_hamiltonian(dark, 0.0, FREE)
        self.assertEqual((h["coordinates"].metric_rank, np.shape(h["hamiltonian"]), len(h["basis"])), (1, (2, 2), 2))
        gram = np.array([[ref.tangent_metric(dark, a, b) for b in h["basis"]] for a in h["basis"]])
        self.assertApprox(gram, np.eye(2), atol=1e-10)

    def test_rotated_dark_state(self):
        """test/excitation_eigsolve.jl:160-166: a rotated dark state has a roundoff
        eigenvalue of rho; the iterative solver (metric_tol = 0, density gauge) keeps two
        directions and agrees with the dense matrix (1e-9), free model, 64 kinetic nodes."""
        g = np.linalg.qr(np.random.default_rng(3).standard_normal((2, 2)))[0]
        rotated = make_state(g.T @ np.array([[0, 1], [0, 0]]) @ g, np.zeros((2, 2)))
        dense = ref.dense_hamiltonian(rotated, 0.0, FREE, assembly="action", kinetic_nodes=64)
        reference = np.linalg.eigvalsh(dense["hamiltonian"])
        r = ref.eigsolve(rotated, 0.0, FREE, nev=1, kinetic_nodes=64)
        self.assertEqual((r["dimension"], r["metric_rank"]), (2, 1))
        self.assertApprox(r["energies"][0], reference[0], atol=1e-9)

    def test_input_and_numerical_failures(self):
        """Applicable lines of "Excitation input and numerical failures"
        (test/excitations.jl:147, 152-155, 158) and test/excitation_action.jl:27, 33 and
        test/excitation_eigsolve.jl:210; ArgumentError and DimensionMismatch are ValueError
        here. A Hermiticity defect above hermitian_tol is a SolverFailure
        (src/excitations.jl:185-186)."""
        s, w1 = vacuum(), np.ones((1, 1), dtype=complex)
        generic = generic_state()
        ctx = ref.make_context(generic, 0.0)
        failures = [lambda: models.phi4_terms(math.nan),
                    lambda: ref.spectrum(s, math.nan, FREE),
                    lambda: ref.spectrum(s, 0.0, FREE, nev=0),
                    lambda: ref.spectrum(s, 0.0, FREE, nev=2),
                    lambda: ref.dense_hamiltonian(s, 0.0, FREE, metric_tol=-1.0),
                    lambda: ref.dense_hamiltonian(s, 0.0, FREE, assembly="invalid"),
                    lambda: ref.make_pair(ref.make_context(s, 0.0), np.zeros((2, 2)), np.zeros((1, 1))),
                    lambda: ref.hamiltonian_action(ctx, w1, PHI4),
                    lambda: ref.hamiltonian_action(ctx, gaussian(1, 2, 2), FREE, kinetic_nodes=1)]
        for k, failure in enumerate(failures):
            with self.assertRaises(ValueError, msg=f"failure {k}"):
                failure()
        with self.assertRaises(SolverFailure):
            ref.dense_hamiltonian(generic, 0.3, FREE, hermitian_tol=1e-30)


class SectorsAndIterativeSolver(Case):
    """Item 5: port of "Sector Hamiltonians and the iterative solver"
    (test/excitation_eigsolve.jl:169-300) on z2_state(), ExcitationModel(:phi4,1.),
    32 kinetic nodes. Dense sector spectra are computed once per (momentum, sector)."""

    NODES = 32
    OPTIONS = DavidsonOptions(tol=1e-8, maxops=60, maxspace=10)

    @classmethod
    def setUpClass(cls):
        cls.s, cls.cache = z2_state(), {}

    def dense(self, p, sector):
        """eigvals(excitation_hamiltonian(s,p;model,sector,assembly=:action,kinetic_nodes=nodes).hamiltonian),
        with the Julia defaults parity = nothing for 'all' and :auto otherwise, original gauge."""
        if (p, sector) not in self.cache:
            h = ref.dense_hamiltonian(self.s, p, PHI4, sector=sector, assembly="action", kinetic_nodes=self.NODES)
            c = h["coordinates"]
            self.assertEqual((len(h["basis"]), c.sector, c.gauge), (16 if sector == "all" else 8, sector, "original"))
            self.assertEqual(c.parity is None, sector == "all")
            self.assertLess(h["hermiticity_error"], 1e-7)
            self.cache[p, sector] = np.linalg.eigvalsh(h["hamiltonian"])
        return self.cache[p, sector]

    def check_levels(self, p, sector, gauge, nev):
        """Body of the loop of test/excitation_eigsolve.jl:177-195, plus the bookkeeping of
        solve_levels (src/excitation_eigsolve.jl:377-400)."""
        s, reference = self.s, self.dense(p, sector)
        r = ref.eigsolve(s, p, PHI4, sector=sector, gauge=gauge, nev=nev, kinetic_nodes=self.NODES,
                         davidson_options=self.OPTIONS)
        self.assertTrue(r["converged"], f"stop_reason {r['stop_reason']}, residuals {r['residuals']}")
        self.assertApprox(r["energies"], reference[:nev], atol=1e-8)
        self.assertEqual(r["start"], "field" if sector == "odd" else "kinetic")
        self.assertLess(r["preconditioner"]["skew"], 1e-8)
        self.assertEqual(len(r["validation"]), nev)
        for k, v in enumerate(r["validation"]):
            self.assertApprox(v["standard_energy"], r["energies"][k], atol=1e-8)
            # Julia: both pairs at 1e-8. Tighter here, from the values measured over p in (0, .6),
            # both sectors and gauges, nev <= 2: |bilinear - standard| <= 6.0e-11 and
            # |refined bilinear - refined| <= 7.2e-12, while the two settings (windows 14 and
            # 24) differ by 3.8e-10 to 1.4e-8 for the first level: the pairs cannot be exchanged.
            self.assertLess(abs(v["bilinear_energy"] - v["standard_energy"]), 5e-10)
            self.assertLess(abs(v["refined_bilinear_energy"] - v["refined_energy"]), 1e-10)
            if k == 0:
                self.assertGreater(abs(v["refined_energy"] - v["standard_energy"]), 2e-10)
            self.assertLess(v["max_imaginary"], 1e-8)
            four = [v[key] for key in ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")]
            self.assertApprox(v["energy_spread"], max(four) - min(four), atol=1e-14)
        for k in range(nev):
            for l in range(nev):
                self.assertApprox(ref.tangent_metric(s, r["vectors"][k], r["vectors"][l]), float(k == l), atol=1e-10)
        pair = ref.make_pair(ref.make_context(s, p), r["vectors"][0], r["vectors"][0])
        self.assertApprox(ref.matrix_element(pair, PHI4, kinetic_nodes=self.NODES), r["energies"][0], atol=1e-8)
        c = r["coordinates"]
        self.assertEqual((r["sector"], r["gauge"], r["dimension"], len(c), r["metric_rank"]), (sector, gauge, 8, 8, 4))
        self.assertEqual((r["momentum"], r["nev"], r["kinetic_nodes"], r["refined_kinetic_nodes"]), (p, nev, 32, 64))
        self.assertEqual((r["stop_reason"], len(r["history"]), r["action_calls"]),
                         ("converged", r["numops"], r["numops"] + nev))    # one fresh standard action per level
        self.assertEqual((r["preconditioner"]["nodes"], r["preconditioner"]["floor"]), (8, 1.0))
        self.assertApprox(r["validated_energies"], [v["refined_bilinear_energy"] for v in r["validation"]],
                          atol=0.0, rtol=0.0)
        self.assertEqual((r["refined_converged"], len(r["corrections"])), (None, 0))
        self.assertEqual(bool(r["accepted"]), all(v["accepted"] for v in r["validation"]))
        for k in range(nev):
            w = tc.original_tangent(c, tc.tangent_matrix(c, r["coefficients"][k]))
            self.assertApprox(r["vectors"][k], w, atol=1e-13)
        return r

    def test_sector_spectra(self):
        """test/excitation_eigsolve.jl:174-176 at p = 0.6: the odd and even sector spectra
        together are the full spectrum (atol 1e-9 on the sorted vector)."""
        full, odd, even = (self.dense(0.6, sector) for sector in ("all", "odd", "even"))
        self.assertApprox(np.sort(np.r_[odd, even]), full, atol=1e-9)

    def test_eigsolve_odd_density(self):
        """test/excitation_eigsolve.jl:177-195 at p = 0.6 with nev = 1 (lowest level, 1e-8)."""
        self.check_levels(0.6, "odd", "density", 1)

    def test_eigsolve_odd_original(self):
        self.check_levels(0.6, "odd", "original", 1)

    def test_eigsolve_even_density(self):
        self.check_levels(0.6, "even", "density", 1)

    def test_eigsolve_even_original(self):
        self.check_levels(0.6, "even", "original", 1)

    @slow
    def test_julia_sector_loop(self):
        """test/excitation_eigsolve.jl:173-196 in full: p in (0, .6), sectors x gauges, nev = 2,
        DavidsonOptions(tol=1e-8, maxops=60, maxspace=10)."""
        for p in (0.0, 0.6):
            full, odd, even = (self.dense(p, sector) for sector in ("all", "odd", "even"))
            self.assertApprox(np.sort(np.r_[odd, even]), full, atol=1e-9)
            for sector in ("odd", "even"):
                for gauge in ("density", "original"):
                    with self.subTest(p=p, sector=sector, gauge=gauge):
                        self.check_levels(p, sector, gauge, 2)

    def test_momentum_and_integrator_reuse(self):
        """test/excitation_eigsolve.jl:197-206: a context moved to p = 0.6 shares the pinned
        Poisson factor and reproduces a fresh context (action atol 1e-10); new integration
        settings leave the base untouched; at p = 0 the three factors are one object. Also
        ExcitationContext(base; ...) branches of src/excitation_environments.jl:58-71."""
        s = self.s
        base, fresh = ref.make_context(s, 0.0), ref.make_context(s, 0.6)
        moved = ref.retarget(base, momentum=0.6)
        w = gaussian(5, 4, 4)
        self.assertApprox(ref.hamiltonian_action(moved, w, PHI4, kinetic_nodes=self.NODES),
                          ref.hamiltonian_action(fresh, w, PHI4, kinetic_nodes=self.NODES), atol=1e-10)
        self.assertIs(moved.poisson, base.poisson)
        self.assertEqual((moved.momentum, base.momentum, moved.tol), (0.6, 0.0, base.tol))
        self.assertIsNot(moved.ketfactor, base.poisson)
        y = gaussian(6, 16)
        self.assertApprox(lu_solve(moved.ketfactor, y), lu_solve(fresh.ketfactor, y), atol=1e-12)
        self.assertApprox(lu_solve(moved.brafactor, y), lu_solve(fresh.brafactor, y), atol=1e-12)
        tight = ref.retarget(base, integrator=IntegratorParameters(max_point=20))
        self.assertEqual((tight.state.integrator.max_point, base.state.integrator.max_point), (20, 14))
        self.assertIs(base.ketfactor, base.poisson)
        self.assertIs(base.brafactor, base.poisson)
        self.assertIs(tight.poisson, base.poisson)
        self.assertIs(tight.ketfactor, base.ketfactor)
        self.assertEqual(tight.momentum, 0.0)
        same, back = ref.retarget(moved), ref.retarget(moved, momentum=0.0)
        self.assertIs(same.ketfactor, moved.ketfactor)
        self.assertIs(same.brafactor, moved.brafactor)
        self.assertIs(back.ketfactor, base.poisson)
        self.assertIs(back.brafactor, base.poisson)
        with self.assertRaises(ValueError):
            ref.retarget(base, momentum=math.nan)
        loose = ref.make_context(s, 0.0, tol=3e-9)                        # base.tol, line 70
        self.assertEqual((ref.retarget(loose, momentum=0.3).tol, ref.retarget(loose, integrator=IntegratorParameters()).tol),
                         (3e-9, 3e-9))

    def test_fixed_kinetic_grid_is_resolved(self):
        """test/excitation_eigsolve.jl:207-209 with adaptive refinement replaced by node
        doubling: 256 fixed kinetic nodes agree with 512 (atol 1e-8). Only the kinetic terms
        depend on the grid."""
        base, w = ref.make_context(self.s, 0.6), gaussian(5, 4, 4)
        self.assertApprox(ref.hamiltonian_action(base, w, FREE, kinetic_nodes=256),
                          ref.hamiltonian_action(base, w, FREE, kinetic_nodes=512), atol=1e-8)

    def test_field_seed_and_preconditioner(self):
        """test/excitation_eigsolve.jl:229-244: W = 1 + [R,X] with (L* + ip)X = R + R'
        removes the left-gauge defect of the raw insertion (R, 1) at every p (1e-9) and is
        the field seed (1e-10); the coarse preconditioner matrix is the kinetic-only odd
        sector matrix on 8 nodes (1e-10) with floor 1. Added: the same matrix column by
        column from excitation_kinetic_action_sum on the coordinate basis (1e-10), skew,
        spectral data and `apply` (src/excitation_eigsolve.jl:181-201); only the kinetic
        terms of the model enter."""
        s = self.s
        for p in (0.0, 0.6):
            c = tc.make_coordinates(s)
            t = c.state
            ctx = ref.make_context(t, p)
            x = -ref.action_dual_poisson(ctx, t.R + adj(t.R), shifted=True)
            w = np.eye(4) + t.R @ x - x @ t.R
            self.assertLess(norm(t.R + t.Q @ x - x @ t.Q - 1j * p * x + adj(t.R) @ w), 1e-9)
            seed = ref.field_seed(ctx, c)
            self.assertEqual(np.shape(seed), (8,))
            self.assertApprox(tc.tangent_matrix(c, seed), w, atol=1e-10)
            pre = ref.kinetic_preconditioner(ctx, c, PHI4, nodes=8)
            k = np.asarray(pre["matrix"])
            kinetic = ref.dense_hamiltonian(s, p, FREE, sector="odd", gauge="density", metric_tol=0.0,
                                            assembly="action", kinetic_nodes=8)["hamiltonian"]
            self.assertApprox((k + adj(k)) / 2, kinetic, atol=1e-10)
            self.assertEqual((pre["floor"], pre["nodes"]), (1.0, 8))
            columns = []
            for basis in tc.tangent_basis(c):
                v, rw = ket_environment(ctx, basis)
                columns.append(tc.covector_coordinates(c, ref.kinetic_action_sum(ctx, basis, v, rw, FREE, 8)))
            self.assertApprox(k, np.column_stack(columns), atol=1e-10)
            self.assertLess(pre["skew"], 1e-8)
            self.assertApprox(pre["skew"], norm(k - adj(k)) / max(1, norm(k)), atol=1e-14)
            values, vectors = np.linalg.eigh((k + adj(k)) / 2)
            stored = np.asarray(pre["vectors"])
            self.assertApprox(pre["values"], values, atol=1e-10)
            self.assertApprox((stored * np.asarray(pre["values"])) @ adj(stored), (k + adj(k)) / 2, atol=1e-10)
            r = gaussian(9, 8)
            self.assertApprox(pre["apply"](r), vectors @ ((adj(vectors) @ r) / np.maximum(1.0, values)), atol=1e-10)
            self.assertApprox(ref.kinetic_preconditioner(ctx, c, FREE, nodes=8)["matrix"], k, atol=1e-13)
            self.assertEqual(ref.kinetic_preconditioner(ctx, c, PHI4, nodes=8, floor=2.5)["floor"], 2.5)
            # Unequal coefficients (lines 184, 196): only spec.kinetic terms enter, not the
            # quadratic monomial that the action sends to the same resolvent sum, and the
            # floor is the LARGEST kinetic coefficient.
            weighted = [(0.5, models.kinetic_spec(False)), (2.0, models.kinetic_spec(True)),
                        (0.37, models.monomial_spec(2)), (1.0, models.monomial_spec(4))]
            heavy = ref.kinetic_preconditioner(ctx, c, weighted, nodes=8)
            self.assertEqual(heavy["floor"], 2.0)
            columns = []
            for basis in tc.tangent_basis(c):
                v, rw = ket_environment(ctx, basis)
                columns.append(tc.covector_coordinates(c, ref.kinetic_action_sum(ctx, basis, v, rw, weighted[:2], 8)))
            self.assertApprox(heavy["matrix"], np.column_stack(columns), atol=1e-13)
            self.assertGreater(norm(heavy["matrix"] - k), 0.1)
            self.assertApprox(heavy["apply"](r), heavy["vectors"] @ ((adj(heavy["vectors"]) @ r) / np.maximum(2.0, heavy["values"])),
                              atol=1e-12)
            self.assertGreater(norm(heavy["apply"](r) - heavy["vectors"] @ ((adj(heavy["vectors"]) @ r) / np.maximum(0.5, heavy["values"]))),
                               1e-3)

    def test_operator(self):
        """ExcitationOperator (src/excitation_eigsolve.jl:146-172): x -> [F(W(x)) B]_sector,
        `calls` counts applications, and dot(x, op(y)) = dot(W(x), F(W(y))). For an odd W the
        even-sector covector of F vanishes (parity sectors decouple; exact up to roundoff)."""
        s, p = self.s, 0.6
        c = tc.make_coordinates(s)
        ctx = ref.make_context(c.state, p)
        op = ref.excitation_operator(ctx, c, PHI4, kinetic_nodes=self.NODES)
        x, y = gaussian(8, len(c)), gaussian(9, len(c))
        self.assertEqual(op.calls, 0)
        hy = op(y)
        self.assertEqual(op.calls, 1)
        f = ref.hamiltonian_action(ctx, tc.tangent_matrix(c, y), PHI4, kinetic_nodes=self.NODES)
        self.assertApprox(hy, tc.covector_coordinates(c, f), atol=1e-12)
        self.assertApprox(np.vdot(x, hy), np.vdot(tc.tangent_matrix(c, x), f), atol=1e-12)
        even = tc.make_coordinates(s, sector="even")
        self.assertApprox(even.whitening, c.whitening, atol=0.0, rtol=0.0)
        self.assertLess(norm(tc.covector_coordinates(even, f)), 1e-10)

    def test_request_validation(self):
        """test/excitation_eigsolve.jl:290-300 and check_levels_request
        (src/excitation_eigsolve.jl:260-300): cheap requests fail before any dense
        factorization. ArgumentError and DimensionMismatch are ValueError here; Julia's
        unknown keyword (kinetic_node=3) is a TypeError in Python; max_bond has no
        counterpart."""
        s = self.s
        requests = [dict(start=[gaussian(k, 8) for k in range(8)]),       # at most maxspace-1 = 7 starts
                    dict(start=gaussian(1, 7)),                           # wrong coordinate length
                    dict(refined_kinetic_nodes=1),
                    dict(validate=1),
                    dict(nev=2, davidson_options=DavidsonOptions(maxspace=2)),
                    dict(nev=9),                                          # sector dimension is 8
                    dict(refine_steps=1, nev=2),
                    dict(sector="even", start="field"),                   # the field insertion is odd
                    dict(start="kinetic", preconditioner="none"),
                    dict(start="invalid"),
                    dict(validation_tol=0.0)]
        for k, request in enumerate(requests):
            with self.assertRaises(ValueError, msg=f"request {k}: {sorted(request)}"):
                ref.eigsolve(s, 0.0, PHI4, **request)
        with self.assertRaises(TypeError):
            ref.eigsolve(s, 0.0, PHI4, kinetic_node=3)
        with self.assertRaises(ValueError):
            ref.eigsolve(s, math.nan, PHI4)

    def no_parity_levels(self, s, p, dimension, start):
        """Body of test/excitation_eigsolve.jl:276-283: dense reference, unvalidated solve in
        the full tangent space, and a warm start from the dense eigenvector. Julia passes the
        tangent W itself; the contract takes coordinate vectors, so W is converted with
        tangent_coordinates(c, coordinate_tangent(c, W)) as initial_vectors does."""
        dense = ref.spectrum(s, p, PHI4, nev=2, assembly="action", kinetic_nodes=self.NODES)
        r = ref.eigsolve(s, p, PHI4, nev=2, kinetic_nodes=self.NODES, validate=False)
        self.assertEqual((r["sector"], r["dimension"], bool(r["converged"])), ("all", dimension, True))
        self.assertApprox(r["energies"], dense["energies"], atol=1e-8)
        self.assertIsNone(r["validation"])
        self.assertIsNone(r["validated_energies"])
        self.assertEqual((r["action_calls"], r["refined_kinetic_nodes"], r["refined_converged"]), (r["numops"], None, None))
        self.assertEqual((len(r["corrections"]), bool(r["accepted"])), (0, True))   # accepted = converged without validation
        c = r["coordinates"]
        x0 = tc.tangent_coordinates(c, tc.coordinate_tangent(c, dense["vectors"][0]))
        warm = ref.eigsolve(s, p, PHI4, kinetic_nodes=self.NODES, validate=False, start=start(x0))
        self.assertEqual(warm["start"], "supplied")
        self.assertLessEqual(warm["numops"], 2)
        self.assertApprox(warm["energies"][0], dense["energies"][0], atol=1e-8)
        return dense, r

    def test_no_parity_and_supplied_start(self):
        """Fast D = 2 version of test/excitation_eigsolve.jl:272-283 (generic_state(), p = 0.5,
        full tangent space of dimension 4, start supplied as one bare coordinate vector), and
        the unpreconditioned random start (preconditioner 'none', start 'random')."""
        s = generic_state(max_point=14, n_quad=48, atol=1e-11, rtol=1e-10)
        dense, r = self.no_parity_levels(s, 0.5, 4, lambda x: x)
        self.assertEqual((r["start"], r["gauge"]), ("field", "density"))
        plain = ref.eigsolve(s, 0.5, PHI4, nev=2, kinetic_nodes=self.NODES, validate=False,
                             preconditioner="none", start="random")
        self.assertEqual((plain["start"], plain["preconditioner"]), ("random", None))
        self.assertApprox(plain["energies"], dense["energies"], atol=1e-8)

    def test_no_parity_d3(self):
        """test/excitation_eigsolve.jl:272-283: RandomState(3;seed=5,scaling=.5) becomes
        random_state(3, seed=5, scaling=0.5) (transfer gap 0.16, cond(rho) = 2.7); p = 0.5,
        dimension 9 > maxspace, start supplied as a list of one coordinate vector."""
        s = tuned(random_state(3, seed=5, scaling=0.5), max_point=14, n_quad=48, atol=1e-11, rtol=1e-10)
        self.no_parity_levels(s, 0.5, 9, lambda x: [x])

    def test_density_gauge_matrix_and_vectors(self):
        """excitation_hamiltonian with gauge=:density (src/excitations.jl:167-189): the matrix
        is built on the rotated state but the returned basis and Ritz vectors are in the
        gauge of the input state. Not a Julia test; the spectrum is compared with the
        original-gauge one at the solver tolerance 1e-8, and odd vectors commute with Z as
        in test/excitation_eigsolve.jl:131."""
        s, p = self.s, 0.6
        result = ref.spectrum(s, p, PHI4, nev=2, sector="odd", gauge="density", metric_tol=0.0, kinetic_nodes=self.NODES)
        c, basis, vectors = result["coordinates"], result["basis"], result["vectors"]
        self.assertEqual((c.sector, c.gauge, len(basis)), ("odd", "density", 8))
        self.assertApprox(result["energies"], self.dense(p, "odd")[:2], atol=1e-8)
        gram = np.array([[ref.tangent_metric(s, a, b) for b in basis] for a in basis])
        self.assertApprox(gram, np.eye(8), atol=1e-10)
        for k, w in enumerate(tc.tangent_basis(c)):
            self.assertApprox(basis[k], tc.original_tangent(c, w), atol=1e-13)
        z = np.diag([1.0, 1.0, -1.0, -1.0])
        for k in range(2):
            self.assertLess(norm(z @ vectors[k] @ z - vectors[k]), 1e-12)
            self.assertApprox(ref.tangent_metric(s, vectors[k], vectors[k]), 1.0, atol=1e-10)
        pair = ref.make_pair(ref.make_context(s, p), vectors[0], vectors[0])
        self.assertApprox(ref.matrix_element(pair, PHI4, kinetic_nodes=self.NODES), result["energies"][0], atol=1e-8)

    def test_validation_settings(self):
        """test/excitation_eigsolve.jl:245-250: nev = 4 with default options is converged and
        accepted, every level accepted and consistent, refined kinetic nodes doubled. The
        davidson and refined_integration fields are not among the contract keys and are
        checked only when present; the dense comparison (1e-8) is added."""
        r = ref.eigsolve(self.s, 0.0, PHI4, nev=4, kinetic_nodes=self.NODES)
        self.assertTrue(r["converged"] and r["accepted"])
        self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"], len(r["energies"])), (32, 64, 4))
        self.assertTrue(all(v["accepted"] and v["consistent"] for v in r["validation"]))
        self.assertApprox(r["energies"], self.dense(0.0, "odd")[:4], atol=1e-8)
        if "davidson" in r:
            self.assertEqual(r["davidson"].maxspace, 10)
        if "refined_integration" in r:
            fine = r["refined_integration"]
            self.assertEqual((fine.max_point, fine.n_quad), (24, 96))
            self.assertApprox(fine.atol, 1e-12)
            self.assertApprox(fine.rtol, 1e-11)

    def test_strict_validation_is_not_accepted(self):
        """test/excitation_eigsolve.jl:258-259 without excitation_gap: at validation_tol = 1e-16
        the level cannot be accepted (accepted_energies would throw SolverFailure)."""
        r = ref.eigsolve(self.s, 0.0, PHI4, kinetic_nodes=self.NODES, validation_tol=1e-16)
        self.assertFalse(r["accepted"] or r["validation"][0]["accepted"] or r["validation"][0]["consistent"])
        self.assertApprox(r["energies"][0], self.dense(0.0, "odd")[0], atol=1e-8)

    def test_validation_contexts_and_grids(self):
        """The four validation energies (src/excitation_eigsolve.jl:357-359): fresh actions on
        (standard context, n kinetic nodes) and (refined context, 2n), and the bilinear form
        on the same two. With n = 4 the two settings are 1.2e-3 apart, while the action and
        the bilinear form of one setting agree to 3e-12 and 9e-14 (asserted at 1e-10; a
        refined bilinear energy on the standard window would be 8e-9 off, on n nodes 1e-3)."""
        r = ref.eigsolve(self.s, 0.6, PHI4, kinetic_nodes=4, davidson_options=self.OPTIONS)
        v = r["validation"][0]
        self.assertEqual((r["converged"], r["kinetic_nodes"], r["refined_kinetic_nodes"]), (True, 4, 8))
        self.assertLess(abs(v["bilinear_energy"] - v["standard_energy"]), 1e-10)
        self.assertLess(abs(v["refined_bilinear_energy"] - v["refined_energy"]), 1e-10)
        self.assertGreater(abs(v["refined_energy"] - v["standard_energy"]), 1e-4)
        self.assertApprox(v["energy_spread"], abs(v["refined_energy"] - v["standard_energy"]), atol=1e-10)
        self.assertFalse(v["consistent"] or v["accepted"] or r["accepted"])
        self.assertLess(v["standard_residual"], 1e-7)

    def test_refine_steps(self):
        """test/excitation_eigsolve.jl:251-257: a loose search (tol 1e-2, 3 operator calls,
        random start) followed by refine_steps = 3 corrections under the refined action.
        Added, for the correction loop of src/excitation_eigsolve.jl:336-353: here every
        correction is accepted (fresh refined residuals 0.37, 0.057, 0.016, 0.0036), each costs
        one standard action op(d), and step 1 is rebuilt from the last Ritz vector with the
        Julia formulas (1e-10). The validation row (lines 363-370) is then the discriminating
        one: four consistent energies with a residual above both limits, so not accepted."""
        ritz = []
        loose = ref.eigsolve(self.s, 0.0, PHI4, kinetic_nodes=self.NODES, refine_steps=3, start="random",
                             davidson_options=DavidsonOptions(tol=1e-2, maxops=3, stall_window=0),
                             callback=lambda row, u, au: ritz.append(u[:, 0].copy()))
        corrections, validation = loose["corrections"], loose["validation"][0]
        self.assertTrue(len(corrections) > 0 and corrections[0]["step"] == 0)
        self.assertApprox(loose["energies"][0], validation["refined_energy"], atol=1e-10)
        self.assertApprox(loose["residuals"][0], validation["refined_residual"], atol=1e-10)
        self.assertLessEqual(loose["residuals"][0], corrections[0]["residual"])
        self.assertEqual(bool(loose["converged"]), loose["stop_reason"] == "converged")
        self.assertIsInstance(loose["refined_converged"], (bool, np.bool_))
        self.assertEqual(loose["start"], "random")
        residuals = [correction["residual"] for correction in corrections]
        self.assertEqual([(correction["step"], correction["accepted"]) for correction in corrections],
                         [(step, True) for step in range(4)])
        self.assertTrue(all(b < a for a, b in zip(residuals, residuals[1:])) and residuals[-1] < 1e-2 * residuals[0])
        self.assertEqual((loose["energies"][0], loose["residuals"][0]), (corrections[-1]["energy"], residuals[-1]))
        self.assertEqual(loose["action_calls"], loose["numops"] + 3 + 1)   # op(d) per step, one fresh validation action
        self.assertEqual((loose["refined_converged"], loose["converged"]), (True, False))   # 0.0036 <= tol = 1e-2
        self.assertLess(max(validation["energy_spread"], validation["max_imaginary"]), 1e-7)
        self.assertTrue(validation["consistent"] and not validation["residual_target_met"])
        self.assertFalse(validation["accepted"] or loose["accepted"])
        c, x = loose["coordinates"], ritz[-1]
        ctx = ref.make_context(c.state, 0.0)
        op = ref.excitation_operator(ctx, c, PHI4, kinetic_nodes=self.NODES)
        reference = ref.excitation_operator(ref.retarget(ctx, integrator=loose["refined_integration"]), c, PHI4,
                                            kinetic_nodes=2 * self.NODES)
        apply = ref.kinetic_preconditioner(ctx, c, PHI4, nodes=8)["apply"]
        h = reference(x)
        energy = (np.vdot(x, h) / np.vdot(x, x)).real
        r = h - energy * x
        self.assertApprox([corrections[0]["energy"], corrections[0]["residual"]], [energy, norm(r) / norm(x)], atol=1e-12)
        d = apply(r)
        d = d - x * np.vdot(x, d)
        d = d / norm(d)
        shifted = op(d) - energy * d
        trial = x - np.vdot(shifted, r) / np.vdot(shifted, shifted).real * d
        trial = trial / norm(trial)
        h = reference(trial)
        energy = np.vdot(trial, h).real
        self.assertApprox([corrections[1]["energy"], corrections[1]["residual"]], [energy, norm(h - energy * trial)],
                          atol=1e-10)

    @slow
    def test_restarts(self):
        """test/excitation_eigsolve.jl:260-271: the 18-dimensional odd sector of a D = 6 Z2
        state forces restarts; fresh and thick restarts agree with the dense matrix (1e-8).
        RandomStateZ2(6;seed=21,scaling=.5) becomes random_state(6, seed=21, scaling=0.5,
        z2=True): transfer gap 0.11, cond(rho) = 40. About 45 s; kept in the slow tier with
        the full sector loop (about 65 s)."""
        s6 = tuned(random_state(6, seed=21, scaling=0.5, z2=True), max_point=14, n_quad=48, atol=1e-11, rtol=1e-10)
        odd6 = np.linalg.eigvalsh(ref.dense_hamiltonian(s6, 0.0, PHI4, sector="odd", assembly="action",
                                                        kinetic_nodes=self.NODES)["hamiltonian"])
        for options in (DavidsonOptions(tol=1e-8, maxspace=6, stall_window=0),
                        DavidsonOptions(tol=1e-8, maxspace=8, retain=3, refresh=False, stall_window=0)):
            r = ref.eigsolve(s6, 0.0, PHI4, nev=2, kinetic_nodes=self.NODES, davidson_options=options, validate=False)
            self.assertTrue(r["converged"], f"stop_reason {r['stop_reason']}, residuals {r['residuals']}")
            self.assertEqual(r["dimension"], 18)
            self.assertTrue(any(row["cycle"] > 1 for row in r["history"]))
            self.assertApprox(r["energies"], odd6[:2], atol=1e-8)


class IntegrationAdditions(Case):
    """Added at integration with dense_reference.py open (the classes above were written
    without reading it): the checks of the oracle author's smoke script that no test above
    covered, each tied to the Julia line that defines the behaviour, and at the final check
    the solver rules that no assertion pinned. generic_state() with the window of the solver
    tests, p = 0.5, 32 kinetic nodes and no validation, unless a test names another state."""

    NODES = 32

    @classmethod
    def setUpClass(cls):
        cls.s, cls.p = generic_state(max_point=14, n_quad=48, atol=1e-11, rtol=1e-10), 0.5
        cls.dense = ref.spectrum(cls.s, cls.p, PHI4, nev=2, assembly="action", kinetic_nodes=cls.NODES)

    def solve(self, **kw):
        return ref.eigsolve(self.s, self.p, PHI4, kinetic_nodes=self.NODES, validate=False, **kw)

    def test_tangent_starts(self):
        """test/excitation_eigsolve.jl:280-283 literally: start=dense3.vectors[1] is a D x D
        tangent W in the gauge of the INPUT state, mapped to the (density-gauge) coordinates by
        tangent_coordinates(c, coordinate_tangent(c, W)) (src/excitation_eigsolve.jl:213-215);
        a collection may mix tangents and coordinate vectors (line 235). The contract names
        coordinate vectors only: tangents are the oracle's Julia-faithful extension."""
        dense = self.dense
        warm = self.solve(start=dense["vectors"][0])
        self.assertEqual((warm["start"], warm["gauge"]), ("supplied", "density"))
        self.assertLessEqual(warm["numops"], 2)
        self.assertApprox(warm["energies"][0], dense["energies"][0], atol=1e-8)
        c = warm["coordinates"]
        self.assertGreater(norm(c.rotation - np.eye(2)), 0.1)             # the gauge map is not trivial here
        x1 = tc.tangent_coordinates(c, tc.coordinate_tangent(c, dense["vectors"][1]))
        both = self.solve(nev=2, start=[dense["vectors"][0], x1])
        self.assertEqual((both["start"], bool(both["converged"])), ("supplied", True))
        self.assertLessEqual(both["numops"], 3)
        self.assertApprox(both["energies"], dense["energies"], atol=1e-8)
        with self.assertRaises(ValueError):                               # DimensionMismatch, lines 285-286
            self.solve(start=np.ones((3, 3)))

    def test_callable_preconditioner(self):
        """preconditioner may be a function (src/excitation_eigsolve.jl:275, 314): no coarse
        kinetic matrix is built, so start = 'auto' is the field seed alone (lines 218, 225)."""
        r = self.solve(nev=2, preconditioner=lambda residual: 0.5 * residual)
        self.assertEqual((r["start"], r["preconditioner"], bool(r["converged"])), ("field", None, True))
        self.assertEqual(r["action_calls"], r["numops"])
        self.assertApprox(r["energies"], self.dense["energies"], atol=1e-8)

    def test_explicit_start_kinds(self):
        """start = :field and :kinetic requested by name (src/excitation_eigsolve.jl:219-231);
        above they are only reached through :auto."""
        for start in ("field", "kinetic"):
            r = self.solve(start=start)
            self.assertEqual((r["start"], bool(r["converged"])), (start, True))
            self.assertApprox(r["energies"][0], self.dense["energies"][0], atol=1e-8)

    def test_refinement_without_validation(self):
        """refine_steps > 0 with validate = false (src/excitation_eigsolve.jl:325-353, 377-378):
        the refined operator exists (twice the kinetic nodes), every accepted correction lowers
        the fresh refined residual and a rejection ends the loop, the first level reports the
        last accepted correction, each correction costs one standard action op(d) (line 342),
        and accepted = converged && refined_converged. The refined window is 10 longer, so the
        refined energy is not compared with the standard dense matrix."""
        options, ritz = DavidsonOptions(tol=1e-9, maxops=40), []
        r = self.solve(refine_steps=2, davidson_options=options, callback=lambda row, u, au: ritz.append(u[:, 0].copy()))
        corrections = r["corrections"]
        self.assertIsNone(r["validation"])
        self.assertIsNone(r["validated_energies"])
        self.assertGreater(len(corrections), 1, "no correction was attempted: nothing is tested")
        self.assertEqual([c["step"] for c in corrections], list(range(len(corrections))))
        self.assertTrue(all(c["accepted"] for c in corrections[:-1]))
        kept = [c for c in corrections if c["accepted"]]
        self.assertTrue(all(b["residual"] < a["residual"] for a, b in zip(kept, kept[1:])))
        self.assertEqual((r["energies"][0], r["residuals"][0]), (kept[-1]["energy"], kept[-1]["residual"]))
        self.assertEqual(r["action_calls"], r["numops"] + len(corrections) - 1)
        self.assertEqual((r["kinetic_nodes"], r["refined_kinetic_nodes"]), (32, 64))
        self.assertEqual(bool(r["refined_converged"]), r["residuals"][0] <= options.tol)
        self.assertEqual(bool(r["accepted"]), bool(r["converged"]) and bool(r["refined_converged"]))
        # Step 0 is the Davidson vector under the refined action (lines 336-337), rebuilt here
        # from refined_integration (lines 239-241: window + 10, twice n_quad, tolerances / 10).
        fine, c, x = r["refined_integration"], r["coordinates"], ritz[-1]
        self.assertEqual((fine.max_point, fine.n_quad), (24, 96))
        self.assertApprox([fine.atol, fine.rtol], [1e-12, 1e-11], atol=1e-24)
        reference = ref.excitation_operator(ref.retarget(ref.make_context(c.state, self.p), integrator=fine), c, PHI4,
                                            kinetic_nodes=64)
        h = reference(x)
        energy = (np.vdot(x, h) / np.vdot(x, x)).real
        self.assertApprox(corrections[0]["energy"], energy, atol=1e-12)
        self.assertApprox(corrections[0]["residual"], norm(h - energy * x) / norm(x), atol=1e-12)

    def test_refinement_stops_at_the_target(self):
        """Line 339: no correction is attempted once the fresh refined residual meets the
        Davidson tolerance. The search exhausts the four-dimensional space (residual 7e-10)
        and the refined residual of step 0 is 3.7e-7, below tol = 1e-5: the loop stops there
        (at tol = 1e-9, in the test above, corrections are attempted)."""
        r = self.solve(refine_steps=2, davidson_options=DavidsonOptions(tol=1e-5))
        self.assertLess(r["history"][-1]["residual"], 1e-8)
        self.assertEqual([(c["step"], c["accepted"]) for c in r["corrections"]], [(0, True)])
        self.assertLessEqual(r["corrections"][0]["residual"], 1e-5)
        self.assertEqual((r["refined_converged"], bool(r["accepted"]), r["action_calls"]), (True, True, r["numops"]))

    def test_validation_row_logic(self):
        """The validation row and the acceptance (src/excitation_eigsolve.jl:356-371, 377-378) on
        the vacuum level, with forged fresh residuals, imaginary parts and one shifted energy:
        the residual is the LARGER of the standard and refined ones, residual_target_met
        compares it with validation_tol and `accepted` with acceptance_residual, and
        `consistent` needs the energy spread and all four imaginary parts within
        validation_tol. Returned as (residual_target_met, consistent, accepted)."""
        fresh, bilinear = ref.fresh_level, ref.bilinear_level

        def row(standard=0.0, refined=0.0, imaginary=(0.0, 0.0, 0.0, 0.0), shift=0.0, **kw):
            def forged_fresh(op, x):
                fine, level = op.kinetic_nodes == 512, fresh(op, x)
                return dict(level, residual=refined if fine else standard, imaginary=imaginary[fine],
                            energy=level["energy"] + shift * fine)

            def forged_bilinear(ctx, op, x, *, kinetic_nodes):
                return dict(bilinear(ctx, op, x, kinetic_nodes=kinetic_nodes), imaginary=imaginary[2 + (kinetic_nodes == 512)])

            with mock.patch.object(ref, "fresh_level", forged_fresh), mock.patch.object(ref, "bilinear_level", forged_bilinear):
                r = ref.eigsolve(vacuum(), 0.7, FREE, **kw)
            v = r["validation"][0]
            self.assertEqual((r["converged"], r["accepted"]), (True, v["accepted"]))
            self.assertApprox(v["max_imaginary"], max(abs(z) for z in imaginary), atol=1e-15)
            self.assertApprox(v["energy_spread"], abs(shift), atol=1e-11)
            return v["residual_target_met"], v["consistent"], v["accepted"]

        self.assertEqual(row(), (True, True, True))
        self.assertEqual(row(standard=3e-8, refined=3e-8), (True, True, True))
        for residuals in (dict(standard=3e-6), dict(refined=3e-6)):
            self.assertEqual(row(**residuals), (False, True, True), residuals)
        for residuals in (dict(standard=3e-5), dict(refined=3e-5)):
            self.assertEqual(row(**residuals), (False, True, False), residuals)
            self.assertEqual(row(acceptance_residual=1e-4, **residuals), (False, True, True), residuals)
        self.assertEqual(row(refined=3e-6, validation_tol=1e-5), (True, True, True))
        for k in range(4):
            for value in (1e-3, -1e-3):
                imaginary = tuple(value * (j == k) for j in range(4))
                self.assertEqual(row(imaginary=imaginary), (True, False, False), imaginary)
            self.assertEqual(row(imaginary=imaginary, validation_tol=1e-2, acceptance_residual=1e-1), (True, True, True))
        self.assertEqual(row(shift=1e-3), (True, False, False))
        self.assertEqual(row(shift=-1e-3, validation_tol=1e-2), (True, True, True))

    def test_preconditioner_report(self):
        """preconditioner_nodes reaches kinetic_preconditioner and the result reports its
        nodes, skew, floor and LOWEST coarse eigenvalue (src/excitation_eigsolve.jl:312-313,
        390-391)."""
        r = self.solve(preconditioner_nodes=4)
        c, report = r["coordinates"], r["preconditioner"]
        ctx = ref.make_context(c.state, self.p)
        four, eight = (ref.kinetic_preconditioner(ctx, c, PHI4, nodes=nodes) for nodes in (4, 8))
        self.assertEqual((set(report), report["nodes"], report["floor"]), ({"nodes", "skew", "floor", "minimum"}, 4, 1.0))
        self.assertApprox([report["minimum"], report["skew"]], [four["values"].min(), four["skew"]], atol=1e-13)
        self.assertGreater(four["values"].max() - four["values"].min(), 0.1)
        self.assertGreater(abs(eight["values"].min() - four["values"].min()), 1e-6)
        self.assertEqual(self.solve()["preconditioner"]["nodes"], 8)

    def test_initial_vectors(self):
        """initial_vectors (src/excitation_eigsolve.jl:211-237) on z2_state() at p = 0.6, nev = 3:
        :field (and :auto in the odd sector) is the field seed followed by the nev-1 LOWEST
        coarse kinetic modes; :kinetic (and :auto in the even sector, where the field
        insertion vanishes) is the nev lowest modes; :random is nev vectors drawn in turn
        from default_rng(1234) (Julia: MersenneTwister(1234)), also the fallback of :auto
        without a kinetic preconditioner."""
        s, p, nev = z2_state(), 0.6, 3
        for sector in ("odd", "even"):
            c = tc.make_coordinates(s, sector=sector)
            ctx = ref.make_context(c.state, p)
            op = ref.excitation_operator(ctx, c, PHI4, kinetic_nodes=self.NODES)
            pre = ref.kinetic_preconditioner(ctx, c, PHI4, nodes=8)
            self.assertTrue(np.all(np.diff(pre["values"]) > 0))
            modes, seed = [pre["vectors"][:, k] for k in range(nev)], ref.field_seed(ctx, c)
            rng = np.random.default_rng(1234)
            noise = [(rng.standard_normal(len(c)) + 1j * rng.standard_normal(len(c))) / math.sqrt(2) for _ in range(nev)]
            if sector == "odd":
                self.assertGreater(norm(seed), 0.5)
                expected = [("auto", pre, [seed] + modes[:2], "field"), ("field", pre, [seed] + modes[:2], "field"),
                            ("auto", None, [seed], "field"), ("field", None, [seed], "field")]
            else:
                self.assertLess(norm(seed), 1e-6)
                expected = [("auto", pre, modes, "kinetic"), ("auto", None, noise, "random")]
                with self.assertRaises(ValueError):
                    ref.initial_vectors(op, "field", nev, pre)
            expected += [("kinetic", pre, modes, "kinetic"), ("random", pre, noise, "random"), ("random", None, noise, "random")]
            for start, preconditioner, vectors, kind in expected:
                found, label = ref.initial_vectors(op, start, nev, preconditioner)
                self.assertEqual((label, len(found)), (kind, len(vectors)), f"{sector} {start}")
                for a, b in zip(found, vectors):
                    self.assertApprox(a, b, atol=1e-14, msg=f"{sector} {start}")
            with self.assertRaises(ValueError):
                ref.initial_vectors(op, "kinetic", nev, None)
            self.assertEqual(len(ref.initial_vectors(op, "kinetic", 20, pre)[0]), len(c))   # at most the dimension

    def test_supplied_refined_settings(self):
        """refined=IntegratorParameters(...) replaces refined_integration(...) (line 326): the
        refined context owns a copy of it and both refined numbers of the validation row are
        evaluated with it (window 18 instead of the default 24: the refined energy moves by
        more than the comparison tolerance)."""
        refined = IntegratorParameters(n_quad=64, max_point=18.0, atol=1e-12, rtol=1e-11)
        r = ref.eigsolve(self.s, self.p, PHI4, kinetic_nodes=self.NODES, refined=refined)
        self.assertEqual(r["refined_integration"], refined)
        self.assertIsNot(r["refined_integration"], refined)
        self.assertEqual((r["integration"].max_point, r["integration"].n_quad, r["refined_kinetic_nodes"]), (14, 48, 64))
        c, x, v = r["coordinates"], r["coefficients"][0], r["validation"][0]
        ctx = ref.make_context(c.state, self.p)

        def levels(integrator):
            fine = ref.retarget(ctx, integrator=integrator)
            reference = ref.excitation_operator(fine, c, PHI4, kinetic_nodes=64)
            return [ref.fresh_level(reference, x)["energy"], ref.bilinear_level(fine, reference, x, kinetic_nodes=64)["energy"]]

        self.assertApprox([v["refined_energy"], v["refined_bilinear_energy"]], levels(refined), atol=1e-13)
        default = levels(ref.refined_integration(ctx.state.integrator))
        self.assertGreater(abs(default[0] - v["refined_energy"]), 1e-11)

    def test_parity_tolerance_is_forwarded(self):
        """parity_tol reaches TangentCoordinates from excitation_hamiltonian and from
        excitation_eigsolve (src/excitations.jl:161, 167; src/excitation_eigsolve.jl:469, 473):
        z2_state() with K[1,3] and K[3,1] moved by 1e-9 (parity error 2.7e-9) has sectors only
        with a looser tolerance. Free model on 8 kinetic nodes."""
        base = z2_state()
        k = base.K.copy()
        k[0, 2] += 1e-9
        k[2, 0] += 1e-9
        s = make_state(base.R, k, integrator=replace(base.integrator))
        with self.assertRaises(ValueError):
            ref.dense_hamiltonian(s, 0.0, FREE, sector="odd", kinetic_nodes=8)
        h = ref.dense_hamiltonian(s, 0.0, FREE, sector="odd", kinetic_nodes=8, parity_tol=1e-6)
        self.assertEqual((len(h["basis"]), h["sector"]), (8, "odd"))
        r = ref.eigsolve(s, 0.0, FREE, kinetic_nodes=8, validate=False, parity_tol=1e-6)
        self.assertEqual((r["sector"], r["dimension"], bool(r["converged"])), ("odd", 8, True))
        self.assertTrue(1e-9 < r["parity_error"] < 1e-8)
        self.assertApprox(r["energies"][0], np.linalg.eigvalsh(h["hamiltonian"])[0], atol=1e-7)
        strict = ref.eigsolve(s, 0.0, FREE, kinetic_nodes=8, validate=False)
        self.assertEqual((strict["sector"], strict["dimension"], strict["parity"]), ("all", 16, None))
        # In the even sector the field insertion is now small but not zero: below the 1e-6 of
        # line 225 the automatic start is the kinetic one.
        even = ref.eigsolve(s, 0.0, FREE, kinetic_nodes=8, validate=False, parity_tol=1e-6, sector="even")
        c = even["coordinates"]
        self.assertTrue(1e-12 < norm(ref.field_seed(ref.make_context(c.state, 0.0), c)) < 1e-7)
        self.assertEqual((even["start"], even["sector"], bool(even["converged"])), ("kinetic", "even", True))

    def test_context_is_a_snapshot(self):
        """ExcitationContext deep-copies the state (src/excitation_environments.jl:21);
        ExcitationContext(base; ...) shares the matrices and the generator read-only but owns
        a copy of the settings even when they are unchanged (share_state, lines 46-48, 61)."""
        s = generic_state(max_point=14, n_quad=48)
        ctx = ref.make_context(s, 0.3)
        self.assertIsNot(ctx.state, s)
        self.assertIsNot(ctx.state.R, s.R)
        s.integrator.max_point = 99.0
        self.assertEqual(ctx.state.integrator.max_point, 14)
        same = ref.retarget(ctx)
        self.assertIs(same.state.R, ctx.state.R)
        self.assertIs(same.state.rho, ctx.state.rho)
        self.assertIs(same.transfer, ctx.transfer)
        self.assertIsNot(same.state.integrator, ctx.state.integrator)
        self.assertEqual(same.state.integrator, ctx.state.integrator)

    def test_remaining_failures(self):
        """Error paths not asserted above; ArgumentError and DimensionMismatch are ValueError.
        The oracle's bond limit is 16 for every entry point (Julia: max_bond = 16 for the dense
        routines, 64 for excitation_eigsolve)."""
        s, p = self.s, self.p
        ctx, w, nan = ref.make_context(s, p), gaussian(2, 2, 2), np.full((2, 2), np.nan)
        c = tc.make_coordinates(s)
        cctx = ref.make_context(c.state, p)
        big = states.State(*(np.zeros((17, 17), dtype=complex),) * 4)     # rejected before any factorization
        other = ref.make_context(random_state(3, seed=5, scaling=0.5), 0.0)
        failures = [lambda: ref.make_context(big, 0.0),                   # src/excitation_environments.jl:20
                    lambda: ref.dense_hamiltonian(big, 0.0, FREE),        # src/excitations.jl:166
                    lambda: ref.eigsolve(big, 0.0, FREE),                 # src/excitation_eigsolve.jl:471
                    lambda: ref.hamiltonian_action(ctx, nan, PHI4),       # src/excitation_action.jl:208
                    lambda: ref.make_pair(ctx, w, nan),                   # src/excitation_environments.jl:106
                    lambda: ref.make_pair(ctx, w, np.ones((3, 3))),       # src/excitation_environments.jl:105
                    lambda: ref.dense_hamiltonian(s, p, PHI4, hermitian_tol=0.0),    # src/excitations.jl:163
                    lambda: ref.dense_hamiltonian(s, p, PHI4, sector="odd"),         # src/excitation_eigsolve.jl:76-77
                    lambda: self.solve(preconditioner="other"),           # src/excitation_eigsolve.jl:275-276
                    lambda: self.solve(preconditioner=None),              # `nothing` is not :none, same lines
                    lambda: self.solve(refined=IntegratorParameters(n_quad=4)),      # :292-293, src/observables.jl:66-67
                    lambda: self.solve(refined="fine"),                   # :292-293
                    lambda: self.solve(preconditioner_nodes=1),           # :296
                    lambda: self.solve(acceptance_residual=0.0),          # :297-298
                    lambda: ref.kinetic_preconditioner(cctx, c, [(1.0, models.monomial_spec(4))]),   # :184-185
                    lambda: ref.kinetic_preconditioner(cctx, c, PHI4, floor=0.0),    # :196-197
                    lambda: ref.excitation_operator(other, c, PHI4)]      # :159
        for k, failure in enumerate(failures):
            with self.assertRaises(ValueError, msg=f"failure {k}"):
                failure()
        # Not Julia: excitation_kinetic_sum (src/excitations.jl:64-86) sums the two mixed source
        # orders whatever the spec; the oracle refuses a non-kinetic spec (contract: "kinetic
        # specs only") instead of returning half of a :phi^2: element.
        with self.assertRaises(ValueError):
            ref.kinetic_sum(ref.make_pair(ctx, w, w), [(1.0, models.monomial_spec(2))], 8)


if __name__ == "__main__":
    unittest.main()
