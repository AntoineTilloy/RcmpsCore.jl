"""Tests of packed.py: the Z2-packed term tables of the action ODE system.

The reference is potential.ActionSystem (every channel live, unpacked); the oracle of
the complete action is dense_reference.hamiltonian_action. Grid: :phi^n: for n in
(2, 3, 4), both sectors, p in (0, 0.6), S = 1 and S = exp(linspace(-2.5, 1.5, D)), on
states.random_state(4 and 6, z2=True) (cond(rho) = 9.0 and 22.9, transfer gap 0.71 and
0.99).

RightHandSide  (1) packed rhs at random packed states against ActionSystem.rhs on the
               unpacked state; the dropped channels are stationary and y_{C-1} is unused.
               Independently of ActionSystem: against action_rhs! in physical variables
               (test_potential.julia_rhs), also for D = 2, n = 5, p < 0 and S = sqrt(diag
               rho), and on the stored D = 16 Julia state (cond(rho) = 3.3e7).
Packing        (2) pack/unpack and the block layout, with classes derived here by
               counting factors of R.
Integration    (3) trajectories, PotentialAction.bulk and HamiltonianAction against the
               unpacked system and the oracle. Routine runs (about a minute) take a subset
               of the grid; RCMPS_EXCITATION_SLOW=1 runs all of it (about five minutes).
               The adaptive comparisons measure the step control, whose error norm differs
               between the layouts; test_fixed_steps isolates the packing on one common
               step sequence. Also the chunked pullback, :phi^5: at p < 0 and D = 2, and
               the driver on the packed system (nev = 2).
Tables         (4) the flat tables() contract through `table_rhs`, a loop interpreter
               written here that reads nothing but the flat arrays and the constants.
Rejections     (5) non-Z2 states, tangents without a class or not gauge fixed, specs
               that are not chains.

RCMPS_EXCITATION_REPORT=1 prints the largest measured errors at the end.
"""
from __future__ import annotations

import copy
import functools
import itertools
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coordinates as tc
import dense_reference as ref
import eigsolve as es
import packed as pk
import states
from hamiltonian import HamiltonianAction
from kinetic import TangentFrame
from models import Edge, Observable, kinetic_spec, monomial_spec, phi4_terms, quadrature, transformed_source
from ode import integrate_host
from packed import PackedActionSystem, has_class
from potential import ActionSystem, PotentialAction
from scaled import ScaledState, stack
from states import IntegratorParameters, adj, make_state, random_state
from test_potential import julia_rhs, to_hat    # action_rhs! written out in physical variables

slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (about five minutes)")
norm = np.linalg.norm
DIMS, DEGREES, MOMENTA, SCALINGS, SECTORS = (4, 6), (2, 3, 4), (0.0, 0.6), ("none", "explicit"), ("odd", "even")
GRID = list(itertools.product(DIMS, SCALINGS, MOMENTA, SECTORS, DEGREES))
# ODE tolerances ten times below atol 1e-13, rtol 1e-12. There the comparisons of Integration
# measure the integrators (some 400 steps each): over the grid the packed and unpacked bulk
# actions differ by up to 4.1e-11, and the ODE terms of the oracle are 5.3e-11 from their
# converged values (D = 6, S = 1, p = 0.6).
INTEGRATOR = IntegratorParameters(n_quad=32, max_point=16.0, atol=1e-14, rtol=1e-13)
TOL = 1e-13                                             # relative residual of the linear solves
TERMS = phi4_terms(1.3) + [(0.7, monomial_spec(3))]     # packed ODE terms: :phi^4: and :phi^3:
NODES = 16
BULK = [(4, "explicit", 0.6, 2), (4, "none", 0.0, 3), (6, "explicit", 0.6, 4)]
ORACLE = [(4, "none", 0.0), (6, "explicit", 0.6)]
PREFIX = ("x{}", "y{}", "a1_{}", "a0_{}")
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "test", "fixtures",
                       "phi4_g2_D16_z2_state.jld2")
BLOCKS = {2: 12, 3: 18, 4: 26}        # packed blocks and products per right-hand side of :phi^n:
PRODUCTS = {2: 78, 3: 128, 4: 190}
WORST = {}


def record(name, value, msg=""):
    if float(value) >= WORST.get(name, (0.0, ""))[0]:
        WORST[name] = (float(value), msg)


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for name, (value, msg) in WORST.items():
            print(f"{name}: {value:.3e}  [{msg.strip()}]", file=sys.stderr)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


@functools.lru_cache(maxsize=None)
def scaled(dim, scaling):
    state = random_state(dim, z2=True).with_integrator(INTEGRATOR)
    return ScaledState(state, scaling if scaling in ("none", "density") else np.exp(np.linspace(-2.5, 1.5, dim)))


def class_tangents(dim):
    """Random hat tangents: block-diagonal ('odd' sector) and block-off-diagonal ('even')."""
    w, d = gaussian(5, dim, dim), dim // 2
    odd = w.copy()
    odd[:d, d:] = odd[d:, :d] = 0
    return {"odd": odd, "even": w - odd}


@functools.lru_cache(maxsize=None)
def frame(dim, scaling, p):
    w = class_tangents(dim)
    return TangentFrame(scaled(dim, scaling), p, stack([w[sector] for sector in SECTORS]), tol=TOL)


def tangent(dim, scaling, p, sector):
    f, t = frame(dim, scaling, p), SECTORS.index(sector)
    return tuple(np.ascontiguousarray(m[:, t, :]) for m in (f.wh, f.vh, f.rw))


def systems(dim, scaling, p, sector, n):
    ss, spec, w = scaled(dim, scaling), monomial_spec(n), tangent(dim, scaling, p, sector)
    return PackedActionSystem(ss, p, spec, w), ActionSystem(ss, p, spec, w)


def random_packed(system, seed, lead=()):
    """Random complex blocks, Hermitian where the packing assumes it: the diagonal blocks
    of xh_k and a1_k (the off-diagonal partner is Hermitian by construction)."""
    y = gaussian(seed, *lead, *system.shape)
    for b, (kind, _, i, j, _) in enumerate(system.layout):
        if kind in (0, 2) and i == j:
            y[..., b, :, :] = (y[..., b, :, :] + adj(y[..., b, :, :])) / 2
    return y


def rk4(system, steps, endpoint, stride):
    """Classical RK4 on `steps` equal steps over (-a, a); the state after every `stride` steps."""
    def f(t, y):
        return system.rhs(y, 2 * abs(t), transformed_source(t))

    y, h, out = system.initial.copy(), 2 * endpoint / steps, []
    for i in range(steps):
        t = -endpoint + i * h
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % stride == 0:
            out.append(y)
    return np.stack(out)


def expected_classes(n, sector):
    """Class (0 block-diagonal, 1 block-off-diagonal) of every live matrix by counting its
    factors of R: xh_k ~ R^k, yh_k ~ W R^(k+1), a1_k ~ R^(n-k), a0t_k ~ W R^(n-k+1)."""
    w = SECTORS.index(sector)
    classes = {(0, k): k % 2 for k in range(1, n + 1)}
    classes.update({(1, k): (w + k + 1) % 2 for k in range(1, n)})
    classes.update({(2, k): (n - k) % 2 for k in range(n)})
    classes.update({(3, k): (w + n - k + 1) % 2 for k in range(n)})
    return classes


def table_rhs(t, constants, y, jac, source):
    """The packed right-hand side from the flat tables alone, term by term as a kernel
    runs it: out_b = jac (G0 + T0 C[r0] + T1 C[r1] + mu Y_b) + source (coefficient G1 + C[fixed])."""
    def operand(side, i):
        index = t[f"term_{side}_index"][i]
        if t[f"term_{side}_kind"][i] == 0:
            return constants[index]
        return y[index].conj().T if t[f"term_{side}_adjoint"][i] else y[index]

    d, out = y.shape[-1], np.empty_like(y)
    for b in range(t["n_blocks"]):
        sums = np.zeros((4, d, d), dtype=np.complex128)
        for i in range(t["term_first"][b], t["term_first"][b + 1]):
            sums[t["term_group"][i]] += operand("left", i) @ operand("right", i)
        value = sums[0] + complex(*t["momentum"][b]) * y[b]
        for g in range(2):
            if t["temp_right"][b, g] >= 0:
                value = value + sums[2 + g] @ constants[t["temp_right"][b, g]]
        inserted = t["coefficient"][b] * sums[1]
        if t["fixed"][b] >= 0:
            inserted = inserted + constants[t["fixed"][b]]
        out[b] = jac * value + source * inserted
    return out


class Case(unittest.TestCase):
    def assertClose(self, x, y, rtol, name, msg="", scale=None):
        """|x - y|_2 <= rtol |y|_2 (or rtol * scale), recording the relative error."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        error, scale = norm(x - y), norm(y) if scale is None else scale
        record(name, error / scale if scale > 0 else error, msg)
        self.assertLessEqual(error, rtol * scale, f"{msg} {name}: |x-y| = {error:.3e}, |y| = {scale:.3e}")


class RightHandSide(Case):
    def test_matches_unpacked(self):
        """(1) unpack(packed rhs(y)) equals ActionSystem.rhs(unpack(y)) on every live matrix
        (complete D x D matrices: class zeros and Hermitian partners included), for the
        Lindblad part alone, the source part alone, and both."""
        for case in GRID:
            packed, full = systems(*case)
            self.assertEqual(packed.sector, case[3])
            y = random_packed(packed, 11)
            unpacked = packed.unpack(y)
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                want = full.rhs(unpacked, jac, source)
                got = packed.unpack(packed.rhs(y, jac, source), constants=False)
                self.assertFalse(got[~packed.live].any())
                for kind, k in zip(*np.nonzero(packed.live)):
                    self.assertClose(got[kind, k], want[kind, k], 1e-12, "rhs: packed vs unpacked",
                                     f"{case} {PREFIX[kind].format(k)} jac={jac} source={source}")

    def test_julia_formulas(self):
        """(1) Independently of potential.ActionSystem: the packed right-hand side against
        action_rhs! (src/excitation_action.jl:19-54) evaluated in physical variables,
        x = S xh S, y = yh S, a1, a0 = a0t S^-1, W = Wh S^-1, V = Vh S^-1, on every live
        matrix (1e-12). The grid adds what GRID lacks: D = 2 (blocks of size 1), :phi^5:,
        a negative momentum and S = sqrt(diag rho)."""
        grid = itertools.product((2, 4, 6), ("none", "explicit", "density"), (0.0, 0.6, -1.3), SECTORS, (2, 3, 4, 5))
        for case in grid:
            dim, scaling, p, sector, n = case
            packed, full = systems(*case)
            ss, s = full.ss, full.ss.s_host
            wh, vh, _ = tangent(dim, scaling, p, sector)
            y = random_packed(packed, 15)
            physical = packed.unpack(y)
            physical[0] *= np.outer(s, s)
            physical[1] *= s
            physical[3] /= s
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                want = to_hat(ss, julia_rhs(ss.state, p, packed.spec, wh / s, vh / s, physical, jac, source))
                got = packed.unpack(packed.rhs(y, jac, source), constants=False)
                for kind, k in zip(*np.nonzero(packed.live)):
                    self.assertClose(got[kind, k], want[kind, k], 1e-12, "rhs: packed vs action_rhs! (physical)",
                                     f"{case} {PREFIX[kind].format(k)} jac={jac} source={source}")

    @unittest.skipUnless(os.path.exists(FIXTURE), "the D = 16 Julia fixture is not available")
    def test_stored_state(self):
        """(1) A realistically conditioned state, which the random ones are not: the D = 16
        phi^4 g = 2 Julia ground state in the density gauge, cond(rho) = 3.3e7, with
        S = sqrt(diag rho) (1.7e-4 to 0.99) and S = 1. Right-hand side only (no
        integration): packed against ActionSystem on every live matrix of :phi^4: (1e-12)."""
        chart = tc.make_coordinates(states.load_jld2(FIXTURE))
        w, d = gaussian(5, 16, 16), 8
        odd = w.copy()
        odd[:d, d:] = odd[d:, :d] = 0
        for scaling, p in itertools.product(("density", "none"), (0.0, 0.5)):
            ss = ScaledState(chart.state, scaling)
            frame = TangentFrame(ss, p, stack([odd, w - odd]), tol=TOL)
            for t, sector in enumerate(SECTORS):
                triple = tuple(np.ascontiguousarray(m[:, t, :]) for m in (frame.wh, frame.vh, frame.rw))
                packed, full = (cls(ss, p, monomial_spec(4), triple) for cls in (PackedActionSystem, ActionSystem))
                self.assertEqual((packed.sector, len(packed.blocks), packed.d), (sector, 26, 8))
                y = random_packed(packed, 17)
                want = full.rhs(packed.unpack(y), 0.7, 0.3)
                got = packed.unpack(packed.rhs(y, 0.7, 0.3), constants=False)
                for kind, k in zip(*np.nonzero(packed.live)):
                    self.assertClose(got[kind, k], want[kind, k], 1e-12, "rhs: packed vs unpacked, D = 16 fixture",
                                     f"S={scaling} p={p} {sector} {PREFIX[kind].format(k)}")

    def test_constant_channels(self):
        """(1) With x_0 = rh, y_0 = rw (TangentFrame), a1_{C-1} = 1 and a0_{C-1} = 0 the unpacked
        right-hand sides of these four matrices vanish, whatever the live blocks are: to
        1e-12 of the sum of the norms of their terms. This is what allows dropping them."""
        for case in GRID:
            dim, scaling, p, sector, n = case
            packed, full = systems(*case)
            ss, (wh, vh, rw) = full.ss, tangent(dim, scaling, p, sector)
            ql, rl, qld, rld = ss.Q["L"], ss.R["L"], ss.Qd["L"], ss.Rd["L"]
            qh, rhh, qhd, rhd = ss.Q["H"], ss.R["H"], ss.Qd["H"], ss.Rd["H"]
            rh = ss.rh
            scales = {(0, 0): norm(qh @ rh) + norm(rh @ qhd) + norm(rhh @ rh @ rhd),
                      (1, 0): (norm(ql @ rw) + norm(rw @ qhd) + norm(rl @ rw @ rhd) + abs(p) * norm(rw)
                               + norm(wh @ rh @ rhd) + norm(vh @ rh)),
                      (2, n): norm(qld) + norm(ql) + norm(rld @ rl),
                      (3, n): norm(vh) + norm(rld @ wh)}
            jac = 1.3
            out = full.rhs(packed.unpack(random_packed(packed, 12)), jac, 0.4)
            for (kind, k), scale in scales.items():
                error = norm(out[kind, k])
                record("rhs of the constant channels / term norms", error / (jac * scale),
                       f"{case} {PREFIX[kind].format(k)}")
                self.assertLessEqual(error, 1e-12 * jac * scale, f"{case} {PREFIX[kind].format(k)}")

    def test_last_ket_channel_unused(self):
        """y_{C-1} enters neither the right-hand side of a live matrix nor the pullback
        (a1_{C-1} = 1 commutes with R), so it is not integrated."""
        points, weights = quadrature(IntegratorParameters(n_quad=8, max_point=4.0))
        for case in GRID:
            n = case[-1]
            packed, full = systems(*case)
            samples = packed.unpack(random_packed(packed, 13, (len(points) + 1,)))
            changed = samples.copy()
            changed[:, 1, n] = gaussian(14, *changed[:, 1, n].shape)
            a, b = full.rhs(samples[0], 0.7, 0.3), full.rhs(changed[0], 0.7, 0.3)
            self.assertTrue(np.array_equal(a[packed.live], b[packed.live]), case)
            for u, v in zip(full.pullback(samples, points, weights)[:2], full.pullback(changed, points, weights)[:2]):
                self.assertClose(v, u, 1e-14, "pullback: dependence on y_{C-1}", str(case))
            got = packed.pullback(packed.pack(samples), points, weights)
            want = full.pullback(samples, points, weights)
            for u, v in zip((*got[:2], *got[2]), (*want[:2], *want[2])):
                self.assertTrue(np.array_equal(u, v), case)


class Packing(Case):
    def test_round_trip(self):
        """(2) pack(unpack(y)) == y exactly, with leading axes; unpack restores the constant
        channels, Hermitian partners and y_{C-1} = 0, and every matrix has its class."""
        for case in GRID:
            dim, scaling, p, sector, n = case
            packed, full = systems(*case)
            d, (_, _, rw) = dim // 2, tangent(dim, scaling, p, sector)
            y = random_packed(packed, 21, (2, 3))
            unpacked = packed.unpack(y)
            self.assertEqual(unpacked.shape, (2, 3, 4, n + 1, dim, dim))
            self.assertTrue(np.array_equal(packed.pack(unpacked), y), case)
            self.assertTrue(np.array_equal(packed.unpack(packed.pack(unpacked)), unpacked), case)
            for value, kind, k in ((full.ss.rh, 0, 0), (rw, 1, 0), (np.eye(dim), 2, n)):
                self.assertTrue((unpacked[..., kind, k, :, :] == value).all(), case)
            self.assertFalse(unpacked[..., 3, n, :, :].any() or unpacked[..., 1, n, :, :].any())
            bare = packed.unpack(y, constants=False)
            self.assertTrue(np.array_equal(bare[..., packed.live, :, :], unpacked[..., packed.live, :, :]))
            self.assertFalse(bare[..., ~packed.live, :, :].any())
            for (kind, k), cls in expected_classes(n, sector).items():
                m = unpacked[1, 2, kind, k]
                self.assertTrue(has_class(m, cls) and not has_class(m, 1 - cls), f"{case} {kind} {k}")
                if kind in (0, 2):
                    self.assertTrue(np.array_equal(m, adj(m)), f"{case} {kind} {k}")
            self.assertEqual(packed.d, d)

    def test_initial(self):
        """The packed initial data are zero and unpack to ActionSystem.initial."""
        for case in GRID:
            packed, full = systems(*case)
            self.assertFalse(packed.initial.any())
            self.assertEqual(packed.initial.shape, packed.shape)
            self.assertTrue(np.array_equal(packed.unpack(packed.initial), full.initial), case)

    def test_layout(self):
        """Names, order and layout of the blocks from the classes derived here; 12 of the
        26 blocks of :phi^4: are Hermitian-packed."""
        for dim, sector, n in itertools.product(DIMS, SECTORS, DEGREES):
            packed, _ = systems(dim, "none", 0.6, sector, n)
            names, layout = [], []
            for (kind, k), cls in expected_classes(n, sector).items():
                rows = (0,) if kind in (0, 2) and cls == 1 else (0, 1)
                for i in rows:
                    names.append(PREFIX[kind].format(k) + f"[{i}{i ^ cls}]")
                    layout.append((kind, k, i, i ^ cls, len(rows) == 1))
            self.assertEqual([b.name for b in packed.blocks], names)
            self.assertEqual(packed.layout, layout)
            self.assertEqual(len(packed.blocks), BLOCKS[n])
            live = np.zeros((4, n + 1), dtype=bool)
            live[0, 1:], live[1, 1:n], live[2, :n], live[3, :n] = True, True, True, True
            self.assertTrue(np.array_equal(packed.live, live))
        packed, _ = systems(4, "none", 0.6, "even", 4)
        self.assertEqual(sum(kind in (0, 2) for kind, *_ in packed.layout), 12)
        self.assertEqual([b.name for b in packed.blocks], [
            "x1[01]", "x2[00]", "x2[11]", "x3[01]", "x4[00]", "x4[11]",
            "y1[01]", "y1[10]", "y2[00]", "y2[11]", "y3[01]", "y3[10]",
            "a1_0[00]", "a1_0[11]", "a1_1[01]", "a1_2[00]", "a1_2[11]", "a1_3[01]",
            "a0_0[00]", "a0_0[11]", "a0_1[01]", "a0_1[10]", "a0_2[00]", "a0_2[11]", "a0_3[01]", "a0_3[10]"])

    def test_zero_tangent(self):
        """tangent=None (the tangent-independent trajectories) is the 'odd' layout with zero
        tangent blocks; its y and a0 blocks have a vanishing right-hand side at y = a0 = 0."""
        for dim, n in itertools.product(DIMS, DEGREES):
            ss, spec = scaled(dim, "explicit"), monomial_spec(n)
            packed, full = PackedActionSystem(ss, 0.6, spec), ActionSystem(ss, 0.6, spec)
            self.assertEqual(packed.sector, "odd")
            odd, _ = systems(dim, "explicit", 0.6, "odd", n)
            self.assertEqual([b.name for b in packed.blocks], [b.name for b in odd.blocks])
            y = random_packed(packed, 22)
            for b, (kind, *_) in enumerate(packed.layout):
                if kind in (1, 3):
                    y[b] = 0
            got = packed.unpack(packed.rhs(y, 0.7, 0.3), constants=False)
            want = full.rhs(packed.unpack(y), 0.7, 0.3)
            self.assertFalse(got[1].any() or got[3].any())
            for kind, k in zip(*np.nonzero(packed.live)):
                self.assertClose(got[kind, k], want[kind, k], 1e-12, "rhs: packed vs unpacked", f"{dim} {n} {kind} {k}")


class Integration(Case):
    def compare_bulk(self, dim, scaling, p, n):
        """Bulk action to 1e-10; the end covector a0 (which decays for p != 0, and feeds the
        boundary term of F through a Poisson solve) to 1e-10 of max(|a0|, |F|)."""
        ss, spec, f = scaled(dim, scaling), monomial_spec(n), frame(dim, scaling, p)
        results = []
        for system in (None, PackedActionSystem):
            action = PotentialAction(ss, p, spec, INTEGRATOR, integrate=integrate_host, system=system, tol=TOL)
            results.append(action.bulk(f))
            self.assertEqual(action.trajectories, 1 + f.count)
        (f0, a0), (f1, a1) = results
        for t, sector in enumerate(SECTORS):
            msg = f"D={dim} S={scaling} p={p} n={n} {sector}"
            self.assertClose(f1[:, t, :], f0[:, t, :], 1e-10, "bulk action: packed vs unpacked", msg)
            self.assertClose(a1[:, t, :], a0[:, t, :], 1e-10, "boundary covector a0: packed vs unpacked, over |F|",
                             msg, max(norm(a0[:, t, :]), norm(f0[:, t, :])))

    def compare_oracle(self, dim, scaling, p):
        ss, w = scaled(dim, scaling), class_tangents(dim)
        action = HamiltonianAction(ss, p, TERMS, integrator=INTEGRATOR, kinetic_nodes=NODES, tol=TOL,
                                   integrate=integrate_host, system=PackedActionSystem)
        self.assertEqual(len(action.potentials), 2)
        fh = action(stack([w[sector] for sector in SECTORS]))
        ctx, s = ref.make_context(ss.state, p), ss.s_host
        for t, sector in enumerate(SECTORS):
            want = ref.hamiltonian_action(ctx, w[sector] / s[None, :], TERMS, kinetic_nodes=NODES)
            self.assertClose(fh[:, t, :] * s[None, :], want, 1e-11, "Hamiltonian action: packed vs oracle",
                             f"D={dim} S={scaling} p={p} {sector}")

    def test_trajectory(self):
        """(3) The packed trajectory equals the unpacked one at every quadrature node (each live
        matrix to 1e-10 over all nodes), whose constant channels stay at rh, rw, 1, 0; so do
        the pullbacks (bulk, a0, a1, xout), for a tangent and for the tangent-independent run."""
        dim, scaling, p, sector, n = 4, "explicit", 0.6, "even", 3
        ss, spec = scaled(dim, scaling), monomial_spec(n)
        points, weights = quadrature(INTEGRATOR)
        endpoint = math.sqrt(INTEGRATOR.max_point)
        for w in (tangent(dim, scaling, p, sector), None):
            packed, full = PackedActionSystem(ss, p, spec, w), ActionSystem(ss, p, spec, w)
            want = integrate_host(full, points, endpoint, INTEGRATOR)
            samples = integrate_host(packed, points, endpoint, INTEGRATOR)
            self.assertEqual(samples.shape, (len(points) + 1,) + packed.shape)
            got = packed.unpack(samples)
            for kind, k in itertools.product(range(4), range(n + 1)):
                msg = f"{PREFIX[kind].format(k)} tangent={w is not None}"
                if packed.live[kind, k]:
                    self.assertClose(got[:, kind, k], want[:, kind, k], 1e-10, "trajectory: packed vs unpacked", msg)
                elif (kind, k) != (1, n):     # constant channel, against all channels of its kind
                    self.assertClose(got[:, kind, k], want[:, kind, k], 1e-10, "trajectory: constant channels",
                                     msg, norm(want[:, kind]))
            f1, a1, pair1 = packed.pullback(samples, points, weights)
            f0, a0, pair0 = full.pullback(want, points, weights)
            self.assertClose(f1, f0, 1e-10, "pullback: bulk, packed vs unpacked")
            # The end values a0_0, a1_0, x_{C-1} at +a have decayed unless they overlap a zero
            # mode: their error is measured against the largest sample of the same matrix.
            for u, v, (kind, k) in ((a1, a0, (3, 0)), (pair1[0], pair0[0], (2, 0)), (pair1[1], pair0[1], (0, n))):
                self.assertClose(u, v, 1e-10, "pullback: end values, over the trajectory peak",
                                 scale=max(norm(m) for m in want[:, kind, k]))

    def test_fixed_steps(self):
        """(3) The packing alone. Under adaptive control the two layouts take different
        steps (the unpacked state carries class zeros and constant channels, which dilute
        its RMS error), so the 1e-10 of test_trajectory cannot tell a packing error of
        1e-11 from integrator noise. On one common sequence of 800 RK4 steps every live
        matrix agrees to 1e-13 of its peak over the trajectory (roundoff), and the
        constant channels of the unpacked run stay at rh, rw, 1, 0 (1e-12 of their kind)."""
        endpoint = math.sqrt(INTEGRATOR.max_point)
        for sector, n in itertools.product(SECTORS, (3, 4)):
            packed, full = systems(4, "explicit", 0.6, sector, n)
            got, want = packed.unpack(rk4(packed, 800, endpoint, 40)), rk4(full, 800, endpoint, 40)
            self.assertTrue(np.isfinite(want).all() and len(want) == 20)
            for kind, k in itertools.product(range(4), range(n + 1)):
                msg = f"{sector} n={n} {PREFIX[kind].format(k)}"
                error = max(norm(a - b) for a, b in zip(got[:, kind, k], want[:, kind, k]))
                if packed.live[kind, k]:
                    peak = max(norm(m) for m in want[:, kind, k])
                    record("fixed steps: packed vs unpacked, over the peak", error / peak, msg)
                    self.assertLessEqual(error, 1e-13 * peak, msg)
                elif (kind, k) != (1, n):
                    self.assertLessEqual(error, 1e-12 * max(norm(m) for m in want[:, kind]), msg)

    def test_chunked_pullback(self):
        """(3) pullback unpacks the samples in mirror-symmetric sets of at most 2 chunk
        nodes plus the endpoint, not all at once (20 D^2 numbers per sample of :phi^4:
        against 6.5 D^2 packed: 5 GiB at D = 128, n_quad = 512). Same bulk (1e-13),
        identical end values, for even and odd numbers of nodes; with the default chunk
        a 160-node pullback unpacks at most 129 samples at a time."""
        for nodes, drop, chunks in ((32, 0, (1, 5, 16, 64)), (32, 1, (1, 5, 16)), (160, 0, (7, 64, None))):
            points, weights = quadrature(IntegratorParameters(n_quad=nodes // 2, max_point=4.0))
            points, weights = points[drop:], weights[drop:]
            n = len(points)
            for sector in SECTORS:
                packed, full = systems(4, "explicit", 0.6, sector, 4)
                samples = random_packed(packed, 16, (n + 1,))
                want = full.pullback(packed.unpack(samples), points, weights)
                sizes, unpack = [], packed.unpack
                packed.unpack = lambda x, constants=True: (sizes.append(len(x)), unpack(x, constants))[1]
                for chunk in chunks:
                    del sizes[:]
                    got = packed.pullback(samples, points, weights, **({} if chunk is None else dict(chunk=chunk)))
                    size = 64 if chunk is None else chunk
                    self.assertEqual(max(sizes), min(n, 2 * size) + 1, f"{n} nodes, chunk {chunk}")
                    self.assertEqual(sum(sizes) - len(sizes), n)
                    self.assertClose(got[0], want[0], 1e-13, "pullback: chunked vs all nodes",
                                     f"{n} nodes, chunk {chunk}")
                    for u, v in zip((got[1], *got[2]), (want[1], *want[2])):
                        self.assertTrue(np.array_equal(u, v))

    def test_degree_five_negative_momentum(self):
        """(3) What the grid lacks, in one case: :phi^5: next to phi^4, p = -0.9, D = 2
        (blocks of size 1) and S = sqrt(diag rho); packed Hamiltonian action against the
        oracle (1e-11), both sectors."""
        ss, p = scaled(2, "density"), -0.9
        terms = phi4_terms(1.3) + [(0.4, monomial_spec(5))]
        w = class_tangents(2)
        action = HamiltonianAction(ss, p, terms, integrator=INTEGRATOR, kinetic_nodes=NODES, tol=TOL,
                                   integrate=integrate_host, system=PackedActionSystem)
        fh = action(stack([w[sector] for sector in SECTORS]))
        ctx, s = ref.make_context(ss.state, p), ss.s_host
        for t, sector in enumerate(SECTORS):
            want = ref.hamiltonian_action(ctx, w[sector] / s[None, :], terms, kinetic_nodes=NODES)
            self.assertClose(fh[:, t, :] * s[None, :], want, 1e-11, "Hamiltonian action: phi^5, p < 0, D = 2", sector)

    def test_driver_levels(self):
        """(3) excitation_eigsolve on the packed system for nev = 2 at p != 0, both signs
        and both sectors: the oracle's levels (1e-8) in the same number of operations.
        State and window of test_eigsolve (random_state(4, seed=11, scaling=0.5, z2=True),
        max_point 14, atol 1e-13, rtol 1e-12), on which 32 pullback nodes give a
        Hermitian operator; the states of this file need more."""
        state = random_state(4, seed=11, scaling=0.5, z2=True).with_integrator(
            IntegratorParameters(n_quad=32, max_point=14.0, atol=1e-13, rtol=1e-12))
        terms = phi4_terms(1.0)
        for sector, p in (("odd", -0.7), ("even", 0.5)):
            request = dict(nev=2, sector=sector, kinetic_nodes=NODES, validate=False)
            dense = ref.eigsolve(state, p, terms, **request)
            r = es.excitation_eigsolve(state, p, terms, scaling="density", integrate=integrate_host,
                                       system=PackedActionSystem, **request)
            msg = f"{sector} p={p}"
            self.assertTrue(r["converged"] and dense["converged"], msg)
            self.assertEqual(r["numops"], dense["numops"], msg)
            self.assertClose(np.array(r["energies"]), np.array(dense["energies"]), 1e-8,
                             "driver levels: packed vs oracle", msg, 1.0)

    def test_bulk(self):
        """(3) PotentialAction.bulk with the packed system and ode.integrate_host against
        the unpacked system, both sectors in one frame."""
        for case in BULK:
            self.compare_bulk(*case)

    def test_hamiltonian_oracle(self):
        """(3) HamiltonianAction with the packed system for :phi^4: and :phi^3: (kinetic part
        through the resolvent) against the dense oracle, F = Fh S and W = Wh S^-1, to 1e-11.
        The oracle integrates with the same tolerances (SciPy DOP853, the two trajectories
        apart)."""
        for case in ORACLE:
            self.compare_oracle(*case)

    @slow
    def test_bulk_grid(self):
        for case in itertools.product(DIMS, SCALINGS, MOMENTA, DEGREES):
            if case not in BULK:
                self.compare_bulk(*case)

    @slow
    def test_hamiltonian_oracle_grid(self):
        for case in itertools.product(DIMS, SCALINGS, MOMENTA):
            if case not in ORACLE:
                self.compare_oracle(*case)


class Tables(Case):
    INTEGER = ("term_first", "term_group", "term_left_kind", "term_left_index", "term_left_adjoint",
               "term_right_kind", "term_right_index", "term_right_adjoint", "temp_right", "fixed",
               "left_constants")

    def test_interpreter(self):
        """(4) `table_rhs`, which reads only tables() and the constants, reproduces rhs(y)
        (to 1e-14: the two differ in the order of the sums only)."""
        for case in GRID:
            packed, _ = systems(*case)
            t, constants = packed.tables(), np.asarray(packed.constants)
            y = gaussian(31, *packed.shape)      # no Hermiticity needed: the same table both ways
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                got, want = packed.rhs(y, jac, source), table_rhs(t, constants, y, jac, source)
                self.assertEqual(got.shape, packed.shape)
                for b, block in enumerate(packed.blocks):
                    self.assertClose(got[b], want[b], 1e-14, "rhs: interpreter vs flat tables", f"{case} {block.name}")

    def test_structure(self):
        """(4) Shapes and types of the flat arrays, exactly one state operand per term,
        terms sorted by group, two-sided groups and their right factors, left_constants,
        counts, and agreement with the Block objects."""
        for case in GRID:
            n = case[-1]
            packed, _ = systems(*case)
            t = packed.tables()
            nb, nc, count = t["n_blocks"], t["n_constants"], len(t["term_group"])
            self.assertEqual((nb, nc), (len(packed.blocks), len(packed.constants)))
            self.assertEqual(packed.constants.shape, (nc, packed.d, packed.d))
            for key in self.INTEGER:
                self.assertEqual(t[key].dtype, np.int32, key)
            for key in ("coefficient", "momentum"):
                self.assertEqual(t[key].dtype, np.float64, key)
            shapes = dict(term_first=(nb + 1,), temp_right=(nb, 2), coefficient=(nb,), momentum=(nb, 2),
                          fixed=(nb,), left_constants=(nb, 4))
            shapes.update({key: (count,) for key in self.INTEGER[1:8]})
            self.assertEqual({key: t[key].shape for key in shapes}, shapes)
            first = t["term_first"]
            self.assertTrue(first[0] == 0 and first[-1] == count and (np.diff(first) > 0).all())
            self.assertTrue((t["term_left_kind"] + t["term_right_kind"] == 1).all())
            referenced = set(t["temp_right"][t["temp_right"] >= 0]) | set(t["fixed"][t["fixed"] >= 0])
            for side in ("left", "right"):
                kind, index, adjoint = (t[f"term_{side}_{field}"] for field in ("kind", "index", "adjoint"))
                self.assertTrue(np.isin(kind, (0, 1)).all() and np.isin(adjoint, (0, 1)).all())
                self.assertTrue((index >= 0).all() and (index < np.where(kind == 0, nc, nb)).all())
                self.assertFalse(adjoint[kind == 0].any())
                referenced |= set(index[kind == 0])
            self.assertEqual(referenced, set(range(nc)), "every constant is used")
            self.assertTrue(((t["temp_right"] >= -1) & (t["temp_right"] < nc)).all())
            self.assertTrue(((t["fixed"] >= -1) & (t["fixed"] < nc)).all())
            products = 0
            for b, block in enumerate(packed.blocks):
                part = slice(first[b], first[b + 1])
                groups = t["term_group"][part]
                self.assertTrue((np.diff(groups) >= 0).all() and np.isin(groups, range(4)).all())
                for g in range(2):
                    self.assertEqual(t["temp_right"][b, g] >= 0, (groups == pk.GROUP_TEMP0 + g).any())
                used = sorted(set(t["term_left_index"][part][t["term_left_kind"][part] == 0]))
                self.assertLessEqual(len(used), 4)
                self.assertEqual(list(t["left_constants"][b]), used + [-1] * (4 - len(used)))
                kind, k = packed.layout[b][:2]
                reduced = k == (1 if kind < 2 else n - 1)      # source from a constant channel: fixed
                self.assertEqual(t["fixed"][b] >= 0, reduced, block.name)
                self.assertEqual((groups == pk.GROUP_SRC).sum(), 0 if reduced else (2, 3, 2, 3)[kind])
                here = len(groups) + (t["temp_right"][b] >= 0).sum()
                self.assertEqual(here, (6, 9, 6, 10)[kind] - (2, 3, 2, 3)[kind] * reduced, block.name)
                products += here
                self.assertEqual(len(block.terms), len(groups))
                for i, term in zip(range(first[b], first[b + 1]), block.terms):
                    self.assertEqual(sum(o.kind == "state" for o in (term.left, term.right)), 1)
                    self.assertEqual(
                        [t[key][i] for key in self.INTEGER[1:8]],
                        [term.group, term.left.kind == "state", term.left.index, term.left.adjoint,
                         term.right.kind == "state", term.right.index, term.right.adjoint])
                self.assertEqual((tuple(t["temp_right"][b]), t["coefficient"][b], complex(*t["momentum"][b]),
                                  t["fixed"][b]), (block.temp_right, block.coefficient, block.momentum, block.fixed))
            self.assertEqual((t["products"], packed.products), (products, products))
            self.assertEqual((nb, products), (BLOCKS[n], PRODUCTS[n]), case)

    def test_values(self):
        """mu = -ip (ket blocks), +ip (a0 blocks), 0 otherwise; coefficients of the chain; the
        constants are the named blocks of Q, R, Wh, Vh (each once, state first, then tangent,
        then fixed sources), and the fixed sources are those of ActionSystem.rhs."""
        for case in GRID:
            dim, scaling, p, sector, n = case
            packed, full = systems(*case)
            ss, d, (wh, vh, rw) = full.ss, dim // 2, tangent(dim, scaling, p, sector)
            for block, (kind, k, i, j, _) in zip(packed.blocks, packed.layout):
                self.assertEqual(block.momentum, (0, -1j * p, 0, 1j * p)[kind], block.name)
                self.assertEqual(block.coefficient, k if kind < 2 else k + 1, block.name)
                self.assertIsInstance(block.coefficient, float)
            dense = {"Q_L": ss.Q["L"], "Q_L'": ss.Qd["L"], "Q_H": ss.Q["H"], "Q_H'": ss.Qd["H"],
                     "R_L": ss.R["L"], "R_L'": ss.Rd["L"], "R_H": ss.R["H"], "R_H'": ss.Rd["H"], "W": wh, "V": vh}
            names = packed.constant_names
            self.assertEqual(len(set(names)), len(names))
            self.assertEqual([name[:-4] for name in names[:20]], [key for key in dense for _ in range(2)])
            self.assertTrue(all(name.startswith("fixed:") for name in names[20:]))
            for name, value in zip(names[:20], packed.constants[:20]):
                i, j = int(name[-3]), int(name[-2])
                self.assertTrue(np.array_equal(value, dense[name[:-4]][i * d:(i + 1) * d, j * d:(j + 1) * d]), name)
            rh, rl, rld, rhh, rhd = ss.rh, ss.R["L"], ss.Rd["L"], ss.R["H"], ss.Rd["H"]
            fixed = {(0, 1): rhh @ rh + rh @ rhd, (1, 1): rl @ rw + wh @ rh + rw @ rhd,
                     (2, n - 1): n * (rl + rld), (3, n - 1): n * wh}
            seen = 0
            for block, (kind, k, i, j, _) in zip(packed.blocks, packed.layout):
                if block.fixed >= 0:
                    self.assertEqual(names[block.fixed], f"fixed:{block.name}")
                    want = fixed[(kind, k)][i * d:(i + 1) * d, j * d:(j + 1) * d]
                    error = norm(packed.constants[block.fixed] - want)
                    self.assertLessEqual(error, 1e-14 * norm(fixed[(kind, k)]), f"{case} {block.name}")
                    seen += 1
            self.assertEqual(seen, 6)

    def test_structure_is_universal(self):
        """The integer tables depend on the degree and on the sector only: one kernel
        configuration serves every tangent of a sector, every momentum and every scaling."""
        for n, sector in itertools.product(DEGREES, SECTORS):
            base = systems(4, "none", 0.0, sector, n)[0]
            t0 = base.tables()
            others = [systems(dim, scaling, p, sector, n)[0]
                      for dim, scaling, p in ((4, "explicit", 0.6), (6, "none", 0.6))]
            ss = scaled(4, "none")
            other = TangentFrame(ss, 0.0, stack([(1 + 2j) * class_tangents(4)[sector] @ ss.Q["L"]]), tol=TOL)
            others.append(PackedActionSystem(ss, 0.0, monomial_spec(n),
                                             tuple(m[:, 0, :] for m in (other.wh, other.vh, other.rw))))
            if sector == "odd":
                others.append(PackedActionSystem(ss, 0.6, monomial_spec(n)))
            for system in others:
                t = system.tables()
                self.assertEqual(system.sector, sector)
                self.assertEqual(system.constant_names, base.constant_names)
                self.assertEqual([b.name for b in system.blocks], [b.name for b in base.blocks])
                for key in (*self.INTEGER, "coefficient"):
                    self.assertTrue(np.array_equal(t[key], t0[key]), key)
                self.assertEqual((t["n_blocks"], t["n_constants"], t["products"]),
                                 (t0["n_blocks"], t0["n_constants"], t0["products"]))
        odd, even = (systems(4, "none", 0.0, sector, 4)[0].tables() for sector in SECTORS)
        self.assertFalse(np.array_equal(odd["term_right_index"], even["term_right_index"]))

    def test_left_constant_limit(self):
        """tables() refuses a block with more left constants than the kernel holds."""
        packed, _ = systems(4, "none", 0.0, "odd", 4)
        limit = pk.MAX_LEFT
        try:
            pk.MAX_LEFT = 3
            with self.assertRaises(ValueError):
                packed.tables()
        finally:
            pk.MAX_LEFT = limit
        self.assertEqual(packed.tables()["left_constants"].shape, (26, 4))


class Rejections(Case):
    def test_state(self):
        """(5) States without exact balanced Z2 block zeros: generic, odd D, Z2 in another
        index order, and a roundoff-sized entry in rh (nothing is projected)."""
        spec = monomial_spec(4)
        z2 = random_state(4, z2=True)
        order = [0, 2, 1, 3]
        interleaved = make_state(z2.R[np.ix_(order, order)], z2.K[np.ix_(order, order)])
        for state in (random_state(4), random_state(3), interleaved):
            with self.assertRaises(ValueError):
                PackedActionSystem(ScaledState(state, "none"), 0.0, spec)
        ss = copy.copy(scaled(4, "none"))
        ss.rh = ss.rh.copy()
        ss.rh[0, 2] = 1e-300
        with self.assertRaises(ValueError):
            PackedActionSystem(ss, 0.0, spec)
        PackedActionSystem(scaled(4, "none"), 0.0, spec)

    def test_tangent(self):
        """(5) A tangent mixing the classes, and a triple (wh, vh, rw) of inconsistent classes."""
        ss, spec = scaled(4, "explicit"), monomial_spec(4)
        wh, vh, rw = tangent(4, "explicit", 0.6, "odd")
        other = tangent(4, "explicit", 0.6, "even")
        mixed = wh.copy()
        mixed[0, 3] = 1e-300
        for bad in ((mixed, vh, rw), (wh + other[0], vh + other[1], rw + other[2]), (wh, other[1], rw),
                    (wh, vh, other[2]), (other[0], vh, rw)):
            with self.assertRaises(ValueError):
                PackedActionSystem(ss, 0.6, spec, bad)
        self.assertEqual(PackedActionSystem(ss, 0.6, spec, (wh, vh, rw)).sector, "odd")
        self.assertEqual(PackedActionSystem(ss, 0.6, spec, other).sector, "even")

    def test_gauge_fixing(self):
        """(5) The reductions (y_0 = rw and a0_{C-1} = 0 constant) hold for vh = -R' wh only
        (src/excitation_action.jl:212): a class-consistent vh that is not the gauge part of
        wh was accepted and silently integrated other equations than ActionSystem (bulk
        3.7e-6 off for a relative perturbation of 1e-6). rw is a Poisson solution and is
        not checked: the same perturbation of rw moves the bulk by 2e-5."""
        ss, spec = scaled(4, "explicit"), monomial_spec(4)
        for sector in SECTORS:
            wh, vh, rw = tangent(4, "explicit", 0.6, sector)
            dv = gaussian(41, 4, 4)     # a perturbation of the class of vh
            if has_class(vh, pk.DG):
                dv[:2, 2:] = dv[2:, :2] = 0
            else:
                dv[:2, :2] = dv[2:, 2:] = 0
            for bad in ((wh, vh + 1e-6 * dv, rw), (wh, 2 * vh, rw), (wh, np.zeros_like(vh), rw)):
                with self.assertRaises(ValueError) as failure:
                    PackedActionSystem(ss, 0.6, spec, bad)
                self.assertIn("gauge-fixed", str(failure.exception))
            self.assertEqual(PackedActionSystem(ss, 0.6, spec, (wh, vh * (1 + 1e-14), rw)).sector, sector)
        PackedActionSystem(ss, 0.6, spec)     # no tangent: wh = vh = 0

    def test_complex_coefficients(self):
        """(5) Edge.coefficient is complex-typed (ComplexF64 in src/observables.jl:7-12): a
        chain with coefficients complex(k+1) equals monomial_spec(n), is accepted by
        ActionSystem, and must give the same packed tables (it raised TypeError)."""
        ss, plain = scaled(4, "none"), monomial_spec(4)
        spec = Observable(5, tuple(Edge(e.src, e.dst, e.left, complex(e.coefficient)) for e in plain.edges))
        self.assertEqual(spec, plain)
        triple = tangent(4, "none", 0.6, "even")
        got, want = (PackedActionSystem(ss, 0.6, chain, triple) for chain in (spec, plain))
        for block in got.blocks:
            self.assertIsInstance(block.coefficient, float)
        for key, value in want.tables().items():
            self.assertTrue(np.array_equal(got.tables()[key], value), key)
        self.assertTrue(np.array_equal(got.constants, want.constants))

    def test_spec(self):
        """(5) Commutator and kinetic specs, :phi^n: with n < 2, and a chain with other edges."""
        ss = scaled(4, "none")
        chain = monomial_spec(4)
        bad = [Observable(chain.channels, chain.edges, commutator=True), kinetic_spec(False), kinetic_spec(True),
               monomial_spec(0), monomial_spec(1), Observable(chain.channels, chain.edges[:-1]),
               Observable(chain.channels, chain.edges, kinetic=True), Observable(6, chain.edges)]
        for spec in bad:
            with self.assertRaises(ValueError):
                PackedActionSystem(ss, 0.0, spec)
            with self.assertRaises(ValueError):
                PackedActionSystem(ss, 0.0, spec, tangent(4, "none", 0.0, "even"))
        with self.assertRaises(ValueError):       # the default of PotentialAction stays the unpacked system
            PotentialAction(ss, 0.0, kinetic_spec(False), INTEGRATOR, integrate=integrate_host,
                            system=PackedActionSystem)


if __name__ == "__main__":
    unittest.main()
