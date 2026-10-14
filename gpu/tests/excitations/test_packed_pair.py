"""Tests of packed_pair.py: the Z2-packed four-block ODE of the diagonal element
<Phi(W)|:phi^n:|Phi(W)>.

The reference is the unpacked system of bilinear.py (Pair, PairSystem and
BilinearElement.observable with ode.integrate_host). Grid: Z2 states of D = 4 and 6, both
sectors (block-diagonal and block-off-diagonal W), p = 0 and 0.5, :phi^2: and :phi^4:,
S = 1, S = sqrt(diag rho) and S = exp(linspace(-2.5, 1.5, D)). One physical tangent W
per (D, sector); the hat tangent of a scaling is Wh = W S.

States: "gap" = states.random_state(D, z2=True) (transfer gap 0.71 and 0.99, cond rho 9
and 23; the Poisson solves of the environments end at roundoff) and "weak" =
random_state(D, z2=True, scaling=0.5, seed=11) (gap 0.20 and 0.21, cond rho 14 and 28),
on which the residual of the environment solves shows.

Element       (1) diagonal_observable against the unpacked observable, (5) real and
              independent of S. About 75 s: the two trajectories of the 48 cases.
RightHandSide (2) packed rhs against the packed projection of PairSystem.rhs on physical
              states (x1, x4 Hermitian, b3 = y2'), and the stationary channel 0.
Value         (3) value(final) against Pair.contract.
Tables        (6) layout, widths and counts of tables().
Rejections    (4) the ValueErrors.

RCMPS_EXCITATION_REPORT=1 prints the largest measured errors at the end.
"""
from __future__ import annotations

import functools
import itertools
import math
import os
import sys
import unittest

import numpy as np

from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations.bilinear import TYPES, BilinearElement, Pair, PairSystem
from rcmps_gpu.excitations.models import Observable, kinetic_spec, monomial_spec
from rcmps_gpu.excitations.packed_pair import PackedDiagonalPair, diagonal_observable
from rcmps_gpu.excitations.scaled import ScaledState
from rcmps_gpu.excitations.states import IntegratorParameters, adj, make_state, random_state

norm = np.linalg.norm
STATES = {"gap": {}, "weak": dict(scaling=0.5, seed=11)}
DIMS, SCALINGS, SECTORS, MOMENTA, DEGREES = (4, 6), ("none", "density", "explicit"), ("odd", "even"), (0.0, 0.5), (2, 4)
GRID = list(itertools.product(DIMS, SCALINGS, SECTORS, MOMENTA, DEGREES))
INTEGRATOR = IntegratorParameters(n_quad=24, max_point=12.0, atol=1e-13, rtol=1e-12)
TOL = 1e-14                                  # relative residual of the environment solves
BLOCK = (0, 1, 3)                            # four-block index of the packed kinds x, y, z
BLOCKS, PRODUCTS = {2: 10, 4: 20}, {2: 90, 4: 192}
INTEGER = ("term_first", "term_group", "term_left_kind", "term_left_index", "term_left_adjoint",
           "term_right_kind", "term_right_index", "term_right_adjoint", "temp_right", "fixed", "left_constants")
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
def scaled(name, dim, scaling):
    state = random_state(dim, z2=True, **STATES[name]).with_integrator(INTEGRATOR)
    return ScaledState(state, np.exp(np.linspace(-2.5, 1.5, dim)) if scaling == "explicit" else scaling)


def physical(dim, sector):
    """Physical tangent W: block-diagonal ('odd' sector) or block-off-diagonal ('even')."""
    w, d = gaussian(5, dim, dim), dim // 2
    odd = w.copy()
    odd[:d, d:] = odd[d:, :d] = 0
    return odd if sector == "odd" else w - odd


@functools.lru_cache(maxsize=None)
def cached(kind, name, dim, scaling, p, tol, sector=None, n=None):
    """One object per key: the BilinearElement, the pair of BilinearElement.pair(wh, wh),
    the packed system, and the (packed, unpacked) values of the term, both through
    ode.integrate_host: diagonal_observable and BilinearElement.observable."""
    if kind == "element":
        return BilinearElement(scaled(name, dim, scaling), p, [], integrator=INTEGRATOR,
                               integrate=ode.integrate_host, tol=tol)
    if kind == "pair":
        wh = physical(dim, sector) * scaled(name, dim, scaling).s_host[None, :]
        return cached("element", name, dim, scaling, p, tol).pair(wh, wh)
    q, spec = cached("pair", name, dim, scaling, p, tol, sector), monomial_spec(n)
    if kind == "system":
        return PackedDiagonalPair(q, spec)
    return (diagonal_observable(q, spec, INTEGRATOR, ode.integrate_host),
            cached("element", name, dim, scaling, p, tol).observable(q, spec))


def element(name, dim, scaling, p, tol=TOL):
    return cached("element", name, dim, scaling, p, tol)


def pair(name, dim, scaling, sector, p, tol=TOL):
    return cached("pair", name, dim, scaling, p, tol, sector)


def system(name, dim, scaling, sector, p, n):
    return cached("system", name, dim, scaling, p, TOL, sector, n)


def values(name, dim, scaling, sector, p, n, tol=TOL):
    return cached("values", name, dim, scaling, p, tol, sector, n)


def random_packed(s, seed):
    """Random packed state on the physical subspace: Hermitian diagonal blocks of x and z
    (an off-diagonal block stands for a Hermitian matrix by construction)."""
    y = gaussian(seed, *s.shape)
    for b, (kind, _, i, j, _) in enumerate(s.layout):
        if kind in (0, 2) and i == j:
            y[b] = (y[b] + adj(y[b])) / 2
    return y


def unpack(s, y):
    """Four-block state (4, D, C, D) of PairSystem for a packed state: x1 and x4 from their
    stored blocks with the Hermitian partners, x2 = y, b3 = y', and channel 0 = pair.right."""
    d, c = s.d, s.spec.channels
    out = np.zeros((4, 2 * d, c, 2 * d), dtype=np.complex128)
    out[:, :, 0, :] = s.pair.right[:, :, 0, :]
    for b, (kind, k, i, j, partner) in enumerate(s.layout):
        rows, cols = slice(i * d, (i + 1) * d), slice(j * d, (j + 1) * d)
        out[BLOCK[kind], rows, k, cols] = y[b]
        if partner:
            out[BLOCK[kind], cols, k, rows] = adj(y[b])
        if kind == 1:
            out[2, cols, k, rows] = adj(y[b])
    return out


def project(s, x):
    """Stored blocks (n_blocks, d, d) of a four-block array (4, D, C, D)."""
    d = s.d
    return np.stack([x[BLOCK[kind], i * d:(i + 1) * d, k, j * d:(j + 1) * d] for kind, k, i, j, _ in s.layout])


def dagger(x):
    """Conjugate transpose of every channel of a block (D, channels, D)."""
    return np.conj(np.swapaxes(x, 0, 2))


def expected_layout(n, sector):
    """Names and layout from the classes obtained by counting factors of R (0 block-diagonal,
    1 block-off-diagonal): x_k ~ R^k, y_k ~ W R^(k+1), z_k ~ W W' R^k; channel by channel."""
    w, names, layout = SECTORS.index(sector), [], []
    for k in range(1, n + 1):
        for kind, cls in enumerate((k % 2, (w + k + 1) % 2, k % 2)):
            rows = (0,) if kind != 1 and cls == 1 else (0, 1)
            for i in rows:
                names.append(f"{'xyz'[kind]}{k}[{i}{i ^ cls}]")
                layout.append((kind, k, i, i ^ cls, len(rows) == 1))
    return names, layout


class Case(unittest.TestCase):
    def assertClose(self, x, y, rtol, name, msg="", scale=None):
        """|x - y|_2 <= rtol |y|_2 (or rtol * scale), recording the relative error."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        error, scale = norm(x - y), norm(y) if scale is None else scale
        record(name, error / scale if scale > 0 else error, msg)
        self.assertLessEqual(error, rtol * scale, f"{msg} {name}: |x-y| = {error:.3e}, |y| = {scale:.3e}")


class Element(Case):
    def test_matches_unpacked(self):
        """(1) diagonal_observable(pair, spec, integrator, ode.integrate_host) equals
        BilinearElement.observable(pair, spec), the unpacked four-block ODE, to 1e-11
        relative over the grid (environments solved to 1e-14)."""
        for case in GRID:
            packed, unpacked = values("gap", *case)
            self.assertIsInstance(packed, complex)
            self.assertClose(packed, unpacked, 1e-11, "element: packed vs unpacked", str(case))
            self.assertIs(system("gap", *case).pair, pair("gap", *case[:4]))    # the environments are shared
        self.assertEqual(element("gap", 4, "none", 0.0).trajectories, len(SECTORS) * len(DEGREES))

    def test_real(self):
        """(5) The diagonal element of a Hermitian form is real: |Im| <= 1e-9 |value|."""
        for case in GRID:
            value = values("gap", *case)[0]
            record("element: |Im| / |value|", abs(value.imag) / abs(value), str(case))
            self.assertLessEqual(abs(value.imag), 1e-9 * abs(value), case)

    def test_scaling_independent(self):
        """(5) For one physical tangent W = Wh S^-1 the element does not depend on S: the
        values at S = sqrt(diag rho) and S = exp(linspace(-2.5, 1.5, D)) equal the one at
        S = 1 to 1e-10 relative."""
        for dim, sector, p, n in itertools.product(DIMS, SECTORS, MOMENTA, DEGREES):
            want = values("gap", dim, "none", sector, p, n)[0]
            for scaling in SCALINGS[1:]:
                self.assertClose(values("gap", dim, scaling, sector, p, n)[0], want, 1e-10,
                                 "element: scaling vs S = 1", f"D={dim} {sector} p={p} n={n} S={scaling}")

    def test_environment_residual(self):
        """(1) The packed system keeps the pair environments as constants while the unpacked
        ODE integrates them, so the two values differ at first order in the residual r of the
        environment solves, not at the ODE tolerance. Weak-gap state, D = 6, S = exp(linspace),
        even sector, p = 0, :phi^2:: relative difference below 3e3 r (measured 3.0e-10 = 350 r
        with tol = 1e-12, 2.0e-12 = 220 r with tol = 1e-14), and below 1e-11 with tol = 1e-14."""
        case = ("weak", 6, "explicit", "even", 0.0)
        for tol, rtol in ((1e-12, math.inf), (1e-14, 1e-11)):
            packed, unpacked = values(*case, 2, tol=tol)
            residual = max(float(info["residuals"].max()) for info in pair(*case, tol=tol).infos)
            self.assertLessEqual(residual, tol)
            self.assertClose(packed, unpacked, min(rtol, 3e3 * residual),
                             f"element: packed vs unpacked, weak gap, tol={tol:g}", f"residual {residual:.1e}")

    def test_final_state(self):
        """(1) At t = +a every stored block of the packed trajectory equals the block of the
        unpacked one, to 1e-11 of the norm of the final state (the first channels have
        decayed there), and channel 0 of the unpacked one has not moved (1e-11)."""
        s, endpoint = system("gap", 4, "explicit", "even", 0.5, 4), math.sqrt(INTEGRATOR.max_point)
        statistics = {}
        got = ode.integrate_host(s, (), endpoint, INTEGRATOR, statistics=statistics)
        self.assertEqual(got.shape, (1,) + s.shape)
        self.assertGreater(statistics["accepted"], 50)
        full = ode.integrate_host(PairSystem(s.pair, s.spec), (), endpoint, INTEGRATOR)[-1]
        want = project(s, full)
        for b, block in enumerate(s.blocks):
            self.assertClose(got[-1, b], want[b], 1e-11, "final state: packed vs unpacked", block.name, norm(want))
        self.assertClose(full[:, :, 0, :], s.pair.right[:, :, 0, :], 1e-11, "final state: channel 0 stays put")


class RightHandSide(Case):
    def test_matches_unpacked(self):
        """(2) On a physical packed state the packed right-hand side equals, block by block
        to 1e-12, PairSystem.rhs of the unpacked state; the unpacked one stays physical
        (x1 and x4 Hermitian, b3 = y2', exact class zeros). Lindblad part, source part, both."""
        for name, case in itertools.product(STATES, GRID):
            s = system(name, *case)
            full, d = PairSystem(s.pair, s.spec), s.d
            y = random_packed(s, 11)
            unpacked = unpack(s, y)
            self.assertTrue(np.array_equal(project(s, unpacked), y))
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                want, got = full.rhs(unpacked, jac, source), s.rhs(y, jac, source)
                self.assertEqual(got.shape, s.shape)
                msg = f"{name} {case} jac={jac} source={source}"
                for block, u, v in zip(s.blocks, got, project(s, want)):
                    self.assertClose(u, v, 1e-12, "rhs: packed vs unpacked", f"{msg} {block.name}")
                live = want[:, :, 1:, :]
                self.assertClose(live[2], dagger(live[1]), 1e-12, "rhs: bra block vs adjoint of the ket block", msg)
                for q in (0, 3):
                    self.assertClose(live[q], dagger(live[q]), 1e-12, "rhs: Hermiticity of x1 and x4", msg)
                kept = np.zeros(want.shape, dtype=bool)
                for kind, k, i, j, partner in s.layout:
                    for q, r, c in ((BLOCK[kind], i, j),) + (((BLOCK[kind], j, i),) if partner else ()) + (
                            ((2, j, i),) if kind == 1 else ()):
                        kept[q, r * d:(r + 1) * d, k, c * d:(c + 1) * d] = True
                kept[:, :, 0, :] = True
                self.assertFalse(want[~kept].any(), f"{msg}: a block excluded by the Z2 class is not zero")

    def test_constant_channel(self):
        """(2) Channel 0 holds the pair environments (rh, ket, bra and double environment):
        the unpacked right-hand side vanishes there, to 1e-12 of |Q x| + |x Q'| per block,
        whatever the live channels are; and the bra environment is the adjoint of the ket
        one, the double one Hermitian (1e-12), as the packing assumes."""
        for name, case in itertools.product(STATES, GRID):
            s = system(name, *case)
            ss, right = s.ss, s.pair.right[:, :, 0, :]
            out = PairSystem(s.pair, s.spec).rhs(unpack(s, random_packed(s, 12)), 1.3, 0.4)
            for q, (a, b) in enumerate(TYPES):
                scale = 1.3 * (norm(ss.Q[a] @ right[q]) + norm(right[q] @ ss.Qd[b]))
                record("rhs of channel 0 / (|Q x| + |x Q'|)", norm(out[q, :, 0, :]) / scale, f"{name} {case} block {q}")
                self.assertLessEqual(norm(out[q, :, 0, :]), 1e-12 * scale, f"{name} {case} block {q}")
            self.assertClose(right[2], adj(right[1]), 1e-12, "environment: bra vs adjoint of ket", f"{name} {case}")
            self.assertClose(right[3], adj(right[3]), 1e-12, "environment: Hermiticity of x4", f"{name} {case}")
            self.assertFalse(s.initial.any())
            self.assertEqual(s.initial.shape, s.shape)


class Value(Case):
    def test_contract(self):
        """(3) value(final) = tr(z_{C-1}) + tr(hh x_{C-1}) equals Pair.contract on the last
        channel of the unpacked state (1e-13); only the blocks of the last channel enter."""
        for name, case in itertools.product(STATES, GRID):
            s = system(name, *case)
            y = random_packed(s, 21)
            got = s.value(y)
            want = complex(s.pair.contract(unpack(s, y)[:, :, -1:, :])[0])
            self.assertIsInstance(got, complex)
            self.assertClose(got, want, 1e-13, "value vs Pair.contract", f"{name} {case}")
            other = random_packed(s, 22)
            last = [b for b, (kind, k, *_) in enumerate(s.layout) if k == s.spec.channels - 1 and kind != 1]
            other[last] = y[last]
            self.assertEqual(s.value(other), got, f"{name} {case}")


class Tables(Case):
    def test_layout(self):
        """(6) Blocks channel by channel (x_k, y_k, z_k), with the classes derived here:
        10 blocks for :phi^2:, 20 for :phi^4:, in both sectors."""
        for name, case in itertools.product(STATES, GRID):
            sector, n = case[2], case[4]
            s = system(name, *case)
            names, layout = expected_layout(n, sector)
            self.assertEqual([b.name for b in s.blocks], names, case)
            self.assertEqual(s.layout, layout, case)
            self.assertEqual((s.sector, s.d, s.shape), (sector, case[0] // 2, (BLOCKS[n], case[0] // 2, case[0] // 2)))
        self.assertEqual([b.name for b in system("gap", 4, "none", "even", 0.0, 2).blocks],
                         ["x1[01]", "y1[01]", "y1[10]", "z1[01]", "x2[00]", "x2[11]", "y2[00]", "y2[11]",
                          "z2[00]", "z2[11]"])

    def test_tables(self):
        """(6) tables(): 3 two-sided columns and 5 left-constant columns (both needed by the
        z blocks), int32 / float64 arrays, n_blocks and the products of one right-hand side
        (per block 6, 9, 16 for x, y, z, less 2, 3, 4 at k = 1 where the source is a fixed
        block: 90 and 192), 32 constants, mu = -ip on the ket blocks, coefficient k."""
        for name, case in itertools.product(STATES, GRID):
            p, n = case[3], case[4]
            s = system(name, *case)
            t = s.tables()
            nb, count = BLOCKS[n], len(t["term_group"])
            self.assertEqual((t["n_blocks"], len(s.blocks), t["n_constants"], len(s.constants)), (nb, nb, 32, 32))
            self.assertEqual(s.constants.shape, (32, s.d, s.d))
            self.assertEqual(len(set(s.constant_names)), 32)
            self.assertEqual({key: t[key].shape for key in ("term_first", "temp_right", "left_constants",
                                                            "coefficient", "momentum", "fixed")},
                             dict(term_first=(nb + 1,), temp_right=(nb, 3), left_constants=(nb, 5),
                                  coefficient=(nb,), momentum=(nb, 2), fixed=(nb,)))
            for key in INTEGER:
                self.assertEqual(t[key].dtype, np.int32, key)
                if key.startswith("term_") and key != "term_first":
                    self.assertEqual(t[key].shape, (count,), key)
            self.assertEqual((t["coefficient"].dtype, t["momentum"].dtype), (np.float64, np.float64))
            self.assertTrue((t["term_left_kind"] + t["term_right_kind"] == 1).all())
            products = 0
            for b, (kind, k, *_) in enumerate(s.layout):
                groups = t["term_group"][t["term_first"][b]:t["term_first"][b + 1]]
                self.assertTrue((np.diff(groups) >= 0).all() and np.isin(groups, range(5)).all())
                used, temps = (t["left_constants"][b] >= 0).sum(), (t["temp_right"][b] >= 0).sum()
                self.assertEqual((used, temps), ((2, 1), (4, 1), (5, 3))[kind], s.blocks[b].name)
                self.assertEqual(t["fixed"][b] >= 0, k == 1, s.blocks[b].name)
                here = len(groups) + temps
                self.assertEqual(here, (6, 9, 16)[kind] - (2, 3, 4)[kind] * (k == 1), s.blocks[b].name)
                products += here
                self.assertEqual(tuple(t["momentum"][b]), (0.0, -p if kind == 1 else 0.0))
                self.assertEqual(t["coefficient"][b], k)
            self.assertEqual((t["products"], s.products), (products, products))
            self.assertEqual(products, PRODUCTS[n], case)

    def test_structure_is_universal(self):
        """(6) The integer tables depend on the degree and on the sector only (not on D, p,
        S or the state): one compiled kernel per (d, 10 or 20 blocks, 5 panels, 3 groups)."""
        for sector, n in itertools.product(SECTORS, DEGREES):
            base = system("gap", 4, "none", sector, 0.0, n)
            for other in (system("gap", 6, "explicit", sector, 0.5, n), system("weak", 4, "density", sector, 0.5, n)):
                self.assertEqual(other.constant_names, base.constant_names)
                for key in (*INTEGER, "coefficient"):
                    self.assertTrue(np.array_equal(other.tables()[key], base.tables()[key]), key)
        odd, even = (system("gap", 4, "none", sector, 0.0, 4).tables() for sector in SECTORS)
        self.assertFalse(np.array_equal(odd["term_right_index"], even["term_right_index"]))


class Rejections(Case):
    def test_bra_differs(self):
        """(4) U != W, by one rounding error, by a factor or by its sector: ValueError."""
        ss, spec = scaled("gap", 4, "none"), monomial_spec(4)
        wh = physical(4, "odd")
        PackedDiagonalPair(Pair(ss, 0.5, wh, wh), spec)
        ulp = wh.copy()
        ulp[np.unravel_index(np.abs(wh).argmax(), wh.shape)] *= 1 + 1e-15
        self.assertFalse(np.array_equal(ulp, wh))
        for uh in (ulp, 2 * wh, physical(4, "even")):
            for bra, ket in ((uh, wh), (wh, uh)):
                with self.assertRaisesRegex(ValueError, "U must equal W"):
                    PackedDiagonalPair(Pair(ss, 0.5, bra, ket), spec)

    def test_spec(self):
        """(4) A commutator spec, kinetic specs, :phi^n: with n < 2 and a chain with other
        edges: ValueError; so are invalid integration controls in diagonal_observable."""
        q, chain = pair("gap", 4, "none", "odd", 0.5), monomial_spec(4)
        self.assertEqual(PackedDiagonalPair(q, chain).sector, "odd")
        bad = [Observable(chain.channels, chain.edges, commutator=True), kinetic_spec(False), kinetic_spec(True),
               monomial_spec(0), monomial_spec(1), Observable(chain.channels, chain.edges[:-1]),
               Observable(chain.channels, chain.edges, kinetic=True), Observable(6, chain.edges)]
        for spec in bad:
            with self.assertRaisesRegex(ValueError, "monomial chain"):
                PackedDiagonalPair(q, spec)
            with self.assertRaisesRegex(ValueError, "monomial chain"):
                diagonal_observable(q, spec, INTEGRATOR, ode.integrate_host)
        for controls in (dict(max_point=0.0), dict(atol=0.0), dict(rtol=-1e-12), dict(maxiters=0)):
            with self.assertRaisesRegex(ValueError, "integration controls"):
                diagonal_observable(q, chain, IntegratorParameters(**controls), ode.integrate_host)

    def test_state(self):
        """(4) An odd D, a state without Z2 symmetry, a Z2 state in another index order and a
        tangent mixing the two classes: ValueError (nothing is projected)."""
        spec = monomial_spec(4)
        with self.assertRaisesRegex(ValueError, "even bond dimension"):
            PackedDiagonalPair(Pair(ScaledState(random_state(3), "none"), 0.5, *[gaussian(6, 3, 3)] * 2), spec)
        z2 = random_state(4, z2=True)
        order = [0, 2, 1, 3]
        interleaved = make_state(z2.R[np.ix_(order, order)], z2.K[np.ix_(order, order)])
        for state in (random_state(4), interleaved):
            for sector in SECTORS:
                wh = physical(4, sector)
                with self.assertRaisesRegex(ValueError, "Z2 block zeros"):
                    PackedDiagonalPair(Pair(ScaledState(state, "none"), 0.5, wh, wh), spec)
        ss = scaled("gap", 4, "density")
        mixed = physical(4, "odd")
        mixed[0, 3] = 1e-300
        for wh in (mixed, physical(4, "odd") + physical(4, "even")):
            with self.assertRaisesRegex(ValueError, "definite Z2 class"):
                PackedDiagonalPair(Pair(ss, 0.5, wh, wh), spec)
        self.assertEqual([PackedDiagonalPair(Pair(ss, 0.5, wh, wh), spec).sector
                          for wh in (physical(4, "odd"), physical(4, "even"))], list(SECTORS))


if __name__ == "__main__":
    unittest.main()
