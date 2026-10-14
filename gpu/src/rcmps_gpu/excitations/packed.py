"""Z2-packed, reduced action ODE system for monomial terms, as explicit term tables.

The equations are those of potential.ActionSystem (port of action_rhs!,
src/excitation_action.jl) in the scaled variables of scaled.py: xh_k (HH), yh_k (LH),
a1_k (dual LL), a0t_k (dual LH), with tangent Wh, Vh = -R' Wh and ket environment rw.
This module only rewrites them block by block; it is the specification that the fused
CUDA kernel executes, one d x d complex multiply-add per term, d = D/2.

Classes. For a Z2-symmetric state in the balanced layout (R block-off-diagonal, Q and
rh block-diagonal, S diagonal) and a tangent of definite class, every matrix of the
system is block-diagonal ("Dg") or block-off-diagonal ("O"): block row i holds one
d x d block, in block column i ^ class. Classes add mod 2 under products, so a block
of a product is ONE product of one block of each factor. A Dg tangent is the 'odd'
sector, an O tangent the 'even' one (coordinates.py); a zero tangent counts as Dg.

Reductions for a monomial chain :phi^n:, n >= 2 (models.monomial_spec, C = n+1
channels). The forward channel 0 is constant (xh_0 = rh, yh_0 = rw) and so is the
adjoint channel C-1 (a1 = 1, a0t = 0): their right-hand sides vanish, and they enter
the others only through fixed source blocks. The ket block yh_{C-1} is not used by
the pullback. The live matrices are xh_k (k = 1..C-1), yh_k (k = 1..C-2), a1_k and
a0t_k (k = 0..C-2). xh_k and a1_k are Hermitian: an O one is stored as its (0,1)
block, whose conjugate transpose is the (1,0) block, a Dg one as its two diagonal
blocks. yh_k and a0t_k store both blocks.

Table contract. With C the array `constants` and Y the packed state, block b obeys
    out_b = jac (G0 + T0 C[temp_right[0]] + T1 C[temp_right[1]] + mu Y_b)
            + source (coefficient G1 + C[fixed]),
where Gg and Tg are the sums of left @ right over the terms of group g of the block
(GROUP_JAC, GROUP_SRC for G0, G1; GROUP_TEMP0, GROUP_TEMP1 for T0, T1), jac = 2|t| and
source = 2|t| J(t^2). Every term has exactly one state operand; an index -1 in
`temp_right` or `fixed` means no such contribution. In this form
    xh_k:   jac (Q_H x + x Q_H' + (R_H x) R_H') + source k (R_H x_{k-1} + x_{k-1} R_H')
    yh_k:   jac (Q y + y Q_H' + (R y + Wh x_k) R_H' - i p y + Vh x_k)
            + source k (R y_{k-1} + Wh x_{k-1} + y_{k-1} R_H')
    a1_k:   jac (Q' a + a Q + (R' a) R) + source (k+1) (a_{k+1} R + R' a_{k+1})
    a0t_k:  jac (Q' b + b Q_H + (R' b) R_H + (R' a1_k) Wh + i p b + a1_k Vh)
            + source (k+1) (b_{k+1} R_H + a1_{k+1} Wh + R' b_{k+1})
(Q, R in the L gauge), i.e. 6, 9, 6 and 10 products per block, less 2, 3, 2, 3 where the
source comes from a constant channel and is a fixed block (k = 1 forward, k = C-2
adjoint). Blocks and products per right-hand side for n = 2, 3, 4: 12, 18, 26 and
78, 128, 190, in both sectors; the layout depends on the degree and the sector only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import monomial_spec
from .potential import ActionSystem
from .states import adj

GROUP_JAC, GROUP_SRC, GROUP_TEMP0, GROUP_TEMP1 = 0, 1, 2, 3
DG, OFF = 0, 1                        # matrix classes; the class of a product is the XOR
SECTORS = {DG: "odd", OFF: "even"}    # class of the tangent Wh -> parity sector
MAX_LEFT = 4                          # distinct left constants of a block held by the kernel


@dataclass(frozen=True)
class Operand:
    kind: str               # 'const' | 'state'
    index: int              # into `constants`, or packed block index
    adjoint: bool = False   # conjugate transpose of the stored d x d block; 'state' only


@dataclass(frozen=True)
class Term:
    group: int
    left: Operand
    right: Operand


@dataclass
class Block:
    name: str               # e.g. 'x1[01]', 'y2[00]', 'a1_0[11]', 'a0_3[10]'
    terms: tuple            # sorted by group
    temp_right: tuple       # constant indices of the right factors of T0, T1; -1 if unused
    coefficient: float      # multiplies the source group
    momentum: complex       # mu
    fixed: int              # constant index of the fixed source block (coefficient included)


class _Matrix:
    """A class-definite D x D matrix: a constant (dense `value`) or a live unknown,
    whose stored block rows have the packed indices `stored`."""
    def __init__(self, name, cls, value=None, hermitian=False):
        self.name, self.cls, self.value, self.hermitian = name, cls, value, hermitian
        self.stored = {}

    @property
    def rows(self):
        return (0,) if self.hermitian and self.cls == OFF else (0, 1)


def has_class(m, cls):
    """Exact zeros in the blocks that the class excludes (balanced layout)."""
    d = m.shape[0] // 2
    excluded = (m[:d, d:], m[d:, :d]) if cls == DG else (m[:d, :d], m[d:, d:])
    return not any(bool((block != 0).any()) for block in excluded)


class PackedActionSystem:
    """Packed counterpart of potential.ActionSystem(ss, momentum, spec, tangent), with
    the same role in potential.PotentialAction and ode.integrate_host.

    `tangent` = (wh, vh, rw) must be gauge fixed as kinetic.TangentFrame builds it
    (vh = -R' wh, rw the ket environment): that is what makes channel 0 and C-1 constant.
    vh is checked (1e-12 relative); rw, a Poisson solution, is trusted to its solver
    tolerance, and its error enters the packed result at first order (a relative 1e-6
    moved the bulk by 2e-5 at D = 4).
    `blocks` (list[Block]) and `constants` ((n_const, d, d): the blocks of Q and R in
    both gauges and of their adjoints, then those of Wh and Vh, then the fixed sources,
    each once; names in `constant_names`) are the tables. Their structure depends on
    the degree and on the sector only. `layout[b]` = (kind, channel, block row, block
    column, partner) locates packed block b in the ActionSystem state (4, C, D, D),
    `partner` marking a Hermitian O matrix stored as its (0,1) block; `live[kind, channel]`
    flags the matrices that are integrated. `initial` is zero: the nonzero initial data
    of ActionSystem are the constant channels. `sector` is 'odd' or 'even'.
    Raises ValueError unless Q, R, rh and the tangent have exact block zeros (nothing is
    projected), vh = -R' wh, and `spec` is a non-commutator monomial chain with n >= 2.
    """
    def __init__(self, ss, momentum, spec, tangent=None):
        xp = ss.xp
        if (spec.commutator or spec.kinetic or spec.channels < 3
                or spec.edges != monomial_spec(spec.channels - 1).edges):
            raise ValueError("the packed action system needs a monomial chain :phi^n: with n >= 2")
        if ss.D % 2:
            raise ValueError("the packed action system needs an even bond dimension")
        self.ss, self.p, self.spec, self.xp, self.d = ss, float(momentum), spec, xp, ss.D // 2
        self.reference = ref = ActionSystem(ss, momentum, spec, tangent)
        c, p = spec.channels, self.p

        # Constants. The classes of V, rw and of every live matrix follow from the equations.
        ql, qld = _Matrix("Q_L", DG, ss.Q["L"]), _Matrix("Q_L'", DG, ss.Qd["L"])
        qh, qhd = _Matrix("Q_H", DG, ss.Q["H"]), _Matrix("Q_H'", DG, ss.Qd["H"])
        rl, rld = _Matrix("R_L", OFF, ss.R["L"]), _Matrix("R_L'", OFF, ss.Rd["L"])
        rhh, rhd = _Matrix("R_H", OFF, ss.R["H"]), _Matrix("R_H'", OFF, ss.Rd["H"])
        x = [_Matrix("x0", DG, ss.rh)]
        if not all(has_class(m.value, m.cls) for m in (ql, qld, qh, qhd, rl, rld, rhh, rhd, x[0])):
            raise ValueError("the packed action system needs exact balanced Z2 block zeros in Q, R "
                             "and rh; no projection is performed")
        classes = [cls for cls in (DG, OFF) if has_class(ref.wh, cls)]
        if not classes:
            raise ValueError("the tangent has no definite Z2 class")
        w = _Matrix("W", classes[0], ref.wh)
        v = _Matrix("V", rld.cls ^ w.cls, ref.vh)
        y = [_Matrix("y0", v.cls ^ x[0].cls, ref.rw)]
        if not (has_class(v.value, v.cls) and has_class(y[0].value, y[0].cls)):
            raise ValueError("vh and rw do not have the Z2 class implied by wh")
        gauge = ss.Rd["L"] @ ref.wh
        if float(xp.linalg.norm(ref.vh + gauge)) > 1e-12 * float(xp.linalg.norm(ref.vh) + xp.linalg.norm(gauge)):
            raise ValueError("the packed action system needs a gauge-fixed tangent, vh = -R' wh")
        self.sector = SECTORS[w.cls]
        a1 = {c - 1: _Matrix(f"a1_{c - 1}", DG, ss.eye)}
        a0 = {c - 1: _Matrix(f"a0_{c - 1}", a1[c - 1].cls ^ v.cls, xp.zeros_like(ss.eye))}

        # Live matrices and their equations: (matrix, jac pairs, two-sided groups, mu).
        for k in range(1, c):
            x.append(_Matrix(f"x{k}", rhh.cls ^ x[k - 1].cls, hermitian=True))
        for k in range(1, c - 1):
            y.append(_Matrix(f"y{k}", rl.cls ^ y[k - 1].cls))
        for k in range(c - 2, -1, -1):
            a1[k] = _Matrix(f"a1_{k}", a1[k + 1].cls ^ rl.cls, hermitian=True)
            a0[k] = _Matrix(f"a0_{k}", a0[k + 1].cls ^ rhh.cls)
        equations = [(0, k, x[k], [(qh, x[k]), (x[k], qhd)], [([(rhh, x[k])], rhd)], 0)
                     for k in range(1, c)]
        equations += [(1, k, y[k], [(ql, y[k]), (y[k], qhd), (v, x[k])],
                       [([(rl, y[k]), (w, x[k])], rhd)], -1j * p) for k in range(1, c - 1)]
        equations += [(2, k, a1[k], [(qld, a1[k]), (a1[k], ql)], [([(rld, a1[k])], rl)], 0)
                      for k in range(c - 1)]
        equations += [(3, k, a0[k], [(qld, a0[k]), (a0[k], qh), (a1[k], v)],
                       [([(rld, a0[k])], rhh), ([(rld, a1[k])], w)], 1j * p) for k in range(c - 1)]

        # Source insertions of ActionSystem.rhs; channel C-1 of y is dropped.
        source = {m: [] for _, _, m, *_ in equations}
        coefficient = dict.fromkeys(source, 0.0)
        for e in spec.edges:
            s, t = e.src, e.dst
            if e.left:
                pairs = [(x[t], (rhh, x[s])), (a1[s], (a1[t], rl)), (a0[s], (a0[t], rhh)),
                         (a0[s], (a1[t], w))]
                pairs += [(y[t], (rl, y[s])), (y[t], (w, x[s]))] if t < c - 1 else []
            else:
                pairs = [(x[t], (x[s], rhd)), (a1[s], (rld, a1[t])), (a0[s], (rld, a0[t]))]
                pairs += [(y[t], (y[s], rhd))] if t < c - 1 else []
            for m, pair in pairs:
                source[m].append(pair)
                coefficient[m] = complex(e.coefficient).real    # an integer k+1 of the chain

        self.layout = []
        for kind, k, m, *_ in equations:
            for row in m.rows:
                m.stored[row] = len(self.layout)
                self.layout.append((kind, k, row, row ^ m.cls, len(m.rows) == 1))
        self._pool, self._slot = [], {}
        for m in (ql, qld, qh, qhd, rl, rld, rhh, rhd, w, v):    # constant order: state, tangent, fixed
            for row in (0, 1):
                self._operand(m, row)
        self.blocks = [self._block(m, row, jac, temps, source[m], coefficient[m], mu)
                       for _, _, m, jac, temps, mu in equations for row in m.rows]
        self.constants = xp.ascontiguousarray(xp.stack(self._pool))
        self.constant_names = list(self._slot)
        del self._pool
        self.live = np.zeros((4, c), dtype=bool)
        for kind, k, *_ in self.layout:
            self.live[kind, k] = True
        self.initial = xp.zeros((len(self.blocks), self.d, self.d), dtype=xp.complex128)
        self.shape = self.initial.shape
        self._compile()

    # ---- class algebra -----------------------------------------------------------------
    def _dense(self, m, row):
        d, col = self.d, row ^ m.cls
        return self.xp.ascontiguousarray(m.value[row * d:(row + 1) * d, col * d:(col + 1) * d])

    def _constant(self, name, block):
        if name not in self._slot:
            self._slot[name] = len(self._pool)
            self._pool.append(block)
        return self._slot[name]

    def _operand(self, m, row):
        """Block (row, row ^ class) of m as a table operand."""
        if m.value is not None:
            return Operand("const", self._constant(f"{m.name}[{row}{row ^ m.cls}]", self._dense(m, row)))
        if row in m.stored:
            return Operand("state", m.stored[row])
        return Operand("state", m.stored[row ^ 1], True)    # (1,0) block of a Hermitian O matrix

    def _terms(self, group, pairs, row, cls):
        """Block row `row` of sum a @ b over `pairs`, each product of class `cls`:
        the middle block index of a product is fixed by the class of its left factor."""
        out = []
        for a, b in pairs:
            if a.cls ^ b.cls != cls or (a.value is None) == (b.value is None):
                raise AssertionError(f"inconsistent term {a.name} {b.name}")
            out.append(Term(group, self._operand(a, row), self._operand(b, row ^ a.cls)))
        return out

    def _block(self, m, row, jac, temps, source, coefficient, mu):
        """Table of block row `row` of the live matrix m. Source pairs of two constants
        (a constant channel feeding a live one) are summed into the fixed block."""
        name = f"{m.name}[{row}{row ^ m.cls}]"
        terms, temp_right, fixed = self._terms(GROUP_JAC, jac, row, m.cls), [-1, -1], None
        for i, (pairs, right) in enumerate(temps):
            cls = m.cls ^ right.cls
            terms += self._terms(GROUP_TEMP0 + i, pairs, row, cls)
            temp_right[i] = self._operand(right, row ^ cls).index
        for a, b in source:
            if a.value is None or b.value is None:
                terms += self._terms(GROUP_SRC, [(a, b)], row, m.cls)
            elif a.cls ^ b.cls != m.cls:
                raise AssertionError(f"inconsistent fixed source {a.name} {b.name}")
            else:
                product = coefficient * (self._dense(a, row) @ self._dense(b, row ^ a.cls))
                fixed = product if fixed is None else fixed + product
        index = -1 if fixed is None else self._constant(f"fixed:{name}", fixed)
        return Block(name, tuple(sorted(terms, key=lambda t: t.group)), tuple(temp_right),
                     coefficient, complex(mu), index)

    # ---- tables and their interpreter --------------------------------------------------
    def tables(self):
        """Flat host arrays for the fused kernel. The terms of block b are
        term_first[b]:term_first[b+1], sorted by term_group; term_{left,right}_kind is 0
        (constant) or 1 (state), with _index into `constants` or the packed state and
        _adjoint for the conjugate transpose of a state block (int32 throughout).
        Per block: temp_right (n_blocks, 2), coefficient, momentum (n_blocks, 2: re, im),
        fixed, and left_constants (n_blocks, 4): the sorted distinct constants used as
        left operands, padded with -1. Counts: n_blocks, n_constants, and products, the
        d x d products of one right-hand side (terms plus two-sided products)."""
        blocks = self.blocks
        terms = [t for b in blocks for t in b.terms]

        def column(f):
            return np.array([f(t) for t in terms], dtype=np.int32)

        left = np.full((len(blocks), MAX_LEFT), -1, dtype=np.int32)
        for i, b in enumerate(blocks):
            used = sorted({t.left.index for t in b.terms if t.left.kind == "const"})
            if len(used) > MAX_LEFT:
                raise ValueError(f"block {b.name} has {len(used)} left constants; the kernel holds {MAX_LEFT}")
            left[i, :len(used)] = used
        temp_right = np.array([b.temp_right for b in blocks], dtype=np.int32)
        return dict(
            term_first=np.cumsum([0] + [len(b.terms) for b in blocks]).astype(np.int32),
            term_group=column(lambda t: t.group),
            term_left_kind=column(lambda t: t.left.kind == "state"),
            term_left_index=column(lambda t: t.left.index),
            term_left_adjoint=column(lambda t: t.left.adjoint),
            term_right_kind=column(lambda t: t.right.kind == "state"),
            term_right_index=column(lambda t: t.right.index),
            term_right_adjoint=column(lambda t: t.right.adjoint),
            temp_right=temp_right,
            coefficient=np.array([b.coefficient for b in blocks], dtype=np.float64),
            momentum=np.array([[b.momentum.real, b.momentum.imag] for b in blocks], dtype=np.float64),
            fixed=np.array([b.fixed for b in blocks], dtype=np.int32),
            left_constants=left,
            n_blocks=len(blocks), n_constants=len(self.constant_names),
            products=len(terms) + int((temp_right >= 0).sum()))

    def _compile(self):
        """Gather indices of `rhs`, from the flat tables alone. Operands are drawn from
        the pool [constants, Y, Y']; padded slots point to a zero matrix. `_temp` and
        `_fixed` are copies of rows of `constants`: a system is built per tangent, and
        overwriting `constants` in place instead would leave them stale."""
        xp, d, t = self.xp, self.d, self.tables()
        nc, nb, count = t["n_constants"], t["n_blocks"], len(t["term_group"])

        def pool(side):
            index, state = t[f"term_{side}_index"], t[f"term_{side}_kind"]
            return xp.asarray(index + state * (nc + nb * t[f"term_{side}_adjoint"]))

        self._left, self._right = pool("left"), pool("right")
        members = [[[] for _ in range(4)] for _ in range(nb)]
        for b in range(nb):
            for i in range(t["term_first"][b], t["term_first"][b + 1]):
                members[b][t["term_group"][i]].append(i)
        slots = np.full((nb, 4, max(len(m) for row in members for m in row)), count)
        for b, row in enumerate(members):
            for g, m in enumerate(row):
                slots[b, g, :len(m)] = m
        self._slots = xp.asarray(slots)
        self._zero = xp.zeros((1, d, d), dtype=xp.complex128)
        padded = xp.concatenate([self.constants, self._zero])    # index -1: no contribution
        self._temp = [xp.ascontiguousarray(padded[xp.asarray(t["temp_right"][:, i])]) for i in range(2)]
        self._fixed = xp.ascontiguousarray(padded[xp.asarray(t["fixed"])])
        self._coefficient = xp.asarray(t["coefficient"])[:, None, None]
        self._mu = xp.asarray(t["momentum"][:, 0] + 1j * t["momentum"][:, 1])[:, None, None]
        self.products = t["products"]

    def rhs(self, y, jac, source):
        """d/dt of the packed state (n_blocks, d, d) for scalars jac = 2|t| and
        source = 2|t| J(t^2): the table contract, one batched product per stage."""
        xp = self.xp
        pool = xp.concatenate([self.constants, y, adj(y)])
        products = xp.concatenate([pool[self._left] @ pool[self._right], self._zero])
        g = products[self._slots].sum(axis=2)                    # (n_blocks, group, d, d)
        return (jac * (g[:, GROUP_JAC] + g[:, GROUP_TEMP0] @ self._temp[0]
                       + g[:, GROUP_TEMP1] @ self._temp[1] + self._mu * y)
                + source * (self._coefficient * g[:, GROUP_SRC] + self._fixed))

    # ---- relation to the unpacked system -----------------------------------------------
    def unpack(self, packed, constants=True):
        """(..., n_blocks, d, d) -> ActionSystem layout (..., 4, C, D, D): Hermitian
        partner blocks restored, y_{C-1} = 0, and the constant channels rh, rw, 1, 0
        (left at zero with constants=False, for a right-hand side)."""
        xp, d = self.xp, self.d
        out = xp.zeros(packed.shape[:-3] + self.reference.shape, dtype=xp.complex128)
        if constants:
            c, initial = self.spec.channels, self.reference.initial
            for kind, k in ((0, 0), (1, 0), (2, c - 1)):
                out[..., kind, k, :, :] = initial[kind, k]
        for b, (kind, k, i, j, partner) in enumerate(self.layout):
            block = packed[..., b, :, :]
            out[..., kind, k, i * d:(i + 1) * d, j * d:(j + 1) * d] = block
            if partner:
                out[..., kind, k, j * d:(j + 1) * d, i * d:(i + 1) * d] = adj(block)
        return out

    def pack(self, unpacked):
        """Stored blocks of an ActionSystem-layout state (..., 4, C, D, D)."""
        xp, d = self.xp, self.d
        out = xp.empty(unpacked.shape[:-4] + self.shape, dtype=xp.complex128)
        for b, (kind, k, i, j, _) in enumerate(self.layout):
            out[..., b, :, :] = unpacked[..., kind, k, i * d:(i + 1) * d, j * d:(j + 1) * d]
        return out

    def pullback(self, samples, points, weights, *, chunk=64):
        """ActionSystem.pullback on the unpacked samples. The bulk is a sum over the nodes
        that pairs node i (forward blocks) with its mirror n-1-i (covector blocks), so it
        is accumulated over mirror-symmetric sets of at most 2 `chunk` nodes: only those
        samples and the endpoint are unpacked at a time (20 D^2 numbers per sample for
        :phi^4:, against 6.5 D^2 packed). The end values are those of the endpoint sample."""
        xp, n = self.xp, len(points)
        if n <= 2 * chunk:
            return self.reference.pullback(self.unpack(samples), points, weights)
        points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
        half, out = (n + 1) // 2, 0
        for first in range(0, half, chunk):
            last = min(half, first + chunk)
            lower, upper = slice(first, last), slice(max(last, n - last), n - first)
            nodes = np.r_[lower, upper]
            part = xp.concatenate([samples[lower], samples[upper], samples[n:n + 1]])
            bulk, a0, pair = self.reference.pullback(self.unpack(part), points[nodes], weights[nodes])
            out = out + bulk
        return out, a0, pair
