"""Z2-packed, reduced four-block ODE of the diagonal bilinear element <Phi(W)|H|Phi(W)>.

The equations are those of bilinear.PairSystem (port of excitation_rhs!,
src/excitations.jl) for U = W, as term tables in the format of packed.py so that
the fused kernel (fused.py) integrates them: this is the independent bilinear energy
of the validation step at bond dimensions where the host-controlled integrator is
too slow.

For U = W the bra block is the conjugate transpose of the ket block (b3h = y2h') and
the doubly inserted block x4 is Hermitian, so each live channel k = 1..C-1 holds
    x_k = x1h (HH, Hermitian), y_k = y2h (LH), z_k = x4 (LL, Hermitian),
with channel 0 constant (rh, the ket environment and the double environment of the
pair). In the table contract of packed.py, with n the overlap and S the scaling:
    x_k: jac (Q_H x + x Q_H' + (R_H x) R_H') + source k (R_H x_{k-1} + x_{k-1} R_H')
    y_k: jac (Q y + y Q_H' + (R y + Wh x_k) R_H' - i p y + Vh x_k)
         + source k (R y_{k-1} + Wh x_{k-1} + y_{k-1} R_H')
    z_k: jac (Q z + z Q' + Vh y' + y Vh' + (R z + Wh y') R' + (R y + Wh x_k) Wh' + (-n S x_k) S)
         + source k (R z_{k-1} + Wh y_{k-1}' + z_{k-1} R' + y_{k-1} Wh')
The value of the term is tr(z_{C-1}) + tr(hh x_{C-1}) at t = +a (excitation_contract).
Three two-sided groups and five left constants per block are needed for z.
"""
from __future__ import annotations

import math

import numpy as np

from models import monomial_spec
from packed import DG, OFF, SECTORS, Block, Operand, Term, _Matrix, has_class
from states import SolverFailure, adj

GROUP_JAC, GROUP_SRC, GROUP_TEMP = 0, 1, 2   # two-sided groups are GROUP_TEMP + i
TEMPS, MAX_LEFT = 3, 5


class PackedDiagonalPair:
    """Packed value-only system of one monomial term for a bilinear.Pair with U = W.
    Same interface as packed.PackedActionSystem for the integrators (xp, d, blocks,
    constants, initial, rhs, tables); `value(final)` contracts the state at t = +a."""
    def __init__(self, pair, spec):
        ss = pair.ss
        xp = ss.xp
        if (spec.commutator or spec.kinetic or spec.channels < 3
                or spec.edges != monomial_spec(spec.channels - 1).edges):
            raise ValueError("the packed pair system needs a monomial chain :phi^n: with n >= 2")
        if ss.D % 2:
            raise ValueError("the packed pair system needs an even bond dimension")
        if float(xp.abs(pair.uh - pair.wh).max()) != 0.0:
            raise ValueError("the packed pair system is the diagonal element: U must equal W")
        self.pair, self.ss, self.spec, self.xp, self.d = pair, ss, spec, xp, ss.D // 2
        c, p = spec.channels, pair.p
        s = xp.diag(ss.s).astype(xp.complex128)

        ql, qld = _Matrix("Q_L", DG, ss.Q["L"]), _Matrix("Q_L'", DG, ss.Qd["L"])
        qh, qhd = _Matrix("Q_H", DG, ss.Q["H"]), _Matrix("Q_H'", DG, ss.Qd["H"])
        rl, rld = _Matrix("R_L", OFF, ss.R["L"]), _Matrix("R_L'", OFF, ss.Rd["L"])
        rhh, rhd = _Matrix("R_H", OFF, ss.R["H"]), _Matrix("R_H'", OFF, ss.Rd["H"])
        x = [_Matrix("x0", DG, ss.rh)]
        if not all(has_class(m.value, m.cls) for m in (ql, qld, qh, qhd, rl, rld, rhh, rhd, x[0])):
            raise ValueError("the packed pair system needs exact balanced Z2 block zeros in Q, R and rh")
        classes = [cls for cls in (DG, OFF) if has_class(pair.wh, cls)]
        if not classes:
            raise ValueError("the tangent has no definite Z2 class")
        w, wd = _Matrix("W", classes[0], pair.wh), _Matrix("W'", classes[0], adj(pair.wh))
        v, vd = _Matrix("V", OFF ^ w.cls, pair.vh), _Matrix("V'", OFF ^ w.cls, adj(pair.vh))
        y = [_Matrix("y0", v.cls, pair.right[1][:, 0, :])]
        z = [_Matrix("z0", DG, pair.right[3][:, 0, :])]
        ns = _Matrix("-nS", DG, -pair.overlap * s)
        sm = _Matrix("S", DG, s)
        if not all(has_class(m.value, m.cls) for m in (v, y[0], z[0])):
            raise ValueError("the pair environments do not have the Z2 class implied by the tangent")
        self.sector = SECTORS[w.cls]
        for k in range(1, c):
            x.append(_Matrix(f"x{k}", rhh.cls ^ x[k - 1].cls, hermitian=True))
            y.append(_Matrix(f"y{k}", rl.cls ^ y[k - 1].cls))
            z.append(_Matrix(f"z{k}", rl.cls ^ z[k - 1].cls, hermitian=True))
        self._adjoint_of = {}
        yd = [self._dagger(m) for m in y]
        equations = []
        for k in range(1, c):
            equations.append((0, k, x[k], [(qh, x[k]), (x[k], qhd)], [([(rhh, x[k])], rhd)], 0))
            equations.append((1, k, y[k], [(ql, y[k]), (y[k], qhd), (v, x[k])],
                              [([(rl, y[k]), (w, x[k])], rhd)], -1j * p))
            equations.append((2, k, z[k], [(ql, z[k]), (z[k], qld), (v, yd[k]), (y[k], vd)],
                              [([(rl, z[k]), (w, yd[k])], rld), ([(rl, y[k]), (w, x[k])], wd),
                               ([(ns, x[k])], sm)], 0))
        source = {m: [] for _, _, m, *_ in equations}
        coefficient = dict.fromkeys(source, 0.0)
        for e in spec.edges:
            a, b = e.src, e.dst
            if e.left:
                pairs = [(x[b], (rhh, x[a])), (y[b], (rl, y[a])), (y[b], (w, x[a])),
                         (z[b], (rl, z[a])), (z[b], (w, yd[a]))]
            else:
                pairs = [(x[b], (x[a], rhd)), (y[b], (y[a], rhd)), (z[b], (z[a], rld)), (z[b], (y[a], wd))]
            for m, pair_ in pairs:
                source[m].append(pair_)
                coefficient[m] = float(e.coefficient)

        self.layout = []
        for kind, k, m, *_ in equations:
            for row in m.rows:
                m.stored[row] = len(self.layout)
                self.layout.append((kind, k, row, row ^ m.cls, len(m.rows) == 1))
        self._pool, self._slot = [], {}
        self.blocks = [self._block(m, row, jac, temps, source[m], coefficient[m], mu)
                       for _, _, m, jac, temps, mu in equations for row in m.rows]
        self.constants = xp.ascontiguousarray(xp.stack(self._pool))
        self.constant_names = list(self._slot)
        del self._pool
        self.initial = xp.zeros((len(self.blocks), self.d, self.d), dtype=xp.complex128)
        self.shape = self.initial.shape
        self._last = (x[c - 1], z[c - 1])
        self._tables = self._build_tables()
        self._compile()

    # ---- class algebra (as packed.PackedActionSystem, plus daggered non-Hermitian operands) ----
    def _dagger(self, m):
        """The conjugate transpose of a matrix, as an operand source."""
        d = _Matrix(m.name + "'", m.cls, None if m.value is None else adj(m.value))
        self._adjoint_of[d] = m
        return d

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
        if m in self._adjoint_of:       # block (row, col) of y' is the adjoint of block (col, row) of y
            return Operand("state", self._adjoint_of[m].stored[row ^ m.cls], True)
        if row in m.stored:
            return Operand("state", m.stored[row])
        return Operand("state", m.stored[row ^ 1], True)    # (1,0) block of a Hermitian O matrix

    def _terms(self, group, pairs, row, cls):
        out = []
        for a, b in pairs:
            if a.cls ^ b.cls != cls or (a.value is None) == (b.value is None):
                raise AssertionError(f"inconsistent term {a.name} {b.name}")
            out.append(Term(group, self._operand(a, row), self._operand(b, row ^ a.cls)))
        return out

    def _block(self, m, row, jac, temps, source, coefficient, mu):
        name = f"{m.name}[{row}{row ^ m.cls}]"
        terms, temp_right, fixed = self._terms(GROUP_JAC, jac, row, m.cls), [-1] * TEMPS, None
        for i, (pairs, right) in enumerate(temps):
            cls = m.cls ^ right.cls
            terms += self._terms(GROUP_TEMP + i, pairs, row, cls)
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

    # ---- tables and their interpreter --------------------------------------------------------
    def _build_tables(self):
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
            left_constants=left, n_blocks=len(blocks), n_constants=len(self.constant_names),
            products=len(terms) + int((temp_right >= 0).sum()))

    def tables(self):
        """Flat host arrays for the fused kernel (format of packed.PackedActionSystem.tables,
        with three two-sided groups and five left-constant slots)."""
        return self._tables

    def _compile(self):
        xp, d, t = self.xp, self.d, self._tables
        nc, nb, count, groups = t["n_constants"], t["n_blocks"], len(t["term_group"]), 2 + TEMPS

        def pool(side):
            index, state = t[f"term_{side}_index"], t[f"term_{side}_kind"]
            return xp.asarray(index + state * (nc + nb * t[f"term_{side}_adjoint"]))

        self._left, self._right = pool("left"), pool("right")
        members = [[[] for _ in range(groups)] for _ in range(nb)]
        for b in range(nb):
            for i in range(t["term_first"][b], t["term_first"][b + 1]):
                members[b][t["term_group"][i]].append(i)
        slots = np.full((nb, groups, max(len(m) for row in members for m in row)), count)
        for b, row in enumerate(members):
            for g, m in enumerate(row):
                slots[b, g, :len(m)] = m
        self._slots = xp.asarray(slots)
        self._zero = xp.zeros((1, d, d), dtype=xp.complex128)
        padded = xp.concatenate([self.constants, self._zero])
        self._temp = [xp.ascontiguousarray(padded[xp.asarray(t["temp_right"][:, i])]) for i in range(TEMPS)]
        self._fixed = xp.ascontiguousarray(padded[xp.asarray(t["fixed"])])
        self._coefficient = xp.asarray(t["coefficient"])[:, None, None]
        self._mu = xp.asarray(t["momentum"][:, 0] + 1j * t["momentum"][:, 1])[:, None, None]
        self.products = t["products"]

    def rhs(self, y, jac, source):
        """d/dt of the packed state (n_blocks, d, d): the table contract."""
        xp = self.xp
        pool = xp.concatenate([self.constants, y, adj(y)])
        products = xp.concatenate([pool[self._left] @ pool[self._right], self._zero])
        g = products[self._slots].sum(axis=2)
        two_sided = sum(g[:, GROUP_TEMP + i] @ self._temp[i] for i in range(TEMPS))
        return (jac * (g[:, GROUP_JAC] + two_sided + self._mu * y)
                + source * (self._coefficient * g[:, GROUP_SRC] + self._fixed))

    # ---- value -----------------------------------------------------------------------------
    def _full(self, m, packed):
        """Dense D x D matrix of a live Hermitian matrix from its stored blocks."""
        xp, d = self.xp, self.d
        out = xp.zeros((2 * d, 2 * d), dtype=xp.complex128)
        for row in (0, 1):
            col = row ^ m.cls
            block = packed[m.stored[row]] if row in m.stored else adj(packed[m.stored[row ^ 1]])
            out[row * d:(row + 1) * d, col * d:(col + 1) * d] = block
        return out

    def value(self, final):
        """excitation_contract at t = +a: tr(x4) + tr(hh x1h) in the last channel."""
        xp = self.xp
        x, z = (self._full(m, final) for m in self._last)
        return complex(xp.trace(z) + xp.sum(self.pair.left.T * x))


def diagonal_observable(pair, spec, integrator, integrate):
    """Port of excitation_observable for U = W through the packed system: the value of one
    monomial term. `integrate(system, points, endpoint, integrator)` is ode.integrate_host or
    fused.fused_integrator(cp)."""
    if not (integrator.max_point > 0 and integrator.atol > 0 and integrator.rtol > 0 and integrator.maxiters > 0):
        raise ValueError("invalid integration controls")
    system = PackedDiagonalPair(pair, spec)
    final = integrate(system, (), math.sqrt(integrator.max_point), integrator)[-1]
    if system.xp is np and hasattr(final, "get"):     # host pair integrated on a device
        final = final.get()
    if not bool(system.xp.isfinite(final).all()):
        raise SolverFailure("nonfinite excitation trajectory")
    return system.value(final)
