"""Matrix-free kinetic (resolvent) part of the Hamiltonian action, batched over theta nodes.

Port of excitation_kinetic_action_sum (src/excitation_action.jl) to the scaled
variables of scaled.py, with two structural changes that leave the result unchanged:

* Every resolvent is a batched preconditioned GMRES solve; no transfer matrix,
  Schur form or LU factor exists.
* The tangent-independent halves are computed once. For each source operator
  c (R, or [Q,R] for the commutator term) the forward family x0(z) = (1-zL)^-1 (rho c')
  and the covector family a1(z) = (1-zL*)^-1 c do not depend on W (the other two
  CPU solves are their adjoints), and neither do the boundary solutions built
  from them. An action then needs, per node and per c, two ket solves and two
  covector solves instead of the CPU's eight.

Variable types (see scaled.py): x0 = S xh S (HH), ket-inserted y = yh S (LH),
a1 unscaled (dual LL), a0 = a0t S^-1 (dual LH), W = Wh S^-1, V = Vh S^-1, F = Fh S.
With scaling "none" every hat quantity is the CPU one.
"""
from __future__ import annotations

import numpy as np

from models import is_quadratic, kinetic_grid
from scaled import lmul, rmul, stack
from states import adj


def node_sum(x, weights):
    """sum_n weights[n] x[:, n, :]"""
    return (x * weights[None, :, None]).sum(axis=1)


def contract(a, b, weights, xp):
    """sum_n weights[n] a[:, n, :] @ b[:, n, :] as one GEMM."""
    d, n, _ = a.shape
    right = xp.ascontiguousarray(xp.moveaxis(b * weights[None, :, None], 1, 0)).reshape(n * d, d)
    return a.reshape(d, n * d) @ right


def batch_adjoint(x, xp):
    return xp.ascontiguousarray(xp.conj(xp.swapaxes(x, 0, 2)))


class Boundary:
    """Tangent-independent boundary data of one or several Hamiltonian terms.

    For accumulated covectors (a0, a1) and forward output xout of a term, the CPU
    action_boundaries adds pull(e, rho) + pull(d, rw) + d w rho + w G with d = dual
    Poisson(a1), G = Poisson(xout), e = shifted dual Poisson(a0 + ket_trace_dual(d)).
    a1 and xout never depend on W, so `d` and `g` are stored; boundaries are linear,
    so those of several terms add and one e solve serves the whole Hamiltonian.
    """
    def __init__(self, d, g):
        self.d, self.g = d, g

    def __add__(self, other):
        return Boundary(self.d + other.d, self.g + other.g)

    def scaled(self, factor):
        return Boundary(factor * self.d, factor * self.g)


class TangentFrame:
    """A batch of tangents with everything every term needs: Wh, Vh = -R' Wh and the
    ket Poisson environment rw = (-L + i p)^-1 (V rho + W rho R'), as hat variables."""
    def __init__(self, ss, momentum, wh, *, tol=1e-12, **controls):
        xp = ss.xp
        self.ss, self.p, self.count = ss, float(momentum), wh.shape[1]
        self.wh = xp.ascontiguousarray(wh)
        self.vh = -lmul(ss.Rd["L"], self.wh)
        rh = ss.rh
        source = rmul(self.vh, rh) + rmul(rmul(self.wh, rh), ss.Rd["H"])
        self.rw, self.rw_info = ss.poisson(source, "L", "H", 1j * self.p, tol=tol, **controls)

    def ket(self, x, t):
        """ket_insertion for tangent t on a batch of HH matrices -> LH."""
        ss = self.ss
        return lmul(self.vh[:, t, :], x) + rmul(lmul(self.wh[:, t, :], x), ss.Rd["H"])

    def ket_trace_dual(self, a, t):
        """a v + R' a w for tangent t on a batch of dual LL matrices -> dual LH."""
        ss = self.ss
        return rmul(a, self.vh[:, t, :]) + lmul(ss.Rd["L"], rmul(a, self.wh[:, t, :]))


def finish_boundary(frame, boundary, a0, *, tol=1e-12, **controls):
    """Boundary terms of the action for every tangent of `frame`, given the summed
    tangent-dependent covectors a0 (batch, dual LH) of all terms."""
    ss, xp = frame.ss, frame.ss.xp
    n = frame.count
    d = boundary.d
    rhs = a0.copy()
    for t in range(n):
        rhs[:, t, :] += frame.ket_trace_dual(d[:, None, :], t)[:, 0, :]
    e, info = ss.poisson(rhs, "L", "H", -1j * frame.p, dual=True, tol=tol, **controls)
    rl, rhh, rh = ss.R["L"], ss.R["H"], ss.rh
    out = rmul(rmul(e, rhh) - lmul(rl, e), rh)                 # pull(e, rho)
    out += lmul(d @ rl - rl @ d, frame.rw)                      # pull(d, rw)
    out += lmul(d, rmul(frame.wh, rh)) + rmul(frame.wh, boundary.g)   # d w rho + w G
    return out, info


class KineticAction:
    """Kinetic and quadratic-monomial terms through the theta resolvent quadrature.

    `terms`: [(coefficient, Observable)] with kinetic specs and/or quadratic monomials.
    `nodes` fixed theta nodes make this part exactly linear in W up to solver tolerance.
    """
    def __init__(self, ss, momentum, terms, nodes, *, tol=1e-12, **controls):
        xp = ss.xp
        self.ss, self.p, self.tol, self.controls = ss, float(momentum), tol, controls
        theta, weights = kinetic_grid(nodes)
        self.nodes = int(nodes)
        self.z_host, self.weights_host = np.cos(theta), weights
        self.z, self.weights = xp.asarray(self.z_host), xp.asarray(weights)
        self.terms = []
        for coefficient, spec in terms:
            if spec.kinetic:
                orders = ((False, True), (True, False))
            elif is_quadratic(spec):
                orders = ((False, False), (False, True), (True, False), (True, True))
            else:
                raise ValueError("only kinetic specs and quadratic monomials use the resolvent")
            self.terms.append((float(coefficient), bool(spec.commutator), orders))
        if not self.terms:
            raise ValueError("no kinetic term")
        self.sources = sorted({commutator for _, commutator, _ in self.terms})
        self.stats = dict(solves=0, iterations=0, max_residual=0.0)
        self._prepare()

    # ---- tangent-independent part ----------------------------------------------------
    def operators(self, commutator):
        """(c, ch): the source operator in the L gauge and in the hat gauge."""
        ss = self.ss
        if not commutator:
            return ss.R["L"], ss.R["H"]
        return (ss.Q["L"] @ ss.R["L"] - ss.R["L"] @ ss.Q["L"],
                ss.Q["H"] @ ss.R["H"] - ss.R["H"] @ ss.Q["H"])

    def _record(self, info):
        self.stats["solves"] += len(info["iterations"])
        self.stats["iterations"] += int(info["iterations"].sum())
        self.stats["max_residual"] = max(self.stats["max_residual"], float(info["residuals"].max()))

    def _prepare(self):
        ss, xp, n = self.ss, self.ss.xp, self.nodes
        rh = ss.rh
        rhs_x = stack([rh @ adj(self.operators(c)[1]) for c in self.sources for _ in range(n)], xp)
        rhs_a = stack([self.operators(c)[0] for c in self.sources for _ in range(n)], xp)
        z = xp.concatenate([self.z] * len(self.sources))
        x0, info = ss.resolvent(rhs_x, "H", "H", z, tol=self.tol, **self.controls)
        self._record(info)
        a1, info = ss.resolvent(rhs_a, "L", "L", z, dual=True, tol=self.tol, **self.controls)
        self._record(info)
        s = ss.s
        self.x0, self.a1, self.pull1, self.kappa = {}, {}, {}, {}
        for i, c in enumerate(self.sources):
            part = slice(i * n, (i + 1) * n)
            right, second = xp.ascontiguousarray(x0[:, part, :]), xp.ascontiguousarray(a1[:, part, :])
            self.x0[c] = {False: right, True: batch_adjoint(right, xp)}      # keyed by `left`
            self.a1[c] = {True: second, False: batch_adjoint(second, xp)}    # keyed by `secondleft`
            self.pull1[c] = {k: rmul(v, ss.R["L"]) - lmul(ss.R["L"], v) for k, v in self.a1[c].items()}
        # Scalars kappa = sum_n w z tr(a1 x0) and the tangent-independent boundary sums.
        total_a1 = xp.zeros_like(rh)
        total_x0 = xp.zeros_like(rh)
        wz = self.weights * self.z
        for coefficient, c, orders in self.terms:
            op, oph = self.operators(c)
            for left, second in orders:
                weight = coefficient / 2
                x, a = self.x0[c][left], self.a1[c][second]
                scaled_x = s[:, None, None] * x * s[None, None, :]
                self.kappa[(coefficient, c, left, second)] = weight * xp.sum(
                    wz[None, :, None] * xp.swapaxes(a, 0, 2) * scaled_x)
                a_sum, x_sum = node_sum(a, self.weights), node_sum(x, self.weights)
                total_a1 += weight * (a_sum @ op if left else adj(op) @ a_sum)
                total_x0 += weight * (oph @ x_sum if second else x_sum @ adj(oph))
        d, info = ss.poisson(xp.ascontiguousarray(total_a1[:, None, :]), "L", "L", dual=True,
                             tol=self.tol, **self.controls)
        self._record(info)
        g, info = ss.poisson(xp.ascontiguousarray(total_x0[:, None, :]), "H", "H", tol=self.tol,
                             **self.controls)
        self._record(info)
        self.boundary = Boundary(d[:, 0, :], g[:, 0, :])

    # ---- tangent-dependent part ------------------------------------------------------
    def source_operators(self, frame, t, commutator):
        """cwh for tangent t: W itself, or its commutator lift (hat variables)."""
        ss, p = self.ss, self.p
        wh, vh = frame.wh[:, t, :], frame.vh[:, t, :]
        if not commutator:
            return wh
        return (ss.Q["L"] @ wh - wh @ ss.Q["H"] + vh @ ss.R["H"] - ss.R["L"] @ vh - 1j * p * wh)

    def bra_source_pull(self, y, commutator):
        ss = self.ss
        if not commutator:
            return y
        return (ss.Qd["L"] @ y - y @ ss.Qd["H"] - ss.R["L"] @ y @ ss.Rd["H"]
                + ss.R["L"] @ ss.Rd["L"] @ y + 1j * self.p * y)

    def bulk(self, frame):
        """Bulk action and accumulated covector a0 (dual LH) for every tangent of `frame`.
        The boundary terms are added by `finish_boundary` with `self.boundary`."""
        ss, xp, n, p = self.ss, self.ss.xp, self.nodes, self.p
        z, weights = self.z, self.weights
        wz = weights * z
        rh = ss.rh
        count = frame.count
        keys = [(t, c, side) for t in range(count) for c in self.sources for side in (False, True)]
        rhs_y, rhs_a = [], []
        cw = {}
        for t, c, side in keys:
            op, oph = self.operators(c)
            cw[(t, c)] = self.source_operators(frame, t, c)
            rw = frame.rw[:, t, :]
            b1 = op @ rw + cw[(t, c)] @ rh if side else rw @ adj(oph)       # side = left
            rhs_y.append(b1[:, None, :] + z[None, :, None] * frame.ket(self.x0[c][side], t))
            a0 = cw[(t, c)] if side else xp.zeros_like(rh)                   # side = secondleft
            rhs_a.append(a0[:, None, :] + z[None, :, None] * frame.ket_trace_dual(self.a1[c][side], t))
        zs = xp.concatenate([z] * len(keys))
        y, info = ss.resolvent(xp.ascontiguousarray(xp.concatenate(rhs_y, axis=1)), "L", "H", zs,
                               1 + 1j * zs * p, tol=self.tol, **self.controls)
        self._record(info)
        a, info = ss.resolvent(xp.ascontiguousarray(xp.concatenate(rhs_a, axis=1)), "L", "H", zs,
                               1 - 1j * zs * p, dual=True, tol=self.tol, **self.controls)
        self._record(info)
        index = {key: slice(i * n, (i + 1) * n) for i, key in enumerate(keys)}
        out = xp.zeros_like(frame.wh)
        total_a0 = xp.zeros_like(frame.wh)
        rl, rhh = ss.R["L"], ss.R["H"]
        for t in range(count):
            wh, rw = frame.wh[:, t, :], frame.rw[:, t, :]
            wrh = wh @ rh
            f = xp.zeros_like(rh)
            a0_total = xp.zeros_like(rh)
            for coefficient, c, orders in self.terms:
                op, oph = self.operators(c)
                for left, second in orders:
                    weight = coefficient / 2
                    y1 = xp.ascontiguousarray(y[:, index[(t, c, left)], :])
                    a0 = xp.ascontiguousarray(a[:, index[(t, c, second)], :])
                    x0, a1 = self.x0[c][left], self.a1[c][second]
                    a0_sum, a1_sum = node_sum(a0, weights), node_sum(a1, weights)
                    if not second:
                        f += weight * self.bra_source_pull(node_sum(y1, weights), c)
                    f += weight * (contract(rmul(a0, rhh) - lmul(rl, a0), x0, wz, xp)
                                   + contract(self.pull1[c][second], y1, wz, xp)
                                   + contract(rmul(a1, wh), x0, wz, xp))
                    f -= self.kappa[(coefficient, c, left, second)] * wrh
                    if left:
                        a0_total += weight * (a0_sum @ oph + a1_sum @ cw[(t, c)])
                    else:
                        f += weight * self.bra_source_pull(a0_sum @ rh + a1_sum @ rw, c)
                        a0_total += weight * (adj(op) @ a0_sum)
            out[:, t, :] = f
            total_a0[:, t, :] = a0_total
        return out, total_a0

    def action(self, frame):
        """Complete kinetic action Fh (batch) for the tangents of `frame`."""
        out, a0 = self.bulk(frame)
        boundary, info = finish_boundary(frame, self.boundary, a0, tol=self.tol, **self.controls)
        self._record(info)
        return out + boundary


def kinetic_terms(terms):
    """The terms of a model that the action route sends to the resolvent quadrature."""
    return [(c, spec) for c, spec in terms if spec.kinetic or is_quadratic(spec)]
