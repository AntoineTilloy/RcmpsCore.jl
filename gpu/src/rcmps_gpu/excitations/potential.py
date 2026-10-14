"""Potential (ODE) part of the Hamiltonian action in scaled variables: reference path.

Port of excitation_observable_action (src/excitation_action.jl). One term is a
forward trajectory of the pairs (x_k, y_k) and an adjoint trajectory of the covector
pairs (a0_k, a1_k), k over the channels of the Observable, on t in (-a, a) with
x = t|t|, followed by a quadrature pullback and boundary terms.

Variable types (scaled.py): x_k = S xh_k S (HH), y_k = yh_k S (LH), a1_k unscaled
(dual LL), a0_k = a0t_k S^-1 (dual LH). xh_k and a1_k do not depend on the tangent:
they are the ground-state trajectories.

This module keeps every channel live and unpacked; it is the reference for the
packed fused GPU kernel, which integrates the same equations. Both trajectories
are integrated as one system here (one shared adaptive step), as on the GPU; the
CPU control integrates them separately, which differs at integration tolerance.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .kinetic import Boundary
from .models import quadrature, transformed_source
from .scaled import lmul, rmul
from .states import SolverFailure, adj

BLOCKS = ("x", "y", "a1", "a0")   # state[block, channel] is a (D, D) matrix


class ActionSystem:
    """Right-hand side, initial data and pullback of one term for one tangent.

    `tangent` = (wh, vh, rw) hat variables, or None for the tangent-independent
    part alone (wh = 0). State layout: array (4, channels, D, D) in the order BLOCKS.
    """
    def __init__(self, ss, momentum, spec, tangent=None):
        xp = ss.xp
        self.ss, self.p, self.spec, self.xp = ss, float(momentum), spec, xp
        d = ss.D
        zero = xp.zeros((d, d), dtype=xp.complex128)
        self.wh, self.vh, self.rw = (zero, zero, zero) if tangent is None else tangent
        ql, rl, qh, rhh = ss.Q["L"], ss.R["L"], ss.Q["H"], ss.R["H"]
        if spec.commutator:
            self.c, self.ch = ql @ rl - rl @ ql, qh @ rhh - rhh @ qh
            self.cw = (ql @ self.wh - self.wh @ qh + self.vh @ rhh - rl @ self.vh
                       - 1j * self.p * self.wh)
        else:
            self.c, self.ch, self.cw = rl, rhh, self.wh
        self.cd, self.chd = adj(self.c), adj(self.ch)
        self.shape = (4, spec.channels, d, d)
        initial = xp.zeros(self.shape, dtype=xp.complex128)
        initial[0, 0] = ss.rh
        initial[1, 0] = self.rw
        initial[2, spec.channels - 1] = ss.eye
        self.initial = initial

    def rhs(self, y, jac, source):
        """d/dt of the stacked state for scalars jac = 2|t| and source = 2|t| J(t^2)."""
        ss, xp, p = self.ss, self.xp, self.p
        x, k, a1, a0 = y[0], y[1], y[2], y[3]
        ql, rl, qld, rld = ss.Q["L"], ss.R["L"], ss.Qd["L"], ss.Rd["L"]
        qh, rhh, qhd, rhd = ss.Q["H"], ss.R["H"], ss.Qd["H"], ss.Rd["H"]
        out = xp.empty_like(y)
        out[0] = jac * (qh @ x + x @ qhd + rhh @ x @ rhd)
        out[1] = jac * (ql @ k + k @ qhd + (rl @ k + self.wh @ x) @ rhd - 1j * p * k + self.vh @ x)
        out[2] = jac * (qld @ a1 + a1 @ ql + rld @ a1 @ rl)
        out[3] = jac * (qld @ a0 + a0 @ qh + rld @ a0 @ rhh + 1j * p * a0
                        + a1 @ self.vh + rld @ a1 @ self.wh)
        for edge in self.spec.edges:
            f = edge.coefficient * source
            src, dst = edge.src, edge.dst
            if edge.left:
                out[0, dst] += f * (self.ch @ x[src])
                out[1, dst] += f * (self.c @ k[src] + self.cw @ x[src])
                out[2, src] += f * (a1[dst] @ self.c)
                out[3, src] += f * (a0[dst] @ self.ch + a1[dst] @ self.cw)
            else:
                out[0, dst] += f * (x[src] @ self.chd)
                out[1, dst] += f * (k[src] @ self.chd)
                out[2, src] += f * (self.cd @ a1[dst])
                out[3, src] += f * (self.cd @ a0[dst])
        return out

    def bra_source_pull(self, y):
        ss = self.ss
        if not self.spec.commutator:
            return y
        return (ss.Qd["L"] @ y - y @ ss.Qd["H"] - ss.R["L"] @ y @ ss.Rd["H"]
                + ss.R["L"] @ ss.Rd["L"] @ y + 1j * self.p * y)

    def pullback(self, samples, points, weights, *, chunk=128):
        """Quadrature pullback. `samples[i]` is the state at points[i] for i < len(points)
        and at the endpoint +a for the last entry. Returns the bulk Fh, the boundary
        covector a0t (dual LH), and the tangent-independent boundary pair (a1, xout).
        Nodes are processed in batches of `chunk` (batched products, bounded memory)."""
        ss, xp = self.ss, self.xp
        n = len(points)
        rl, rhh, s = ss.R["L"], ss.R["H"], ss.s
        points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
        jacobian = weights * 2 * np.abs(points)
        source = weights * np.array([transformed_source(t) for t in points])
        right_edges = [edge for edge in self.spec.edges if not edge.left]
        bulk, traces = xp.zeros_like(ss.rh), 0.0
        edge_sum = xp.zeros_like(ss.rh)
        for first in range(0, n, chunk):
            last = min(n, first + chunk)
            forward = samples[first:last]
            mirrored = samples[n - last:n - first][::-1]    # adjoint at the mirrored nodes
            x, k, a1, a0 = forward[:, 0], forward[:, 1], mirrored[:, 2], mirrored[:, 3]
            w = xp.asarray(jacobian[first:last])[:, None, None, None]
            terms = (a0 @ rhh - rl @ a0) @ x + (a1 @ rl - rl @ a1) @ k + a1 @ self.wh @ x
            bulk += (w * terms).sum(axis=(0, 1))
            traces = traces + (w * xp.swapaxes(a1, -1, -2) * (s[:, None] * x * s[None, :])).sum()
            j = xp.asarray(source[first:last])[:, None, None]
            for edge in right_edges:
                edge_sum += edge.coefficient * (j * (a0[:, edge.dst] @ x[:, edge.src]
                                                     + a1[:, edge.dst] @ k[:, edge.src])).sum(axis=0)
        out = bulk - traces * (self.wh @ ss.rh) + self.bra_source_pull(edge_sum)
        end = samples[n]
        return out, end[3, 0], (end[2, 0], end[0, self.spec.channels - 1])


def integrate_scipy(system, points, endpoint, integrator):
    """Samples of the stacked state at `points` and at +endpoint by SciPy's DOP853,
    stopping at t = 0 where the source has a cusp (tstops=[0.] on the CPU)."""
    if system.xp is not np:
        raise ValueError("the SciPy reference integrator is CPU only")
    rtol = max(integrator.rtol, 2.3e-14)   # SciPy clips at 100 eps; do it explicitly
    shape = system.shape

    def fun(t, y):
        return system.rhs(y.reshape(shape), 2 * abs(t), transformed_source(t)).reshape(-1)

    points = np.asarray(points)
    y, pieces = system.initial.reshape(-1), []
    for lower, upper in ((-endpoint, 0.0), (0.0, endpoint)):
        inside = points[(points > lower) & (points <= upper)] if upper == 0 else points[points > lower]
        solution = solve_ivp(fun, (lower, upper), y, method="DOP853", t_eval=np.r_[inside, upper],
                             rtol=rtol, atol=integrator.atol)
        if not solution.success:
            raise SolverFailure(f"excitation action ODE failed: {solution.message}")
        if solution.nfev > integrator.maxiters:
            raise SolverFailure("excitation action ODE failed: MaxIters")
        values = solution.y.T
        pieces.append(values[:-1])
        y = values[-1]
    samples = np.concatenate(pieces + [y[None, :]]).reshape((len(points) + 1,) + shape)
    if not np.isfinite(samples[-1]).all():
        raise SolverFailure("nonfinite excitation action trajectory")
    return samples


class PotentialAction:
    """One ODE term of the Hamiltonian (a monomial; a kinetic spec is accepted for
    cross-checks). `integrate(system, points, endpoint, integrator)` returns the samples
    (default: the SciPy reference); `system(ss, momentum, spec, tangent)` builds the
    equations (default: ActionSystem). The coefficient of the term is applied by the caller."""
    def __init__(self, ss, momentum, spec, integrator, *, integrate=integrate_scipy, system=None,
                 tol=1e-12, **controls):
        self.ss, self.p, self.spec, self.integrator = ss, float(momentum), spec, integrator
        self.integrate, self.tol, self.controls = integrate, tol, controls
        self.system = system or ActionSystem   # e.g. the Z2-packed system of packed.py
        self.points, self.weights = quadrature(integrator)
        self.endpoint = math.sqrt(integrator.max_point)
        self.trajectories = 0
        # Tangent-independent boundary data from the ground-state trajectories (W = 0).
        _, _, (a1, xout) = self._run(self.system(ss, momentum, spec))
        xp = ss.xp
        d, _ = ss.poisson(xp.ascontiguousarray(a1[:, None, :]), "L", "L", dual=True, tol=tol, **controls)
        g, _ = ss.poisson(xp.ascontiguousarray(xout[:, None, :]), "H", "H", tol=tol, **controls)
        self.boundary = Boundary(d[:, 0, :], g[:, 0, :])

    def _run(self, system):
        samples = self.integrate(system, self.points, self.endpoint, self.integrator)
        self.trajectories += 1
        return system.pullback(samples, self.points, self.weights)

    def bulk(self, frame):
        """Bulk action and boundary covector a0t for every tangent of `frame`."""
        xp = self.ss.xp
        out, a0 = xp.zeros_like(frame.wh), xp.zeros_like(frame.wh)
        for t in range(frame.count):
            tangent = tuple(xp.ascontiguousarray(m[:, t, :]) for m in (frame.wh, frame.vh, frame.rw))
            out[:, t, :], a0[:, t, :], _ = self._run(self.system(self.ss, self.p, self.spec, tangent))
        return out, a0
