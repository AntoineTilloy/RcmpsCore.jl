"""Dense small-D reference of the tangent-space excitation solver (test ORACLE, D <= 16).

A literal NumPy port, formula by formula, of src/excitation_environments.jl,
src/excitation_action.jl, src/excitations.jl and of the solver part of
src/excitation_eigsolve.jl. It is never used by the production (matrix-free) solver.

Conventions
- W, U, V, F are (D, D) complex128 arrays in the gauge of `ctx.state`; V = -R'W.
- The excitation matrix element is H(U, W) = vdot(U, F(W)) = tr(U' F(W)).
- `terms` is a list of (coefficient, models.Observable), as models.phi4_terms returns.
- Julia's block axis comes first here: tangent environments are (4, D, D) arrays ordered
  (00, ket, bra, ket+bra), ODE states are (channels, blocks, D, D). Channels are zero based.
- Dense factors use the column-major vec of states.transfer_matrix. Resolvents are dense LU
  solves of (1 + i z shift p) - z L per node instead of Julia's shifted Schur form.
- Fixed kinetic theta grids only (Julia's kinetic_nodes=nothing refinement is not ported),
  kinetic_backend=:resolvent only.
- ODEs: scipy DOP853 on the halves (-a, 0) and (0, a) (Julia: Vern7 with tstops=[0.]),
  independent of the GPU runtime's integrator. rtol is clipped at 2.3e-14 (SciPy's floor);
  `maxiters` bounds the attempted steps through a count of right-hand sides (`integrate`).
  States saved at the quadrature points are DOP853 dense-output values (t_eval), less
  accurate than the steps themselves: for an ill-conditioned rho this, times the whitening,
  limits the action (2e-7 in white coordinates at D = 16, cond(rho) = 3e7, atol 1e-13;
  see test_stored_references.py). Bilinear elements save no intermediate state.
- Julia's ArgumentError and DimensionMismatch become ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

from rcmps_gpu.excitations import coordinates as tc  # `coordinates` below is always a TangentCoordinates argument
from rcmps_gpu.excitations import davidson as dv
from rcmps_gpu.excitations.models import default_terms, is_quadratic, kinetic_grid, quadrature, transformed_source
from rcmps_gpu.excitations.states import (IntegratorParameters, SolverFailure, State, adj, lindblad, lindblad_adjoint,
                    transfer_matrix, unvec, vec)

MAX_BOND = 16
norm = np.linalg.norm


def _solve(factor, b, adjoint=False):
    """reshape(F \\ vec(b), D, D), or F' \\ vec(b), for an LU factor F."""
    return unvec(lu_solve(factor, vec(b), trans=2 if adjoint else 0, check_finite=False), len(b))


def _check_tangents(s, *tangents):
    if any(np.shape(x) != s.R.shape for x in tangents):
        raise ValueError("tangent dimensions do not match the state")
    if not all(np.isfinite(x).all() for x in tangents):
        raise ValueError("nonfinite tangent")
    return [np.array(x, dtype=np.complex128) for x in tangents]


def check_kinetic_nodes(n):
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 2):
        raise ValueError("kinetic_nodes must be an integer of at least 2")
    return int(n)


# --- src/excitation_environments.jl -------------------------------------------------------

@dataclass
class Context:
    """ExcitationContext: pinned LU factors of -L+P, -L+ip+P and -L-ip+P, P = vec(rho) vec(1)',
    on a snapshot of the state. At p=0 the three factors are the same object."""
    state: State
    momentum: float
    tol: float
    transfer: np.ndarray
    poisson: tuple
    ketfactor: tuple
    brafactor: tuple


def transfer_factors(t, l, p, poisson=None):
    identity = np.eye(t.D * t.D, dtype=np.complex128)
    projector = np.outer(vec(t.rho), vec(np.eye(t.D)))
    if poisson is None:
        poisson = lu_factor(-l + projector)
    if p == 0:
        return poisson, poisson, poisson
    return poisson, lu_factor(-l + 1j * p * identity + projector), lu_factor(-l - 1j * p * identity + projector)


def make_context(state, p=0.0, *, tol=1e-10):
    """ExcitationContext(s, p; tol). The spectrum of L must hold one zero mode and otherwise
    decaying modes only (peripheral modes are rejected: boundary integrals need mixing)."""
    if not (math.isfinite(p) and math.isfinite(tol) and tol > 0):
        raise ValueError("invalid momentum or environment tolerance")
    if state.D > MAX_BOND:
        raise ValueError(f"dense excitation reference is limited to D<={MAX_BOND}")
    t = copy.deepcopy(state)
    l = transfer_matrix(t.Q, t.R)
    values = np.linalg.eigvals(l)
    scale = max(1.0, norm(l))
    null = np.abs(values) <= tol * scale
    if not (null.sum() == 1 and (null | (values.real < -tol * scale)).all()):
        raise SolverFailure("excitation environments require a unique, mixing transfer fixed point")
    poisson, ket, bra = transfer_factors(t, l, p)
    return Context(t, float(p), float(tol), l, poisson, ket, bra)


def retarget(ctx, *, momentum=None, integrator=None):
    """ExcitationContext(base; momentum, integrator): the Poisson factor is reused; the two
    shifted factors are recomputed only when the momentum changes. Own copy of the settings."""
    momentum = ctx.momentum if momentum is None else momentum
    if not math.isfinite(momentum):
        raise ValueError("invalid momentum")
    t = ctx.state.with_integrator(ctx.state.integrator if integrator is None else integrator)
    p = float(momentum)
    if p == ctx.momentum:
        ket, bra = ctx.ketfactor, ctx.brafactor
    else:
        _, ket, bra = transfer_factors(t, ctx.transfer, p, poisson=ctx.poisson)
    return Context(t, p, ctx.tol, ctx.transfer, ctx.poisson, ket, bra)


def excitation_poisson(ctx, b, shift=0):
    """x with (-L + i shift p) x = b - tr(b) rho and tr(x) = 0; shift in (0, 1, -1)."""
    s = ctx.state
    rhs = b - np.trace(b) * s.rho
    f = ctx.poisson if shift == 0 else ctx.ketfactor if shift == 1 else ctx.brafactor
    x = _solve(f, rhs)
    x = x - np.trace(x) * s.rho
    residual = -lindblad(s.Q, s.R, x) + (1j * shift * ctx.momentum) * x - rhs
    if not norm(residual) <= 20 * ctx.tol * max(1, norm(rhs)):
        raise SolverFailure("tangent boundary solve residual too large")
    return x


def tangent_metric(state, u, w):
    """The physical complex tangent inner product tr(w rho u'), with no regularization."""
    if not np.shape(u) == np.shape(w) == state.R.shape:
        raise ValueError("tangent dimensions do not match the state")
    return np.trace(w @ state.rho @ adj(u))


def ket_insertion(s, v, w, x):
    return v @ x + w @ x @ adj(s.R)


def bra_insertion(s, v, u, x):
    return x @ adj(v) + s.R @ x @ adj(u)


@dataclass
class Pair:
    """ExcitationPair: `right` holds the (00, ket, bra, connected double) environments as a
    (4, D, D) array; `left00` is the adjoint tail environment."""
    context: Context
    w: np.ndarray
    u: np.ndarray
    v: np.ndarray
    vbra: np.ndarray
    overlap: complex
    right: np.ndarray
    left00: np.ndarray


def make_pair(ctx, u, w):
    s = ctx.state
    u, w = _check_tangents(s, u, w)
    identity = np.eye(s.D, dtype=np.complex128)
    v, vbra = -adj(s.R) @ w, -adj(s.R) @ u
    n = tangent_metric(s, u, w)
    right = np.zeros((4, s.D, s.D), dtype=np.complex128)
    right[0] = s.rho
    right[1] = excitation_poisson(ctx, ket_insertion(s, v, w, s.rho), shift=1)
    right[2] = excitation_poisson(ctx, bra_insertion(s, vbra, u, s.rho), shift=-1)
    f = ket_insertion(s, v, w, right[2]) + bra_insertion(s, vbra, u, right[1]) + w @ s.rho @ adj(u)
    right[3] = excitation_poisson(ctx, f)
    b = adj(u) @ w - n * identity
    h = _solve(ctx.poisson, b, adjoint=True)
    h = h - np.trace(s.rho @ h) * identity
    if not norm(-lindblad_adjoint(s.Q, s.R, h) - b) <= 20 * ctx.tol * max(1, norm(b)):
        raise SolverFailure("tangent right-tail solve residual too large")
    return Pair(ctx, w, u, v, vbra, n, right, h)


def excitation_transfer(pair, x):
    """Connected four-block generator on x of shape (..., 4, D, D) (excitation_transfer!)."""
    s, p = pair.context.state, pair.context.momentum
    rh, uh, vbrah = adj(s.R), adj(pair.u), adj(pair.vbra)
    out = lindblad(s.Q, s.R, x)
    out[..., 1, :, :] -= 1j * p * x[..., 1, :, :]
    out[..., 2, :, :] += 1j * p * x[..., 2, :, :]
    for src, dst in ((0, 1), (2, 3)):
        out[..., dst, :, :] += pair.v @ x[..., src, :, :] + pair.w @ x[..., src, :, :] @ rh
    for src, dst in ((0, 2), (1, 3)):
        out[..., dst, :, :] += x[..., src, :, :] @ vbrah + s.R @ x[..., src, :, :] @ uh
    out[..., 3, :, :] += pair.w @ x[..., 0, :, :] @ uh - pair.overlap * x[..., 0, :, :]
    return out


def excitation_contract(pair, x):
    return np.trace(x[3]) + np.trace(pair.left00 @ x[0])


def excitation_operators(pair, commutator):
    s, p = pair.context.state, pair.context.momentum
    if commutator:
        c = s.Q @ s.R - s.R @ s.Q
        cw = s.Q @ pair.w - pair.w @ s.Q + pair.v @ s.R - s.R @ pair.v - 1j * p * pair.w
        cu = s.Q @ pair.u - pair.u @ s.Q + pair.vbra @ s.R - s.R @ pair.vbra - 1j * p * pair.u
        return c, cw, cu
    return s.R, pair.w, pair.u


def excitation_source_add(out, x, ops, left, a):
    """out += a * (normal-ordered annihilation (left) or creation (right) source on x); in place
    on the (4, D, D) array or view `out`."""
    c, cw, cu = ops
    if left:
        out += a * (c @ x)
        out[1] += a * (cw @ x[0])
        out[3] += a * (cw @ x[2])
    else:
        out += a * (x @ adj(c))
        out[2] += a * (x[0] @ adj(cu))
        out[3] += a * (x[1] @ adj(cu))
    return out


def resolvent_factor(ctx, z, shift):
    """LU of (1 + i z shift p) - z L (Julia: the same shifted matrix in the Schur basis of L)."""
    l = ctx.transfer
    return lu_factor((1 + 1j * z * shift * ctx.momentum) * np.eye(len(l), dtype=np.complex128) - z * l)


def resolvent_factors(ctx, z):
    f0 = resolvent_factor(ctx, z, 0)
    if ctx.momentum == 0:
        return f0, f0, f0
    return f0, resolvent_factor(ctx, z, 1), resolvent_factor(ctx, z, -1)


def excitation_resolvent(pair, b, z, factors=None):
    s = pair.context.state
    f0, fket, fbra = resolvent_factors(pair.context, z) if factors is None else factors
    x = np.empty_like(b)
    x[0] = _solve(f0, b[0])
    x[1] = _solve(fket, b[1] + z * ket_insertion(s, pair.v, pair.w, x[0]))
    x[2] = _solve(fbra, b[2] + z * bra_insertion(s, pair.vbra, pair.u, x[0]))
    rhs = b[3] + z * (ket_insertion(s, pair.v, pair.w, x[2]) + bra_insertion(s, pair.vbra, pair.u, x[1])
                      + pair.w @ x[0] @ adj(pair.u) - pair.overlap * x[0])
    x[3] = _solve(f0, rhs)
    return x


# --- ODE integration ----------------------------------------------------------------------

def quadrature_check(p):
    if not (p.max_point > 0 and p.atol > 0 and p.rtol > 0 and p.maxiters > 0):
        raise ValueError("invalid integration controls")


def integrate(rhs, u0, integrator, points, label):
    """Julia: solve(ODEProblem(rhs!, u0, (-a, a)); saveat=vcat(points, a), save_start=false,
    tstops=[0.], maxiters). Returns the saved states, shape (len(points) + 1, *u0.shape); the
    last one is at t = a. `points` are the (nonzero) save points inside (-a, a), ascending.
    Julia's step budget is emulated on right-hand-side evaluations: DOP853 spends 12 per
    attempted step, 3 more on a step that holds save points and 2 to start each half, so
    more than 15 maxiters + 4 of them certify more than maxiters attempted steps."""
    p = integrator
    a = math.sqrt(p.max_point)
    points = np.asarray(points, dtype=float)
    shape, y, saved = u0.shape, u0.reshape(-1), []
    budget, evaluations = 15 * p.maxiters + 4, 0

    def flat(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > budget:
            raise SolverFailure(f"{label} ODE failed: MaxIters")
        return rhs(t, y.reshape(shape)).reshape(-1)

    for t0, t1, inner in ((-a, 0.0, points[points < 0]), (0.0, a, points[points > 0])):
        sol = solve_ivp(flat, (t0, t1), y, method="DOP853", t_eval=np.r_[inner, t1],
                        rtol=max(p.rtol, 2.3e-14), atol=p.atol)
        if not sol.success:
            raise SolverFailure(f"{label} ODE failed: {sol.message}")
        y = sol.y[:, -1].copy()
        saved.append(sol.y.T[:len(inner)] if t1 == 0 else sol.y.T)
    out = np.concatenate(saved).reshape(-1, *shape)
    if not np.isfinite(out[-1]).all():
        raise SolverFailure(f"nonfinite {label} trajectory")
    return out


# --- src/excitation_action.jl -------------------------------------------------------------

def action_operators(ctx, w, v, commutator):
    s = ctx.state
    if not commutator:
        return s.R, w
    return s.Q @ s.R - s.R @ s.Q, s.Q @ w - w @ s.Q + v @ s.R - s.R @ v - 1j * ctx.momentum * w


def action_rhs(ctx, w, v, spec, c, cw, adjoint):
    """action_rhs! on u of shape (channels, 2, D, D): blocks (ground, ket) forward, and their
    covectors for the adjoint (which is integrated forward too: the sources are even in t)."""
    s, p = ctx.state, ctx.momentum
    rh, ch = adj(s.R), adj(c)

    def forward(t, u):
        jac, j = 2 * abs(t), transformed_source(t)
        du = lindblad(s.Q, s.R, u)
        x = u[:, 0]
        du[:, 1] += -1j * p * u[:, 1] + v @ x + w @ x @ rh
        du *= jac
        for edge in spec.edges:
            a = edge.coefficient * j
            if edge.left:
                du[edge.dst] += a * (c @ u[edge.src])
                du[edge.dst, 1] += a * (cw @ u[edge.src, 0])
            else:
                du[edge.dst] += a * (u[edge.src] @ ch)
        return du

    def backward(t, u):
        jac, j = 2 * abs(t), transformed_source(t)
        du = lindblad_adjoint(s.Q, s.R, u)
        y = u[:, 1]
        du[:, 0] += 1j * p * u[:, 0] + y @ v + rh @ y @ w
        du *= jac
        for edge in spec.edges:
            a = edge.coefficient * j
            src, dst = edge.dst, edge.src
            if edge.left:
                du[dst] += a * (u[src] @ c)
                du[dst, 0] += a * (u[src, 1] @ cw)
            else:
                du[dst] += a * (ch @ u[src])
        return du

    return backward if adjoint else forward


def action_trajectory(ctx, w, v, rw, spec, points, adjoint):
    s = ctx.state
    u0 = np.zeros((spec.channels, 2, s.D, s.D), dtype=np.complex128)
    if adjoint:
        u0[-1, 1] = np.eye(s.D)
    else:
        u0[0, 0], u0[0, 1] = s.rho, rw
    c, cw = action_operators(ctx, w, v, spec.commutator)
    return integrate(action_rhs(ctx, w, v, spec, c, cw, adjoint), u0, s.integrator, points,
                     "excitation action")


def bra_transfer_pull(s, a, x):
    return (a @ s.R - s.R @ a) @ x


def ket_trace_dual(s, v, w, a):
    return a @ v + adj(s.R) @ a @ w


def bra_source_pull(ctx, x, commutator):
    s = ctx.state
    if not commutator:
        return x
    qh, rh = adj(s.Q), adj(s.R)
    return qh @ x - x @ qh - s.R @ x @ rh + s.R @ rh @ x + 1j * ctx.momentum * x


def action_dual_poisson(ctx, b, shifted=False):
    """x with (-L* - i p [shifted]) x = b - tr(rho b) and tr(rho x) = 0."""
    s = ctx.state
    identity = np.eye(s.D, dtype=np.complex128)
    rhs = b - np.trace(s.rho @ b) * identity
    x = _solve(ctx.ketfactor if shifted else ctx.poisson, rhs, adjoint=True)
    x = x - np.trace(s.rho @ x) * identity
    residual = -lindblad_adjoint(s.Q, s.R, x) - (1j * ctx.momentum * x if shifted else 0) - rhs
    if not norm(residual) <= 20 * ctx.tol * max(1, norm(rhs)):
        raise SolverFailure("excitation action boundary residual too large")
    return x


def action_boundaries(ctx, w, v, rw, a0, a1, xout):
    s = ctx.state
    d = action_dual_poisson(ctx, a1)
    e = action_dual_poisson(ctx, a0 + ket_trace_dual(s, v, w, d), shifted=True)
    return (bra_transfer_pull(s, e, s.rho) + bra_transfer_pull(s, d, rw) + d @ w @ s.rho
            + w @ excitation_poisson(ctx, xout))


def observable_action(ctx, w, v, rw, spec):
    """excitation_observable_action: forward and adjoint trajectories saved at the quadrature
    points, quadrature pullback of the bra tangent, and the two boundary terms."""
    s = ctx.state
    points, weights = quadrature(s.integrator)
    forward = action_trajectory(ctx, w, v, rw, spec, points, False)
    backward = action_trajectory(ctx, w, v, rw, spec, points, True)
    x, a = forward[:-1], backward[-2::-1]  # a[i] is Julia's backward.u[length(points)-i+1]
    dx = weights * 2 * np.abs(points)
    j = weights * np.array([transformed_source(t) for t in points])
    wrho = w @ s.rho
    pulled = ((a @ s.R - s.R @ a) @ x).sum(axis=2) + a[:, :, 1] @ w @ x[:, :, 0]
    traces = np.einsum("ikab,ikba->i", a[:, :, 1], x[:, :, 0])
    out = np.einsum("i,ikab->ab", dx, pulled) - (dx @ traces) * wrho
    for edge in spec.edges:
        if edge.left:
            continue
        b = np.einsum("i,iab->ab", j, a[:, edge.dst, 0] @ x[:, edge.src, 0]
                      + a[:, edge.dst, 1] @ x[:, edge.src, 1])
        out += edge.coefficient * bra_source_pull(ctx, b, spec.commutator)  # linear in b
    out += action_boundaries(ctx, w, v, rw, backward[-1, 0, 0], backward[-1, 0, 1], forward[-1, -1, 0])
    return out


def kinetic_action_sum(ctx, w, v, rw, terms, n):
    """excitation_kinetic_action_sum: kinetic specs (two source orders) and quadratic monomials
    (four orders) on an n-point theta grid, with one boundary pullback for all of them."""
    s = ctx.state
    rho = s.rho
    zero = np.zeros((s.D, s.D), dtype=np.complex128)
    out, wrho = zero.copy(), w @ rho
    total_a0, total_a1, total_x0 = zero.copy(), zero.copy(), zero.copy()
    prepared = [(coefficient, spec, *action_operators(ctx, w, v, spec.commutator))
                for coefficient, spec in terms]
    for theta, nodeweight in zip(*kinetic_grid(n)):
        z = math.cos(theta)
        factor = resolvent_factor(ctx, z, 0)
        shiftedfactor = factor if ctx.momentum == 0 else resolvent_factor(ctx, z, 1)
        for coefficient, spec, c, cw in prepared:
            ch = adj(c)
            orders = (((False, True), (True, False)) if spec.kinetic else
                      ((False, False), (False, True), (True, False), (True, True)))
            weight = nodeweight * coefficient / 2
            for left, secondleft in orders:
                # First source and forward resolvent, retaining the ground and ket blocks.
                b0 = c @ rho if left else rho @ ch
                b1 = c @ rw + cw @ rho if left else rw @ ch
                x0 = _solve(factor, b0)
                x1 = _solve(shiftedfactor, b1 + z * ket_insertion(s, v, w, x0))
                xout = c @ x0 if secondleft else x0 @ ch
                # Reverse second source; the terminal mixed-block covector is 1.
                if not secondleft:  # second source is a bra source
                    a0, a1 = zero, ch
                    out += weight * bra_source_pull(ctx, x1, spec.commutator)
                else:
                    a0, a1 = cw, c
                a1 = _solve(factor, a1, adjoint=True)
                a0 = _solve(shiftedfactor, a0 + z * ket_trace_dual(s, v, w, a1), adjoint=True)
                out += (weight * z) * (bra_transfer_pull(s, a0, x0) + bra_transfer_pull(s, a1, x1)
                                       + a1 @ w @ x0 - np.trace(a1 @ x0) * wrho)
                # Reverse first source and accumulate the two boundary covectors.
                if left:
                    a0, a1 = a0 @ c + a1 @ cw, a1 @ c
                else:
                    out += weight * bra_source_pull(ctx, a0 @ rho + a1 @ rw, spec.commutator)
                    a0, a1 = ch @ a0, ch @ a1
                total_a0 += weight * a0
                total_a1 += weight * a1
                total_x0 += weight * xout
    out += action_boundaries(ctx, w, v, rw, total_a0, total_a1, total_x0)
    return out


def hamiltonian_action(ctx, w, terms, *, kinetic_nodes=256):
    """F(W) with H(U, W) = vdot(U, F) (excitation_hamiltonian_action, kinetic_backend=:resolvent,
    fixed nodes): kinetic specs and quadratic monomials through the resolvent sum, every other
    term through its ODE action."""
    s = ctx.state
    w, = _check_tangents(s, w)
    kinetic_nodes = check_kinetic_nodes(kinetic_nodes)
    v = -adj(s.R) @ w
    rw = excitation_poisson(ctx, ket_insertion(s, v, w, s.rho), shift=1)
    kinetics = [term for term in terms if term[1].kinetic or is_quadratic(term[1])]
    potentials = [term for term in terms if not (term[1].kinetic or is_quadratic(term[1]))]
    out = np.zeros((s.D, s.D), dtype=np.complex128)
    for coefficient, spec in potentials:
        out += coefficient * observable_action(ctx, w, v, rw, spec)
    if kinetics:
        out += kinetic_action_sum(ctx, w, v, rw, kinetics, kinetic_nodes)
    if not np.isfinite(out).all():
        raise SolverFailure("nonfinite Hamiltonian action")
    return out


# --- src/excitations.jl -------------------------------------------------------------------

def observable_element(pair, spec):
    """excitation_observable: value-only four-block ODE, contracted at t = a."""
    s = pair.context.state
    quadrature_check(s.integrator)
    u0 = np.zeros((spec.channels, 4, s.D, s.D), dtype=np.complex128)
    u0[0] = pair.right
    ops = excitation_operators(pair, spec.commutator)

    def rhs(t, u):
        jac, j = 2 * abs(t), transformed_source(t)
        du = excitation_transfer(pair, u)
        du *= jac
        for edge in spec.edges:
            excitation_source_add(du[edge.dst], u[edge.src], ops, edge.left, edge.coefficient * j)
        return du

    final = integrate(rhs, u0, s.integrator, (), "excitation")[-1]
    return excitation_contract(pair, final[-1])


def kinetic_sum(pair, terms, n):
    """excitation_kinetic_sum (bilinear). `terms` must hold kinetic specs only, as selected by
    `matrix_element`: only the two mixed source orders are summed."""
    if not all(spec.kinetic for _, spec in terms):
        raise ValueError("kinetic_sum accepts kinetic specs only")
    result = 0j
    sources = []
    for weight, spec in terms:
        ops = excitation_operators(pair, spec.commutator)
        bs = [excitation_source_add(np.zeros_like(pair.right), pair.right, ops, left, 1)
              for left in (False, True)]
        sources.append((weight, ops, bs))
    for theta, w in zip(*kinetic_grid(n)):
        z = math.cos(theta)
        factors = resolvent_factors(pair.context, z)
        for weight, ops, bs in sources:
            for left, b in zip((False, True), bs):
                x = excitation_resolvent(pair, b, z, factors)
                y = excitation_source_add(np.zeros_like(x), x, ops, not left, 1)
                result += (weight * w / 2) * excitation_contract(pair, y)
    return result


def matrix_element(pair, terms, *, kinetic_nodes=256):
    """excitation_element with kinetic_backend=:resolvent and fixed nodes: kinetic specs through
    the resolvent sum, every other term (quadratic monomials included) through its ODE."""
    result = 0j
    kinetics = [term for term in terms if term[1].kinetic]
    if kinetics:
        result += kinetic_sum(pair, kinetics, check_kinetic_nodes(kinetic_nodes))
    for weight, spec in terms:
        if spec.kinetic:
            continue
        result += weight * observable_element(pair, spec)
    if not np.isfinite(result):
        raise SolverFailure("nonfinite excitation matrix element")
    return complex(result)


def dense_hamiltonian(state, p, terms, *, sector="all", parity=None, gauge="original", metric_tol=1e-12,
                      assembly="action", kinetic_nodes=64, environment_tol=1e-10, hermitian_tol=1e-7,
                      parity_tol=1e-12):
    """excitation_hamiltonian with a FIXED kinetic grid: the dense Hamiltonian in an orthonormal
    tangent basis, assembled from actions ('action') or entry by entry ('bilinear').
    parity=None means no parity for sector='all' and 'auto' otherwise (the Julia default).
    The returned basis is in the gauge of the input state."""
    if not (math.isfinite(hermitian_tol) and hermitian_tol > 0):
        raise ValueError("invalid Hermiticity tolerance")
    if assembly not in ("bilinear", "action"):
        raise ValueError("assembly must be 'bilinear' or 'action'")
    if not math.isfinite(p):
        raise ValueError("invalid momentum or environment tolerance")
    if state.D > MAX_BOND:
        raise ValueError(f"dense excitation reference is limited to D<={MAX_BOND}")
    terms = default_terms(state) if terms is None else terms
    if parity is None and sector != "all":
        parity = "auto"
    coordinates = tc.make_coordinates(state, sector=sector, parity=parity, parity_tol=parity_tol,
                                      gauge=gauge, metric_tol=metric_tol)
    ctx = make_context(coordinates.state, p, tol=environment_tol)
    basis = tc.tangent_basis(coordinates)
    n = len(basis)
    h = np.empty((n, n), dtype=np.complex128)
    for j in range(n):
        if assembly == "action":
            action = hamiltonian_action(ctx, basis[j], terms, kinetic_nodes=kinetic_nodes)
            for i in range(n):
                h[i, j] = np.vdot(basis[i], action)
        else:
            for i in range(n):
                h[i, j] = matrix_element(make_pair(ctx, basis[i], basis[j]), terms,
                                         kinetic_nodes=kinetic_nodes)
    defect = float(norm(h - adj(h)) / max(1, norm(h)))
    if not defect <= hermitian_tol:
        raise SolverFailure(f"excitation Hamiltonian Hermiticity defect {defect} exceeds {hermitian_tol}")
    return dict(hamiltonian=(h + adj(h)) / 2, basis=[tc.original_tangent(coordinates, w) for w in basis],
                momentum=float(p), hermiticity_error=defect, metric_rank=coordinates.metric_rank,
                discarded_metric_weight=coordinates.discarded_metric_weight, sector=coordinates.sector,
                coordinates=coordinates)


def spectrum(state, p, terms, *, nev=1, **kw):
    """excitation_spectrum: lowest Ritz energies of `dense_hamiltonian`, with normalized W
    vectors (gauge of the input state) and algebraic residuals."""
    if not nev > 0:
        raise ValueError("nev must be positive")
    result = dense_hamiltonian(state, p, terms, **kw)
    h, basis = result["hamiltonian"], result["basis"]
    n = len(h)
    if not nev <= n:
        raise ValueError(f"nev={nev} exceeds the tangent dimension {n}")
    values, eigenvectors = np.linalg.eigh(h)
    energies, coefficients = values[:nev], eigenvectors[:, :nev]
    vectors = [sum(coefficients[i, j] * basis[i] for i in range(n)) for j in range(nev)]
    residuals = [float(norm(h @ coefficients[:, j] - energies[j] * coefficients[:, j])) for j in range(nev)]
    return dict(result, energies=energies, vectors=vectors, coefficients=coefficients, residuals=residuals)


# --- src/excitation_eigsolve.jl -----------------------------------------------------------

class ExcitationOperator:
    """Hermitian excitation operator in white coordinates, x -> [F(W(x)) B]_sector, with a fixed
    kinetic grid. `calls` counts applications."""

    def __init__(self, context, coordinates, terms, kinetic_nodes=256):
        if context.state.D != coordinates.state.D:
            raise ValueError("context and coordinates differ")
        self.context, self.coordinates, self.terms = context, coordinates, terms
        self.kinetic_nodes = check_kinetic_nodes(kinetic_nodes)
        self.calls = 0

    def __call__(self, x):
        f = hamiltonian_action(self.context, tc.tangent_matrix(self.coordinates, x), self.terms,
                               kinetic_nodes=self.kinetic_nodes)
        self.calls += 1
        return tc.covector_coordinates(self.coordinates, f)

    def with_context(self, context, kinetic_nodes=None):
        return ExcitationOperator(context, self.coordinates, self.terms,
                                  self.kinetic_nodes if kinetic_nodes is None else kinetic_nodes)


def excitation_operator(ctx, coordinates, terms, *, kinetic_nodes=256):
    return ExcitationOperator(ctx, coordinates, terms, kinetic_nodes)


def kinetic_preconditioner(ctx, coordinates, terms, *, nodes=8, floor=None):
    """Positive preconditioner from the coarse kinetic matrix K (kinetic specs of `terms` only)
    on a `nodes`-point theta grid: r -> sum_k v_k <v_k, r> / max(floor, kappa_k). The floor
    defaults to the largest kinetic coefficient."""
    c, s, n = coordinates, ctx.state, len(coordinates)
    kinetics = [term for term in terms if term[1].kinetic]
    if not kinetics:
        raise ValueError("the kinetic preconditioner requires kinetic terms")
    nodes = check_kinetic_nodes(nodes)
    columns = []
    for unit in np.eye(n):
        w = tc.tangent_matrix(c, unit)
        v = -adj(s.R) @ w
        rw = excitation_poisson(ctx, ket_insertion(s, v, w, s.rho), shift=1)
        columns.append(tc.covector_coordinates(c, kinetic_action_sum(ctx, w, v, rw, kinetics, nodes)))
    k = np.column_stack(columns)
    if not np.isfinite(k).all():
        raise SolverFailure("nonfinite kinetic preconditioner")
    values, vectors = np.linalg.eigh((k + adj(k)) / 2)
    lower = max(coefficient for coefficient, _ in kinetics) if floor is None else float(floor)
    if not (math.isfinite(lower) and lower > 0):
        raise ValueError("preconditioner floor must be positive")

    def apply(r):
        return vectors @ ((adj(vectors) @ r) / np.maximum(lower, values))

    return dict(apply=apply, matrix=k, values=values, vectors=vectors,
                skew=float(norm(k - adj(k)) / max(1, norm(k))), floor=lower, nodes=nodes)


def field_seed(ctx, coordinates):
    """White coordinates of the gauge-fixed field insertion phi(0)|Psi> at the context momentum:
    W = 1 + [R, X] with (L* + i p) X = R + R' (trace-projected). Only a start vector."""
    s = ctx.state
    x = -action_dual_poisson(ctx, s.R + adj(s.R), shifted=True)
    return tc.tangent_coordinates(coordinates, np.eye(s.D, dtype=np.complex128) + s.R @ x - x @ s.R)


STARTS = ("auto", "field", "kinetic", "random")


def start_list(start):
    """One coordinate vector (1-D) or one tangent ((D, D) array, as a Julia matrix), else a
    sequence of them."""
    if isinstance(start, np.ndarray) and start.ndim <= 2:
        return [start]
    starts = list(start)
    return [np.asarray(starts)] if starts and all(np.ndim(x) == 0 for x in starts) else starts


def initial_vectors(op, start, nev, pre):
    c = op.coordinates
    n = len(c)

    def as_coordinates(x):
        x = np.asarray(x)
        if x.ndim == 2:  # a tangent in the gauge of the input state
            if x.shape != (c.state.D, c.state.D):
                raise ValueError("start tangent has the wrong size")
            return tc.tangent_coordinates(c, tc.coordinate_tangent(c, x.astype(np.complex128)))
        if x.shape != (n,):
            raise ValueError("start coordinates have the wrong length")
        return x.astype(np.complex128)

    def kinetic(k):
        return [] if pre is None else [pre["vectors"][:, j].copy() for j in range(min(k, n))]

    if isinstance(start, str):
        if start not in STARTS:
            raise ValueError("start must be 'auto', 'field', 'kinetic', 'random', coordinates or tangents")
        if start in ("auto", "field"):
            x = field_seed(op.context, c)
            # The field insertion has unit norm up to its gauge part; ~0 means another sector.
            if norm(x) > 1e-6:
                return [x] + kinetic(nev - 1), "field"
            if start == "field":
                raise ValueError(f"the field insertion vanishes in the {c.sector} sector")
        if start in ("auto", "kinetic") and pre is not None:
            return kinetic(nev), "kinetic"
        if start == "kinetic":
            raise ValueError("start='kinetic' requires preconditioner='kinetic'")
        rng = np.random.default_rng(1234)  # Julia: MersenneTwister(1234)
        return [(rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
                for _ in range(nev)], "random"
    return [as_coordinates(x) for x in start_list(start)], "supplied"


def refined_integration(p):
    return IntegratorParameters(n_quad=2 * p.n_quad, max_point=p.max_point + 10, atol=p.atol / 10,
                                rtol=p.rtol / 10, maxiters=p.maxiters)


def fresh_level(op, x):
    """One fresh action: a Rayleigh quotient with an exact (for that action) residual."""
    h = op(x)
    value = np.vdot(x, h) / np.vdot(x, x)
    return dict(energy=float(value.real), imaginary=float(value.imag),
                residual=float(norm(h - value.real * x) / norm(x)), action=h)


def bilinear_level(ctx, op, x, *, kinetic_nodes):
    w = tc.tangent_matrix(op.coordinates, x)
    value = (matrix_element(make_pair(ctx, w, w), op.terms, kinetic_nodes=kinetic_nodes)
             / tangent_metric(ctx.state, w, w))
    return dict(energy=float(value.real), imaginary=float(value.imag))


def check_levels_request(coordinates, *, nev=1, davidson_options=None, start="auto", preconditioner="kinetic",
                         refine_steps=0, kinetic_nodes=256, preconditioner_nodes=8,
                         acceptance_residual=1e-5, validation_tol=1e-7, validate=True, refined=None,
                         refined_kinetic_nodes=None):
    """Validate the cheap request before any dense factorization; returns the Davidson options."""
    n, d = len(coordinates), coordinates.state.D
    integer = (int, np.integer)
    if not (isinstance(nev, integer) and 1 <= nev <= n):
        raise ValueError(f"nev={nev} must lie between 1 and the sector dimension {n}")
    options = dv.DavidsonOptions(maxspace=max(8, 2 * nev + 2)) if davidson_options is None else davidson_options
    dv.check_davidson(options, nev)
    if not (isinstance(refine_steps, integer) and refine_steps >= 0 and (refine_steps == 0 or nev == 1)):
        raise ValueError("refine_steps must be a nonnegative integer and requires nev=1")
    if not (callable(preconditioner) or preconditioner in ("kinetic", "none")):
        raise ValueError("preconditioner must be 'kinetic', 'none', or a function")
    if isinstance(start, str):
        if start not in STARTS:
            raise ValueError("start must be 'auto', 'field', 'kinetic', 'random', coordinates or tangents")
        if start == "kinetic" and preconditioner != "kinetic":
            raise ValueError("start='kinetic' requires preconditioner='kinetic'")
    else:
        starts = start_list(start)
        for x in starts:
            if np.shape(x) not in ((n,), (d, d)):
                raise ValueError(f"start entries must be length-{n} coordinates or {d}x{d} tangents")
        if not 1 <= len(starts) < options.maxspace:
            raise ValueError(f"between 1 and maxspace-1={options.maxspace - 1} start vectors are accepted")
    if not isinstance(validate, bool):
        raise ValueError("validate must be true or false")
    if refined is not None:
        if not isinstance(refined, IntegratorParameters):
            raise ValueError("refined must be None or IntegratorParameters")
        quadrature(refined)
    if refined_kinetic_nodes is not None:
        check_kinetic_nodes(refined_kinetic_nodes)
    check_kinetic_nodes(kinetic_nodes)
    if preconditioner == "kinetic":
        check_kinetic_nodes(preconditioner_nodes)
    if not (math.isfinite(acceptance_residual) and acceptance_residual > 0
            and math.isfinite(validation_tol) and validation_tol > 0):
        raise ValueError("acceptance_residual and validation_tol must be positive")
    return options


def solve_levels(coordinates, ctx, terms, *, nev=1, kinetic_nodes=256, preconditioner="kinetic",
                 preconditioner_nodes=8, start="auto", davidson_options=None, validate=True, refined=None,
                 refined_kinetic_nodes=None, refine_steps=0, validation_tol=1e-7, acceptance_residual=1e-5,
                 callback=None, context_seconds=0.0):
    """solve_levels: preconditioned Davidson search, optional refinement of the first level
    under the refined action, and validation by fresh actions and bilinear energies."""
    options = check_levels_request(
        coordinates, nev=nev, davidson_options=davidson_options, start=start, preconditioner=preconditioner,
        refine_steps=refine_steps, kinetic_nodes=kinetic_nodes, preconditioner_nodes=preconditioner_nodes,
        acceptance_residual=acceptance_residual, validation_tol=validation_tol, validate=validate,
        refined=refined, refined_kinetic_nodes=refined_kinetic_nodes)
    op = ExcitationOperator(ctx, coordinates, terms, kinetic_nodes)
    tick = time.time()
    pre = (kinetic_preconditioner(ctx, coordinates, terms, nodes=preconditioner_nodes)
           if preconditioner == "kinetic" else None)
    apply = pre["apply"] if pre is not None else (lambda r: r) if preconditioner == "none" else preconditioner
    preconditioner_seconds = time.time() - tick
    starts, startkind = initial_vectors(op, start, nev, pre)
    tick = time.time()
    result = dv.davidson(op, starts, nev=nev, precondition=apply, options=options, callback=callback)
    solve_seconds = time.time() - tick
    coefficients = [np.array(x, dtype=np.complex128) for x in result["vectors"]]
    energies, residuals, refined_converged = list(result["energies"]), list(result["residuals"]), None
    tick = time.time()
    reference = refined_ctx = None
    corrections = []
    if validate or refine_steps > 0:
        refined_params = refined_integration(ctx.state.integrator) if refined is None else refined
        refined_nodes = 2 * op.kinetic_nodes if refined_kinetic_nodes is None else refined_kinetic_nodes
        refined_ctx = retarget(ctx, integrator=refined_params)
        reference = op.with_context(refined_ctx, kinetic_nodes=refined_nodes)
    if refine_steps > 0:
        # Preconditioned corrections under the refined action, accepted only when the fresh
        # residual decreases. The refined vector's own refined Rayleigh quotient and residual
        # replace the Davidson values.
        x = coefficients[0]
        level = fresh_level(reference, x)
        corrections.append(dict(step=0, energy=level["energy"], residual=level["residual"], accepted=True))
        for step in range(1, refine_steps + 1):
            if level["residual"] <= options.tol:
                break
            r = level["action"] - level["energy"] * x
            d = apply(r)
            d = d - x * np.vdot(x, d)
            d = d / norm(d)
            shifted = op(d) - level["energy"] * d
            alpha = np.vdot(shifted, r) / np.vdot(shifted, shifted).real
            trial = x - alpha * d
            trial = trial / norm(trial)
            candidate = fresh_level(reference, trial)
            accepted = candidate["residual"] < level["residual"]
            corrections.append(dict(step=step, energy=candidate["energy"], residual=candidate["residual"],
                                    accepted=accepted))
            if not accepted:
                break
            x, level = trial, candidate
        coefficients[0] = x
        energies[0], residuals[0] = level["energy"], level["residual"]
        refined_converged = level["residual"] <= options.tol
    validation = None
    if validate:
        validation = []
        for x in coefficients:
            standard, fine = fresh_level(op, x), fresh_level(reference, x)
            bilinear = bilinear_level(ctx, op, x, kinetic_nodes=op.kinetic_nodes)
            refined_bilinear = bilinear_level(refined_ctx, reference, x, kinetic_nodes=reference.kinetic_nodes)
            values = (standard["energy"], fine["energy"], bilinear["energy"], refined_bilinear["energy"])
            imaginary = max(abs(level["imaginary"]) for level in (standard, fine, bilinear, refined_bilinear))
            spread = max(values) - min(values)
            residual = max(standard["residual"], fine["residual"])
            consistent = spread <= validation_tol and imaginary <= validation_tol
            validation.append(dict(
                standard_energy=standard["energy"], standard_residual=standard["residual"],
                refined_energy=fine["energy"], refined_residual=fine["residual"],
                bilinear_energy=bilinear["energy"], refined_bilinear_energy=refined_bilinear["energy"],
                energy_spread=spread, max_imaginary=imaginary,
                residual_target_met=residual <= validation_tol, consistent=consistent,
                accepted=consistent and residual <= acceptance_residual))
    validation_seconds = time.time() - tick
    # Validated levels are accepted on the research criteria (energy agreement, imaginary
    # parts, fresh residuals); without validation only a converged search (and refinement)
    # is accepted. Every requested level must be present.
    accepted = len(coefficients) == nev and (
        (result["converged"] and (True if refined_converged is None else refined_converged))
        if validation is None else all(v["accepted"] for v in validation))
    vectors = [tc.original_tangent(coordinates, tc.tangent_matrix(coordinates, x)) for x in coefficients]
    return dict(
        energies=energies, vectors=vectors, coefficients=coefficients, residuals=residuals,
        converged=result["converged"], stop_reason=result["stop_reason"],
        refined_converged=refined_converged, accepted=bool(accepted), nev=nev,
        validation_tol=validation_tol, acceptance_residual=acceptance_residual,
        numops=result["numops"], action_calls=op.calls, history=result["history"],
        momentum=ctx.momentum, sector=coordinates.sector, parity=coordinates.parity,
        parity_error=coordinates.parity_error, gauge=coordinates.gauge, dimension=len(coordinates),
        metric_rank=coordinates.metric_rank, discarded_metric_weight=coordinates.discarded_metric_weight,
        metric_defect=coordinates.metric_defect, density_condition=coordinates.density_condition,
        start=startkind, davidson=options,
        preconditioner=None if pre is None else dict(nodes=pre["nodes"], skew=pre["skew"], floor=pre["floor"],
                                                     minimum=float(pre["values"].min())),
        corrections=corrections, validation=validation,
        validated_energies=None if validation is None else [v["refined_bilinear_energy"] for v in validation],
        integration=ctx.state.integrator,
        refined_integration=None if refined_ctx is None else refined_ctx.state.integrator,
        kinetic_nodes=op.kinetic_nodes,
        refined_kinetic_nodes=None if reference is None else reference.kinetic_nodes,
        timings=dict(context=context_seconds, preconditioner=preconditioner_seconds, solve=solve_seconds,
                     validation=validation_seconds),
        coordinates=coordinates)


def eigsolve(state, p=0.0, terms=None, *, nev=1, sector="auto", parity="auto", gauge="density",
             metric_tol=0.0, environment_tol=1e-10, kinetic_nodes=256, preconditioner="kinetic",
             preconditioner_nodes=8, start="auto", davidson_options=None, validate=True, refined=None,
             refined_kinetic_nodes=None, refine_steps=0, validation_tol=1e-7, acceptance_residual=1e-5,
             callback=None, parity_tol=1e-12):
    """excitation_eigsolve on the dense factors: lowest `nev` levels at momentum p by the
    preconditioned Davidson iteration on Hamiltonian actions. terms=None is
    models.default_terms(state) (phi4 states only). `start` is 'auto' (field insertion, else the lowest
    coarse kinetic modes), 'field', 'kinetic', 'random', white coordinate vectors, or (D, D)
    tangents in the gauge of `state`. Returns the dict of `solve_levels`."""
    if not math.isfinite(p):
        raise ValueError("invalid momentum")
    if state.D > MAX_BOND:
        raise ValueError(f"dense excitation reference is limited to D<={MAX_BOND}")
    terms = default_terms(state) if terms is None else terms
    request = dict(nev=nev, kinetic_nodes=kinetic_nodes, preconditioner=preconditioner,
                   preconditioner_nodes=preconditioner_nodes, start=start, davidson_options=davidson_options,
                   validate=validate, refined=refined, refined_kinetic_nodes=refined_kinetic_nodes,
                   refine_steps=refine_steps, validation_tol=validation_tol,
                   acceptance_residual=acceptance_residual)
    tick = time.time()
    coordinates = tc.make_coordinates(state, sector=sector, parity=parity, parity_tol=parity_tol,
                                                  gauge=gauge, metric_tol=metric_tol)
    check_levels_request(coordinates, **request)
    ctx = make_context(coordinates.state, p, tol=environment_tol)
    return solve_levels(coordinates, ctx, terms, callback=callback, context_seconds=time.time() - tick,
                        **request)
