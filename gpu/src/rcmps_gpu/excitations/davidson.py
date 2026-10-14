"""Restarted Hermitian Rayleigh-Ritz (Davidson) iteration: port of src/davidson.jl.

NumPy only. The control flow follows the Julia source line by line (pending/stale
queues, fresh and thick restarts, expansion and breakdown rules, maxops accounting).

Conventions: vectors are 1-D complex128 arrays; `count`, `cycle` and `numops` are
counts starting at one, as in Julia, while root indices are zero based; the inner
product is vdot(u, v) = u^H v. History rows are dicts. Julia's ArgumentError and
DimensionMismatch become ValueError, SolverFailure is states.SolverFailure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from states import SolverFailure

EPS = float(np.finfo(float).eps)
norm = np.linalg.norm


@dataclass(frozen=True)
class DavidsonOptions:
    """Controls of `davidson` (DavidsonOptions of src/davidson.jl).

    `maxspace` bounds the search space. At a restart the `retain` lowest Ritz vectors
    are kept (None means nev). With `refresh` their actions are recomputed (fresh
    restart), otherwise recombined from stored columns (thick restart). The search
    stops as 'converged' (every targeted residual <= tol), 'maxops', 'stalled' (see
    `davidson_stalled`; stall_window=0 disables it) or 'breakdown' (no direction
    orthogonal to the search space remains while a residual exceeds tol).
    """
    tol: float = 1e-7
    maxops: int = 200
    maxspace: int = 8
    retain: int | None = None
    refresh: bool = True
    hermitian_tol: float = 1e-7
    stall_window: int = 20
    stall_minops: int = 40
    stall_rtol: float = 0.1
    stall_residual_limit: float | None = None
    stall_energy_atol: float = 1e-9


def stall_limit(o):
    return max(1e-5, 10 * o.tol) if o.stall_residual_limit is None else o.stall_residual_limit


def check_davidson(o, nev):
    """Validate the controls and return `retain` (check_davidson)."""
    retain = nev if o.retain is None else o.retain
    if not nev >= 1:
        raise ValueError("nev must be positive")
    if not (math.isfinite(o.tol) and o.tol > 0 and o.maxops >= 1
            and math.isfinite(o.hermitian_tol) and o.hermitian_tol > 0):
        raise ValueError("invalid Davidson controls")
    if not nev <= retain < o.maxspace:
        raise ValueError(f"require nev <= retain < maxspace (nev={nev}, retain={retain}, maxspace={o.maxspace})")
    if not o.maxops >= nev:
        raise ValueError("maxops must be at least nev")
    if not o.stall_window >= 0:
        raise ValueError("stall_window must be nonnegative")
    if o.stall_window > 0:
        limit = stall_limit(o)
        if not (o.stall_minops >= 1 and math.isfinite(o.stall_rtol) and 0 < o.stall_rtol < 1
                and math.isfinite(o.stall_energy_atol) and o.stall_energy_atol >= 0):
            raise ValueError("invalid Davidson stagnation controls")
        if not (math.isfinite(limit) and limit >= o.tol):
            raise ValueError(f"stall_residual_limit ({limit}) must be at least tol ({o.tol})")
    return retain


def davidson_stalled(history, o):
    """Stall rule (davidson_stalled). Rows need `residual` (largest targeted residual) and
    `energies`. Only rows that carry every targeted root enter: a partially rebuilt space
    after a fresh restart does not. Over those rows (at least stall_minops and more than
    stall_window of them) it requires: last residual <= stall_limit, best residual improved
    by less than a fraction stall_rtol over the last stall_window rows, and every energy
    varied by at most stall_energy_atol over the last stall_window+1 rows."""
    window = o.stall_window
    if not (window > 0 and len(history) > 0):
        return False
    nev = len(history[-1]["energies"])
    full = [row for row in history if len(row["energies"]) == nev]
    n = len(full)
    if not n >= max(o.stall_minops, window + 1):
        return False
    if not full[-1]["residual"] <= stall_limit(o):
        return False
    old_best = min(row["residual"] for row in full[:n - window])
    best = min(row["residual"] for row in full)
    if not best >= (1 - o.stall_rtol) * old_best:
        return False
    recent = full[n - window - 1:]
    for k in range(nev):
        values = [row["energies"][k] for row in recent]
        if not max(values) - min(values) <= o.stall_energy_atol:
            return False
    return True


def orthogonalize(x, basis):
    """In place; two modified Gram-Schmidt passes avoid losing a new direction to roundoff."""
    for _ in range(2):
        for b in basis:
            x -= b * np.vdot(b, x)
    return x


def start_vectors(x0):
    """One vector (1-D array or sequence of numbers) or a sequence of vectors."""
    if isinstance(x0, np.ndarray):
        if x0.ndim != 1:
            raise ValueError("x0 must be one vector or a sequence of vectors, not a matrix")
        return [x0]
    starts = list(x0)
    return [starts] if starts and all(np.ndim(x) == 0 for x in starts) else starts


def columns(a):
    return [a[:, k].copy() for k in range(a.shape[1])]


def davidson(op, x0, *, nev=1, precondition=None, options=DavidsonOptions(), callback=None):
    """Lowest `nev` eigenpairs of a Hermitian linear map known only through `op(x)`.

    `x0` is a start vector or a sequence of start vectors (at most maxspace-1 independent
    ones). `precondition(r)` maps a residual to a correction direction (None = identity);
    the expansion vector is -precondition(r). The projected matrix V^H A V is formed from
    the actual action columns; its anti-Hermitian part must stay below
    options.hermitian_tol, otherwise SolverFailure is raised. Residuals are recomputed from
    stored actions at every step. `callback(row, U, AU)` receives the history row and n x m
    matrices whose columns are the normalized current Ritz vectors and their actions.

    Returns a dict: energies, vectors (normalized), actions, residuals, converged,
    stop_reason ('converged', 'stalled', 'maxops' or 'breakdown'), numops, history. A
    'maxops' stop always returns nev pairs. Where Julia draws the deterministic direction
    for an invariant subspace smaller than nev from MersenneTwister(count), this port
    uses np.random.default_rng(count).
    """
    o = options
    retain = check_davidson(o, nev)
    starts = start_vectors(x0)
    if not starts:
        raise ValueError("at least one start vector is required")
    pending = []
    for x in starts:
        y = np.array(x, dtype=np.complex128)
        if y.ndim != 1:
            raise ValueError("start vectors must be one-dimensional")
        if not np.isfinite(y).all():
            raise ValueError("start vectors must be finite")
        if pending and len(y) != len(pending[0]):
            raise ValueError("start vectors differ in length")
        scale = norm(y)
        orthogonalize(y, pending)
        if scale > 0 and norm(y) > math.sqrt(EPS) * scale:
            pending.append(y / norm(y))
    if not pending:
        raise ValueError("start vectors must be nonzero and independent")
    if not len(pending) < o.maxspace:
        raise ValueError(f"at most maxspace-1={o.maxspace - 1} start vectors are accepted")
    n = len(pending[0])
    if not nev <= n:
        raise ValueError(f"nev={nev} exceeds the dimension {n}")
    stale = [None for _ in pending]
    vs, avs, history, cycle = [], [], [], 1
    for count in range(1, o.maxops + 1):
        x, previous = pending.pop(0), stale.pop(0)
        orthogonalize(x, vs)
        x /= norm(x)
        ax = np.array(op(x), dtype=np.complex128)
        if not (ax.shape == (n,) and np.isfinite(ax).all()):
            raise SolverFailure("invalid operator action")
        change = 0.0 if previous is None else float(norm(ax - previous))
        vs.append(x)
        avs.append(ax)
        v, av = np.column_stack(vs), np.column_stack(avs)
        projected = v.conj().T @ av
        defect = float(norm(projected - projected.conj().T) / max(1, norm(projected)))
        if not defect <= o.hermitian_tol:
            raise SolverFailure(f"Davidson projected operator is not Hermitian: {defect}")
        ritz = np.linalg.eigh((projected + projected.conj().T) / 2)[1]  # ascending
        m = min(nev, len(vs))
        u, au = v @ ritz[:, :m], av @ ritz[:, :m]
        scale = norm(u, axis=0)
        u /= scale
        au /= scale
        energies = [float(np.vdot(u[:, k], au[:, k]).real) for k in range(m)]
        r = au - u * np.array(energies)
        residuals = [float(norm(r[:, k])) for k in range(m)]
        row = dict(count=count, cycle=cycle, dimension=len(vs), energies=energies, residuals=residuals,
                   energy=energies[0], residual=max(residuals), hermiticity=defect,
                   orthogonality=float(norm(v.conj().T @ v - np.eye(len(vs)))), refresh_change=change)
        history.append(row)
        if callback is not None:
            callback(row, u, au)
        converged = m == nev and row["residual"] <= o.tol
        stalled = not converged and m == nev and davidson_stalled(history, o)

        def finish(reason):
            return dict(energies=energies, vectors=columns(u), actions=columns(au), residuals=residuals,
                        converged=reason == "converged", stop_reason=reason, numops=count, history=history)

        if converged:
            return finish("converged")
        if stalled:
            return finish("stalled")
        if count == o.maxops:
            return finish("maxops")
        if pending:  # still rebuilding the retained space after a fresh restart
            continue
        if len(vs) >= o.maxspace:
            keep = ritz[:, :min(retain, len(vs))]
            kept, keptactions = v @ keep, av @ keep
            cycle += 1
            # A fresh rebuild needs nev calls before all targeted roots are present again.
            if o.refresh and o.maxops - count < nev:
                return finish("maxops")
            if o.refresh:
                vs, avs = [], []
                pending, stale = columns(kept), columns(keptactions)
                continue
            vs, avs = columns(kept), columns(keptactions)

        # Expand along the lowest unconverged root whose residual leaves the search space.
        def expansion(k):
            y = -np.array(r[:, k] if precondition is None else precondition(r[:, k].copy()),
                          dtype=np.complex128)
            if not (y.shape == (n,) and np.isfinite(y).all()):
                raise SolverFailure("invalid preconditioned residual")
            orthogonalize(y, vs)
            if norm(y) < 1e-13 * max(1, residuals[k]):
                y = orthogonalize(r[:, k].copy(), vs)
            return y if norm(y) > EPS * max(1, residuals[k]) else None

        candidates = [k for k in range(m) if residuals[k] > o.tol] or [m - 1]
        y = None
        for k in candidates:
            y = expansion(k)
            if y is not None:
                break
        if y is None:
            # Every targeted residual lies in the search space: with every root present no
            # direction is left (e.g. a spanned sector at the anti-Hermitian floor).
            if m == nev:
                return finish("breakdown")
            # An invariant subspace smaller than nev: continue with a deterministic direction.
            rng = np.random.default_rng(count)
            y = orthogonalize((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2), vs)
        pending.append(y / norm(y))
        stale.append(None)
