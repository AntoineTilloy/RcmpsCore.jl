"""Isolated CPU stationary-state correction using the existing strict solver.

For A(X)=-L(X)+tr(X)I/D, solve A(delta)=I/D-A(rho_previous)
with exactly the same no-jump/Sherman-Morrison right preconditioner and GMRES
    controls as numerics.stationary_state (tightened if the correction RHS norm
    exceeds one). The full A(rho)=I/D residual is checked
before the same normalization, Hermiticity, positivity and stationary-residual
gates.  There is no density cutoff, extra shift, relaxed tolerance or fallback.
This module does not import an observable backend and never uses a GPU.
"""

from time import perf_counter

import numpy as np

from .numerics import NoJumpInverse, SolverFailure, _right_solve, adj, lindblad


def stationary_state_correction(q, r, rho_previous, *, tolerance=1e-11,
                                maxiter=200):
    """Return ``(rho, stats)`` without mutating inputs or silently solving cold.

    ``rho_previous`` must be a finite, normalized Hermitian complex128 density.
    It need not be stationary for the new ``q,r``.  The GMRES absolute tolerance
    is unchanged; a nearby state's smaller correction RHS can therefore need
    fewer iterations without weakening the accuracy of the full solution.
    """
    begin = perf_counter()
    q, r, previous = (np.asarray(x) for x in (q, r, rho_previous))
    if (q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape != r.shape
            or q.shape != previous.shape):
        raise ValueError("q, r and rho_previous must be equally sized square matrices")
    if any(x.dtype != np.complex128 for x in (q, r, previous)):
        raise ValueError("stationary correction requires complex128 inputs")
    if any(not np.isfinite(x).all() for x in (q, r, previous)):
        raise ValueError("stationary correction inputs must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0 or maxiter < 1:
        raise ValueError("tolerance and maxiter must be positive")
    if (abs(np.trace(previous)-1) > 100*tolerance
            or np.linalg.norm(previous-adj(previous)) > 100*tolerance):
        raise ValueError("rho_previous must be normalized and Hermitian")
    if np.linalg.eigvalsh(previous).min() < -10*tolerance:
        raise ValueError("rho_previous must be positive within the existing tolerance")

    d = len(q)
    identity = np.eye(d, dtype=np.complex128)
    inverse = NoJumpInverse(q, r)
    z = inverse(identity)
    denominator = 1+np.trace(z)/d

    def pre(y):
        x = inverse(y)
        return x-np.trace(x)/(d*denominator)*z

    def op(x):
        return -lindblad(q, r, x)+np.trace(x)/d*identity

    rhs = identity/d-op(previous)
    rhs_norm = float(np.linalg.norm(rhs))
    # _right_solve supplies the same number as both GMRES rtol and atol. For a
    # large correction RHS, lower that number so rtol*||rhs|| cannot exceed the
    # original cold solve's absolute goal (||I/D|| <= 1).
    solve_tolerance = tolerance/max(1., rhs_norm)
    setup_seconds = perf_counter()-begin
    solve_begin = perf_counter()
    correction, stats = _right_solve(op, pre, rhs, tolerance=solve_tolerance,
                                     maxiter=maxiter)
    solve_seconds = perf_counter()-solve_begin
    rho = previous+correction
    full_residual = float(np.linalg.norm(op(rho)-identity/d))
    full_rhs_norm = float(np.linalg.norm(identity/d))
    if (not np.isfinite(full_residual)
            or full_residual > 20*tolerance*max(1., full_rhs_norm)):
        raise SolverFailure(f"Full stationary correction residual failed: {full_residual:.3e}")

    # These gates intentionally match numerics.stationary_state, including the
    # order of normalization, Hermiticity checking and symmetrization.
    rho = rho/np.trace(rho)
    if np.linalg.norm(rho-adj(rho)) > 100*tolerance:
        raise SolverFailure("Stationary state is not Hermitian")
    rho = (rho+adj(rho))/2
    residual = np.linalg.norm(lindblad(q, r, rho))
    scale = max(1., 2*np.linalg.norm(q, 2)+np.linalg.norm(r, 2)**2)
    eigenvalues = np.linalg.eigvalsh(rho)
    if residual > tolerance*scale or eigenvalues.min() < -10*tolerance:
        raise SolverFailure(f"Invalid stationary density: residual={residual:.3e}, min={eigenvalues.min():.3e}")
    stats.update(correction_residual=float(stats["residual"]),
                 residual=full_residual, stationary_residual=float(residual),
                 eigmin=float(eigenvalues.min()),
                 correction_rhs_norm=rhs_norm, gmres_tolerance=float(solve_tolerance),
                 correction_norm=float(np.linalg.norm(correction)),
                 full_rhs_norm=full_rhs_norm, preconditioner_shift=float(inverse.shift),
                 setup_seconds=float(setup_seconds), solve_seconds=float(solve_seconds),
                 total_seconds=float(perf_counter()-begin),
                 warm_start=True, fallback_used=False)
    return rho, stats
