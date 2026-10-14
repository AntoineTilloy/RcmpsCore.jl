"""Parity sectors and white tangent coordinates (src/excitation_eigsolve.jl, lines 32-140).

NumPy/SciPy only. A coefficient vector x maps to the gauge-fixed tangent W = M B'
(V = -R'W) with M[rows, cols] = x and B' rho B = 1 on the resolved metric support, so
that tr(W rho W') = |x|^2. `rows`, `cols` are ZERO-BASED and enumerate Julia's `indices`
in its order: kept columns ascending, then rows ascending (column-major).

For a Z2-symmetric state physical parity acts as W -> -Z W Z: block-diagonal W form
the 'odd' sector and block-off-diagonal W the 'even' sector. Coordinates and all
environments use `coordinates.state`; convert tangents with `original_tangent` and
`coordinate_tangent`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.linalg import eigh, lapack, solve_triangular

from .states import SolverFailure, State, adj, excitation_parity, make_state
from .states import parity_error as state_parity_error

EPS, TINY = np.finfo(float).eps, np.finfo(float).tiny  # Julia eps(), floatmin()


@dataclass
class TangentCoordinates:
    state: State            # working gauge: rotated for gauge='density', the input otherwise
    rotation: np.ndarray    # unitary u, state.R = u' R_input u; identity for gauge='original'
    whitening: np.ndarray   # B; column j lies in the parity block of j
    rows: np.ndarray
    cols: np.ndarray
    parity: np.ndarray | None
    parity_error: float     # of the input state; nan without parity
    sector: str
    gauge: str
    metric_rank: int
    discarded_metric_weight: float
    metric_defect: float
    density_condition: float

    def __len__(self):
        return len(self.rows)


def _eigen(a):
    """eigen(Hermitian(a)): zheevr on the upper triangle, ascending eigenvalues."""
    return eigh(a, lower=False, driver="evr")


def _cholesky(a):
    """Upper factor U, a = U'U, of cholesky(Hermitian(a); check=false); None on failure."""
    u, info = lapack.zpotrf(a, lower=0, clean=1)
    return u if info == 0 else None


def make_coordinates(state, *, sector="auto", parity="auto", gauge="density", metric_tol=0.0,
                     parity_tol=1e-12):
    """TangentCoordinates(s; sector, parity, gauge, metric_tol, parity_tol).

    gauge='density' rotates into the eigenbasis of rho within each parity block (an exact
    unitary gauge change) and whitens by a Cholesky factor; eigenvalues of rho at or below
    max(metric_tol, 10 D eps) max(rho) are dropped, and eigen-whitening on the kept
    directions replaces Cholesky as soon as one is dropped or Cholesky fails."""
    if not (math.isfinite(metric_tol) and 0 <= metric_tol < 1):
        raise ValueError("metric_tol must lie in [0,1)")
    if gauge not in ("density", "original"):
        raise ValueError("gauge must be 'density' or 'original'")
    z = excitation_parity(state, parity, tol=parity_tol)
    if sector == "auto":
        sector = "all" if z is None else "odd"
    if sector not in ("all", "odd", "even"):
        raise ValueError("sector must be 'auto', 'all', 'odd' or 'even'")
    if sector != "all" and z is None:
        raise ValueError(f"the {sector} sector requires a Z2-symmetric state; supply its parity")
    d = state.D
    blocks = [np.arange(d)] if z is None else [np.flatnonzero(z == 1), np.flatnonzero(z == -1)]
    # Block eigenvectors of rho: off-block elements of rho are zero by parity.
    u, lam = np.zeros((d, d), dtype=np.complex128), np.zeros(d)
    for blk in blocks:
        ix = np.ix_(blk, blk)
        lam[blk], u[ix] = _eigen(state.rho[ix])
    cutoff = max(metric_tol, 10 * d * EPS) * lam.max()
    keep = lam > cutoff
    if not keep.any():
        raise SolverFailure("empty tangent metric support")
    if gauge == "density":
        rho = adj(u) @ state.rho @ u
        t = make_state(adj(u) @ state.R @ u, adj(u) @ state.K @ u, (rho + adj(rho)) / 2,
                       integrator=replace(state.integrator), model=state.model,
                       coupling=state.coupling, e0=state.e0, source=state.source)
        rotation = u
    else:
        t, rotation = state, np.eye(d, dtype=np.complex128)
    # Column j of B lies in the parity block of j. Cholesky keeps every column;
    # eigen-whitening keeps the resolved eigen-directions.
    b, colkeep = np.zeros((d, d), dtype=np.complex128), np.ones(d, dtype=bool)
    for blk in blocks:
        ix = np.ix_(blk, blk)
        f = _cholesky(t.rho[ix]) if keep.all() else None
        if f is not None:
            b[ix] = solve_triangular(f, np.eye(len(blk)), lower=False)  # inv(f.U)
        else:
            values, vectors = _eigen(t.rho[ix]) if gauge == "density" else (lam[blk], u[ix])
            mask = values > cutoff
            b[ix] = vectors * (mask / np.sqrt(np.maximum(values, TINY)))
            colkeep[blk] = mask
    columns = np.flatnonzero(colkeep)
    allowed = (np.ones((d, d), dtype=bool) if sector == "all"
               else (z[:, None] == z[None, :]) == (sector == "odd"))
    # Julia: [CartesianIndex(i,j) for j in columns for i in 1:d if allowed(i,j)].
    k, rows = np.nonzero(allowed[:, columns].T)
    cols = columns[k]
    if len(rows) == 0:
        raise SolverFailure("empty tangent sector")
    defect = float(np.linalg.norm(adj(b) @ t.rho @ b - np.diag(colkeep.astype(float))))
    if not defect <= 1e-6:
        raise SolverFailure(f"tangent whitening is unresolved: metric defect {defect}; "
                            "use gauge='density' or a larger metric_tol")
    return TangentCoordinates(
        t, rotation, b, rows, cols, z,
        math.nan if z is None else float(state_parity_error(state, z)), sector, gauge,
        int(keep.sum()), float(np.abs(lam[~keep]).sum()), defect,
        float(lam.max() / lam[keep].min()))


def tangent_matrix(c, x):
    """Tangent W (in the gauge of c.state) of a white coefficient vector."""
    x = np.asarray(x)
    if x.shape != (len(c),):
        raise ValueError("coordinate vector has the wrong length")
    m = np.zeros((c.state.D, c.state.D), dtype=np.complex128)
    m[c.rows, c.cols] = x
    return m @ adj(c.whitening)


def tangent_coordinates(c, w):
    """Metric-orthogonal projection of W (in the gauge of c.state) onto the coordinates."""
    return (w @ c.state.rho @ c.whitening)[c.rows, c.cols]


def covector_coordinates(c, f):
    """Coordinates of a Hamiltonian action F: vdot(x, covector(F)) == vdot(tangent_matrix(x), F)."""
    return (f @ c.whitening)[c.rows, c.cols]


def original_tangent(c, w):
    return c.rotation @ w @ adj(c.rotation)


def coordinate_tangent(c, w):
    return adj(c.rotation) @ w @ c.rotation


def tangent_basis(c):
    return [tangent_matrix(c, row) for row in np.eye(len(c))]
