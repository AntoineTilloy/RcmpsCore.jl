"""States, loaders and elementary Lindblad maps for the GPU excitation solver.

NumPy/SciPy only; importing this module never imports CuPy or initializes CUDA.
It mirrors src/state.jl, src/transfer.jl and src/storage.jl of the CPU package,
which stays the control implementation and is not modified.

Conventions: every matrix is a C-ordered complex128 (D, D) array; `adj(x)` is the
conjugate transpose; Q = -i K - R'R/2 (left-canonical gauge: L*(1) = 0), and the
stationary density rho solves L(rho) = 0 with unit trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import tomllib

import numpy as np
from scipy.linalg import get_lapack_funcs, schur
from scipy.sparse.linalg import LinearOperator, gmres


class SolverFailure(RuntimeError):
    """A numerical solve failed its residual or structure check (src/transfer.jl)."""


@dataclass
class IntegratorParameters:
    """`max_point` is the physical half-window (mass one); `n_quad` the Gauss nodes per half."""
    n_quad: int = 128
    max_point: float = 20.0
    atol: float = 1e-11
    rtol: float = 1e-10
    maxiters: int = 1_000_000


@dataclass
class State:
    R: np.ndarray
    K: np.ndarray
    Q: np.ndarray
    rho: np.ndarray
    integrator: IntegratorParameters = field(default_factory=IntegratorParameters)
    model: str = "phi4"
    coupling: float = math.nan
    e0: float = math.nan
    source: str = ""

    @property
    def D(self):
        return self.R.shape[0]

    def with_integrator(self, integrator):
        """The same matrices (shared, read-only) with other integration settings."""
        return replace(self, integrator=replace(integrator))


def adj(x):
    return x.conj().swapaxes(-1, -2)


def lindblad(q, r, x):
    """L(X) = QX + XQ' + RXR'; broadcasts over leading axes of x."""
    return q @ x + x @ adj(q) + r @ x @ adj(r)


def lindblad_adjoint(q, r, x):
    """Trace dual (= Hilbert-Schmidt adjoint) L*(A) = Q'A + AQ + R'AR."""
    return adj(q) @ x + x @ q + adj(r) @ x @ r


def transfer_matrix(q, r):
    """Dense reference with column-major vec: vec(L(X)) == transfer_matrix(q, r) @ vec(X)."""
    identity = np.eye(len(q), dtype=np.complex128)
    return np.kron(identity, q) + np.kron(q.conj(), identity) + np.kron(r.conj(), r)


def vec(x):
    return x.reshape(-1, order="F")


def unvec(v, d):
    return v.reshape((d, d), order="F")


class NoJumpInverse:
    """(-N)^-1 for the no-jump map N(X) = QX + XQ', by a Schur form of Q and trsyl.

    `adjoint=True` inverts N*(A) = Q'A + AQ. A dummy shift keeps the map invertible
    when Q has a non-decaying mode (src/transfer.jl NoJumpInverse).
    """
    def __init__(self, q, r):
        self.t, self.u = schur(q, output="complex")
        scale = max(1.0, np.linalg.norm(q, 2), np.linalg.norm(r, 2) ** 2)
        decaying = np.real(np.diag(self.t)).max() < -100 * np.finfo(float).eps * scale
        self.shift = 0.0 if decaying else scale
        self.t = np.asfortranarray(self.t - self.shift / 2 * np.eye(len(q)))
        self.trsyl = get_lapack_funcs("trsyl", (self.t,))

    def __call__(self, y, *, adjoint=False):
        c = np.asfortranarray(adj(self.u) @ y @ self.u)
        x, scale, info = self.trsyl(self.t, self.t, c, trana="C" if adjoint else "N",
                                    tranb="N" if adjoint else "C", overwrite_c=1)
        if info < 0 or not np.isfinite(scale) or scale <= 0:
            raise SolverFailure(f"Sylvester inverse failed: info={info}, scale={scale}")
        return self.u @ (-x / scale) @ adj(self.u)


def checked_density(q, r, rho, *, tol=1e-11):
    """Normalize, Hermitize and check a stationary density (src/transfer.jl checked_density)."""
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != q.shape:
        raise ValueError("rho has the wrong dimensions")
    if not np.isfinite(rho).all():
        raise SolverFailure("nonfinite stationary state")
    trace = np.trace(rho)
    if abs(trace) <= np.finfo(float).eps * np.linalg.norm(rho):
        raise SolverFailure("stationary state has zero trace")
    x = rho / trace
    if np.linalg.norm(x - adj(x)) > 100 * tol:
        raise SolverFailure("stationary state is not Hermitian")
    x = (x + adj(x)) / 2
    residual = np.linalg.norm(lindblad(q, r, x))
    scale = max(1.0, 2 * np.linalg.norm(q, 2) + np.linalg.norm(r, 2) ** 2)
    if residual > tol * scale:
        raise SolverFailure(f"stationary residual {residual} exceeds {tol * scale}")
    if np.linalg.eigvalsh(x).min() < -10 * tol:
        raise SolverFailure("stationary state is not positive")
    return x  # No eigenvalue clipping or shift: that would change the stationary equation.


def stationary_state(q, r, *, tol=1e-11, method="auto"):
    """Unit-trace stationary density of L. `dense` solves the pinned D^2 system directly
    (small D); `iterative` is GMRES right-preconditioned by the no-jump inverse with a
    Sherman-Morrison trace correction, as in the GPU optimizer's CPU helper."""
    d = len(q)
    identity = np.eye(d, dtype=np.complex128)
    if method == "auto":
        method = "dense" if d <= 24 else "iterative"
    if method == "dense":
        l = transfer_matrix(q, r)
        singular = np.linalg.svd(l, compute_uv=False)
        # As src/transfer.jl: reject a nonunique stationary state rather than select one.
        if np.count_nonzero(singular < tol * max(1.0, singular[0])) != 1:
            raise SolverFailure("stationary state is nonunique or numerically unresolved")
        v = vec(identity)
        rho = unvec(np.linalg.solve(l + np.outer(v, v.conj()) / d, v / d), d)
    elif method == "iterative":
        inverse = NoJumpInverse(q, r)
        z = inverse(identity)
        denominator = 1 + np.trace(z) / d

        def precondition(y):
            x = inverse(y)
            return x - np.trace(x) / (d * denominator) * z

        def action(x):
            return -lindblad(q, r, x) + np.trace(x) / d * identity

        operator = LinearOperator((d * d, d * d), dtype=np.complex128,
                                  matvec=lambda y: vec(action(precondition(unvec(y, d)))))
        y, info = gmres(operator, vec(identity / d), rtol=tol / 10, atol=tol / 10,
                        restart=min(40, d * d), maxiter=200)
        if info != 0:
            raise SolverFailure(f"stationary GMRES failed: info={info}")
        rho = precondition(unvec(y, d))
    else:
        raise ValueError(f"unknown stationary solver: {method}")
    return checked_density(q, r, rho, tol=tol)


def make_state(R, K, rho=None, *, integrator=None, steady_tol=1e-11, **metadata):
    """State(R, K) of src/state.jl: checks, Q = -i K - R'R/2, and a checked density."""
    R = np.array(R, dtype=np.complex128, order="C")
    K = np.array(K, dtype=np.complex128, order="C")
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] == 0:
        raise ValueError("R must be nonempty and square")
    if K.shape != R.shape:
        raise ValueError("K and R must have equal dimensions")
    if not (np.isfinite(R).all() and np.isfinite(K).all()):
        raise ValueError("nonfinite state parameters")
    if np.linalg.norm(K - adj(K)) > 1e-12 * max(1.0, np.linalg.norm(K)):
        raise ValueError("K must be Hermitian")
    K = (K + adj(K)) / 2
    Q = -1j * K - adj(R) @ R / 2
    rho = (stationary_state(Q, R, tol=steady_tol) if rho is None
           else checked_density(Q, R, rho, tol=steady_tol))
    return State(R, K, Q, rho, integrator or IntegratorParameters(), **metadata)


def _complex(dataset):
    value = dataset[()]
    return value["re"] + 1j * value["im"]


def load_jld2(path, *, rho=None):
    """Read an `rcmps_store` checkpoint (schema 1) with h5py. JLD2 stores Julia's
    column-major matrices, so the HDF5 view is the transpose. rho is not stored and is
    recomputed and checked, exactly as `rcmps_load` does; integration settings in the file
    are the optimizer's and are returned unchanged (override them for excitations)."""
    import h5py

    with h5py.File(path, "r") as f:
        if int(f["schema_version"][()]) != 1:
            raise ValueError("unsupported schema; migrate old JLD with examples/migrate_legacy.jl")
        R, K = _complex(f["R"]).T, _complex(f["K"]).T
        coupling = _complex(f["coupling"])
        model = f["model"][()].decode()
        e0 = float(f["e0"][()])
        row = f["integration"][()]
        integrator = IntegratorParameters(int(row["n_quad"]), float(row["max_point"]),
                                          float(row["atol"]), float(row["rtol"]), int(row["maxiters"]))
    if coupling.imag != 0:
        raise ValueError("model coupling must be real")
    return make_state(R, K, rho, integrator=integrator, model=model, coupling=float(coupling.real),
                      e0=e0, source=str(path))


def load_portable(directory):
    """Read a GPU-optimizer portable export: headerless little-endian column-major
    complex128 R.bin, K.bin, rho.bin and metadata.toml (research/gpu_d64 convention)."""
    directory = Path(directory)
    with (directory / "metadata.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    d = int(metadata["D"])

    def matrix(name):
        path = directory / f"{name}.bin"
        if path.stat().st_size != 16 * d * d:
            raise ValueError(f"{path} does not hold a {d}x{d} complex128 matrix")
        return np.fromfile(path, dtype="<c16").reshape((d, d), order="F")

    integrator = IntegratorParameters(
        int(metadata.get("quadrature", metadata.get("n_quad", 128))),
        float(metadata.get("window", metadata.get("half_window", 20.0))),
        float(metadata.get("atol", 1e-11)), float(metadata.get("rtol", 1e-10)))
    return make_state(matrix("R"), matrix("K"), matrix("rho"), integrator=integrator,
                      model="phi4", coupling=float(metadata["coupling"]),
                      e0=float(metadata.get("energy", metadata.get("saved_energy", math.nan))),
                      source=str(directory))


def parity_error(state, z):
    """Relative violation of Z R Z = -R, Z K Z = K and Z rho Z = rho for Z = diag(z)."""
    zz = np.outer(z, z)
    return max(np.linalg.norm(zz * state.R + state.R) / max(1.0, np.linalg.norm(state.R)),
               np.linalg.norm(zz * state.K - state.K) / max(1.0, np.linalg.norm(state.K)),
               np.linalg.norm(zz * state.rho - state.rho))


def excitation_parity(state, parity="auto", *, tol=1e-12):
    """Diagonal auxiliary Z2 parity as a +-1 integer vector, or None (excitation_parity)."""
    if parity is None:
        return None
    d = state.D
    if isinstance(parity, str):
        if parity != "auto":
            raise ValueError("parity must be 'auto', None, or a +-1 vector")
        if d % 2:
            return None
        z = np.r_[np.ones(d // 2, dtype=int), -np.ones(d // 2, dtype=int)]
        return z if parity_error(state, z) <= tol else None
    z = np.asarray(parity)
    if z.shape != (d,) or not np.isin(z, (-1, 1)).all() or not (z == 1).any() or not (z == -1).any():
        raise ValueError("parity must be a +-1 vector of length D with both signs")
    z = z.astype(int)
    violation = parity_error(state, z)
    if violation > tol:
        raise ValueError(f"state is not Z2 symmetric under the supplied parity (error {violation})")
    return z


def random_state(d, *, scaling=1.0, seed=1234, z2=False):
    """Random test state. NumPy draws, so it does not reproduce Julia's RandomState."""
    rng = np.random.default_rng(seed)

    def gaussian(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)

    if not z2:
        k = gaussian(d, d)
        return make_state(scaling * gaussian(d, d), scaling * (k + adj(k)))
    if d % 2 or d <= 0:
        raise ValueError("Z2 states require a positive even D")
    h = d // 2
    zero = np.zeros((h, h), dtype=np.complex128)
    k1, k2 = gaussian(h, h), gaussian(h, h)
    r = np.block([[zero, gaussian(h, h)], [gaussian(h, h), zero]])
    return make_state(scaling * r, scaling * np.block([[k1 + adj(k1), zero], [zero, k2 + adj(k2)]]))
