"""CPU control/Poisson solves for the isolated complex128 GPU experiment.

These are intentionally small CPU pieces of a hybrid prototype. Observable
propagation/contractions live on the GPU. Nothing imports or changes CPU Julia.
"""
from pathlib import Path
import tomllib
import numpy as np
from scipy.linalg import schur, get_lapack_funcs
from scipy.sparse.linalg import LinearOperator, gmres


class SolverFailure(RuntimeError):
    pass


def load_reference(directory):
    directory = Path(directory)
    with (directory / "metadata.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    d = metadata["D"]
    matrices = {p.stem: np.fromfile(p, dtype="<c16").reshape((d, d), order="F")
                for p in directory.glob("*.bin")}
    return matrices, metadata


def adj(x):
    return x.conj().T


def lindblad(q, r, x, *, adjoint=False):
    if adjoint:
        return adj(q) @ x + x @ q + adj(r) @ x @ r
    return q @ x + x @ adj(q) + r @ x @ adj(r)


class NoJumpInverse:
    def __init__(self, q, r):
        self.t, self.u = schur(q, output="complex")
        scale = max(1., np.linalg.norm(q, 2), np.linalg.norm(r, 2)**2)
        self.shift = scale if np.real(np.diag(self.t)).max() >= -100*np.finfo(float).eps*scale else 0.
        self.t = np.asfortranarray(self.t - self.shift/2*np.eye(len(q)))
        self.trsyl = get_lapack_funcs("trsyl", (self.t,))

    def __call__(self, y, *, adjoint=False):
        c = np.asfortranarray(adj(self.u) @ y @ self.u)
        x, scale, info = self.trsyl(self.t, self.t, c,
                                   trana="C" if adjoint else "N",
                                   tranb="N" if adjoint else "C", overwrite_c=1)
        if info < 0 or not np.isfinite(scale) or scale <= 0:
            raise SolverFailure(f"Sylvester inverse failed: info={info}, scale={scale}")
        return self.u @ (-x/scale) @ adj(self.u)


def _right_solve(action, precondition, b, *, tolerance, maxiter=200):
    d = len(b)
    shape = (d, d)
    count = [0]

    def mv(v):
        count[0] += 1
        return action(precondition(v.reshape(shape, order="F"))).reshape(-1, order="F")

    op = LinearOperator((d*d, d*d), matvec=mv, dtype=np.complex128)
    y, info = gmres(op, b.reshape(-1, order="F"), rtol=tolerance, atol=tolerance,
                    restart=min(40, d*d), maxiter=maxiter)
    x = precondition(y.reshape(shape, order="F"))
    residual = np.linalg.norm(action(x)-b)
    if info != 0 or not np.isfinite(residual) or residual > 20*tolerance*max(1., np.linalg.norm(b)):
        raise SolverFailure(f"Poisson GMRES failed: info={info}, residual={residual:.3e}")
    return x, {"matvecs": count[0], "residual": float(residual)}


def stationary_state(q, r, *, tolerance=1e-11):
    d = len(q)
    identity = np.eye(d, dtype=np.complex128)
    inverse = NoJumpInverse(q, r)
    z = inverse(identity)
    denominator = 1+np.trace(z)/d

    def pre(y):
        x = inverse(y)
        return x - np.trace(x)/(d*denominator)*z

    def op(x):
        return -lindblad(q, r, x) + np.trace(x)/d*identity

    rho, stats = _right_solve(op, pre, identity/d, tolerance=tolerance)
    rho = rho/np.trace(rho)
    if np.linalg.norm(rho-adj(rho)) > 100*tolerance:
        raise SolverFailure("Stationary state is not Hermitian")
    rho = (rho+adj(rho))/2
    residual = np.linalg.norm(lindblad(q, r, rho))
    scale = max(1., 2*np.linalg.norm(q, 2)+np.linalg.norm(r, 2)**2)
    eigenvalues = np.linalg.eigvalsh(rho)
    if residual > tolerance*scale or eigenvalues.min() < -10*tolerance:
        raise SolverFailure(f"Invalid stationary density: residual={residual:.3e}, min={eigenvalues.min():.3e}")
    # No clipping, density cutoff or diagonal shift is allowed here.
    stats.update(stationary_residual=float(residual), eigmin=float(eigenvalues.min()))
    return rho, stats


def boundary_environment(q, r, rho, environment, *, tolerance=1e-11):
    identity = np.eye(len(r), dtype=np.complex128)
    b = environment - np.trace(rho @ environment)*identity
    inverse = NoJumpInverse(q, r)
    z = inverse(identity, adjoint=True)
    denominator = 1+np.trace(rho @ z)

    def pre(y):
        x = inverse(y, adjoint=True)
        return x - np.trace(rho @ x)/denominator*z

    def op(x):
        return -lindblad(q, r, x, adjoint=True) + np.trace(rho @ x)*identity

    return _right_solve(op, pre, b, tolerance=tolerance)


def finish_gradient(q, r, rho, covector, environment, *, metric_reg=1e-3, tolerance=1e-13):
    h, stats = boundary_environment(q, r, rho, environment, tolerance=tolerance)
    c = covector + rho @ adj(r) @ h - rho @ h @ adj(r)
    gradient = adj(np.linalg.solve(rho + metric_reg*np.eye(len(r)), c))
    if not np.isfinite(gradient).all():
        raise SolverFailure("Nonfinite finished gradient")
    return gradient, stats


def metric(rho, x, y, metric_reg=1e-3):
    return float(2*np.vdot(x, y @ (rho+metric_reg*np.eye(len(rho)))).real)


def retract(r, k, direction, alpha):
    rr = r + alpha*direction
    kk = k + 1j*alpha/2*(adj(direction)@r-adj(r)@direction)
    return rr, (kk+adj(kk))/2


def precondition(rho, gradient, metric_reg=1e-3, precond_reg=1e-4):
    identity = np.eye(len(rho))
    return adj(np.linalg.solve(rho+precond_reg*identity,
                              adj(gradient @ (rho+metric_reg*identity))))
