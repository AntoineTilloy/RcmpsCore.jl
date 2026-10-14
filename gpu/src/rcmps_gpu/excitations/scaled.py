"""Matrix-free Lindblad maps and batched shifted solves in diagonally scaled variables.

No D^2 x D^2 object is ever formed. Every linear solve of the excitation algorithm,
(sigma - z L)(X) = B for a resolvent node or (-L + i p + pin)(X) = B for a Poisson
problem, is done by right-preconditioned GMRES on a whole batch of independent
systems at once. The preconditioner is the exact inverse of the no-jump part
X -> sigma X - z (Q X + X Q'), diagonal in the eigenbasis of Q.

Scaled variables. With a positive diagonal S, each index of a matrix variable is
kept either in the stored left-canonical gauge ("L") or in the gauge rescaled by S
("H", for hat): Q_H = S^-1 Q S and R_H = S^-1 R S. A density-like matrix x is
represented by xh with x = S xh S (type HH), a ket-inserted density y by yh with
y = yh S (type LH), a tangent W by Wh = W S, and so on. With S = sqrt(rho) in the
density gauge every such variable is O(1) and the right fixed point becomes
rh ~ 1, which removes the 1/lambda_min roundoff amplification of the unscaled
formulas. S = 1 reproduces the CPU formulas literally (rh = rho).

Two limits of the statement "any positive S gives the same physical result":
* The density. rh is re-solved here, L_HH(rh) = 0 to 1e-13 in hat variables, from the
  guess S^-1 rho S^-1. With S = sqrt(rho) that corrects the relative error of the small
  eigenvalues of the stored rho, about eps cond(rho), which the CPU code keeps: at
  cond(rho) = 1e10 the stored rho is 2.5e-7 (relative to sqrt(rho_ii rho_jj)) from a
  50-digit stationary solve and S rh S is 4e-15 from it. With S = 1 the stored rho
  passes the test and is kept (no iteration), as it did for the unaligned S of the
  tests. Results with "density" and with "none" then differ by about eps cond(rho);
  the first agrees with the CPU formulas evaluated on rho = S rh S (test_bilinear
  RefinedDensity, test_eigsolve IllConditioned).
* The solves. The formulas do not depend on S, but GMRES controls the residual relative
  to the hat right-hand side, so a physical result carries up to the attained residual
  times (Smax/Smin)^2 unless S is aligned with rho, where every hat variable is O(1).
  An unaligned S spanning 1.7 decades (the tests) keeps 1e-11 at tol = 1e-12; one
  spanning 4 decades gives 1e-8 to 1e-6 there and still up to 1e-8 at tol = 1e-14
  (test_bilinear WideScaling).

Batch layout: a batch of n matrices is one C-contiguous array X[row, system, col]
of shape (D, n, D), so that multiplying every system by a constant matrix on
either side is a single large GEMM. Works with NumPy or CuPy (`xp`); the small
Hessenberg least-squares problems of GMRES are solved on the host.
"""
from __future__ import annotations

import math

import numpy as np

from .states import SolverFailure, adj

GAUGES = ("L", "H")
RESTART = 50


def lmul(a, x):
    """a @ x[:, s, :] for every system s; x has shape (D, n, D)."""
    d, n, _ = x.shape
    return (a @ x.reshape(d, n * d)).reshape(d, n, d)


def rmul(x, b):
    """x[:, s, :] @ b for every system s."""
    d, n, _ = x.shape
    return (x.reshape(d * n, d) @ b).reshape(d, n, d)


def stack(matrices, xp=np):
    """List of (D, D) matrices -> batch array (D, n, D)."""
    return xp.ascontiguousarray(xp.stack(matrices, axis=1))


def batch_norm(x, xp=np):
    return xp.sqrt(xp.sum(x.real * x.real + x.imag * x.imag, axis=(0, 2)))


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class ScaledState:
    """Generators in both gauges, their eigenbases, and the hat right fixed point rh.

    `scaling`: "density" uses S = sqrt(diag(rho)) (intended for a state rotated to the
    density gauge, where rho is diagonal), "none" uses S = 1, or pass a positive vector.
    Physical density: rho = S rh S with tr(rho) = 1; left fixed point of every map: 1.
    rh is the refined fixed point, not S^-1 state.rho S^-1 (module docstring): every
    formula is evaluated on S rh S.
    """
    def __init__(self, state, scaling="density", *, xp=np, floor=1e-15, tol=1e-13,
                 max_basis_bytes=None):
        d = state.D
        self.max_basis_bytes = max_basis_bytes  # bound on one GMRES Krylov basis; None = no chunking
        diag = np.real(np.diag(state.rho))
        if isinstance(scaling, str):
            if scaling == "density":
                s = np.sqrt(np.maximum(diag, floor * diag.max()))
            elif scaling == "none":
                s = np.ones(d)
            else:
                raise ValueError("scaling must be 'density', 'none', or a positive vector")
        else:
            s = np.asarray(scaling, dtype=float)
            if s.shape != (d,) or not (np.isfinite(s).all() and (s > 0).all()):
                raise ValueError("scaling must be a positive vector of length D")
        self.xp, self.D, self.state = xp, d, state
        ratio = s[None, :] / s[:, None]
        host = {"L": (state.Q, state.R), "H": (state.Q * ratio, state.R * ratio)}
        self.Q, self.R, self.Qd, self.Rd = {}, {}, {}, {}
        self.lam, self.V, self.Vi, self.Vd, self.Vid = {}, {}, {}, {}, {}
        self.eigenbasis_condition = {}
        for g in GAUGES:
            q, r = host[g]
            lam, v = np.linalg.eig(q)
            vi = np.linalg.inv(v)
            self.eigenbasis_condition[g] = float(np.linalg.cond(v))
            if np.real(lam).max() >= 0:
                raise SolverFailure("Q has a non-decaying mode: the no-jump preconditioner is singular")
            for name, value in (("Q", q), ("R", r), ("Qd", adj(q)), ("Rd", adj(r)), ("lam", lam),
                                ("V", v), ("Vi", vi), ("Vd", adj(v)), ("Vid", adj(vi))):
                getattr(self, name)[g] = xp.ascontiguousarray(xp.asarray(value))
        self.s_host = s
        self.s = xp.asarray(s)
        self.g = {"L": self.s, "H": xp.ones(d)}          # x = (S / g_a) xh (S / g_b) for a primal of type (a, b)
        self.w = {"L": xp.ones(d), "H": self.s}          # left zero functional weights S / g
        self.eye = xp.eye(d, dtype=xp.complex128)
        guess = state.rho / np.outer(s, s)
        self.rh, self.fixed_point = self._fixed_point(xp.asarray((guess + adj(guess)) / 2), tol)

    # ---- typed maps ------------------------------------------------------------------
    def lmap(self, x, a, b, dual=False):
        """Primal L_ab(x) = Q_a x + x Q_b' + R_a x R_b', or its trace dual
        L*_ab(y) = Q_a' y + y Q_b + R_a' y R_b, on a batch."""
        if dual:
            return lmul(self.Qd[a], x) + rmul(x, self.Q[b]) + rmul(lmul(self.Rd[a], x), self.R[b])
        return lmul(self.Q[a], x) + rmul(x, self.Qd[b]) + rmul(lmul(self.R[a], x), self.Rd[b])

    def nojump_inverse(self, c, a, b, sigma, z, dual=False):
        """Solve sigma X - z (Q_a X + X Q_b') = C (or the dual sigma Y - z (Q_a' Y + Y Q_b) = C)
        for per-system scalars sigma, z of shape (n,)."""
        xp = self.xp
        la, lb = self.lam[a][:, None, None], self.lam[b][None, None, :]
        sigma, z = xp.asarray(sigma)[None, :, None], xp.asarray(z)[None, :, None]
        if dual:
            y = rmul(lmul(self.Vd[a], c), self.V[b]) / (sigma - z * (la.conj() + lb))
            return rmul(lmul(self.Vid[a], y), self.Vi[b])
        y = rmul(lmul(self.Vi[a], c), self.Vid[b]) / (sigma - z * (la + lb.conj()))
        return rmul(lmul(self.V[a], y), self.Vd[b])

    # ---- zero modes ------------------------------------------------------------------
    def zero_mode(self, a, b, dual=False):
        """Right zero mode of L_ab (g_a rh g_b), or of the dual map (diag(w_a w_b))."""
        xp = self.xp
        if dual:
            return xp.diag(self.w[a] * self.w[b]).astype(xp.complex128)
        return self.g[a][:, None] * self.rh * self.g[b][None, :]

    def pairing(self, x, a, b, dual=False):
        """Per-system value of the left zero functional: tr(x) in physical variables for a
        primal, tr(rho y) for a dual. Normalized to one on `zero_mode`."""
        xp = self.xp
        if dual:
            weight = (self.g[b][:, None] * self.rh * self.g[a][None, :]).T
            return xp.sum(weight[:, None, :] * x, axis=(0, 2))
        index = xp.arange(self.D)
        return xp.sum((self.w[a] * self.w[b])[:, None] * x[index, :, index], axis=0)

    def project(self, x, a, b, dual=False):
        """Remove the zero-mode component: x - pairing(x) * zero_mode."""
        return x - self.zero_mode(a, b, dual)[:, None, :] * self.pairing(x, a, b, dual)[None, :, None]

    # ---- solves ----------------------------------------------------------------------
    def resolvent(self, rhs, a, b, z, sigma=1.0, *, dual=False, tol=1e-12, **controls):
        """Solve (sigma - z L_ab) X = rhs for a batch; z and sigma are scalars or (n,) arrays."""
        xp, n = self.xp, rhs.shape[1]
        z = xp.asarray(np.broadcast_to(np.asarray(_host(z), dtype=float), (n,)).copy())
        sigma = xp.asarray(np.broadcast_to(np.asarray(_host(sigma), dtype=complex), (n,)).copy())

        restart = min(controls.get("restart", RESTART), self.D * self.D)
        per_system = (restart + 1) * self.D * self.D * 16
        step = n if self.max_basis_bytes is None else max(1, int(self.max_basis_bytes // per_system))
        outputs, infos = [], []
        for start in range(0, n, step):
            part = slice(start, min(n, start + step))
            zc, sc = z[part], sigma[part]

            def operator(x, zc=zc, sc=sc):
                return sc[None, :, None] * x - zc[None, :, None] * self.lmap(x, a, b, dual)

            def precondition(c, zc=zc, sc=sc):
                return self.nojump_inverse(c, a, b, sc, zc, dual)

            x, info = gmres(operator, precondition, xp.ascontiguousarray(rhs[:, part, :]), tol=tol, xp=xp,
                            **controls)
            outputs.append(x); infos.append(info)
        if len(outputs) == 1:
            return outputs[0], infos[0]
        return (xp.ascontiguousarray(xp.concatenate(outputs, axis=1)),
                dict(iterations=np.concatenate([i["iterations"] for i in infos]),
                     residuals=np.concatenate([i["residuals"] for i in infos]),
                     steps=sum(i["steps"] for i in infos)))

    def poisson(self, rhs, a, b, shift=0.0, *, dual=False, tol=1e-12, **controls):
        """Pinned Poisson solve (-L_ab + shift + |zero><pairing|) X = rhs - pairing(rhs) zero,
        returned with its zero-mode component removed. `shift` is +-i p, or zero."""
        xp, n = self.xp, rhs.shape[1]
        zero = self.zero_mode(a, b, dual)[:, None, :]

        def operator(x):
            return (shift * x - self.lmap(x, a, b, dual)
                    + zero * self.pairing(x, a, b, dual)[None, :, None])

        def precondition(c):
            count = c.shape[1]
            return self.nojump_inverse(c, a, b, xp.full(count, shift, dtype=xp.complex128), xp.ones(count), dual)

        restart = min(controls.get("restart", RESTART), self.D * self.D)
        per_system = (restart + 1) * self.D * self.D * 16
        step = n if self.max_basis_bytes is None else max(1, int(self.max_basis_bytes // per_system))
        outputs, infos = [], []
        for start in range(0, n, step):
            part = xp.ascontiguousarray(self.project(rhs[:, start:start + step, :], a, b, dual))
            x, info = gmres(operator, precondition, part, tol=tol, xp=xp, **controls)
            outputs.append(self.project(x, a, b, dual)); infos.append(info)
        if len(outputs) == 1:
            return outputs[0], infos[0]
        return (xp.ascontiguousarray(xp.concatenate(outputs, axis=1)),
                dict(iterations=np.concatenate([i["iterations"] for i in infos]),
                     residuals=np.concatenate([i["residuals"] for i in infos]),
                     steps=sum(i["steps"] for i in infos)))

    def _fixed_point(self, guess, tol):
        """rh with L_HH(rh) = 0 and tr(S rh S) = 1, refined from S^-1 rho S^-1."""
        xp = self.xp
        guess = guess / self._tau(guess)
        r0 = guess[:, None, :]
        ones, zeros = xp.ones(1), xp.zeros(1, dtype=xp.complex128)

        def operator(x):
            tau = xp.sum((self.s * self.s)[:, None] * x[xp.arange(self.D), :, xp.arange(self.D)], axis=0)
            return -self.lmap(x, "H", "H") + r0 * tau[None, :, None]

        def precondition(c):
            return self.nojump_inverse(c, "H", "H", zeros, ones)

        start = xp.ascontiguousarray(r0)
        x, info = gmres(operator, precondition, start, tol=tol, xp=xp, x0=start, relative_to=1.0,
                        accept=math.inf)
        rh = x[:, 0, :]
        rh = (rh + adj(rh)) / 2
        rh = rh / self._tau(rh)
        info["residual"] = float(xp.linalg.norm(self.lmap(xp.ascontiguousarray(rh[:, None, :]), "H", "H")))
        scale = max(1.0, float(2 * xp.linalg.norm(self.Q["H"]) + xp.linalg.norm(self.R["H"]) ** 2))
        if not math.isfinite(info["residual"]) or info["residual"] > 100 * tol * scale:
            raise SolverFailure(f"scaled stationary residual {info['residual']} is too large")
        return rh, info

    def _tau(self, x):
        return self.xp.sum(self.s * self.s * self.xp.diag(x)).real


_PROJECTIONS = {}


def projections(xp):
    """(dots, combine) for the batched Gram-Schmidt steps on a basis of shape
    (vectors, systems, size): dots(basis, w)[i, s] = <basis[i, s], w[s]> and
    combine(h, basis)[s] = sum_i h[i, s] basis[i, s]. On a GPU both are single
    reduction kernels without temporaries of the size of the basis."""
    key = id(xp)
    if key not in _PROJECTIONS:
        if xp.__name__ == "cupy" and hasattr(xp, "ReductionKernel"):
            dot = xp.ReductionKernel("complex128 v, complex128 w", "complex128 h", "conj(v) * w",
                                     "a + b", "h = a", "0", "rcmps_gmres_dot")
            mix = xp.ReductionKernel("complex128 h, complex128 v", "complex128 u", "h * v",
                                     "a + b", "u = a", "0", "rcmps_gmres_combine")
            _PROJECTIONS[key] = (lambda basis, w: dot(basis, w[None], axis=2),
                                 lambda h, basis: mix(h[:, :, None], basis, axis=0))
        else:
            _PROJECTIONS[key] = (lambda basis, w: xp.sum(xp.conj(basis) * w[None], axis=2),
                                 lambda h, basis: xp.sum(h[:, :, None] * basis, axis=0))
    return _PROJECTIONS[key]


def gmres(operator, precondition, rhs, *, tol=1e-12, restart=RESTART, maxiter=250, xp=np, x0=None,
          relative_to=None, accept=20.0, accept_floor=1e-11, stall=8):
    """Right-preconditioned restarted GMRES on a batch of independent systems.

    `rhs`, the iterates and `operator`/`precondition` arguments have the batch layout
    (D, n, D). Convergence is the true residual norm <= tol * ||rhs|| per system
    (or tol * relative_to). A cycle also ends when every unconverged system is within
    1e3 * tol and none has improved by 10% over `stall` iterations, and the iteration
    stops when such a cycle, or one that ended on its estimates, did not halve the true
    residual of any unconverged system: the attainable accuracy has been reached. A
    final true residual above max(accept * tol, accept_floor) raises SolverFailure (the
    CPU code uses the same factor 20, at a tolerance of 1e-10, for Poisson solves; the
    floor admits the roundoff level of the pinned Poisson problems at D >= 96, about
    2e-13, when tol is below it). The attained residuals are returned in any case.
    Returns (x, info) with per-system iteration counts and true relative residuals.
    """
    d, n, _ = rhs.shape
    size = d * d
    dots, combine = projections(xp)
    scale = batch_norm(rhs, xp) if relative_to is None else xp.full(n, float(relative_to))
    scale_host = np.maximum(_host(scale), np.finfo(float).tiny)
    x = xp.zeros_like(rhs) if x0 is None else x0.copy()
    residual = rhs.copy() if x0 is None else rhs - operator(x)
    iterations = np.zeros(n, dtype=int)
    total, relres, previous = 0, None, None
    restart = min(restart, size)
    while True:
        beta = batch_norm(residual, xp)
        beta_host = _host(beta)
        relres = beta_host / scale_host
        if not np.isfinite(relres).all():
            raise SolverFailure("nonfinite GMRES residual")
        active = relres > tol
        if not active.any() or total >= maxiter:
            break
        # A cycle that ended on its estimates (not on the restart length) without halving the
        # true residual of any unconverged system has reached the roundoff floor.
        if previous is not None and not (relres[active] < 0.5 * previous[active]).any():
            break
        basis = xp.zeros((restart + 1, n, size), dtype=xp.complex128)
        safe = xp.asarray(np.where(beta_host > 0, beta_host, 1.0))
        basis[0] = xp.moveaxis(residual / safe[None, :, None], 1, 0).reshape(n, size)
        hess = np.zeros((n, restart + 1, restart), dtype=np.complex128)
        g = np.zeros((n, restart + 1), dtype=np.complex128)
        g[:, 0] = beta_host
        cs = np.zeros((n, restart)); sn = np.zeros((n, restart), dtype=np.complex128)
        best, since, steps = relres.copy(), 0, 0
        for j in range(restart):
            v = xp.ascontiguousarray(xp.moveaxis(basis[j].reshape(n, d, d), 0, 1))
            w = operator(precondition(v))
            w = xp.ascontiguousarray(xp.moveaxis(w, 1, 0)).reshape(n, size)
            before = _host(xp.sqrt(xp.sum(w.real * w.real + w.imag * w.imag, axis=1)))
            column = np.zeros((n, j + 2), dtype=np.complex128)
            for _ in range(2):  # classical Gram-Schmidt, twice
                h = dots(basis[:j + 1], w)
                w = w - combine(h, basis[:j + 1])
                column[:, :j + 1] += _host(h).T
            norm = _host(xp.sqrt(xp.sum(w.real * w.real + w.imag * w.imag, axis=1)))
            if not (np.isfinite(norm).all() and np.isfinite(column).all()):
                raise SolverFailure("nonfinite GMRES Arnoldi vector")
            alive = norm > 10 * np.finfo(float).eps * np.maximum(before, np.finfo(float).tiny)
            column[:, j + 1] = np.where(alive, norm, 0.0)
            factor = xp.asarray(np.where(alive, 1.0 / np.where(alive, norm, 1.0), 0.0))
            basis[j + 1] = w * factor[:, None]
            # Givens rotations, vectorized over the systems.
            for i in range(j):
                t = cs[:, i] * column[:, i] + sn[:, i] * column[:, i + 1]
                column[:, i + 1] = -np.conj(sn[:, i]) * column[:, i] + cs[:, i] * column[:, i + 1]
                column[:, i] = t
            a, b = column[:, j], column[:, j + 1]
            radius = np.hypot(np.abs(a), np.abs(b))
            degenerate = radius == 0
            radius = np.where(degenerate, 1.0, radius)
            phase = np.where(np.abs(a) > 0, a / np.where(np.abs(a) > 0, np.abs(a), 1.0), 1.0)
            cs[:, j] = np.where(degenerate, 1.0, np.abs(a) / radius)
            sn[:, j] = np.where(degenerate, 0.0, phase * np.conj(b) / radius)
            column[:, j] = np.where(degenerate, 1.0, phase * radius)
            hess[:, :j + 1, j] = column[:, :j + 1]
            g[:, j + 1] = -np.conj(sn[:, j]) * g[:, j]
            g[:, j] = cs[:, j] * g[:, j]
            g[degenerate, j] = 0.0
            steps = j + 1
            total += 1
            estimate = np.abs(g[:, j + 1]) / scale_host
            iterations[active & (estimate > tol)] += 1
            unconverged = active & (estimate > tol)
            improved = unconverged & (estimate < 0.9 * best)
            best = np.minimum(best, estimate)
            # Only a plateau close to the target is a roundoff floor; early plateaus are normal.
            near_floor = unconverged.any() and (best[unconverged] < 1e3 * tol).all()
            since = since + 1 if near_floor and not improved.any() else 0
            if not unconverged.any() or since >= stall or total >= maxiter:
                break
        y = np.zeros((n, steps), dtype=np.complex128)
        for i in range(steps - 1, -1, -1):
            y[:, i] = (g[:, i] - np.einsum("nk,nk->n", hess[:, i, i + 1:steps], y[:, i + 1:])) / hess[:, i, i]
        update = combine(xp.asarray(np.ascontiguousarray(y.T)), basis[:steps]).reshape(n, d, d)
        x = x + precondition(xp.ascontiguousarray(xp.moveaxis(update, 0, 1)))
        residual = rhs - operator(x)
        del basis
        previous = relres if steps < restart or since >= stall else None
    limit = max(accept * tol, accept_floor)
    if not (np.isfinite(relres).all() and (relres <= limit).all()):
        raise SolverFailure(f"batched GMRES residual {np.nanmax(relres):.3e} exceeds {limit:.1e} "
                            f"after {total} iterations")
    return x, dict(iterations=iterations, residuals=relres, steps=total)
