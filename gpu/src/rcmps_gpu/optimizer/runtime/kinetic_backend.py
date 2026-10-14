"""Experimental exact kinetic pair, with CPU Arnoldi and batched NumPy/CuPy work.

This module does not change RcmpsCore's CPU implementation.  All storage and
arithmetic use complex128/float64.  ``kinetic_raw`` returns the raw horizontal
covector and stationary-state environment; the caller must add the stationary
Poisson response and solve the regularized metric, exactly as ``finish_gradient``
does in Julia.  An integer ``nodes`` is the angular Gauss count, not a Krylov cap.

``arnoldi_backend='cpu'`` builds the four bases on the host; ``'gpu'`` keeps bases,
Hessenberg coefficients, and all vector recurrence work on CuPy.  The optional
``projection_backend='hessenberg_gpu'`` uses quadratic-cost pivoted projected
solves on device; the default uses a CPU Schur factorization. Reconstruction,
every true physical residual, and all
angular gradient contractions are batched on ``xp``.  Reuse ``result['context']``
only for angular refinement of the same immutable state.  Time fields include
explicit GPU synchronization and distinguish host and device work.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import schur, solve_triangular
from scipy.linalg.blas import zgemv
from scipy.special import roots_legendre


class KineticFailure(RuntimeError):
    """A numerical requirement was not met; no implicit tolerance fallback."""


def _cpu(a, xp):
    return np.array(xp.asnumpy(a) if xp.__name__ == "cupy" else a,
                    dtype=np.complex128, order="F", copy=True)


def _sync(xp):
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


def _scalar(a, xp):
    return float(a.item())


def _physical_batch(vectors, d, xp):
    # Each column is Julia's column-major vec of one physical matrix.
    return xp.ascontiguousarray(vectors.T.reshape((-1, d, d)).swapaxes(1, 2))


def angular_grid(nodes):
    """Return cos(theta), weights including 2/pi, as in Julia kinetic_grid."""
    if isinstance(nodes, (int, np.integer)):
        if nodes < 2:
            raise ValueError("angular node count must be at least 2")
        x, w = roots_legendre(int(nodes))
        return np.cos((np.pi / 4) * (x + 1)), w / 2
    if not isinstance(nodes, (tuple, list)) or len(nodes) != 2:
        raise ValueError("nodes must be an integer or (cosines, weights)")
    c, w = (np.asarray(x, dtype=np.float64) for x in nodes)
    if (c.ndim != 1 or w.shape != c.shape or not len(c) or
            not np.all(np.isfinite(c)) or not np.all(np.isfinite(w)) or
            np.any(c < 0) or np.any(c > 1)):
        raise ValueError("invalid angular shifts/weights")
    return c, w


_HESSENBERG_KERNEL = r'''
#include <cuda_runtime.h>

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
__device__ __forceinline__ double2 csub(double2 a, double2 b) {
    return make_double2(a.x-b.x, a.y-b.y);
}
__device__ __forceinline__ double2 cdiv(double2 a, double2 b) {
    // Smith division avoids unnecessary overflow in |b|^2.
    if (fabs(b.x) >= fabs(b.y)) {
        double ratio=b.y/b.x, den=b.x+b.y*ratio;
        return make_double2((a.x+a.y*ratio)/den, (a.y-a.x*ratio)/den);
    }
    double ratio=b.x/b.y, den=b.y+b.x*ratio;
    return make_double2((a.y+a.x*ratio)/den, (-a.x+a.y*ratio)/den);
}
__device__ __forceinline__ double2 warpsum(double2 a) {
    for (int offset=16; offset>0; offset/=2) {
        a.x += __shfl_down_sync(0xffffffff, a.x, offset);
        a.y += __shfl_down_sync(0xffffffff, a.y, offset);
    }
    return a;
}

extern "C" __global__ void shifted_hessenberg(
    const double2 *H, const double *shifts, double beta, int k,
    double2 *working, double2 *rhs, double2 *solutions, int *failures)
{
    int tid=threadIdx.x, node=blockIdx.x;
    long matrix_size=(long)k*k;
    double2 *u=working+(long)node*matrix_size;
    double2 *b=rhs+(long)node*k;
    double2 *y=solutions+(long)node*k;
    double shift=shifts[node];
    __shared__ int swap_rows, bad;
    __shared__ double2 factor;
    __shared__ double2 partial[8];
    if (tid==0) { bad=0; failures[node]=0; }
    for (long index=tid; index<matrix_size; index+=blockDim.x) {
        double2 h=H[index];
        double2 value=make_double2(-shift*h.x, -shift*h.y);
        if (index/k==index%k) value.x+=1.;
        u[index]=value;
    }
    for (int j=tid; j<k; j+=blockDim.x) {
        b[j]=make_double2(j==0 ? beta : 0., 0.);
        y[j]=make_double2(0., 0.);
    }
    __syncthreads();
    // An upper-Hessenberg column has just two possible pivot rows.
    // Applying the row operations directly to b needs no stored L factors.
    for (int i=0; i<k-1; ++i) {
        if (tid==0) {
            double2 diagonal=u[(long)i*k+i];
            double2 below=u[(long)(i+1)*k+i];
            swap_rows=hypot(below.x,below.y)>hypot(diagonal.x,diagonal.y);
            double2 pivot=swap_rows ? below : diagonal;
            double2 other=swap_rows ? diagonal : below;
            if ((!isfinite(pivot.x)) || (!isfinite(pivot.y)) ||
                (pivot.x==0. && pivot.y==0.)) {
                bad=1; failures[node]=i+1;
            } else {
                factor=cdiv(other,pivot);
                double2 upper=b[i], lower=b[i+1];
                if (swap_rows) {
                    b[i]=lower;
                    b[i+1]=csub(upper,cmul(factor,lower));
                } else b[i+1]=csub(lower,cmul(factor,upper));
            }
        }
        __syncthreads();
        if (bad) return;
        for (int j=i+tid; j<k; j+=blockDim.x) {
            double2 upper=u[(long)i*k+j];
            double2 lower=u[(long)(i+1)*k+j];
            if (swap_rows) {
                u[(long)i*k+j]=lower;
                u[(long)(i+1)*k+j]=j==i ? make_double2(0.,0.) : csub(upper,cmul(factor,lower));
            } else {
                u[(long)(i+1)*k+j]=j==i ? make_double2(0.,0.) : csub(lower,cmul(factor,upper));
            }
        }
        __syncthreads();
    }
    // Warp reductions limit the synchronization cost of each back-substitution
    // row.  The launch has exactly 256 threads (eight complete warps).
    for (int i=k-1; i>=0; --i) {
        double2 sum=make_double2(0.,0.);
        for (int j=i+1+tid; j<k; j+=blockDim.x) {
            double2 term=cmul(u[(long)i*k+j],y[j]);
            sum.x+=term.x; sum.y+=term.y;
        }
        sum=warpsum(sum);
        if ((tid&31)==0) partial[tid/32]=sum;
        __syncthreads();
        if (tid<32) {
            sum=tid<8 ? partial[tid] : make_double2(0.,0.);
            sum=warpsum(sum);
            if (tid==0) {
                double2 pivot=u[(long)i*k+i];
                if ((!isfinite(pivot.x)) || (!isfinite(pivot.y)) ||
                    (pivot.x==0. && pivot.y==0.)) {
                    bad=1; failures[node]=i+1;
                } else {
                    y[i]=cdiv(csub(b[i],sum),pivot);
                    if ((!isfinite(y[i].x)) || (!isfinite(y[i].y))) {
                        bad=1; failures[node]=i+1;
                    }
                }
            }
        }
        __syncthreads();
        if (bad) return;
    }
}
'''


def _hessenberg_shifted_reference(h, shifts, beta):
    """Small NumPy oracle for the pivoted kernel algorithm (k<=64 only)."""
    h = np.asarray(h, dtype=np.complex128)
    k = len(h)
    if h.shape != (k, k) or not 1 <= k <= 64:
        raise ValueError("kernel validation oracle is restricted to 1<=k<=64")
    answers = np.empty((k, len(shifts)), dtype=np.complex128)
    for node, shift in enumerate(shifts):
        u = np.eye(k, dtype=np.complex128)-shift*h
        b = np.zeros(k, dtype=np.complex128)
        b[0] = beta
        for i in range(k-1):
            if abs(u[i+1, i]) > abs(u[i, i]):
                u[[i, i+1], i:] = u[[i+1, i], i:]
                b[[i, i+1]] = b[[i+1, i]]
            if u[i, i] == 0:
                raise np.linalg.LinAlgError("singular projected pivot")
            factor = u[i+1, i]/u[i, i]
            u[i+1, i+1:] -= factor*u[i, i+1:]
            u[i+1, i] = 0
            b[i+1] -= factor*b[i]
        y = np.empty(k, dtype=np.complex128)
        for i in range(k-1, -1, -1):
            if u[i, i] == 0:
                raise np.linalg.LinAlgError("singular projected pivot")
            y[i] = (b[i]-u[i, i+1:]@y[i+1:])/u[i, i]
        answers[:, node] = y
    return answers


@dataclass
class _Basis:
    rhs: np.ndarray
    beta: float
    V: np.ndarray
    H: np.ndarray
    k: int = 0
    exhausted: bool = False
    gpu_V: object = None
    gpu_k: int = -1
    schur_T: object = None
    schur_Z: object = None
    schur_k: int = -1
    host_H: object = None
    projected_H: object = None
    projected_gpu_k: int = -1


class KineticContext:
    """Evaluation-local shared-shift bases; never reuse after state mutation."""

    def __init__(self, q, r, rho, *, xp=np, tolerance=1e-12,
                 initialdim=16, maxdim=1024, arnoldi_backend="cpu",
                 fallback_tolerance=None, projection_backend="schur_cpu",
                 projection_chunk=64):
        fallback_tolerance = tolerance if fallback_tolerance is None else fallback_tolerance
        if (not np.isfinite(tolerance) or tolerance <= 0 or
                not np.isfinite(fallback_tolerance) or fallback_tolerance < tolerance or
                not isinstance(initialdim, (int, np.integer)) or initialdim < 1 or
                not isinstance(maxdim, (int, np.integer)) or maxdim < initialdim):
            raise ValueError("invalid kinetic Krylov controls")
        if arnoldi_backend not in ("cpu", "gpu") or (arnoldi_backend == "gpu" and xp.__name__ != "cupy"):
            raise ValueError("arnoldi_backend='gpu' requires xp=cupy")
        if projection_backend not in ("schur_cpu", "hessenberg_gpu") or (projection_backend == "hessenberg_gpu" and xp.__name__ != "cupy"):
            raise ValueError("projection_backend='hessenberg_gpu' requires xp=cupy")
        if not isinstance(projection_chunk, (int, np.integer)) or projection_chunk < 1:
            raise ValueError("projection_chunk must be a positive integer")
        self.xp = xp
        self.arnoldi_backend = arnoldi_backend
        self.projection_backend = projection_backend
        self.projection_chunk = int(projection_chunk)
        self.projected_kernel = (xp.RawKernel(_HESSENBERG_KERNEL, "shifted_hessenberg")
                                 if projection_backend == "hessenberg_gpu" else None)
        self.q, self.r, self.rho = (_cpu(a, xp) for a in (q, r, rho))
        self.d = self.q.shape[0]
        if (self.q.shape != (self.d, self.d) or self.r.shape != self.q.shape or
                self.rho.shape != self.q.shape or
                any(not np.all(np.isfinite(a)) for a in (self.q, self.r, self.rho))):
            raise ValueError("q, r and rho must be finite square matrices of equal size")
        self.initialdim = min(int(initialdim), self.d**2)
        self.maxdim = min(int(maxdim), self.d**2)
        self.tolerance = float(tolerance)
        self.fallback_tolerance = float(fallback_tolerance)
        self.qh, self.rh = self.q.conj().T, self.r.conj().T
        self.q_device, self.r_device, self.rho_device = (
            xp.asarray(a, dtype=xp.complex128) for a in (self.q, self.r, self.rho))
        self.bases = {}
        self.actions = 0

    def matches(self, q, r, rho, xp):
        return (xp is self.xp and all(np.array_equal(a, _cpu(b, xp))
                for a, b in zip((self.q, self.r, self.rho), (q, r, rho))))

    def basis(self, op, commutator, adjoint):
        key = (commutator, adjoint)
        rhs = np.array(op.conj().T if adjoint else self.rho @ op.conj().T,
                       dtype=np.complex128, order="F").reshape(-1, order="F")
        if key in self.bases:
            basis = self.bases[key]
            if not np.array_equal(rhs, basis.rhs):
                raise KineticFailure("Krylov context was reused after state mutation")
            return basis
        beta = np.linalg.norm(rhs)
        if not np.isfinite(beta):
            raise KineticFailure("nonfinite kinetic right-hand side")
        basis_xp = self.xp if self.arnoldi_backend == "gpu" else np
        V = basis_xp.zeros((len(rhs), self.initialdim + 1), dtype=basis_xp.complex128, order="F")
        H = basis_xp.zeros((self.initialdim + 1, self.initialdim), dtype=basis_xp.complex128, order="F")
        if beta:
            V[:, 0] = basis_xp.asarray(rhs / beta)
        basis = _Basis(rhs, float(beta), V, H, exhausted=(beta == 0))
        self.bases[key] = basis
        return basis

    def extend(self, basis, adjoint, target):
        if self.arnoldi_backend == "gpu":
            return self.extend_gpu(basis, adjoint, target)
        if basis.H.shape[1] < target:
            V = np.zeros((self.d**2, target + 1), dtype=np.complex128, order="F")
            H = np.zeros((target + 1, target), dtype=np.complex128, order="F")
            V[:, :basis.k + 1] = basis.V[:, :basis.k + 1]
            H[:basis.k + 1, :basis.k] = basis.H[:basis.k + 1, :basis.k]
            basis.V, basis.H = V, H
        q, r, qh, rh = self.q, self.r, self.qh, self.rh
        for j in range(basis.k, target):
            x = basis.V[:, j].reshape((self.d, self.d), order="F")
            out = qh @ x + x @ q + rh @ x @ r if adjoint else q @ x + x @ qh + r @ x @ rh
            self.actions += 1
            v = np.array(out, dtype=np.complex128, order="F").reshape(-1, order="F")
            if not np.all(np.isfinite(v)):
                raise KineticFailure("nonfinite kinetic Arnoldi action")
            oldnorm = np.linalg.norm(v)
            block = basis.V[:, :j + 1]
            # Twice reorthogonalize and retain the coefficients of both passes.
            for _ in range(2):
                h = zgemv(1, block, v, trans=2)
                basis.H[:j + 1, j] += h
                v = zgemv(-1, block, h, beta=1, y=v, overwrite_y=1)
            beta = np.linalg.norm(v)
            basis.H[j + 1, j] = beta
            basis.k = j + 1
            if beta <= 10 * np.finfo(np.float64).eps * max(oldnorm, np.finfo(np.float64).eps) or basis.k == self.d**2:
                basis.exhausted = True
                break
            basis.V[:, j + 1] = v / beta
        basis.gpu_k = -1
        basis.schur_k = -1
        basis.host_H = None
        basis.projected_gpu_k = -1

    def extend_gpu(self, basis, adjoint, target):
        """Execute a complete extension chunk without per-vector host reductions.

        The breakdown and finite flags are read once at the chunk boundary.
        A breakdown zeros subsequent vectors, whose unused H columns are ignored.
        This retains the same two-pass projection and true-residual policy as CPU.
        """
        xp, d = self.xp, self.d
        if basis.H.shape[1] < target:
            V = xp.zeros((d*d, target + 1), dtype=xp.complex128, order="F")
            H = xp.zeros((target + 1, target), dtype=xp.complex128, order="F")
            V[:, :basis.k + 1] = basis.V[:, :basis.k + 1]
            H[:basis.k + 1, :basis.k] = basis.H[:basis.k + 1, :basis.k]
            basis.V, basis.H = V, H
        start_k = basis.k
        breakdowns = xp.zeros(target-start_k, dtype=xp.bool_)
        finite = xp.zeros(target-start_k, dtype=xp.bool_)
        q, r = self.q_device, self.r_device
        qh, rh = q.conj().T, r.conj().T
        eps = np.finfo(np.float64).eps
        for j in range(start_k, target):
            x = basis.V[:, j].reshape((d, d), order="F")
            out = qh @ x + x @ q + (rh @ x) @ r if adjoint else q @ x + x @ qh + (r @ x) @ rh
            v = out.reshape(-1, order="F")
            oldnorm = xp.linalg.norm(v)
            block = basis.V[:, :j + 1]
            for _ in range(2):
                # Conjugate the length-D² vector, avoiding a conjugated copy of
                # the full D²-by-k basis at every projection.
                h = (v.conj() @ block).conj()
                basis.H[:j + 1, j] += h
                v = v - block @ h
            beta = xp.linalg.norm(v)
            basis.H[j + 1:j + 2, j] = beta
            finite[j-start_k] = xp.all(xp.isfinite(v)) & xp.isfinite(beta)
            exhausted = beta <= 10 * eps * xp.maximum(oldnorm, eps)
            breakdowns[j-start_k] = exhausted
            basis.V[:, j + 1] = xp.where(exhausted, 0, v / xp.maximum(beta, np.finfo(np.float64).tiny))
        self.actions += target-start_k
        if not bool(xp.all(finite).item()):
            raise KineticFailure("nonfinite kinetic Arnoldi action or projection")
        indices = xp.asnumpy(xp.flatnonzero(breakdowns))
        basis.k = target if len(indices) == 0 else start_k + int(indices[0]) + 1
        basis.exhausted = len(indices) > 0 or basis.k == d*d
        basis.gpu_V = basis.V[:, :basis.k]
        basis.gpu_k = basis.k
        basis.schur_k = -1
        basis.host_H = None
        basis.projected_gpu_k = -1
        _sync(xp)

    def projected(self, basis, shifts):
        if self.projection_backend == "hessenberg_gpu":
            return self.projected_gpu(basis, shifts)
        k = basis.k
        try:
            if basis.schur_k != k:
                basis.host_H = (self.xp.asnumpy(basis.H[:k + 1, :k])
                                if self.arnoldi_backend == "gpu" else basis.H[:k + 1, :k])
                basis.schur_T, basis.schur_Z = schur(basis.host_H[:k, :k], output="complex")
                basis.schur_k = k
            t, z = basis.schur_T, basis.schur_Z
            rhs = basis.beta * z[0, :].conj()
            reduced = np.empty((k, len(shifts)), dtype=np.complex128, order="F")
            identity = np.eye(k, dtype=np.complex128)
            for i, c in enumerate(shifts):
                reduced[:, i] = solve_triangular(identity - c * t, rhs, lower=False,
                                                 check_finite=False)
            y = z @ reduced
            estimate = np.max(np.abs(shifts * basis.host_H[k, k - 1] * y[k - 1, :])) / basis.beta
            if not np.all(np.isfinite(y)):
                return None, np.inf
            return y, float(estimate)
        except np.linalg.LinAlgError:
            return None, np.inf

    def projected_gpu(self, basis, shifts):
        """O(nodes*k²) two-row pivoted elimination, with bounded GPU scratch."""
        xp, k = self.xp, basis.k
        if basis.projected_gpu_k != k:
            basis.projected_H = xp.ascontiguousarray(xp.asarray(basis.H[:k, :k], dtype=xp.complex128))
            basis.projected_gpu_k = k
        shifts_gpu = xp.asarray(shifts, dtype=xp.float64)
        y = xp.empty((k, len(shifts)), dtype=xp.complex128, order="F")
        failures = xp.zeros(len(shifts), dtype=xp.int32)
        chunk = min(len(shifts), self.projection_chunk)
        working = xp.empty((chunk, k, k), dtype=xp.complex128)
        rhs = xp.empty((chunk, k), dtype=xp.complex128)
        for start in range(0, len(shifts), chunk):
            end = min(start+chunk, len(shifts))
            self.projected_kernel((end-start,), (256,),
                (basis.projected_H, shifts_gpu[start:end], np.float64(basis.beta),
                 np.int32(k), working, rhs, y[:, start:end], failures[start:end]))
        valid = xp.all(failures == 0) & xp.all(xp.isfinite(y))
        if not bool(valid.item()):
            return None, np.inf
        subdiagonal = basis.H[k, k-1]
        estimate = xp.max(xp.abs(shifts_gpu*subdiagonal*y[k-1, :]))/basis.beta
        return y, _scalar(estimate, xp)

    def solutions(self, op, commutator, adjoint, shifts, timings, diagnostics):
        xp, d = self.xp, self.d
        basis = self.basis(op, commutator, adjoint)
        label = ("commutator" if commutator else "R") + ("_adjoint" if adjoint else "_forward")
        if basis.beta == 0:
            diagnostics[label] = dict(basis_size=0, true_relative_residual=0.0,
                                      projected_relative_residual=0.0, fallback_used=False)
            return xp.zeros((len(shifts), d, d), dtype=xp.complex128)
        last_residual = np.inf
        while True:
            if basis.k == 0:
                start = perf_counter()
                self.extend(basis, adjoint, self.initialdim)
                timings["arnoldi_seconds"] += perf_counter() - start
            start = perf_counter()
            y, estimate = self.projected(basis, shifts)
            timings["projected_seconds"] += perf_counter() - start
            terminal = basis.exhausted or basis.k == self.maxdim
            if y is not None and (estimate <= self.tolerance or terminal):
                _sync(xp)
                start = perf_counter()
                if basis.gpu_k != basis.k:
                    basis.gpu_V = (basis.V[:, :basis.k] if self.arnoldi_backend == "gpu" else
                                   xp.asarray(basis.V[:, :basis.k], dtype=xp.complex128))
                    basis.gpu_k = basis.k
                yg = xp.asarray(y, dtype=xp.complex128)
                rhs = xp.asarray(basis.rhs.reshape((d, d), order="F"), dtype=xp.complex128)
                cg = xp.asarray(shifts, dtype=xp.float64)
                _sync(xp)
                timings["transfer_seconds"] += perf_counter() - start
                start = perf_counter()
                x = _physical_batch(basis.gpu_V @ yg, d, xp)
                _sync(xp)
                timings["reconstruction_seconds"] += perf_counter() - start
                start = perf_counter()
                q, r = self.q_device, self.r_device
                qh, rh = q.conj().T, r.conj().T
                action = qh @ x + x @ q + (rh @ x) @ r if adjoint else q @ x + x @ qh + (r @ x) @ rh
                residuals = xp.linalg.norm(x - cg[:, None, None] * action - rhs,
                                           axis=(1, 2)) / basis.beta
                worst = _scalar(xp.max(residuals), xp)
                _sync(xp)
                timings["residual_seconds"] += perf_counter() - start
                self.actions += len(shifts)
                last_residual = worst
                fallback = terminal and worst > self.tolerance and worst <= self.fallback_tolerance
                if np.isfinite(worst) and (worst <= self.tolerance or fallback):
                    diagnostics[label] = dict(basis_size=basis.k, true_relative_residual=worst,
                                              projected_relative_residual=estimate,
                                              fallback_used=fallback, residual_target=self.tolerance,
                                              residual_ceiling=self.fallback_tolerance)
                    return x
                if basis.exhausted:
                    raise KineticFailure(f"Arnoldi exhausted at {basis.k} vectors with true relative residual {worst}")
            if terminal:
                raise KineticFailure(f"shifted Arnoldi failed at {self.maxdim} vectors; projected residual {estimate}; last true residual {last_residual}")
            start = perf_counter()
            self.extend(basis, adjoint, min(2 * basis.k, self.maxdim))
            timings["arnoldi_seconds"] += perf_counter() - start


def kinetic_raw(q, r, rho, nodes=128, *, xp=np, tolerance=1e-12,
                initialdim=16, maxdim=1024, gradient=True, context=None,
                arnoldi_backend="cpu", fallback_tolerance=None,
                projection_backend="schur_cpu", projection_chunk=64):
    """Evaluate both exact infinite-line kinetic terms on one angular grid.

    Return ``dict(value, energy, covector, environment, stats, context)``.  Both
    ``value`` and ``energy`` name the sum of the R and [Q,R] terms, with unit
    coefficients as in phi4.  ``stats`` contains strict residuals, basis sizes,
    and synchronized times. ``arnoldi_backend='gpu'`` and
    ``projection_backend='hessenberg_gpu'`` require CuPy. The GPU projection uses
    O(nodes*k²) partial-pivoted upper-Hessenberg elimination; ``projection_chunk``
    bounds scratch storage to chunk*k² complex128 values (1 GiB at 64*1024²).
    An explicit
    ``fallback_tolerance>=tolerance`` accepts only checked terminal residuals and
    records each acceptance; omitting it preserves strict failure. This function
    does not assert angular convergence: double nodes and check the complete
    finished gradient separately.
    """
    _sync(xp)
    total_start = perf_counter()
    shifts, weights = angular_grid(nodes)
    if context is None:
        context = KineticContext(q, r, rho, xp=xp, tolerance=tolerance,
                                 initialdim=initialdim, maxdim=maxdim,
                                 arnoldi_backend=arnoldi_backend,
                                 fallback_tolerance=fallback_tolerance,
                                 projection_backend=projection_backend,
                                 projection_chunk=projection_chunk)
    elif not context.matches(q, r, rho, xp):
        raise KineticFailure("Krylov context was reused after state or backend mutation")
    elif (context.tolerance != tolerance or
          context.arnoldi_backend != arnoldi_backend or
          context.projection_backend != projection_backend or
          context.projection_chunk != projection_chunk or
          context.fallback_tolerance != (tolerance if fallback_tolerance is None else fallback_tolerance) or
          context.initialdim != min(initialdim, context.d**2) or
          context.maxdim != min(maxdim, context.d**2)):
        raise ValueError("controls changed while reusing a kinetic context")
    timings = {name: 0.0 for name in ("arnoldi_seconds", "projected_seconds", "transfer_seconds",
                                    "reconstruction_seconds", "residual_seconds", "contraction_seconds")}
    diagnostics = {}
    actions_before = context.actions
    qg, rg, rhog = context.q_device, context.r_device, context.rho_device
    rh = rg.conj().T
    wg = xp.asarray(weights, dtype=xp.float64)
    w = wg[:, None, None]
    wz = w * xp.asarray(shifts, dtype=xp.float64)[:, None, None]
    value = 0.0
    covector = xp.zeros_like(rg)
    environment = xp.zeros_like(rg)
    for commutator in (False, True):
        op = context.q @ context.r - context.r @ context.q if commutator else context.r
        opg = xp.asarray(op, dtype=xp.complex128)
        x = context.solutions(op, commutator, False, shifts, timings, diagnostics)
        start = perf_counter()
        value += _scalar(xp.sum(wg * xp.real(xp.einsum("ij,nji->n", opg, x))), xp)
        _sync(xp)
        timings["contraction_seconds"] += perf_counter() - start
        if not gradient:
            continue
        aa = context.solutions(op, commutator, True, shifts, timings, diagnostics)
        a = aa.conj().swapaxes(1, 2)
        start = perf_counter()
        # All nodes are independent and share R,Q,rho.  Preserve Julia's
        # multiplication ordering before the angular reduction.
        m = xp.sum(wz * (((x @ rh) @ a) - ((x @ a) @ rh)), axis=0)
        nn = xp.sum(wz * (((a @ rg) @ x) - ((rg @ a) @ x)), axis=0)
        u = xp.sum(w * x, axis=0)
        v = xp.sum(w * (a @ rhog), axis=0)
        env = xp.sum(w * (opg.conj().T @ a), axis=0)
        if commutator:
            m += u @ qg - qg @ u - (rg @ u) @ rh + (u @ rg) @ rh
            nn += qg.conj().T @ v - v @ qg.conj().T - (rg @ v) @ rh + (rg @ rh) @ v
        else:
            m += u
            nn += v
        covector += (m + nn.conj().T) / 2
        environment += (env + env.conj().T) / 2
        _sync(xp)
        timings["contraction_seconds"] += perf_counter() - start
    if (not np.isfinite(value) or (gradient and
            (not bool(xp.all(xp.isfinite(covector)).item()) or
             not bool(xp.all(xp.isfinite(environment)).item())))):
        raise KineticFailure("nonfinite kinetic value or gradient components")
    _sync(xp)
    timings["total_seconds"] = perf_counter() - total_start
    stats = dict(nodes=len(shifts), backends=arnoldi_backend + " Arnoldi; " + projection_backend + "; batched " + xp.__name__,
                 lindblad_actions=context.actions - actions_before, solves=diagnostics,
                 fallback_solves=sum(len(shifts) for v in diagnostics.values() if v["fallback_used"]),
                 maximum_relative_residual=max((v["true_relative_residual"] for v in diagnostics.values()), default=0.0),
                 timings=timings)
    return dict(value=value, energy=value, covector=covector if gradient else None,
                environment=environment if gradient else None, stats=stats, context=context)
