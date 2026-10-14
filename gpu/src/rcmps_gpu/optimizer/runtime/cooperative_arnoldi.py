"""Isolated complex-FP64 kinetic Arnoldi reorthogonalization candidate.

One occupancy-checked cooperative CUDA kernel computes both complex projection
passes, their Hessenberg coefficients, the final norm, finite/breakdown flags
and the next normalized basis vector. The physical Lindblad action remains
the existing CuPy matrix-product expression. The unchanged kinetic backend
performs projected shifted solves, reconstruction, all true physical residuals
and contractions. No residual relaxation or implicit fallback is permitted.

This addresses submission/allocation overhead; it has no measured GPU speedup
yet. CuPy>=14 prevents capturing cuBLAS through its matmul interface, so this
first candidate intentionally avoids graph capture. Cooperative launch can
compete with an overlapping cooperative quartic kernel; complete-objective
latency must be measured separately. V100/A100/H100 feasibility is subject to
actual device cooperative residency and FP64 qualification.

Native FP64 fused multiply-add is enabled by default. Disabling contraction is
an explicit diagnostic variant, not a precision or science requirement; both
variants use the same true-residual, angular-refinement and gradient gates.

NumPy mode is a bounded D<=4 mathematical oracle, never a large CPU objective.
Importing the module does not import CuPy or initialize CUDA.
"""
from __future__ import annotations

from time import perf_counter
import math
import numpy as np

from . import kinetic_backend as baseline


CUDA_SOURCE = r'''
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define FP64_EPS 2.220446049250313080847263336181640625e-16
#define FP64_MIN 2.225073858507201383090232717332404064e-308

// All arithmetic and storage are double/double2. No float accumulator,
// reduced precision instruction, atomics or tolerance-dependent truncation.
__device__ __forceinline__ double block_sum(double value, double *shared) {
    int tid=threadIdx.x;
    shared[tid]=value; __syncthreads();
    for(int stride=blockDim.x/2; stride>0; stride/=2) {
        if(tid<stride) shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    return shared[0];
}

extern "C" __global__ void complex_reorthogonalize(
    const double2 *basis, double2 *work, double2 *hess,
    double2 *coefficients, double *partial_norms, double *norms,
    unsigned char *finite_flags, unsigned char *breakdown_flags,
    double2 *next_vector, int n, int hld, int j, int flag_index)
{
    cg::grid_group grid=cg::this_grid();
    __shared__ double scratch[256];
    int global_thread=blockIdx.x*blockDim.x+threadIdx.x;
    int global_threads=gridDim.x*blockDim.x;
    for(int pass=0; pass<2; ++pass) {
        // One CTA reduces a full complex inner product. A fixed cooperative
        // grid covers longer histories by a column-stride loop; no atomics.
        for(int col=blockIdx.x; col<=j; col+=gridDim.x) {
            double real=0., imag=0., old_squared=0.;
            for(int row=threadIdx.x; row<n; row+=blockDim.x) {
                double2 a=basis[row+(long)n*col], b=work[row];
                real+=a.x*b.x+a.y*b.y;
                imag+=a.x*b.y-a.y*b.x;
                if(pass==0 && col==0) old_squared+=b.x*b.x+b.y*b.y;
            }
            real=block_sum(real,scratch);
            imag=block_sum(imag,scratch);
            if(threadIdx.x==0) {
                coefficients[col]=make_double2(real,imag);
                double2 old=hess[col+(long)hld*j];
                hess[col+(long)hld*j]=make_double2(old.x+real,old.y+imag);
            }
            if(pass==0 && col==0) {
                old_squared=block_sum(old_squared,scratch);
                if(threadIdx.x==0) norms[0]=sqrt(old_squared);
            }
            __syncthreads();
        }
        grid.sync();
        // Coalesced row lanes read each column of the Fortran-order basis.
        // Every thread owns its work-vector entries through both passes.
        double squared=0.;
        for(int row=global_thread; row<n; row+=global_threads) {
            double real=0.,imag=0.;
            for(int col=0; col<=j; ++col) {
                double2 a=basis[row+(long)n*col], b=coefficients[col];
                real+=a.x*b.x-a.y*b.y;
                imag+=a.x*b.y+a.y*b.x;
            }
            double2 old=work[row];
            double2 value=make_double2(old.x-real,old.y-imag);
            work[row]=value;
            if(pass==1) squared+=value.x*value.x+value.y*value.y;
        }
        if(pass==1) {
            squared=block_sum(squared,scratch);
            if(threadIdx.x==0) partial_norms[blockIdx.x]=squared;
        }
        grid.sync();
    }
    if(global_thread==0) {
        double squared=0.;
        for(int block=0; block<gridDim.x; ++block) squared+=partial_norms[block];
        double length=sqrt(squared);
        norms[1]=length;
        hess[j+1+(long)hld*j]=make_double2(length,0.);
        finite_flags[flag_index]=(unsigned char)isfinite(length);
        breakdown_flags[flag_index]=(unsigned char)
            (length<=10.*FP64_EPS*fmax(norms[0],FP64_EPS));
    }
    grid.sync();
    double length=norms[1];
    bool exhausted=breakdown_flags[flag_index]!=0;
    double denominator=fmax(length,FP64_MIN);
    for(int row=global_thread; row<n; row+=global_threads) {
        double2 value=work[row];
        next_vector[row]=exhausted?make_double2(0.,0.):
            make_double2(value.x/denominator,value.y/denominator);
    }
}
'''


def reorthogonalize_reference(vectors, vector, *, prior_coefficients=None):
    """Small independent complex recurrence oracle, restricted to n<=16.

    Uses explicit conjugate scalar products and two projections. This catches
    accidental real-inner-product use, omitted second-pass coefficients and
    incorrectly interpreted column-major views without relying on CUDA.
    """
    vectors = np.asarray(vectors)
    vector = np.asarray(vector)
    n, count = vectors.shape
    if (not 1 <= n <= 16 or not 1 <= count <= n or vector.shape != (n,) or
            vectors.dtype != np.complex128 or vector.dtype != np.complex128):
        raise ValueError("reorthogonalization oracle requires finite complex128 n<=16")
    if not np.isfinite(vectors).all() or not np.isfinite(vector).all():
        raise baseline.KineticFailure("nonfinite Arnoldi oracle input")
    work = vector.copy()
    coefficients = np.zeros(count, dtype=np.complex128)
    if prior_coefficients is not None:
        coefficients[:] = prior_coefficients
    old_norm = float(np.linalg.norm(work))
    for _ in range(2):
        projection = np.asarray([sum(np.conj(vectors[row, col])*work[row]
                                    for row in range(n)) for col in range(count)],
                                dtype=np.complex128)
        coefficients += projection
        work -= np.asarray([sum(vectors[row, col]*projection[col] for col in range(count))
                            for row in range(n)], dtype=np.complex128)
    length = float(np.linalg.norm(work))
    exhausted = bool(length <= 10*np.finfo(np.float64).eps*max(old_norm, np.finfo(np.float64).eps))
    next_vector = np.zeros(n, dtype=np.complex128) if exhausted else work/max(length, np.finfo(np.float64).tiny)
    return next_vector, coefficients, length, exhausted


class CooperativeKineticContext(baseline.KineticContext):
    """Evaluation-local candidate; inherited true-residual acceptance unchanged."""
    def __init__(self, q, r, rho, *, xp=np, max_blocks=64, profile=False, fma=True, **controls):
        if not isinstance(max_blocks, (int, np.integer)) or max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        options = compiler_options(fma)
        if "arnoldi_backend" in controls:
            raise ValueError("candidate selects its own cpu-oracle/GPU Arnoldi backend")
        if controls.get("fallback_tolerance") is not None:
            raise ValueError("cooperative candidate permits no fallback tolerance")
        if xp.__name__ != "cupy" and (xp is not np or len(q) > 4):
            raise ValueError("CPU candidate development is restricted to NumPy D<=4")
        backend = "gpu" if xp.__name__ == "cupy" else "cpu"
        started = perf_counter()
        super().__init__(q, r, rho, xp=xp, arnoldi_backend=backend, **controls)
        self.profile = bool(profile)
        self.fma = bool(fma)
        self.compiler_options = options
        self.phase_events = []
        self.phase_wall = dict(action_submission_seconds=0., reorthogonalization_submission_seconds=0.)
        self.max_blocks = int(max_blocks)
        self.work = xp.empty(self.d**2, dtype=xp.complex128)
        self.coefficients = xp.empty(self.maxdim, dtype=xp.complex128)
        self.norms = xp.empty(2, dtype=xp.float64)
        self.kernel = None
        self.occupancy = None
        if backend == "gpu":
            properties = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
            if not properties.get("cooperativeLaunch", False):
                raise ValueError("cooperative Arnoldi requires cooperative device launch support")
            self.kernel = xp.RawKernel(CUDA_SOURCE, "complex_reorthogonalize",
                options=options, enable_cooperative_groups=True)
            self.kernel.compile()
            active = int(xp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.kernel.kernel.ptr, 256, 0))
            multiprocessors = int(properties["multiProcessorCount"])
            self.capacity = active*multiprocessors
            if self.capacity < 1:
                raise RuntimeError("cooperative Arnoldi has no resident block capacity")
            self.blocks = min(self.max_blocks, self.capacity)
            self.partial_norms = xp.empty(self.blocks, dtype=xp.float64)
            self.occupancy = dict(blocks=self.blocks, max_blocks=self.max_blocks,
                resident_capacity=self.capacity, active_blocks_per_sm=active,
                multiprocessors=multiprocessors, threads_per_block=256,
                registers_per_thread=int(self.kernel.num_regs), shared_bytes=int(self.kernel.shared_size_bytes),
                native_fp64=True, cooperative_grid_guard=True,
                native_fp64_fma_enabled=self.fma, compiler_options=list(options))
        else:
            self.blocks = 1
            self.partial_norms = None
        self.setup_seconds = perf_counter()-started

    def extend(self, basis, adjoint, target):
        xp, d, n = self.xp, self.d, self.d**2
        if basis.H.shape[1] < target:
            vectors = xp.zeros((n, target+1), dtype=xp.complex128, order="F")
            hess = xp.zeros((target+1, target), dtype=xp.complex128, order="F")
            vectors[:, :basis.k+1] = basis.V[:, :basis.k+1]
            hess[:basis.k+1, :basis.k] = basis.H[:basis.k+1, :basis.k]
            basis.V, basis.H = vectors, hess
        start_k = basis.k
        finite = xp.zeros(target-start_k, dtype=xp.uint8)
        breakdowns = xp.zeros(target-start_k, dtype=xp.uint8)
        q, r = self.q_device, self.r_device
        qh, rh = q.conj().T, r.conj().T
        for j in range(start_k, target):
            action_begin = perf_counter()
            if self.profile and xp.__name__ == "cupy":
                a_begin, a_end, o_end = (xp.cuda.Event() for _ in range(3))
                a_begin.record()
            x = basis.V[:, j].reshape((d, d), order="F")
            out = qh@x+x@q+(rh@x)@r if adjoint else q@x+x@qh+(r@x)@rh
            # A stable contiguous work vector makes the RawKernel pointer
            # contract explicit even if the matmul result is a C-order view.
            self.work[:] = out.reshape(-1, order="F")
            self.phase_wall["action_submission_seconds"] += perf_counter()-action_begin
            orth_begin = perf_counter()
            if self.kernel is not None:
                if self.profile:
                    a_end.record()
                blocks = min(self.blocks, j+1)
                if blocks > self.capacity:
                    raise RuntimeError("cooperative Arnoldi grid exceeds resident capacity")
                self.kernel((blocks,), (256,), (basis.V, self.work, basis.H, self.coefficients,
                    self.partial_norms, self.norms, finite, breakdowns, basis.V[:, j+1],
                    np.int32(n), np.int32(basis.H.shape[0]), np.int32(j), np.int32(j-start_k)))
                if self.profile:
                    o_end.record()
                    self.phase_events.append((a_begin, a_end, o_end))
            else:
                next_vector, h, length, exhausted = reorthogonalize_reference(
                    basis.V[:, :j+1], self.work, prior_coefficients=basis.H[:j+1, j])
                basis.H[:j+1, j] = h
                basis.H[j+1, j] = length
                basis.V[:, j+1] = next_vector
                finite[j-start_k] = np.isfinite(length)
                breakdowns[j-start_k] = exhausted
                if exhausted:
                    break
            self.phase_wall["reorthogonalization_submission_seconds"] += perf_counter()-orth_begin
        self.actions += j-start_k+1
        if not bool(xp.all(finite[:j-start_k+1]).item()):
            raise baseline.KineticFailure("nonfinite cooperative Arnoldi action or reorthogonalization")
        indices = xp.flatnonzero(breakdowns[:j-start_k+1])
        indices = xp.asnumpy(indices) if self.kernel is not None else indices
        basis.k = j+1 if len(indices) == 0 else start_k+int(indices[0])+1
        basis.exhausted = len(indices) > 0 or basis.k == n
        basis.gpu_V = basis.V[:, :basis.k] if self.kernel is not None else None
        basis.gpu_k = basis.k if self.kernel is not None else -1
        basis.schur_k = -1
        basis.host_H = None
        basis.projected_gpu_k = -1
        baseline._sync(xp)

    def diagnostics(self):
        result = dict(occupancy=self.occupancy, setup_seconds=float(self.setup_seconds),
                      submission_seconds={name: float(value) for name, value in self.phase_wall.items()},
                      phase_event_profiling=self.profile, compiler_options=list(self.compiler_options),
                      requested_cuda_fp64_fma=self.fma,
                      cuda_kernel_executed=self.kernel is not None)
        if self.phase_events:
            xp = self.xp
            result["device_action_seconds"] = float(sum(xp.cuda.get_elapsed_time(a, b)
                for a, b, c in self.phase_events)/1000)
            result["device_reorthogonalization_seconds"] = float(sum(xp.cuda.get_elapsed_time(b, c)
                for a, b, c in self.phase_events)/1000)
            result["profiled_steps"] = len(self.phase_events)
            result["profiling_includes_event_overhead"] = True
        return result


def kinetic_raw_cooperative(q, r, rho, nodes=128, *, xp=np, tolerance=1e-12,
        initialdim=16, maxdim=1024, projection_backend=None, projection_chunk=64,
        context=None, max_blocks=64, profile=False, fma=True, gradient=True):
    """Same raw kinetic return contract; every inherited true residual retained."""
    projection = projection_backend or ("hessenberg_gpu" if xp.__name__ == "cupy" else "schur_cpu")
    backend = "gpu" if xp.__name__ == "cupy" else "cpu"
    if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-12:
        raise ValueError("candidate requires strict kinetic tolerance <=1e-12")
    started = perf_counter()
    if context is None:
        context = CooperativeKineticContext(q, r, rho, xp=xp, tolerance=tolerance,
            initialdim=initialdim, maxdim=maxdim, projection_backend=projection,
            projection_chunk=projection_chunk, max_blocks=max_blocks, profile=profile, fma=fma)
    elif (not isinstance(context, CooperativeKineticContext) or context.max_blocks != max_blocks or
          context.profile != profile or context.fma != fma):
        raise ValueError("candidate context/controls differ")
    raw = baseline.kinetic_raw(q, r, rho, nodes, xp=xp, tolerance=tolerance,
        initialdim=initialdim, maxdim=maxdim, projection_backend=projection,
        projection_chunk=projection_chunk, arnoldi_backend=backend,
        fallback_tolerance=None, gradient=gradient, context=context)
    raw["stats"]["cooperative_arnoldi"] = context.diagnostics()
    raw["stats"]["backends"] = "cooperative complex FP64 Arnoldi; "+projection+"; batched "+xp.__name__
    raw["stats"]["timings"]["total_seconds"] = float(perf_counter()-started)
    return raw


def compiler_options(fma):
    """Explicit FP64 contraction control; never enable CUDA fast math."""
    if not isinstance(fma, (bool, np.bool_)):
        raise ValueError("fma must be an explicit boolean contraction control")
    return ("--std=c++17", "--fmad=true" if fma else "--fmad=false")
