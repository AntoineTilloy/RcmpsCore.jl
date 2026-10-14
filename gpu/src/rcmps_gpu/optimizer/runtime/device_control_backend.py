"""Scalar DOP853 controller of the native CUDA WHILE/IF-graph experiment.

The experiment ran the adaptive controller, source evaluation, cusp clipping,
quadrature cursor and acceptance copies of a trajectory inside one native
conditional graph. Its integrator, which reached the CUDA driver through
ctypes, was removed. What remains serves persistent_trajectory_backend, whose
cooperative kernel runs the same controller: the status codes and diagnostic
names, the CPU oracles controller_factor and clipped_step, DeviceControlFailure,
and CUDA_SOURCE, the graph kernels whose scalar arithmetic (compiled with
--fmad=false) the trajectory kernel reproduces. Importing this module does not
import CuPy or initialize CUDA.

The NVRTC conditional-function declaration in CUDA_SOURCE follows NVIDIA Warp's
conditional helper module: https://github.com/NVIDIA/warp/blob/main/warp/native/warp.cu
"""
from __future__ import annotations

import math

import numpy as np


STATUS = {0: "success", 1: "max_steps", 2: "step_underflow",
          3: "nonfinite_error", 4: "source_domain", 5: "missing_samples"}
FLOAT_NAMES = ("t", "h", "next_h", "t_new", "last_error", "min_h", "max_h")
INT_NAMES = ("status", "attempts", "accepted", "rejected", "was_rejected",
             "accept_now", "next_point", "dense_steps", "dense_nodes_max")


def controller_factor(error, rejected_since_accept=False):
    """CPU oracle for the exact scalar controller used in the CUDA kernel."""
    if not math.isfinite(error) or error < 0:
        raise ValueError("nonfinite or negative error")
    factor = min(10., .9 * error**(-1./8.)) if error else 10.
    accepted = error <= 1.
    if accepted and rejected_since_accept:
        factor = min(1., factor)
    return accepted, max(.2, factor) if not accepted else factor


def clipped_step(t, h, endpoint):
    """CPU oracle for cusp/end clipping and the ten-ULP step floor."""
    boundary = 0. if t < 0. else endpoint
    h = min(h, boundary-t)
    if h <= 10. * abs(np.nextafter(t, math.inf)-t):
        raise RuntimeError("ODE step underflow")
    return h, boundary if h == boundary-t else t+h


# Explicit declaration is accepted by NVRTC without requiring toolkit include
# discovery; NVIDIA Warp uses this same documented CUDA builtin declaration.
CUDA_SOURCE = r'''
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(
    cudaGraphConditionalHandle handle, unsigned int value);
enum { T=0, H=1, NEXT_H=2, T_NEW=3, ERROR=4, MIN_H=5, MAX_H=6 };
enum { STATUS=0, ATTEMPTS=1, ACCEPTED=2, REJECTED=3, WAS_REJECTED=4,
       ACCEPT_NOW=5, NEXT_POINT=6, DENSE_STEPS=7, DENSE_NODES_MAX=8 };
__device__ double source_fp64(double t, const double* coefficients) {
    double a = fabs(t);
    const double maximum = SOURCE_MAXIMUM;
    int segment = min((int)(a / maximum * SOURCE_SEGMENTS), SOURCE_SEGMENTS-1);
    double z = 2. * a / maximum * SOURCE_SEGMENTS - 2. * segment - 1.;
    const double* c = coefficients + segment * (SOURCE_DEGREE+1);
    double b1=0., b2=0.;
    for (int j=SOURCE_DEGREE; j>0; --j) {
        double b0 = 2. * z * b1 - b2 + c[j];
        b2=b1; b1=b0;
    }
    return z*b1-b2+c[0];
}
extern "C" __global__ void initialize(double2* y, const double2* initial,
    long long n, double* state, long long* counts, double* params,
    const double* coefficients, double a, double initial_h) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) y[i]=initial[i];
    if (i==0) {
        for (int k=0;k<7;++k) state[k]=0.;
        for (int k=0;k<9;++k) counts[k]=0;
        state[T]=-a; state[H]=fmin(initial_h,2.*a); state[MIN_H]=INFINITY;
        params[1]=2.*a; params[2]=source_fp64(-a,coefficients);
    }
}
extern "C" __global__ void prepare(double* state, long long* counts,
    double* params, const double* c, const double* coefficients,
    double a, long long max_steps) {
    if (threadIdx.x || blockIdx.x) return;
    double t=state[T], boundary=t<0. ? 0. : a;
    double h=fmin(state[H],boundary-t);
    counts[ACCEPT_NOW]=0;
    if (counts[ATTEMPTS]>=max_steps) counts[STATUS]=1;
    else if (!isfinite(h) || h<=10.*fabs(nextafter(t,INFINITY)-t)) counts[STATUS]=2;
    if (!counts[STATUS]) {
        ++counts[ATTEMPTS];
        state[MIN_H]=fmin(state[MIN_H],h); state[MAX_H]=fmax(state[MAX_H],h);
    } else h=0.;
    state[H]=h; params[0]=h;
    for (int s=1;s<=12;++s) {
        double ts=t+c[s]*h;
        if (fabs(ts)>SOURCE_MAXIMUM*(1.+1.e-14)) counts[STATUS]=4;
        params[3*s+1]=2.*fabs(ts);
        params[3*s+2]=source_fp64(ts,coefficients);
    }
}
extern "C" __global__ void control(double* state, long long* counts,
    const double* norms, const double* points, long long npoints, long long n,
    double* dense_params, long long* dense_indices, const double* c,
    const double* coefficients, double a, cudaGraphConditionalHandle dense_handle) {
    if (threadIdx.x || blockIdx.x) return;
    cudaGraphSetConditional(dense_handle,0);
    if (counts[STATUS]) return;
    double e5=norms[0], e3=norms[1], h=state[H], t=state[T];
    double denominator=e5+.01*e3;
    double error=denominator>0. ? h*e5/sqrt(denominator*(double)n) : 0.;
    state[ERROR]=error;
    if (!isfinite(e5) || !isfinite(e3) || e5<0. || e3<0. || !isfinite(error)) {
        counts[STATUS]=3; return;
    }
    double factor=error>0. ? fmin(10.,.9*pow(error,-1./8.)) : 10.;
    if (error>1.) {
        ++counts[REJECTED]; counts[WAS_REJECTED]=1;
        state[NEXT_H]=h*fmax(.2,factor);
        return;
    }
    counts[ACCEPT_NOW]=1; ++counts[ACCEPTED];
    double boundary=t<0. ? 0. : a;
    double t_new=h==boundary-t ? boundary : t+h;
    state[T_NEW]=t_new;
    if (counts[WAS_REJECTED]) factor=fmin(1.,factor);
    state[NEXT_H]=h*factor; counts[WAS_REJECTED]=0;
    long long first=counts[NEXT_POINT], last=first;
    while (last<npoints && points[last]<=t_new) ++last;
    dense_indices[0]=first; dense_indices[1]=last;
    if (last>first) {
        ++counts[DENSE_STEPS];
        counts[DENSE_NODES_MAX]=max(counts[DENSE_NODES_MAX],last-first);
        dense_params[0]=h; dense_params[3]=t; dense_params[6]=t_new;
        for (int slot=0;slot<3;++slot) {
            double ts=t+c[13+slot]*h;
            dense_params[3*slot+1]=2.*fabs(ts);
            dense_params[3*slot+2]=source_fp64(ts,coefficients);
        }
        cudaGraphSetConditional(dense_handle,1);
    }
}
extern "C" __global__ void commit(double2* y, double2* f,
    const double2* proposed, const double2* stages, long long n,
    const long long* counts) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n && !counts[STATUS] && counts[ACCEPT_NOW]) {
        y[i]=proposed[i]; f[i]=stages[12*n+i];
    }
}
extern "C" __global__ void finish(double* state, long long* counts,
    const long long* dense_indices, long long npoints, double a,
    long long max_steps, cudaGraphConditionalHandle loop_handle) {
    if (threadIdx.x || blockIdx.x) return;
    if (!counts[STATUS]) {
        if (counts[ACCEPT_NOW]) {
            state[T]=state[T_NEW]; counts[NEXT_POINT]=dense_indices[1];
        }
        state[H]=state[NEXT_H];
        if (state[T]>=a && counts[NEXT_POINT]!=npoints) counts[STATUS]=5;
        if (state[T]<a && counts[ATTEMPTS]>=max_steps) counts[STATUS]=1;
    }
    cudaGraphSetConditional(loop_handle,!counts[STATUS] && state[T]<a);
}
'''


class DeviceControlFailure(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"device DOP853 failed: {diagnostics}")
