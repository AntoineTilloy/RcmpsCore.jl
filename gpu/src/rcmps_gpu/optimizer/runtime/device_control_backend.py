"""Experimental device-controlled DOP853 using native CUDA WHILE/IF graphs.

This module is deliberately separate from the active optimizer. Importing it
does not import CuPy/Triton or initialize CUDA. Constructing the integrator does
compile and capture GPU kernels; call it only during a coordinated GPU pause.

The method, packed equations, seventh-order dense output, and analytic raw
pullback are reused from ode_backend/triton_backend. The adaptive controller,
source evaluation, cusp clipping, quadrature cursor, and acceptance copies run
inside one native conditional graph. One graph launch propagates the complete
trajectory; only the final status is copied to the host. Chebyshev source
coefficients are the independently validated FP64 jax_source coefficients.

Native API reference (12.9 ABI):
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__GRAPH.html
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__STREAM.html
The NVRTC conditional-function declaration follows NVIDIA Warp's conditional
helper module: https://github.com/NVIDIA/warp/blob/main/warp/native/warp.cu

Requires Linux, CUDA driver/runtime >=12.4, CuPy, and Triton >=3.5.1 on SM80 or
SM90. This is an unvalidated experiment until the optional GPU tests pass.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dop

from jax_source import COEFFICIENTS, DEGREE, MAXIMUM, SEGMENTS, source_numpy
from ode_backend import (_GraphAttempt, _GraphDenseOutput, _LinearCombination,
                         _packed_z2_pullback, quadrature)


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


class _ConditionalParams(ct.Structure):
    _fields_ = [("handle", ct.c_ulonglong), ("type", ct.c_int),
                ("size", ct.c_uint), ("phGraph_out", ct.POINTER(ct.c_void_p)),
                ("ctx", ct.c_void_p)]


class _NodeDetails(ct.Union):
    _fields_ = [("reserved1", ct.c_longlong * 29),
                ("conditional", _ConditionalParams)]


class _NodeParams(ct.Structure):
    _fields_ = [("type", ct.c_int), ("reserved0", ct.c_int * 3),
                ("details", _NodeDetails), ("reserved2", ct.c_longlong)]


def native_abi():
    """Expose the header-derived layout without loading a CUDA library."""
    return {"node_bytes": ct.sizeof(_NodeParams),
            "conditional_bytes": ct.sizeof(_ConditionalParams),
            "union_offset": _NodeParams.details.offset,
            "reserved2_offset": _NodeParams.reserved2.offset}


class NativeConditionalGraphs:
    """Minimal CUDA driver bindings, with explicit versioned API signatures."""
    def __init__(self):
        candidates = [ctypes.util.find_library("cuda"), "libcuda.so.1"]
        self.lib = None
        for name in candidates:
            if name:
                try:
                    self.lib = ct.CDLL(name)
                    break
                except OSError:
                    pass
        if self.lib is None:
            raise RuntimeError("libcuda.so.1 is required for native conditional graphs")
        if native_abi() != dict(node_bytes=256, conditional_bytes=32,
                               union_offset=16, reserved2_offset=248):
            raise RuntimeError(f"unsupported native CUDA ABI: {native_abi()}")
        p, n, i = ct.c_void_p, ct.c_size_t, ct.c_int
        pp = ct.POINTER(p)
        self._bind("cuGetErrorName", [i, ct.POINTER(ct.c_char_p)])
        self._bind("cuGetErrorString", [i, ct.POINTER(ct.c_char_p)])
        self._bind("cuDriverGetVersion", [ct.POINTER(i)])
        self._bind("cuCtxGetCurrent", [pp])
        self._bind("cuGraphCreate", [pp, ct.c_uint])
        self._bind("cuGraphDestroy", [p])
        self._bind("cuGraphExecDestroy", [p])
        self._bind("cuGraphConditionalHandleCreate",
                   [ct.POINTER(ct.c_ulonglong), p, p, ct.c_uint, ct.c_uint])
        self._bind("cuGraphAddNode_v2",
                   [pp, p, pp, p, n, ct.POINTER(_NodeParams)])
        self._bind("cuStreamBeginCaptureToGraph", [p, p, pp, p, n, i])
        self._bind("cuStreamGetCaptureInfo_v2",
                   [p, ct.POINTER(i), ct.POINTER(ct.c_ulonglong), pp,
                    ct.POINTER(pp), ct.POINTER(n)])
        self._bind("cuStreamEndCapture", [p, pp])
        self._bind("cuGraphInstantiateWithFlags", [pp, p, ct.c_ulonglong])
        self._bind("cuGraphLaunch", [p, p])
        version = i()
        self.call("cuDriverGetVersion", ct.byref(version))
        self.version = version.value
        if self.version < 12040:
            raise RuntimeError(f"conditional graphs require CUDA >=12.4; driver={self.version}")
        self.context = p()
        self.call("cuCtxGetCurrent", ct.byref(self.context))
        if not self.context.value:
            raise RuntimeError("CuPy must establish a CUDA context before graph construction")

    def _bind(self, name, args):
        try:
            fn = getattr(self.lib, name)
        except AttributeError as exc:
            raise RuntimeError(f"native conditional API unavailable: {name}") from exc
        fn.argtypes, fn.restype = args, ct.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            label, explanation = ct.c_char_p(), ct.c_char_p()
            self.lib.cuGetErrorName(code, ct.byref(label))
            self.lib.cuGetErrorString(code, ct.byref(explanation))
            decode = lambda x: x.value.decode() if x.value else "unavailable"
            raise RuntimeError(f"{name} failed: {code} {decode(label)}: {decode(explanation)}")

    @staticmethod
    def _dependencies(nodes):
        return (ct.c_void_p * len(nodes))(*nodes) if nodes else None

    def create(self):
        graph = ct.c_void_p()
        self.call("cuGraphCreate", ct.byref(graph), 0)
        return graph.value

    def conditional(self, graph, dependencies, kind, default):
        handle = ct.c_ulonglong()
        self.call("cuGraphConditionalHandleCreate", ct.byref(handle), graph,
                  self.context, default, 1)  # CU_GRAPH_COND_ASSIGN_DEFAULT
        params = _NodeParams()
        params.type = 13  # CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.details.conditional.handle = handle.value
        params.details.conditional.type = kind  # IF=0, WHILE=1
        params.details.conditional.size = 1
        params.details.conditional.ctx = self.context
        node = ct.c_void_p()
        self.call("cuGraphAddNode_v2", ct.byref(node), graph,
                  self._dependencies(dependencies), None, len(dependencies), ct.byref(params))
        body = params.details.conditional.phGraph_out[0]
        if not body:
            raise RuntimeError("CUDA returned an empty conditional body graph")
        return node.value, body, handle.value

    def capture(self, stream, graph, dependencies, launch):
        """Append fixed-buffer kernels and return their complete capture frontier."""
        with stream:
            self.call("cuStreamBeginCaptureToGraph", stream.ptr, graph,
                      self._dependencies(dependencies), None, len(dependencies), 1)
            try:
                launch()
                status, capture_id, captured = ct.c_int(), ct.c_ulonglong(), ct.c_void_p()
                frontier, count = ct.POINTER(ct.c_void_p)(), ct.c_size_t()
                self.call("cuStreamGetCaptureInfo_v2", stream.ptr, ct.byref(status),
                          ct.byref(capture_id), ct.byref(captured),
                          ct.byref(frontier), ct.byref(count))
                if status.value != 1 or captured.value != graph:
                    raise RuntimeError("capture did not target the requested conditional body")
                tails = [frontier[k] for k in range(count.value)]
            except BaseException:
                ignored = ct.c_void_p()
                self.lib.cuStreamEndCapture(stream.ptr, ct.byref(ignored))
                raise
            returned = ct.c_void_p()
            self.call("cuStreamEndCapture", stream.ptr, ct.byref(returned))
            if returned.value != graph:
                raise RuntimeError("capture returned a different graph")
        return tails

    def instantiate(self, graph):
        executable = ct.c_void_p()
        self.call("cuGraphInstantiateWithFlags", ct.byref(executable), graph, 0)
        return executable.value


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


class DeviceAdaptiveDOP853:
    """Private-buffer, replayable full-trajectory native conditional graph.

    Constructing this object performs GPU compilation/capture. run() enqueues
    exactly one full-trajectory graph, synchronizes once, and checks every
    controller failure before returning the device samples. Kernels remain
    owned by this instance for the graph's lifetime. The object is not reentrant.
    """
    def __init__(self, rhs, points, *, atol=1e-13, rtol=1e-12,
                 initial_step=None, max_steps=1_000_000):
        cp = rhs.xp
        if cp.__name__ != "cupy":
            raise ValueError("DeviceAdaptiveDOP853 requires CuPy")
        points = np.asarray(points, dtype=np.float64)
        if (points.ndim != 1 or len(points)==0 or np.any(np.diff(points)<=0)
                or not np.all(np.isfinite(points))):
            raise ValueError("points must be finite and strictly increasing")
        self.a = float(points[-1])
        if self.a<=0 or self.a>MAXIMUM or points[0]<-self.a:
            raise ValueError("source coefficients require 0 < half_window <= 40")
        if (not math.isfinite(atol) or not math.isfinite(rtol) or atol<=0
                or rtol<100*np.finfo(float).eps or max_steps<1
                or int(max_steps)!=max_steps):
            raise ValueError("invalid ODE controls")
        self.initial_h = .01 if initial_step is None else float(initial_step)
        if not math.isfinite(self.initial_h) or self.initial_h<=0:
            raise ValueError("initial_step must be positive and finite")
        if cp.cuda.runtime.runtimeGetVersion()<12040:
            raise RuntimeError("CUDA runtime >=12.4 required for conditional graphs")
        self.cp, self.rhs, self.max_steps = cp, rhs, int(max_steps)
        self.graph = self.executable = None
        self.native = None
        started = time.perf_counter()
        self.y = rhs.initial.copy()
        self.f = cp.empty_like(self.y)
        # The prewarm may use the existing exact host source; initialization
        # inside the final graph replaces f with the device-source RHS.
        self.f[:] = rhs(-self.a,self.y)
        self.stages = cp.empty((16,)+self.y.shape,dtype=cp.complex128)
        self.samples = cp.empty((len(points),)+self.y.shape,dtype=cp.complex128)
        self.linear = _LinearCombination(cp,self.y.shape)
        self.attempt = _GraphAttempt(rhs,self.y,self.f,self.stages,self.linear,atol,rtol)
        self.dense = _GraphDenseOutput(self.attempt,self.samples,points)
        self.state = cp.empty(len(FLOAT_NAMES),dtype=cp.float64)
        self.counts = cp.empty(len(INT_NAMES),dtype=cp.int64)
        self.c = cp.asarray(_dop.C,dtype=cp.float64)
        self.coefficients = cp.asarray(COEFFICIENTS,dtype=cp.float64)
        self.atol, self.rtol = float(atol), float(rtol)
        definitions = (f"#define SOURCE_MAXIMUM {MAXIMUM:.17g}\n"
                       f"#define SOURCE_SEGMENTS {SEGMENTS}\n"
                       f"#define SOURCE_DEGREE {DEGREE}\n")
        # Preserve the source's NumPy Clenshaw arithmetic order. No fast math,
        # FP32 constants, or approximate transcendental controller operations.
        self.module = cp.RawModule(code=definitions+CUDA_SOURCE,
            options=("--std=c++11","--fmad=false"),
            name_expressions=("initialize","prepare","control","commit","finish"))
        self.kernels = {name:self.module.get_function(name)
                        for name in ("initialize","prepare","control","commit","finish")}
        cp.cuda.get_current_stream().synchronize()
        self.native = NativeConditionalGraphs()
        try:
            self.graph = self.native.create()
            tails = self.native.capture(self.attempt.capture_stream,self.graph,[],self._initialize)
            self.loop_node,self.body,self.loop_handle = self.native.conditional(
                self.graph,tails,kind=1,default=1)
            # Create the nested IF before capturing its predecessor: the
            # control kernel needs the handle, and the IF gets dependencies
            # afterwards via cuGraphAddDependencies.
            self.dense_node,self.dense_body,self.dense_handle = self.native.conditional(
                self.body,[],kind=0,default=0)
            tails = self.native.capture(self.attempt.capture_stream,self.body,[],self._trial)
            self.native._bind("cuGraphAddDependencies",
                [ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p),ct.c_size_t])
            if tails:
                starts=self.native._dependencies(tails)
                ends=self.native._dependencies([self.dense_node]*len(tails))
                self.native.call("cuGraphAddDependencies",self.body,starts,ends,len(tails))
            self.native.capture(self.attempt.capture_stream,self.dense_body,[],self.dense._launch)
            self.native.capture(self.attempt.capture_stream,self.body,[self.dense_node],self._commit)
            self.executable = self.native.instantiate(self.graph)
        except BaseException:
            self.close()
            raise
        self.setup_seconds = time.perf_counter()-started

    def _initialize(self):
        n=self.y.size
        self.kernels["initialize"](((n+255)//256,),(256,),
            (self.y,self.rhs.initial,np.int64(n),self.state,self.counts,
             self.attempt.params,self.coefficients,np.float64(self.a),np.float64(self.initial_h)))
        self.attempt._rhs_into(0,self.y,self.f)

    def _trial(self):
        self.kernels["prepare"]((1,),(1,),
            (self.state,self.counts,self.attempt.params,self.c,self.coefficients,
             np.float64(self.a),np.int64(self.max_steps)))
        self.attempt._launch()
        self.kernels["control"]((1,),(1,),
            (self.state,self.counts,self.attempt.norms,self.dense.points,
             np.int64(len(self.dense.points)),np.int64(self.y.size),self.dense.params,
             self.dense.indices,self.c,self.coefficients,np.float64(self.a),
             np.uint64(self.dense_handle)))

    def _commit(self):
        n=self.y.size
        self.kernels["commit"](((n+255)//256,),(256,),
            (self.y,self.f,self.attempt.proposed,self.stages,np.int64(n),self.counts))
        self.kernels["finish"]((1,),(1,),
            (self.state,self.counts,self.dense.indices,np.int64(len(self.dense.points)),
             np.float64(self.a),np.int64(self.max_steps),np.uint64(self.loop_handle)))

    def run(self):
        if not self.executable:
            raise RuntimeError("conditional graph is closed")
        cp=self.cp
        stream=cp.cuda.get_current_stream()
        # One graph launch; the device performs all adaptive and dense loops.
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        started=time.perf_counter()
        begin.record(stream)
        self.native.call("cuGraphLaunch",self.executable,stream.ptr)
        end.record(stream)
        end.synchronize()
        elapsed=time.perf_counter()-started
        floats,ints=cp.asnumpy(self.state),cp.asnumpy(self.counts)
        diagnostics=dict(zip(FLOAT_NAMES,map(float,floats)))
        diagnostics.update(zip(INT_NAMES,map(int,ints)))
        diagnostics["status_name"]=STATUS.get(diagnostics["status"],"unknown")
        diagnostics["device_milliseconds"]=float(cp.cuda.get_elapsed_time(begin,end))
        diagnostics["propagation_seconds"]=elapsed
        if diagnostics["status"] or diagnostics["t"]!=self.a:
            raise DeviceControlFailure(diagnostics)
        stats=dict(diagnostics,accepted_steps=diagnostics["accepted"],
            rejected_steps=diagnostics["rejected"],
            rhs_calls=1+12*diagnostics["attempts"]+3*diagnostics["dense_steps"],
            saved_nodes=diagnostics["next_point"],gpu_graph=True,dense_graph=True,
            device_controller=True,host_step_synchronizations=0,
            setup_seconds=self.setup_seconds,arithmetic="complex128/float64",
            source="validated piecewise degree-20 FP64 Chebyshev",
            cuda_driver=self.native.version)
        return self.samples,stats

    def close(self):
        # CUDA owns conditional body graphs and handles; destroying the root
        # destroys them. Keep all arrays/modules alive until exec is destroyed.
        if self.native:
            if self.executable:
                self.native.call("cuGraphExecDestroy",self.executable)
                self.executable=None
            if self.graph:
                self.native.call("cuGraphDestroy",self.graph)
                self.graph=None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def evaluate_quartic_device_control(R,Q,rho,g=1.,*,xp,half_window=30.,n_quad=512,
        atol=1e-13,rtol=1e-12,initial_step=None,max_steps=1_000_000,
        tile_m=16,tile_n=8,num_warps=4,return_samples=False):
    """Standalone FP64 quartic objective and exact analytic raw covector.

    Public invocation starts GPU compilation and execution. Boundary response
    and metric application remain the caller's responsibility, as with every
    existing isolated backend. No CPU objective is evaluated at any D.
    """
    if xp.__name__!="cupy":
        raise ValueError("device control requires CuPy")
    from triton_backend import TritonPackedQuarticRHS
    started=time.perf_counter()
    r,q,rho=(xp.asarray(x,dtype=xp.complex128) for x in (R,Q,rho))
    points,weights=quadrature(half_window,n_quad)
    save_points=np.r_[points,math.sqrt(half_window)]
    rhs=TritonPackedQuarticRHS(q,r,rho,xp,tile_m=tile_m,tile_n=tile_n,num_warps=num_warps)
    integrator=DeviceAdaptiveDOP853(rhs,save_points,atol=atol,rtol=rtol,
        initial_step=initial_step,max_steps=max_steps)
    try:
        samples,stats=integrator.run()
        contraction_started=time.perf_counter()
        ((value,covector,environment),)=_packed_z2_pullback(q,r,rhs,samples,points,weights)
        covector,environment=float(g)*covector,float(g)*environment
        xp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-contraction_started,
            total_seconds=time.perf_counter()-started,packed_z2=True,triton_rhs=True,
            triton_ptx=rhs.ptx_report())
        result=dict(value=float(g)*float(value),energy=float(g)*float(value),
                    covector=covector,environment=environment,stats=stats)
        if return_samples:
            result["samples"]=samples
        return result
    finally:
        integrator.close()
