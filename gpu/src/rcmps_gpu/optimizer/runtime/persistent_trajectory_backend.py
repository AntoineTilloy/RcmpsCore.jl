"""Experimental one-kernel FP64 cooperative DOP853 trajectory.

No existing backend is changed. The row-panel/MMA RHS and error reduction are
extracted from the validated cached-row source; scalar source/controller and
seventh-order interpolation use the same arithmetic order. Constant row panels
remain in shared memory across all attempts. Importing performs no CUDA work.

This is unqualified until the gated actual-GPU tests pass. A queried cooperative
resident-grid guard is mandatory; timeout/max_steps fail before pullback. There
are no software spin barriers, FP32 operations, or implicit solver fallbacks.
"""
from __future__ import annotations

import math
import re
import time
import numpy as np
from scipy.integrate._ivp import dop853_coefficients as dop
from .persistent_rows_cached import cached_row_source
from .device_control_backend import (FLOAT_NAMES, INT_NAMES, STATUS, DeviceControlFailure,
                                    clipped_step, controller_factor)
from .jax_source import COEFFICIENTS, MAXIMUM, SEGMENTS, DEGREE, source_numpy
from .ode_backend import (_PackedZ2QuarticRHS, _UnpackedQuarticRHS, _packed_z2_pullback, _unpacked_pullback,
                          quadrature)

SUPPORTED_PACKED = tuple(range(1, 257))
from .generic_layout import (packed_geometry, dynamic_shared_bytes,
    compilation_options, configure_shared_memory, TILES, select_tiles, unpacked_geometry)
TIMEOUT_STATUS = 6


def geometry(d):
    return packed_geometry(d)







def _validated_parts(d):
    """Extract the unchanged RHS and reduction, rejecting stale source anchors."""
    source = cached_row_source()
    header, body = source.split('extern "C" __global__ void ', 1)
    start = body.index("        double jac=params[3*s+1],source=params[3*s+2];")
    end = body.index("\n    double sum5=0.,sum3=0.;", start)
    rhs = body[start:end]
    if not rhs.endswith("\n    }"):
        raise RuntimeError("unexpected cached RHS closing anchor")
    rhs = rhs[:-6].replace("        double jac=params[3*s+1],source=params[3*s+2];\n", "")
    rhs = rhs.replace("k[s*total+", "out[")
    if "params" in rhs or "s*total" in rhs:
        raise RuntimeError("stage-dependent RHS extraction failed")
    norm = body[end:body.rindex("\n}")]
    if norm.count("for(int w=0;w<4;++w)") != 1:
        raise RuntimeError("unexpected norm reduction anchor")
    norm = norm.replace("for(int w=0;w<4;++w)", f"for(int w=0;w<{geometry(d)['warps']};++w)")
    # In the one-kernel loop the leader computes the controller after the same
    # reduction. The final cross-block dependency is provided by that loop.
    return header, rhs, norm


COORDINATES = r'''
    const int d=PACKED_D,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int tiles=(d+7)/8,matrix_blocks=10*tiles,scalar=blockIdx.x==matrix_blocks;
    const int channel=blockIdx.x/tiles,row_tile=blockIdx.x%tiles;
    const int m=row_tile*8,n=warp*8,row=m+(lane>>2),col=n+2*(lane&3);
    const int total=10*d*d+1,element0=channel*d*d+row*d+col,element1=element0+1;
    const bool valid0=!scalar && row<d && col<d,valid1=!scalar && row<d && col+1<d;
'''

SCALARS = r'''
#define INF64 __longlong_as_double(0x7ff0000000000000LL)
enum { T=0,H=1,NEXT_H=2,T_NEW=3,ERROR=4,MIN_H=5,MAX_H=6 };
enum { STATUS=0,ATTEMPTS=1,ACCEPTED=2,REJECTED=3,WAS_REJECTED=4,
       ACCEPT_NOW=5,NEXT_POINT=6,DENSE_STEPS=7,DENSE_NODES_MAX=8 };
// Scalar controller was compiled --fmad=false in the validated graph backend.
// Explicit round-to-nearest intrinsics retain it while RHS MMA remains native.
__device__ __forceinline__ double add64(double x,double y){return __dadd_rn(x,y);}
__device__ __forceinline__ double mul64(double x,double y){return __dmul_rn(x,y);}
__device__ double source_fp64(double t,const double* coefficients){
    double a=fabs(t),maximum=SOURCE_MAXIMUM;
    int segment=min((int)(mul64(a/maximum,SOURCE_SEGMENTS)),SOURCE_SEGMENTS-1);
    double z=add64(add64(mul64(mul64(2.,a)/maximum,SOURCE_SEGMENTS),-2.*segment),-1.);
    const double* c=coefficients+segment*(SOURCE_DEGREE+1);
    double b1=0.,b2=0.;
    for(int j=SOURCE_DEGREE;j>0;--j){
        double b0=add64(add64(mul64(mul64(2.,z),b1),-b2),c[j]);b2=b1;b1=b0;
    }
    return add64(add64(mul64(z,b1),-b2),c[0]);
}
__device__ void prepare_scalar(double* state,long long* counts,double* params,
    const double* c,const double* source_coefficients,double endpoint,long long max_steps,
    unsigned long long start_clock,unsigned long long maximum_ticks){
    double t=state[T],boundary=t<0.?0.:endpoint,h=fmin(state[H],boundary-t);
    counts[ACCEPT_NOW]=0;
    if(counts[ATTEMPTS]>=max_steps)counts[STATUS]=1;
    else if(!isfinite(h)||h<=mul64(10.,fabs(nextafter(t,INF64)-t)))counts[STATUS]=2;
    else if(maximum_ticks && clock64()-start_clock>=maximum_ticks)counts[STATUS]=6;
    if(!counts[STATUS]){++counts[ATTEMPTS];state[MIN_H]=fmin(state[MIN_H],h);state[MAX_H]=fmax(state[MAX_H],h);}
    else h=0.;
    state[H]=h;params[0]=h;
    for(int s=1;s<=12;++s){
        double ts=add64(t,mul64(c[s],h));
        if(fabs(ts)>mul64(SOURCE_MAXIMUM,1.+1.e-14))counts[STATUS]=4;
        params[3*s+1]=2.*fabs(ts);params[3*s+2]=source_fp64(ts,source_coefficients);
    }
}
__device__ void control_scalar(double* state,long long* counts,const double* norms,
    const double* points,long long npoints,int total,long long* indices,
    const double* c,const double* source_coefficients,double* params,double endpoint){
    if(counts[STATUS])return;
    double e5=norms[0],e3=norms[1],h=state[H],t=state[T];
    double denominator=add64(e5,mul64(.01,e3));
    double error=denominator>0.?mul64(h,e5)/sqrt(mul64(denominator,(double)total)):0.;
    state[ERROR]=error;
    if(!isfinite(e5)||!isfinite(e3)||e5<0.||e3<0.||!isfinite(error)){counts[STATUS]=3;return;}
    double factor=error>0.?fmin(10.,mul64(.9,pow(error,-1./8.))):10.;
    if(error>1.){++counts[REJECTED];counts[WAS_REJECTED]=1;state[NEXT_H]=mul64(h,fmax(.2,factor));return;}
    counts[ACCEPT_NOW]=1;++counts[ACCEPTED];
    double boundary=t<0.?0.:endpoint,t_new=h==boundary-t?boundary:add64(t,h);
    state[T_NEW]=t_new;if(counts[WAS_REJECTED])factor=fmin(1.,factor);
    state[NEXT_H]=mul64(h,factor);counts[WAS_REJECTED]=0;
    long long first=counts[NEXT_POINT],last=first;
    while(last<npoints&&points[last]<=t_new)++last;
    indices[0]=first;indices[1]=last;
    if(last>first){++counts[DENSE_STEPS];counts[DENSE_NODES_MAX]=max(counts[DENSE_NODES_MAX],last-first);
        for(int s=13;s<16;++s){double ts=add64(t,mul64(c[s],h));
            params[3*s+1]=2.*fabs(ts);params[3*s+2]=source_fp64(ts,source_coefficients);}
    }
}
__device__ __forceinline__ double2 addz(double2 x,double2 y){return make_double2(x.x+y.x,x.y+y.y);}
__device__ __forceinline__ double2 subz(double2 x,double2 y){return make_double2(x.x-y.x,x.y-y.y);}
__device__ __forceinline__ double2 scalez(double a,double2 x){return make_double2(a*x.x,a*x.y);}
__device__ void save_dense(int element,const double2* y,const double2* proposed,const double2* f,
    const double2* k,const double* dense_d,const double* points,const long long* indices,
    double t,double h,double t_new,int total,double2* samples){
    double2 delta=subz(proposed[element],y[element]),coefficients[7];
    coefficients[0]=delta;coefficients[1]=subz(scalez(h,f[element]),delta);
    coefficients[2]=subz(scalez(2.,delta),scalez(h,addz(k[12*total+element],f[element])));
    #pragma unroll
    for(int row=0;row<4;++row){double2 value=zeroz();
        for(int s=0;s<16;++s)value=addz(value,scalez(h*dense_d[row*16+s],k[s*total+element]));
        coefficients[3+row]=value;
    }
    for(long long node=indices[0];node<indices[1];++node){
        if(points[node]==t_new)samples[node*total+element]=proposed[element];
        else {double x=(points[node]-t)/h;double2 value=zeroz();
            #pragma unroll
            for(int s=0;s<7;++s){value=addz(value,coefficients[6-s]);value=scalez(s%2==0?x:1.-x,value);}
            samples[node*total+element]=addz(value,y[element]);}
    }
}
'''


# The tile product mma() of the RHS. MMA_TILE is the native FP64 tensor-core instruction of
# the validated source (compute capability 8.0 or later). SCALAR_TILE, the '#else' branch of
# excitations/fused.py, computes the same m8n8k4 fragments with warp shuffles: each
# accumulator is a chain of FP64 FMAs in k order (any GPU with cooperative launch). Both
# need all 32 lanes of a warp, which every call in the block-uniform if(!scalar) of the RHS has.
MMA_TILE = '''__device__ __forceinline__ void mma(double a,double b,double &c0,double &c1){
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+d"(c0),"+d"(c1):"d"(a),"d"(b));
}
'''
SCALAR_TILE = r'''// Scalar tile product with the same fragment layout (any cooperative-launch GPU).
__device__ __forceinline__ void mma(double a,double b,double &c0,double &c1){
    const int lane=threadIdx.x&31,rl=lane>>2,cl=2*(lane&3);
    #pragma unroll
    for(int q=0;q<4;++q){
        double av=__shfl_sync(0xffffffffu,a,(rl<<2)|q);
        double b0=__shfl_sync(0xffffffffu,b,(cl<<2)|q),b1=__shfl_sync(0xffffffffu,b,((cl+1)<<2)|q);
        c0+=av*b0;c1+=av*b1;
    }
}
'''
MATRIX_ARITHMETIC = dict(mma='mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64', scalar='scalar warp-shuffle tiles')


def trajectory_source(d, tiles="mma"):
    """CUDA source of the trajectory kernel for packed dimension d. tiles="mma" is the
    validated source, unchanged (tests/optimizer/golden_cuda.json); "scalar" differs only
    by the definition of mma()."""
    if tiles not in TILES:
        raise ValueError(f"tiles must be one of {', '.join(TILES)}, not {tiles!r}")
    info = geometry(d)
    header, rhs, norm = _validated_parts(d)
    if tiles == "scalar":
        if header.count(MMA_TILE) != 1:
            raise RuntimeError("trajectory tile anchor changed")
        header = header.replace(MMA_TILE, SCALAR_TILE, 1)
    definitions = (f"#define PACKED_D {d}\n#define ROW_WARPS {info['warps']}\n"
                   f"#define SOURCE_MAXIMUM {MAXIMUM:.17g}\n#define SOURCE_SEGMENTS {SEGMENTS}\n"
                   f"#define SOURCE_DEGREE {DEGREE}\n")
    rhs_signature = r'''
__device__ __forceinline__ void trajectory_rhs(const double2* stage,double2* out,double2* base,
    const double2* qr,const double2* rr,const double2* op_right,const double2* fixed,
    const double* coefficients,const int* indices,const int* adjoints,const double2* a,const double2* b,
    const double2* cached_ql,const double2* cached_rl,const double2* cached_ol,double2* row_temp,
    double jac,double source){
    cg::grid_group grid=cg::this_grid();
'''
    rhs_call = ("trajectory_rhs(stage,k+s*total,base,qr,rr,op_right,fixed,coefficients,indices,adjoints,a,b,"
                "cached_ql,cached_rl,cached_ol,row_temp,params[3*s+1],params[3*s+2]);")
    kernel = r'''
extern "C" __global__ void persistent_trajectory(
    const double2* initial,double2* y,double2* f,double2* k,double2* stage,double2* proposed,double2* base,
    const double2* ql,const double2* qr,const double2* rl,const double2* rr,const double2* ol,
    const double2* op_right,const double2* fixed,const double* coefficients,const int* indices,
    const int* adjoints,const double2* a,const double2* b,const double* rk_a,const double* rk_b,
    const double* e5,const double* e3,const double* dense_d,const double* c,const double* source_coefficients,
    const double* points,long long npoints,double2* samples,double* state,long long* counts,
    double* params,long long* dense_indices,double* partials,double* norms,double atol,double rtol,
    double endpoint,double initial_h,long long max_steps,unsigned long long maximum_ticks){
    cg::grid_group grid=cg::this_grid();
    const bool leader=blockIdx.x==0&&threadIdx.x==0;
    unsigned long long start_clock=leader?clock64():0;
    __shared__ double2 row_temp[8*PACKED_D];
    __shared__ double2 cached_ql[8*PACKED_D],cached_rl[8*PACKED_D],cached_ol[8*PACKED_D];
    __shared__ double warp_norms[2*ROW_WARPS];
    COORDINATE_ANCHOR
    if(!scalar){for(int offset=threadIdx.x;offset<8*d;offset+=blockDim.x){
        int ar=m+offset/d,kp=offset%d;cached_ql[offset]=loadz(ql,channel,ar,kp,d);
        cached_rl[offset]=loadz(rl,channel,ar,kp,d);cached_ol[offset]=loadz(ol,channel,ar,kp,d);}}
    for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
        if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}
        if(valid)y[element]=stage[element]=initial[element];}
    if(leader){for(int i=0;i<7;++i)state[i]=0.;for(int i=0;i<9;++i)counts[i]=0;
        state[T]=-endpoint;state[H]=fmin(initial_h,2.*endpoint);state[MIN_H]=INF64;
        params[1]=2.*endpoint;params[2]=source_fp64(-endpoint,source_coefficients);}
    grid.sync();
    trajectory_rhs(stage,k,base,qr,rr,op_right,fixed,coefficients,indices,adjoints,a,b,
                   cached_ql,cached_rl,cached_ol,row_temp,params[1],params[2]);
    for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
        if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}if(valid)f[element]=k[element];}
    grid.sync();
    while(true){
        if(leader)prepare_scalar(state,counts,params,c,source_coefficients,endpoint,max_steps,start_clock,maximum_ticks);
        grid.sync();if(counts[STATUS])break;
        double h=params[0];
        for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
            if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}if(valid)k[element]=f[element];}
        grid.sync();
        for(int s=1;s<=12;++s){const double* weights=s==12?rk_b:rk_a+s*16;int terms=s==12?12:s;
            for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
                if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}
                if(valid){double2 z=y[element];for(int j=0;j<terms;++j){double2 v=k[j*total+element];
                    z.x+=h*weights[j]*v.x;z.y+=h*weights[j]*v.y;}stage[element]=z;if(s==12)proposed[element]=z;}}
            grid.sync();RHS_CALL_ANCHOR
        }
        NORM_ANCHOR
        if(leader)control_scalar(state,counts,norms,points,npoints,total,dense_indices,c,
                                source_coefficients,params,endpoint);
        grid.sync();if(counts[STATUS])break;
        if(counts[ACCEPT_NOW]){
            if(dense_indices[1]>dense_indices[0]){
                for(int s=13;s<16;++s){const double* weights=rk_a+s*16;
                    for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
                        if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}
                        if(valid){double2 z=y[element];for(int j=0;j<s;++j){double2 v=k[j*total+element];
                            z.x+=h*weights[j]*v.x;z.y+=h*weights[j]*v.y;}stage[element]=z;}}
                    grid.sync();RHS_CALL_ANCHOR
                }
                for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
                    if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}
                    if(valid)save_dense(element,y,proposed,f,k,dense_d,points,dense_indices,state[T],h,state[T_NEW],total,samples);}
            }
            for(int slot=0;slot<2;++slot){int element=slot?element1:element0;bool valid=slot?valid1:valid0;
                if(scalar){element=total-1;valid=threadIdx.x==0&&slot==0;}
                if(valid){y[element]=proposed[element];f[element]=k[12*total+element];}}
        }
        // Other CTAs still read T/T_NEW during interpolation. Commit their
        // private components before the leader changes the scalar time.
        grid.sync();
        if(leader){if(counts[ACCEPT_NOW]){state[T]=state[T_NEW];counts[NEXT_POINT]=dense_indices[1];}
            state[H]=state[NEXT_H];
            if(state[T]>=endpoint&&counts[NEXT_POINT]!=npoints)counts[STATUS]=5;
            if(state[T]<endpoint&&counts[ATTEMPTS]>=max_steps)counts[STATUS]=1;}
        grid.sync();if(counts[STATUS]||state[T]>=endpoint)break;
    }
}
'''
    if dynamic_shared_bytes(d):
        declaration=("    __shared__ double2 row_temp[8*PACKED_D];\n"
            "    __shared__ double2 cached_ql[8*PACKED_D],cached_rl[8*PACKED_D],cached_ol[8*PACKED_D];\n"
            "    __shared__ double warp_norms[2*ROW_WARPS];")
        if kernel.count(declaration)!=1:raise RuntimeError('trajectory shared anchor changed')
        kernel=kernel.replace(declaration,'    extern __shared__ double2 dynamic_panels[];\n    double2* row_temp=dynamic_panels;\n    double2* cached_ql=row_temp+8*PACKED_D;\n    double2* cached_rl=cached_ql+8*PACKED_D;\n    double2* cached_ol=cached_rl+8*PACKED_D;\n    double* warp_norms=reinterpret_cast<double*>(cached_ol+8*PACKED_D);',1)
    kernel = kernel.replace("COORDINATE_ANCHOR", COORDINATES).replace("RHS_CALL_ANCHOR", rhs_call)
    kernel = kernel.replace("NORM_ANCHOR", norm)
    return definitions+header+SCALARS+rhs_signature+COORDINATES+rhs+"\n}\n"+kernel


# The rewrites of trajectory_source(D) into the kernel of the seven unpacked D x D channels of
# ode_backend._UnpackedQuarticRHS (states without Z2 symmetry): (anchor, replacement, count).
UNPACKED_REWRITES = (
    ("matrix_blocks=10*tiles", "matrix_blocks=7*tiles", 2),
    ("total=10*d*d+1", "total=7*d*d+1", 2),
    ("indices[10+channel],adjoints[10+channel]", "indices[7+channel],adjoints[7+channel]", 1),
    ("indices[20+channel],adjoints[20+channel]", "indices[14+channel],adjoints[14+channel]", 1),
    # The scalar: Re tr(r F3†) + Re tr(r F3) of channel 2, with a = b = r.
    ("v=stage[3*d*d+i],vt=stage[3*d*d+col*d+row]", "v=stage[2*d*d+i],vt=stage[2*d*d+col*d+row]", 1),
    ("make_double2(8.*source*trace,0.)", "make_double2(4.*source*trace,0.)", 1),
)


def unpacked_trajectory_source(D, tiles="mma"):
    """CUDA source of the trajectory kernel for the seven unpacked D x D channels (any D up to
    256): trajectory_source(D, tiles) with the channel count, gather offsets and scalar of
    UNPACKED_REWRITES; every other line, the controller and the tile product included, is
    the packed kernel's."""
    source = trajectory_source(D, tiles)
    for old, new, count in UNPACKED_REWRITES:
        if source.count(old) != count:
            raise RuntimeError(f"unpacked trajectory anchor changed: {old!r}")
        source = source.replace(old, new)
    return "// Unpacked layout: seven D x D channels (ode_backend._UnpackedQuarticRHS).\n" + source


def _controls(points, atol, rtol, initial_step, max_steps, max_device_seconds):
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 1 or not len(points) or not np.all(np.isfinite(points)) or
            np.any(np.diff(points)<=0)):
        raise ValueError("points must be finite and strictly increasing")
    endpoint = float(points[-1])
    if endpoint<=0 or endpoint>MAXIMUM or points[0]<-endpoint:
        raise ValueError("source coefficients require 0 < half_window <=40")
    initial_h = .01 if initial_step is None else float(initial_step)
    if (not math.isfinite(atol) or atol<=0 or not math.isfinite(rtol) or
            rtol<100*np.finfo(float).eps or max_steps<1 or int(max_steps)!=max_steps or
            not math.isfinite(initial_h) or initial_h<=0 or
            not math.isfinite(max_device_seconds) or max_device_seconds<=0):
        raise ValueError("invalid strict ODE/watchdog controls")
    return points, endpoint, initial_h


class PersistentTrajectoryDOP853:
    """Private replayable trajectory buffers; one cooperative launch per run."""
    def __init__(self, rhs, points, *, atol=1e-13, rtol=1e-12, initial_step=None,
                 max_steps=1_000_000, max_device_seconds=30., tiles="mma", register_cap=None):
        cp = rhs.xp
        if cp.__name__ != "cupy":
            raise ValueError("persistent trajectory requires CuPy")
        if rhs.initial.dtype!=cp.complex128 or not rhs.initial.flags.c_contiguous:
            raise TypeError("packed initial state requires contiguous complex128")
        # The old _GraphAttempt constructor made these contiguous as a side
        # effect. A stand-alone trajectory must establish its own pointer ABI.
        for name in ('ql','qright','rl','rright','ol','oright','fixed','a','b'):
            value=getattr(rhs,name)
            if value.dtype!=cp.complex128:raise TypeError('packed operators require complex128')
            setattr(rhs,name,cp.ascontiguousarray(value))
        points, self.endpoint, self.initial_h = _controls(points,atol,rtol,initial_step,max_steps,max_device_seconds)
        started = time.perf_counter()
        # The unpacked channels (states without Z2 symmetry) have their own grid and source.
        self.unpacked = bool(getattr(rhs, 'unpacked', False))
        layout, source = (unpacked_geometry, unpacked_trajectory_source) if self.unpacked else (geometry, trajectory_source)
        self.cp,self.rhs,self.info,self.closed = cp,rhs,layout(rhs.d),False
        self.atol,self.rtol,self.max_steps = float(atol),float(rtol),int(max_steps)
        self.max_device_seconds = float(max_device_seconds)
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        if tiles not in TILES:raise ValueError(f"tiles must be one of {', '.join(TILES)}, not {tiles!r}")
        if not props.get('cooperativeLaunch',False):
            raise ValueError("the one-kernel trajectory requires cooperative launch")
        self.compute_capability=(int(props['major']),int(props['minor']))
        self.tiles=select_tiles(tiles,*self.compute_capability)   # refuses mma below 8.0
        clock_rate = int(props['clockRate'])  # CUDA property is in kHz.
        self.maximum_ticks = int(math.ceil(self.max_device_seconds*1000*clock_rate))
        # register_cap: the cap the worker selected (generic_layout.register_caps); this
        # solver compiles with it once and refuses a grid that is not resident.
        self.register_cap=register_cap
        self.kernel=cp.RawKernel(source(rhs.d,self.tiles),'persistent_trajectory',
            options=compilation_options(rhs.d,register_cap),enable_cooperative_groups=True)
        self.kernel.compile()
        self.dynamic_shared_bytes=configure_shared_memory(self.kernel,props,self.info,self.tiles)
        resident=int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
            self.kernel.kernel.ptr,self.info['threads'],self.dynamic_shared_bytes))
        sms=int(props['multiProcessorCount']);capacity=resident*sms
        if self.info['blocks']>capacity:
            raise RuntimeError(f"unsafe cooperative trajectory grid {self.info['blocks']}>{capacity}")
        self.occupancy=dict(self.info,active_blocks_per_sm=resident,multiprocessors=sms,resident_capacity=capacity,
            registers=int(self.kernel.num_regs),local_bytes=int(self.kernel.local_size_bytes),
            actual_shared_bytes=int(self.kernel.shared_size_bytes)+self.dynamic_shared_bytes,cooperative_capacity_verified=True,
            static_shared_bytes=int(self.kernel.shared_size_bytes),dynamic_shared_bytes=self.dynamic_shared_bytes,
            requested_register_cap=register_cap if register_cap is not None else 64 if rhs.d>64 else None,
            compiler_options=list(compilation_options(rhs.d,register_cap)),
            matrix_arithmetic=MATRIX_ARITHMETIC[self.tiles],tiles=self.tiles,
            compute_capability=list(self.compute_capability),fast_math=False)
        if self.occupancy['actual_shared_bytes']!=self.info['shared_bytes']:
            raise RuntimeError("unexpected trajectory shared-memory layout")
        self.y,self.f,self.stage,self.proposed,self.base = [cp.empty_like(rhs.initial) for _ in range(5)]
        self.stages=cp.empty((16,)+rhs.initial.shape,dtype=cp.complex128)
        self.samples=cp.empty((len(points),)+rhs.initial.shape,dtype=cp.complex128)
        self.points=cp.asarray(points,dtype=cp.float64)
        self.state,self.counts=cp.empty(7,dtype=cp.float64),cp.empty(9,dtype=cp.int64)
        self.params,self.dense_indices=cp.empty((16,3),dtype=cp.float64),cp.empty(2,dtype=cp.int64)
        self.partials,self.norms=cp.empty(2*self.info['blocks'],dtype=cp.float64),cp.empty(2,dtype=cp.float64)
        self.coefficients=cp.asarray(COEFFICIENTS,dtype=cp.float64)
        self.rk_a,self.rk_b,self.e5,self.e3,self.dense_d,self.c=[cp.asarray(x,dtype=cp.float64)
            for x in (dop.A,dop.B,dop.E5,dop.E3,dop.D,dop.C)]
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds=time.perf_counter()-started

    def run(self):
        if self.closed:raise RuntimeError("persistent trajectory is closed")
        cp,rhs=self.cp,self.rhs
        begin,end=cp.cuda.Event(),cp.cuda.Event();started=time.perf_counter();begin.record()
        self.kernel((self.info['blocks'],),(self.info['threads'],),
            (rhs.initial,self.y,self.f,self.stages,self.stage,self.proposed,self.base,
             rhs.ql,rhs.qright,rhs.rl,rhs.rright,rhs.ol,rhs.oright,rhs.fixed,rhs.coefficients,
             rhs.device_indices,rhs.device_adjoint,rhs.a,rhs.b,self.rk_a,self.rk_b,self.e5,self.e3,
             self.dense_d,self.c,self.coefficients,self.points,np.int64(len(self.points)),self.samples,
             self.state,self.counts,self.params,self.dense_indices,self.partials,self.norms,
             np.float64(self.atol),np.float64(self.rtol),np.float64(self.endpoint),np.float64(self.initial_h),
             np.int64(self.max_steps),np.uint64(self.maximum_ticks)),shared_mem=self.dynamic_shared_bytes)
        end.record();end.synchronize();elapsed=time.perf_counter()-started
        diagnostics=dict(zip(FLOAT_NAMES,map(float,cp.asnumpy(self.state))))
        diagnostics.update(zip(INT_NAMES,map(int,cp.asnumpy(self.counts))))
        diagnostics.update(status_name='device_clock_timeout' if diagnostics['status']==TIMEOUT_STATUS
            else STATUS.get(diagnostics['status'],'unknown'),device_milliseconds=float(cp.cuda.get_elapsed_time(begin,end)),
            propagation_seconds=elapsed)
        if diagnostics['status'] or diagnostics['t']!=self.endpoint:
            raise DeviceControlFailure(diagnostics)
        stats=dict(diagnostics,accepted_steps=diagnostics['accepted'],rejected_steps=diagnostics['rejected'],
            rhs_calls=1+12*diagnostics['attempts']+3*diagnostics['dense_steps'],saved_nodes=diagnostics['next_point'],
            persistent_trajectory=True,device_controller=True,host_step_synchronizations=0,trajectory_kernel_launches=1,
            dense_graph=False,gpu_graph=False,setup_seconds=self.setup_seconds,occupancy=self.occupancy,
            arithmetic='complex128/float64',source='validated piecewise degree20 FP64 Chebyshev',
            max_device_seconds=self.max_device_seconds,device_clock_watchdog_ticks=self.maximum_ticks,
            device_clock_watchdog_is_not_wall_clock=True,implicit_fallbacks=0)
        return self.samples,stats

    def close(self):self.closed=True
    def __enter__(self):return self
    def __exit__(self,*errors):self.close()


def evaluate_quartic_persistent_trajectory(R,Q,rho,g=1.,*,xp,half_window=30.,n_quad=512,
        atol=1e-13,rtol=1e-12,initial_step=None,max_steps=1_000_000,max_device_seconds=30.,return_samples=False,
        tiles='mma',register_cap=None,unpacked=False):
    start=time.perf_counter()
    if xp.__name__!='cupy':raise ValueError('persistent trajectory evaluator is GPU-only')
    if any(x.dtype!=xp.complex128 for x in (R,Q,rho)):raise TypeError('input matrices require complex128')
    if (Q.ndim!=2 or Q.shape[0]!=Q.shape[1] or any(x.shape!=Q.shape for x in (R,rho)) or
            any(not bool(xp.isfinite(x).all().item()) for x in (R,Q,rho))):
        raise ValueError('input matrices require equally shaped finite square complex128 arrays')
    r,q,rho=(xp.asarray(x) for x in (R,Q,rho))
    rhs=(_UnpackedQuarticRHS if unpacked else _PackedZ2QuarticRHS)(q,r,rho,xp)
    points,weights=quadrature(half_window,n_quad)
    with PersistentTrajectoryDOP853(rhs,np.r_[points,math.sqrt(half_window)],atol=atol,rtol=rtol,
            initial_step=initial_step,max_steps=max_steps,max_device_seconds=max_device_seconds,tiles=tiles,
            register_cap=register_cap) as solver:
        samples,stats=solver.run();begin=time.perf_counter()
        ((value,covector,environment),)=(_unpacked_pullback if unpacked else _packed_z2_pullback)(q,r,rhs,samples,points,weights)
        xp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-begin,total_seconds=time.perf_counter()-start)
        result=dict(value=float(g)*value,energy=float(g)*value,covector=float(g)*covector,
                    environment=float(g)*environment,stats=stats)
        if return_samples:result['samples']=samples
        return result


def audit_trajectory_ptx(ptx,tiles):
    """Instruction audit of the PTX of trajectory_source(d, tiles); RuntimeError if it fails.
    Both paths: no FP32 arithmetic. mma: FP64 mma.sync instructions, for a target of compute
    capability 8.0 or later (NVRTC emits them for earlier targets too; only ptxas refuses).
    scalar: no mma instruction of any type, and the warp shuffles and FP64 FMAs of the tile
    product."""
    if tiles not in TILES:raise ValueError(f"tiles must be one of {', '.join(TILES)}, not {tiles!r}")
    target=re.search(r'^\.target\s+sm_(\d+)',ptx,re.M)
    mma=sorted(set(re.findall(r'mma\.sync[^\s;]*\.f64\.f64\.f64\.f64',ptx)))
    any_mma=sorted(set(re.findall(r'\bw?mma\.[^\s;]*',ptx)))
    fp32=sorted(set(re.findall(r'\b(?:add|sub|mul|fma|div|mad|sqrt|rcp|ex2|lg2)\.(?:[a-z0-9]+\.)*f32\b',ptx)))
    shuffles=len(re.findall(r'\bshfl\.sync\.idx\.b32\b',ptx));fmas=len(re.findall(r'\bfma\.rn\.f64\b',ptx))
    failures=[]
    if fp32:failures.append(f'FP32 arithmetic {fp32}')
    if target is None:failures.append('no .target directive')
    if tiles=='mma':
        if not mma:failures.append('no FP64 mma.sync instruction')
        if target is not None and int(target.group(1))<80:
            failures.append(f'mma.sync f64 for target sm_{target.group(1)} (needs compute capability 8.0)')
    else:
        if any_mma:failures.append(f'mma instructions {any_mma} in the scalar tile path')
        if not shuffles or not fmas:failures.append('no warp shuffles or FP64 FMAs of the scalar tile product')
    if failures:raise RuntimeError(f'trajectory FP64 PTX audit failed ({tiles} tiles): '+'; '.join(failures))
    return dict(tiles=tiles,ptx_target=f'sm_{target.group(1)}',f64_mma_instructions=mma,
                fp32_arithmetic_instructions=fp32,shfl_sync_idx_instructions=shuffles,fma_rn_f64_instructions=fmas)


def trajectory_ptx_report(d,cp,directory=None,tiles='mma',register_cap=None,unpacked=False):
    """Gated CUDA NVRTC audit of the tile products and scalar instruction types; unpacked:
    of unpacked_trajectory_source(d), the kernel of the unpacked channels."""
    import importlib.util
    from pathlib import Path
    from cupy.cuda import compiler
    from cupy._core import core
    cuda_path=cp.cuda.get_cuda_path();directories=[] if cuda_path is None else [Path(cuda_path)/'include']
    package=importlib.util.find_spec('nvidia.cuda_runtime')
    if package and package.submodule_search_locations:
        directories.extend(Path(path)/'include' for path in package.submodule_search_locations)
    includes=tuple('-I'+str(path) for path in directories if path.is_dir())
    includes+=tuple(core._get_cccl_include_options())
    arch=cp.cuda.Device().compute_capability
    program=compiler._NVRTCProgram((unpacked_trajectory_source if unpacked else trajectory_source)(d,tiles),
                                   'persistent_trajectory_audit.cu',method='ptx')
    binary,_=program.compile(('--std=c++17','--device-c','--device-as-default-execution-space',
                             '-arch=compute_'+arch)+compilation_options(d,register_cap)[1:]+includes)
    ptx=binary.decode()
    audit=audit_trajectory_ptx(ptx,tiles)
    if directory:
        target=Path(directory);target.mkdir(parents=True,exist_ok=True)
        name=f"persistent_trajectory_{'unpacked_D' if unpacked else 'd'}{d}{'' if tiles=='mma' else '_scalar'}.ptx"
        (target/name).write_text(ptx)
    return dict(audit,arch=arch,controller_uses_explicit_round_to_nearest_intrinsics=True)


def integrate_trajectory_numpy(rhs,points,*,atol=1e-13,rtol=1e-12,initial_step=None,max_steps=1_000_000):
    """Bounded small-D science oracle for the new explicit loop/interpolation."""
    if rhs.xp is not np or rhs.full_d>8:raise ValueError('CPU trajectory oracle is limited to physical D<=8')
    points,a,h=_controls(points,atol,rtol,initial_step,max_steps,30.)
    y=rhs.initial.copy();k=np.empty((16,)+y.shape,dtype=np.complex128)
    samples=np.empty((len(points),)+y.shape,dtype=np.complex128)
    def call(t,z):
        # Same exact packed equations, device-source representation.
        original=rhs(-a if t==-a else t,z)
        source=source_numpy(t);exact_source=__import__('ode_backend',globals(),level=1).transformed_source(t)
        # Isolate the source-linear term without changing the Lindblad term.
        if source==exact_source:return original
        return original+(source-exact_source)*(rhs(0.,z)/__import__('ode_backend',globals(),level=1).transformed_source(0.))
    t=-a;f=call(t,y);attempts=accepted=rejected=dense_steps=next_point=0;was_rejected=False
    while t<a:
        if attempts>=max_steps:raise RuntimeError('max_steps')
        try:h,_=clipped_step(t,h,a)
        except RuntimeError as error:raise RuntimeError('step_underflow') from error
        attempts+=1;k[0]=f
        for s in range(1,13):
            weights=dop.B if s==12 else dop.A[s];terms=12 if s==12 else s
            z=y.copy()
            for j in range(terms):z+=h*weights[j]*k[j]
            k[s]=call(t+dop.C[s]*h,z)
        proposed=z
        err5=sum(dop.E5[s]*k[s] for s in range(13));err3=sum(dop.E3[s]*k[s] for s in range(13))
        scale=atol+rtol*np.maximum(abs(y),abs(proposed))
        e5=float(np.sum(abs(err5/scale)**2));e3=float(np.sum(abs(err3/scale)**2))
        denominator=e5+.01*e3;error=h*e5/math.sqrt(denominator*y.size) if denominator>0 else 0.
        if not math.isfinite(error):raise RuntimeError('nonfinite_error')
        if error>1:
            rejected+=1;was_rejected=True;h*=max(.2,min(10.,.9*error**(-1/8)));continue
        accepted+=1;boundary=0. if t<0 else a;t_new=boundary if h==boundary-t else t+h
        _,factor=controller_factor(error,was_rejected);was_rejected=False
        first=next_point
        while next_point<len(points) and points[next_point]<=t_new:next_point+=1
        if next_point>first:
            dense_steps+=1
            for s in range(13,16):
                z=y.copy()
                for j in range(s):z+=h*dop.A[s,j]*k[j]
                k[s]=call(t+dop.C[s]*h,z)
            delta=proposed-y;coeff=[delta,h*f-delta,2*delta-h*(k[12]+f)]
            for row in range(4):coeff.append(sum(h*dop.D[row,s]*k[s] for s in range(16)))
            for node in range(first,next_point):
                if points[node]==t_new:samples[node]=proposed
                else:
                    x=(points[node]-t)/h;value=np.zeros_like(y)
                    for s in range(7):value=(value+coeff[6-s])*(x if s%2==0 else 1-x)
                    samples[node]=value+y
        y,f=proposed,k[12].copy();t=t_new;h*=factor
    if next_point!=len(points):raise RuntimeError('incomplete_samples')
    return samples,dict(attempts=attempts,accepted=accepted,rejected=rejected,dense_steps=dense_steps,saved_nodes=next_point)
