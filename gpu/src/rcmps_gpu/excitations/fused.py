"""Fused FP64 cooperative kernel for the Z2-packed excitation action trajectory.

One persistent CUDA kernel integrates the whole adaptive DOP853 trajectory of a
packed.PackedActionSystem on the device: step control, the twelve stages, the
error norm and the seventh-order dense output at the quadrature nodes, with a
single launch and no host synchronization per step. It follows the design of the
GPU optimizer's persistent trajectory kernel (rcmps_gpu.optimizer.runtime:
persistent_backend, persistent_rows_cached, persistent_trajectory_backend), whose
controller arithmetic, source approximation, thread-to-tile mapping and native
FP64 tensor-core products (mma.sync m8n8k4) are reproduced here, but the
right-hand side is table driven: every packed block has a list of d x d products
`left @ right` in four groups (packed.py), so the same kernel serves both parity
sectors, any momentum and both gauges.

Layout: one thread block per (packed block, group of eight rows), one warp per
eight-column tile. Constant left operands of a block are cached as 8 x d row
panels in shared memory; the two two-sided intermediates (X R' with X a sum of
products) stay in shared memory between the two product passes of a stage.

Importing this module does not import CuPy. `kernel_source` and `kernel_tables`
are pure Python/NumPy and are tested on the CPU; `compile_ptx` checks the source
with NVRTC alone (no GPU); `FusedTrajectory` needs CuPy and an SM80/SM90 device
(or any cooperative-launch device with use_mma=False, a scalar tile fallback
meant for debugging).
"""
from __future__ import annotations

import ctypes
import importlib.util
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as dop

from ..optimizer.runtime import jax_source
from .states import SolverFailure

PANELS = 4          # default cached constant left-operand row panels per thread block
TEMPS = 2           # default number of two-sided groups (the action system needs 4 and 2)
STATUS = {0: "success", 1: "max_steps", 2: "step_underflow", 3: "nonfinite_error",
          4: "source_domain", 5: "missing_samples", 6: "device_clock_timeout"}
FLOAT_NAMES = ("t", "h", "next_h", "t_new", "last_error", "min_h", "max_h")
INT_NAMES = ("status", "attempts", "accepted", "rejected", "was_rejected", "accept_now",
             "next_point", "dense_steps", "dense_nodes_max")


def source_table():
    """The optimizer's validated piecewise Chebyshev fit of 2|t| J(t^2) (|t| <= sqrt(40))."""
    return jax_source


def geometry(d, n_blocks, panels=PANELS, temps=TEMPS):
    warps = (d + 7) // 8
    return dict(d=d, n_blocks=n_blocks, panels=panels, temps=temps, warps=warps, threads=32 * warps,
                blocks=n_blocks * warps, shared_bytes=(panels + temps) * 8 * d * 16 + 2 * warps * 8,
                total=n_blocks * d * d)


def kernel_tables(tables):
    """Flat device tables from a packed system's tables(): eight int32 per term (group,
    left kind, left index, left adjoint, left panel slot, right kind, right index, right
    adjoint), the group offsets per block (groups: 0 jac, 1 source, 2.. two-sided), and
    the per-block arrays. The numbers of panels and of two-sided groups are the widths of
    `left_constants` and `temp_right`."""
    first, group = np.asarray(tables["term_first"]), np.asarray(tables["term_group"])
    n_blocks, n_terms = len(first) - 1, len(group)
    left_constants = np.asarray(tables["left_constants"], dtype=np.int32).reshape(n_blocks, -1)
    temp_right = np.asarray(tables["temp_right"], dtype=np.int32).reshape(n_blocks, -1)
    groups = 2 + temp_right.shape[1]
    if n_terms and group.max() >= groups:
        raise ValueError("a term refers to a two-sided group without a right factor column")
    terms = np.zeros((n_terms, 8), dtype=np.int32)
    group_first = np.zeros((n_blocks, groups + 1), dtype=np.int32)
    for b in range(n_blocks):
        block = slice(first[b], first[b + 1])
        if np.any(np.diff(group[block]) < 0):
            raise ValueError("terms of a block must be sorted by group")
        group_first[b] = first[b] + np.searchsorted(group[block], np.arange(groups + 1))
        for t in range(first[b], first[b + 1]):
            slot = 0
            if tables["term_left_kind"][t] == 0:
                where = np.flatnonzero(left_constants[b] == tables["term_left_index"][t])
                if len(where) != 1:
                    raise ValueError("left constant of a term is missing from left_constants")
                slot = int(where[0])
            if tables["term_left_kind"][t] + tables["term_right_kind"][t] != 1:
                raise ValueError("every term needs exactly one state operand")
            terms[t] = (group[t], tables["term_left_kind"][t], tables["term_left_index"][t],
                        tables["term_left_adjoint"][t], slot, tables["term_right_kind"][t],
                        tables["term_right_index"][t], tables["term_right_adjoint"][t])
    return dict(terms=terms, group_first=group_first, left_constants=left_constants, temp_right=temp_right,
                coefficient=np.asarray(tables["coefficient"], dtype=np.float64),
                momentum=np.asarray(tables["momentum"], dtype=np.float64).reshape(n_blocks, 2),
                fixed=np.asarray(tables["fixed"], dtype=np.int32))


def emulate_rhs(device_tables, constants, y, jac, source):
    """NumPy evaluation of the right-hand side exactly as the kernel reads its tables."""
    t = device_tables
    temps = t["temp_right"].shape[1]
    out = np.empty_like(y)
    for b in range(len(y)):
        acc = [np.zeros_like(y[0]) for _ in range(2 + temps)]
        for g in range(2 + temps):
            for q in t["terms"][t["group_first"][b, g]:t["group_first"][b, g + 1]]:
                _, lk, li, la, ls, rk, ri, ra = (int(v) for v in q)
                left = constants[t["left_constants"][b, ls]] if lk == 0 else (y[li].conj().T if la else y[li])
                right = constants[ri] if rk == 0 else (y[ri].conj().T if ra else y[ri])
                acc[g] += left @ right
        mu = complex(*t["momentum"][b])
        value = jac * (acc[0] + mu * y[b])
        fixed = constants[t["fixed"][b]] if t["fixed"][b] >= 0 else 0.0
        value += source * (t["coefficient"][b] * acc[1] + fixed)
        for g in range(temps):
            if t["temp_right"][b, g] >= 0:
                value += jac * (acc[2 + g] @ constants[t["temp_right"][b, g]])
        out[b] = value
    return out


CUDA = r'''
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
#define DD (PACKED_D*PACKED_D)
#define TOTAL (NBLOCKS*DD)
#define PANEL (8*PACKED_D)
#define INF64 __longlong_as_double(0x7ff0000000000000LL)
enum { T=0,H=1,NEXT_H=2,T_NEW=3,ERROR=4,MIN_H=5,MAX_H=6 };
enum { STATUS=0,ATTEMPTS=1,ACCEPTED=2,REJECTED=3,WAS_REJECTED=4,
       ACCEPT_NOW=5,NEXT_POINT=6,DENSE_STEPS=7,DENSE_NODES_MAX=8 };
__device__ __forceinline__ double2 zeroz(){ return make_double2(0.,0.); }
__device__ __forceinline__ double2 addz(double2 x,double2 y){return make_double2(x.x+y.x,x.y+y.y);}
__device__ __forceinline__ double2 subz(double2 x,double2 y){return make_double2(x.x-y.x,x.y-y.y);}
__device__ __forceinline__ double2 scalez(double a,double2 x){return make_double2(a*x.x,a*x.y);}
#if USE_MMA
// Native FP64 tensor-core tile product (PTX m8n8k4): lane l holds A[l/4][l%4],
// B[l%4][l/4] and the two accumulators C[l/4][2*(l%4)], C[l/4][2*(l%4)+1].
__device__ __forceinline__ void mma(double a,double b,double &c0,double &c1){
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(c0),"+d"(c1):"d"(a),"d"(b));
}
#else
// Scalar tile product with the same fragment layout (any cooperative-launch GPU).
__device__ __forceinline__ void mma(double a,double b,double &c0,double &c1){
    const int lane=threadIdx.x&31,rl=lane>>2,cl=2*(lane&3);
    #pragma unroll
    for(int q=0;q<4;++q){
        double av=__shfl_sync(0xffffffffu,a,(rl<<2)|q);
        double b0=__shfl_sync(0xffffffffu,b,(cl<<2)|q),b1=__shfl_sync(0xffffffffu,b,((cl+1)<<2)|q);
        c0+=av*b0;c1+=av*b1;
    }
}
#endif
__device__ __forceinline__ void zmm(double2 a,double2 b,double &r0,double &r1,double &i0,double &i1){
    mma(a.x,b.x,r0,r1); mma(-a.y,b.y,r0,r1);
    mma(a.x,b.y,i0,i1); mma(a.y,b.x,i0,i1);
}
__device__ double warp_sum(double v){
    for(int shift=16;shift;shift/=2)v+=__shfl_down_sync(0xffffffffu,v,shift); return v;
}
// Element (row,col) of a stored state block, or of its conjugate transpose.
__device__ __forceinline__ double2 state_element(const double2* stage,int index,int adjoint,int row,int col){
    if(row>=PACKED_D||col>=PACKED_D)return zeroz();
    double2 z=adjoint?stage[index*DD+col*PACKED_D+row]:stage[index*DD+row*PACKED_D+col];
    if(adjoint)z.y=-z.y;
    return z;
}
// The scalar controller of the validated optimizer kernel: explicit round-to-nearest
// operations keep its decisions independent of floating-point contraction.
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
    const double* points,long long npoints,long long* indices,
    const double* c,const double* source_coefficients,double* params,double endpoint){
    if(counts[STATUS])return;
    double e5=norms[0],e3=norms[1],h=state[H],t=state[T];
    double denominator=add64(e5,mul64(.01,e3));
    double error=denominator>0.?mul64(h,e5)/sqrt(mul64(denominator,(double)TOTAL)):0.;
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
__device__ void save_dense(int element,const double2* y,const double2* proposed,const double2* f,
    const double2* k,const double* dense_d,const double* points,const long long* indices,
    double t,double h,double t_new,double2* samples){
    double2 delta=subz(proposed[element],y[element]),coefficients[7];
    coefficients[0]=delta;coefficients[1]=subz(scalez(h,f[element]),delta);
    coefficients[2]=subz(scalez(2.,delta),scalez(h,addz(k[12LL*TOTAL+element],f[element])));
    #pragma unroll
    for(int row=0;row<4;++row){double2 value=zeroz();
        for(int s=0;s<16;++s)value=addz(value,scalez(h*dense_d[row*16+s],k[(long long)s*TOTAL+element]));
        coefficients[3+row]=value;
    }
    for(long long node=indices[0];node<indices[1];++node){
        if(points[node]==t_new)samples[node*TOTAL+element]=proposed[element];
        else {double x=(points[node]-t)/h;double2 value=zeroz();
            #pragma unroll
            for(int s=0;s<7;++s){value=addz(value,coefficients[6-s]);value=scalez(s%2==0?x:1.-x,value);}
            samples[node*TOTAL+element]=addz(value,y[element]);}
    }
}
// out_b = jac (G0 + sum_g Tg C[r_g] + mu Y_b) + source (coefficient G1 + C[fixed]),
// with G0, G1, Tg the sums of left @ right over the terms of each group of block b.
__device__ __forceinline__ void block_rhs(const double2* stage,double2* out,const double2* consts,
    const int* terms,const int* group_first,const int* temp_right,const double* coefficient,
    const double* momentum,const int* fixed,const double2* panels,double2* row_temp,
    double jac,double source,int block,int m,int n,int lane,
    bool valid0,bool valid1,int element0,int element1){
    const int d=PACKED_D,ar=m+(lane>>2),bc=n+(lane>>2),kk=lane&3;
    const int row=m+(lane>>2),col=n+2*(lane&3);
    double acc[NGROUPS][4];
    #pragma unroll
    for(int g=0;g<NGROUPS;++g){
        double r0=0.,r1=0.,i0=0.,i1=0.;
        for(int t=group_first[(NGROUPS+1)*block+g];t<group_first[(NGROUPS+1)*block+g+1];++t){
            const int* q=terms+8*t;
            const int lk=q[1],li=q[2],la=q[3],ls=q[4],rk=q[5],ri=q[6],ra=q[7];
            for(int start=0;start<d;start+=4){
                const int kp=start+kk;
                double2 a,b;
                if(lk==0)a=(ar<d&&kp<d)?panels[ls*PANEL+(ar-m)*d+kp]:zeroz();
                else a=state_element(stage,li,la,ar,kp);
                if(rk==0)b=(kp<d&&bc<d)?consts[ri*DD+kp*d+bc]:zeroz();
                else b=state_element(stage,ri,ra,kp,bc);
                zmm(a,b,r0,r1,i0,i1);
            }
        }
        acc[g][0]=r0;acc[g][1]=r1;acc[g][2]=i0;acc[g][3]=i1;
    }
    const double mr=momentum[2*block],mi=momentum[2*block+1],cf=coefficient[block];
    const int fx=fixed[block];
    double2 y0=valid0?stage[element0]:zeroz(),y1=valid1?stage[element1]:zeroz();
    double2 f0=zeroz(),f1=zeroz();
    if(fx>=0){if(valid0)f0=consts[fx*DD+row*d+col];if(valid1)f1=consts[fx*DD+row*d+col+1];}
    double b0r=jac*(acc[0][0]+mr*y0.x-mi*y0.y)+source*(cf*acc[1][0]+f0.x);
    double b0i=jac*(acc[0][2]+mr*y0.y+mi*y0.x)+source*(cf*acc[1][2]+f0.y);
    double b1r=jac*(acc[0][1]+mr*y1.x-mi*y1.y)+source*(cf*acc[1][1]+f1.x);
    double b1i=jac*(acc[0][3]+mr*y1.y+mi*y1.x)+source*(cf*acc[1][3]+f1.y);
    #pragma unroll
    for(int g=0;g<TEMPS;++g){
        if(valid0)row_temp[g*PANEL+(row-m)*d+col]=make_double2(acc[2+g][0],acc[2+g][2]);
        if(valid1)row_temp[g*PANEL+(row-m)*d+col+1]=make_double2(acc[2+g][1],acc[2+g][3]);
    }
    __syncthreads();
    double t0=0.,t1=0.,ti0=0.,ti1=0.;
    #pragma unroll
    for(int g=0;g<TEMPS;++g){
        const int tr=temp_right[TEMPS*block+g];
        if(tr>=0){
            for(int start=0;start<d;start+=4){
                const int kp=start+kk;
                double2 a=(ar<d&&kp<d)?row_temp[g*PANEL+(ar-m)*d+kp]:zeroz();
                double2 b=(kp<d&&bc<d)?consts[tr*DD+kp*d+bc]:zeroz();
                zmm(a,b,t0,t1,ti0,ti1);
            }
        }
    }
    if(valid0)out[element0]=make_double2(b0r+jac*t0,b0i+jac*ti0);
    if(valid1)out[element1]=make_double2(b1r+jac*t1,b1i+jac*ti1);
}
#define RHS(STAGE,OUT,JAC,SRC) block_rhs(STAGE,OUT,consts,terms,group_first,temp_right,coefficient, \
    momentum,fixed,panels,row_temp,JAC,SRC,block,m,n,lane,valid0,valid1,element0,element1)
extern "C" __global__ void excitation_trajectory(
    const double2* initial,double2* y,double2* f,double2* k,double2* stage,double2* proposed,
    const double2* consts,const int* terms,const int* group_first,const int* temp_right,
    const double* coefficient,const double* momentum,const int* fixed,const int* left_constants,
    const double* rk_a,const double* rk_b,const double* e5,const double* e3,const double* dense_d,
    const double* c,const double* source_coefficients,const double* points,long long npoints,
    double2* samples,double* state,long long* counts,double* params,long long* dense_indices,
    double* partials,double* norms,double atol,double rtol,double endpoint,double initial_h,
    long long max_steps,unsigned long long maximum_ticks){
    cg::grid_group grid=cg::this_grid();
    extern __shared__ double2 shared_block[];
    double2* panels=shared_block;
    double2* row_temp=panels+PANELS*PANEL;
    double* warp_norms=reinterpret_cast<double*>(row_temp+TEMPS*PANEL);
    const int d=PACKED_D,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int block=blockIdx.x/ROW_WARPS,m=(blockIdx.x%ROW_WARPS)*8,n=warp*8;
    const int row=m+(lane>>2),col=n+2*(lane&3);
    const int element0=block*DD+row*d+col,element1=element0+1;
    const bool valid0=row<d&&col<d,valid1=row<d&&col+1<d;
    const bool leader=blockIdx.x==0&&threadIdx.x==0;
    unsigned long long start_clock=leader?clock64():0;
    for(int slot=0;slot<PANELS;++slot){
        const int constant=left_constants[PANELS*block+slot];
        for(int offset=threadIdx.x;offset<PANEL;offset+=blockDim.x){
            const int pr=m+offset/d,pc=offset%d;
            panels[slot*PANEL+offset]=(constant>=0&&pr<d)?consts[constant*DD+pr*d+pc]:zeroz();
        }
    }
    if(valid0)y[element0]=stage[element0]=initial[element0];
    if(valid1)y[element1]=stage[element1]=initial[element1];
    if(leader){for(int i=0;i<7;++i)state[i]=0.;for(int i=0;i<9;++i)counts[i]=0;
        state[T]=-endpoint;state[H]=fmin(initial_h,2.*endpoint);state[MIN_H]=INF64;
        params[1]=2.*endpoint;params[2]=source_fp64(-endpoint,source_coefficients);}
    grid.sync();
    RHS(stage,k,params[1],params[2]);
    if(valid0)f[element0]=k[element0];
    if(valid1)f[element1]=k[element1];
    grid.sync();
    while(true){
        if(leader)prepare_scalar(state,counts,params,c,source_coefficients,endpoint,max_steps,start_clock,maximum_ticks);
        grid.sync();
        if(counts[STATUS])break;
        const double h=params[0];
        if(valid0)k[element0]=f[element0];
        if(valid1)k[element1]=f[element1];
        for(int s=1;s<=12;++s){
            const double* weights=s==12?rk_b:rk_a+s*16;
            const int count=s==12?12:s;
            // Stages alternate between two buffers: thread blocks still reading stage s-1
            // are not disturbed, so the only grid barrier of a stage is the one below.
            double2* buffer=stage+(s&1)*TOTAL;
            #pragma unroll
            for(int slot=0;slot<2;++slot){
                const int element=slot?element1:element0;
                if(slot?valid1:valid0){
                    double2 z=y[element];
                    for(int j=0;j<count;++j){double2 v=k[(long long)j*TOTAL+element];
                        z.x+=h*weights[j]*v.x;z.y+=h*weights[j]*v.y;}
                    buffer[element]=z;if(s==12)proposed[element]=z;
                }
            }
            grid.sync();
            RHS(buffer,k+(long long)s*TOTAL,params[3*s+1],params[3*s+2]);
        }
        double sum5=0.,sum3=0.;
        #pragma unroll
        for(int slot=0;slot<2;++slot){
            const int element=slot?element1:element0;
            if(slot?valid1:valid0){
                double2 z5=zeroz(),z3=zeroz();
                for(int s=0;s<13;++s){double2 z=k[(long long)s*TOTAL+element];
                    z5.x+=e5[s]*z.x;z5.y+=e5[s]*z.y;z3.x+=e3[s]*z.x;z3.y+=e3[s]*z.y;}
                double2 old=y[element],next=proposed[element];
                double scale=atol+rtol*fmax(hypot(old.x,old.y),hypot(next.x,next.y));
                sum5+=(z5.x*z5.x+z5.y*z5.y)/(scale*scale);
                sum3+=(z3.x*z3.x+z3.y*z3.y)/(scale*scale);
            }
        }
        sum5=warp_sum(sum5);sum3=warp_sum(sum3);
        if(lane==0){warp_norms[2*warp]=sum5;warp_norms[2*warp+1]=sum3;}
        __syncthreads();
        if(threadIdx.x==0){double v5=0.,v3=0.;
            for(int w=0;w<ROW_WARPS;++w){v5+=warp_norms[2*w];v3+=warp_norms[2*w+1];}
            partials[2*blockIdx.x]=v5;partials[2*blockIdx.x+1]=v3;}
        grid.sync();
        if(leader){double s5=0.,s3=0.;
            for(int b=0;b<gridDim.x;++b){s5+=partials[2*b];s3+=partials[2*b+1];}
            norms[0]=s5;norms[1]=s3;
            control_scalar(state,counts,norms,points,npoints,dense_indices,c,source_coefficients,params,endpoint);}
        grid.sync();
        if(counts[STATUS])break;
        if(counts[ACCEPT_NOW]){
            if(dense_indices[1]>dense_indices[0]){
                for(int s=13;s<16;++s){
                    const double* weights=rk_a+s*16;
                    double2* buffer=stage+(s&1)*TOTAL;
                    #pragma unroll
                    for(int slot=0;slot<2;++slot){
                        const int element=slot?element1:element0;
                        if(slot?valid1:valid0){
                            double2 z=y[element];
                            for(int j=0;j<s;++j){double2 v=k[(long long)j*TOTAL+element];
                                z.x+=h*weights[j]*v.x;z.y+=h*weights[j]*v.y;}
                            buffer[element]=z;
                        }
                    }
                    grid.sync();
                    RHS(buffer,k+(long long)s*TOTAL,params[3*s+1],params[3*s+2]);
                }
                if(valid0)save_dense(element0,y,proposed,f,k,dense_d,points,dense_indices,state[T],h,state[T_NEW],samples);
                if(valid1)save_dense(element1,y,proposed,f,k,dense_d,points,dense_indices,state[T],h,state[T_NEW],samples);
            }
            if(valid0){y[element0]=proposed[element0];f[element0]=k[12LL*TOTAL+element0];}
            if(valid1){y[element1]=proposed[element1];f[element1]=k[12LL*TOTAL+element1];}
        }
        // Other thread blocks read T and T_NEW during interpolation: commit first.
        grid.sync();
        if(leader){if(counts[ACCEPT_NOW]){state[T]=state[T_NEW];counts[NEXT_POINT]=dense_indices[1];}
            state[H]=state[NEXT_H];
            if(state[T]>=endpoint&&counts[NEXT_POINT]!=npoints)counts[STATUS]=5;
            if(state[T]<endpoint&&counts[ATTEMPTS]>=max_steps)counts[STATUS]=1;}
        grid.sync();
        if(counts[STATUS]||state[T]>=endpoint)break;
    }
}
'''


def kernel_source(d, n_blocks, *, use_mma=True, panels=PANELS, temps=TEMPS):
    """CUDA source specialized by the packed block size d, the number of packed blocks, and
    the numbers of cached left panels and of two-sided groups of the system."""
    source = source_table()
    info = geometry(d, n_blocks, panels, temps)
    return (f"#define PACKED_D {d}\n#define NBLOCKS {n_blocks}\n#define ROW_WARPS {info['warps']}\n"
            f"#define PANELS {panels}\n#define TEMPS {temps}\n#define NGROUPS {2 + temps}\n"
            f"#define USE_MMA {int(bool(use_mma))}\n"
            f"#define SOURCE_MAXIMUM {source.MAXIMUM:.17g}\n#define SOURCE_SEGMENTS {source.SEGMENTS}\n"
            f"#define SOURCE_DEGREE {source.DEGREE}\n") + CUDA


REGISTER_CAPS = (None, 160, 120, 96, 64)


def compile_options(cap=None):
    """A register cap trades spills for resident thread blocks (the optimizer kernel caps at 64
    for d > 64); FusedTrajectory takes the loosest cap whose whole grid is resident."""
    return ("--std=c++17",) if cap is None else ("--std=c++17", f"--maxrregcount={cap}")


def compile_ptx(d, n_blocks, *, use_mma=True, arch="compute_80", include_dirs=(), panels=PANELS, temps=TEMPS):
    """Compile the kernel with NVRTC alone (no GPU, no CuPy): returns (ptx, log).
    Needs libnvrtc and the CUDA runtime headers, e.g. from the nvidia-cuda-nvrtc-cu12 and
    nvidia-cuda-runtime-cu12 wheels, and a directory holding the libcudacxx header
    `nv/target` in `include_dirs`. Raises RuntimeError with the compiler log on failure."""
    roots = []
    for name in ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"{name} is not installed")
        roots.append(Path(list(spec.submodule_search_locations)[0]))
    library = sorted((roots[0] / "lib").glob("libnvrtc.so*"))
    if not library:
        raise RuntimeError("libnvrtc was not found")
    nvrtc = ctypes.CDLL(str(library[0]), mode=ctypes.RTLD_GLOBAL)
    program = ctypes.c_void_p()
    text = kernel_source(d, n_blocks, use_mma=use_mma, panels=panels, temps=temps).encode()

    def check(code, action):
        if code:
            nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"NVRTC {action} failed: {nvrtc.nvrtcGetErrorString(code).decode()}")

    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), text, b"excitation_trajectory.cu", 0, None, None),
          "create")
    options = [o.encode() for o in compile_options() + (
        "--device-as-default-execution-space", f"-arch={arch}", f"-I{roots[1] / 'include'}")
        + tuple(f"-I{path}" for path in include_dirs)]
    array = (ctypes.c_char_p * len(options))(*options)
    code = nvrtc.nvrtcCompileProgram(program, len(options), array)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if code:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("NVRTC compilation failed:\n" + log.value.decode())
    check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "PTX size")
    ptx = ctypes.create_string_buffer(size.value)
    check(nvrtc.nvrtcGetPTX(program, ptx), "PTX")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return ptx.value.decode(), log.value.decode()


def audit_ptx(ptx, *, use_mma=True):
    """FP64 tensor-core instructions present and no FP32 arithmetic (optimizer convention)."""
    mma = sorted(set(re.findall(r"mma\.sync[^\s;]*\.f64\.f64\.f64\.f64", ptx)))
    fp32 = sorted(set(re.findall(r"\b(?:add|sub|mul|fma|div|mad|sqrt|rcp|ex2|lg2)\.(?:[a-z0-9]+\.)*f32\b", ptx)))
    if fp32 or (use_mma and not mma):
        raise RuntimeError(f"FP64 PTX audit failed: mma={mma}, fp32={fp32}")
    return dict(f64_mma_instructions=mma, fp32_arithmetic_instructions=fp32)


class FusedTrajectory:
    """Compiled trajectory kernel for one (d, number of packed blocks, panels, two-sided
    groups) with its buffers. One cooperative launch per trajectory; the whole grid must
    be resident."""
    def __init__(self, cp, d, n_blocks, *, panels=PANELS, temps=TEMPS, use_mma=True,
                 max_device_seconds=600.0, initial_step=0.01):
        self.cp, self.info, self.use_mma = cp, geometry(d, n_blocks, panels, temps), bool(use_mma)
        self.max_device_seconds, self.initial_step = float(max_device_seconds), float(initial_step)
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        capability = (int(props["major"]), int(props["minor"]))
        if not props.get("cooperativeLaunch", False):
            raise ValueError("the trajectory kernel requires cooperative launch")
        if self.use_mma and capability not in ((8, 0), (9, 0)):
            raise ValueError("native FP64 MMA requires an SM80 or SM90 device; use use_mma=False to debug")
        started = time.perf_counter()
        shared = self.info["shared_bytes"]
        if shared > int(props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"])):
            raise RuntimeError(f"kernel shared memory {shared} exceeds the device limit")
        if self.info["threads"] > int(props["maxThreadsPerBlock"]):
            raise RuntimeError("packed dimension exceeds the thread-block limit")
        source = kernel_source(d, n_blocks, use_mma=use_mma, panels=panels, temps=temps)
        tried = []
        for cap in REGISTER_CAPS:
            self.options = compile_options(cap)
            self.kernel = cp.RawKernel(source, "excitation_trajectory", options=self.options,
                                       enable_cooperative_groups=True)
            self.kernel.compile()
            if shared > int(props["sharedMemPerBlock"]):
                self.kernel.max_dynamic_shared_size_bytes = shared
                if int(self.kernel.max_dynamic_shared_size_bytes) < shared:
                    raise RuntimeError("dynamic shared-memory opt-in did not take effect")
            resident = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                self.kernel.kernel.ptr, self.info["threads"], shared))
            capacity = resident * int(props["multiProcessorCount"])
            tried.append((cap, int(self.kernel.num_regs), capacity))
            if self.info["blocks"] <= capacity:
                break
        else:
            raise RuntimeError(f"unsafe cooperative grid: {self.info['blocks']} blocks are not resident "
                               f"for any register cap (cap, registers, capacity): {tried}")
        self.maximum_ticks = int(math.ceil(self.max_device_seconds * 1000 * int(props["clockRate"])))
        self.occupancy = dict(self.info, active_blocks_per_sm=resident, resident_capacity=capacity,
                              multiprocessors=int(props["multiProcessorCount"]),
                              registers=int(self.kernel.num_regs), local_bytes=int(self.kernel.local_size_bytes),
                              compute_capability=capability, compiler_options=list(self.options),
                              register_caps_tried=tried,
                              matrix_arithmetic=("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
                                                 if self.use_mma else "scalar warp-shuffle tiles"),
                              fast_math=False)
        source = source_table()
        self.source_maximum = source.MAXIMUM
        self.coefficients = cp.asarray(source.COEFFICIENTS, dtype=cp.float64)
        self.rk = [cp.asarray(x, dtype=cp.float64) for x in (dop.A, dop.B, dop.E5, dop.E3, dop.D, dop.C)]
        total = self.info["total"]
        self.y, self.f, self.proposed = (cp.empty(total, dtype=cp.complex128) for _ in range(3))
        self.stage = cp.empty(2 * total, dtype=cp.complex128)   # two alternating stage buffers
        self.k = cp.empty(16 * total, dtype=cp.complex128)
        self.state, self.counts = cp.empty(7, dtype=cp.float64), cp.empty(9, dtype=cp.int64)
        self.params, self.dense_indices = cp.empty(48, dtype=cp.float64), cp.empty(2, dtype=cp.int64)
        self.partials, self.norms = cp.empty(2 * self.info["blocks"], dtype=cp.float64), cp.empty(2, dtype=cp.float64)
        self.samples = None
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds = time.perf_counter() - started
        self.last = None

    def integrate(self, system, points, endpoint, integrator):
        """Samples (len(points) + 1, n_blocks, d, d) of `system` at `points` and at +endpoint."""
        cp, info = self.cp, self.info
        d, n_blocks, total = info["d"], info["n_blocks"], info["total"]
        if system.d != d:
            raise ValueError("system does not match the compiled kernel")
        endpoint = float(endpoint)
        if not 0 < endpoint <= self.source_maximum:
            raise ValueError("the device source approximation requires max_point <= 40")
        rtol = float(integrator.rtol)
        if not (integrator.atol > 0 and rtol >= 100 * np.finfo(float).eps):
            raise ValueError("invalid ODE tolerances for the device controller")
        save = np.r_[np.asarray(points, dtype=np.float64), endpoint]
        if np.any(np.diff(save) <= 0) or save[0] <= -endpoint:
            raise ValueError("points must increase strictly inside (-endpoint, endpoint)")
        tables = kernel_tables(system.tables())
        if (len(tables["fixed"]), tables["left_constants"].shape[1], tables["temp_right"].shape[1]) != (
                n_blocks, info["panels"], info["temps"]):
            raise ValueError("system tables do not match the compiled kernel")
        device = {name: cp.asarray(np.ascontiguousarray(value)) for name, value in tables.items()}
        constants = cp.ascontiguousarray(cp.asarray(system.constants, dtype=cp.complex128))
        initial = cp.ascontiguousarray(cp.asarray(system.initial, dtype=cp.complex128)).reshape(-1)
        if self.samples is None or self.samples.shape != (len(save), total):
            self.samples = cp.empty((len(save), total), dtype=cp.complex128)
        nodes = cp.asarray(save)
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        started = time.perf_counter()
        begin.record()
        self.kernel((info["blocks"],), (info["threads"],),
                    (initial, self.y, self.f, self.k, self.stage, self.proposed, constants,
                     device["terms"], device["group_first"], device["temp_right"], device["coefficient"],
                     device["momentum"], device["fixed"], device["left_constants"],
                     self.rk[0], self.rk[1], self.rk[2], self.rk[3], self.rk[4], self.rk[5],
                     self.coefficients, nodes, np.int64(len(save)), self.samples, self.state, self.counts,
                     self.params, self.dense_indices, self.partials, self.norms,
                     np.float64(integrator.atol), np.float64(rtol), np.float64(endpoint),
                     np.float64(self.initial_step), np.int64(integrator.maxiters),
                     np.uint64(self.maximum_ticks)), shared_mem=info["shared_bytes"])
        end.record()
        end.synchronize()
        diagnostics = dict(zip(FLOAT_NAMES, map(float, cp.asnumpy(self.state))))
        diagnostics.update(zip(INT_NAMES, map(int, cp.asnumpy(self.counts))))
        diagnostics.update(status_name=STATUS.get(diagnostics["status"], "unknown"),
                           device_milliseconds=float(cp.cuda.get_elapsed_time(begin, end)),
                           seconds=time.perf_counter() - started,
                           rhs_calls=1 + 12 * diagnostics["attempts"] + 3 * diagnostics["dense_steps"])
        self.last = diagnostics
        if diagnostics["status"] or diagnostics["t"] != endpoint:
            raise SolverFailure(f"excitation action ODE failed on the device: {diagnostics}")
        return self.samples.reshape(len(save), n_blocks, d, d)


def fused_integrator(cp, **options):
    """`integrate(system, points, endpoint, integrator)` for packed systems (e.g.
    potential.PotentialAction with system=packed.PackedActionSystem); kernels are compiled
    once per (d, number of blocks, panels, two-sided groups)."""
    cache = {}

    def integrate(system, points, endpoint, integrator):
        tables = system.tables()
        n_blocks = len(tables["term_first"]) - 1
        panels = np.asarray(tables["left_constants"]).reshape(n_blocks, -1).shape[1]
        temps = np.asarray(tables["temp_right"]).reshape(n_blocks, -1).shape[1]
        key = (system.d, n_blocks, panels, temps)
        if key not in cache:
            cache[key] = FusedTrajectory(cp, system.d, n_blocks, panels=panels, temps=temps, **options)
        return cache[key].integrate(system, points, endpoint, integrator)

    integrate.kernels = cache
    return integrate
