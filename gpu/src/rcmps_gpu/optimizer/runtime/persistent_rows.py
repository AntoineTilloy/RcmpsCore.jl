"""Isolated cooperative FP64 RK variant: one CTA owns an 8-row group.

Four warps cover all columns of each packed block (D/2<=32). The complex jump
intermediate stays in 4KB shared memory, so its dependency uses __syncthreads.
RK stage and completed-RHS dependencies still use cooperative grid.sync.
The original persistent kernel/backend and all validated files are unchanged.
"""
from __future__ import annotations
import math
import time
import numpy as np
from .persistent_backend import CUDA,PersistentAttempt,PersistentDeviceDOP853
from .device_control_backend import DeviceAdaptiveDOP853
from .ode_backend import _packed_z2_pullback,quadrature


def _replace(source,old,new):
    if source.count(old)!=1:raise RuntimeError(f"row kernel source anchor is not unique: {old!r}")
    return source.replace(old,new,1)


def row_source():
    source=_replace(CUDA,"void persistent_attempt(","void persistent_rows_attempt(")
    source=_replace(source,
        "    const int lane=threadIdx.x,tiles=(d+7)/8,matrix_blocks=10*tiles*tiles;",
        "    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,tiles=(d+7)/8,matrix_blocks=10*tiles;\n"
        "    __shared__ double2 row_temp[8*32];\n    __shared__ double warp_norms[8];")
    source=_replace(source,
        "    const int channel=blockIdx.x/(tiles*tiles),tile=blockIdx.x%(tiles*tiles);\n"
        "    const int m=tile/tiles*8,n=tile%tiles*8;",
        "    const int channel=blockIdx.x/tiles,row_tile=blockIdx.x%tiles;\n"
        "    const int m=row_tile*8,n=warp*8;")
    source=_replace(source,"if(scalar && lane==0)k[total-1]=f[total-1];",
                     "if(scalar && threadIdx.x==0)k[total-1]=f[total-1];")
    # There are two deliberately identical component-selection sites: stage
    # formation and norm formation. Both must exclude idle scalar-block warps.
    marker="if(scalar){element=total-1;valid=lane==0 && slot==0;}"
    if source.count(marker)!=2:raise RuntimeError("unexpected scalar selection sites")
    source=source.replace(marker,"if(scalar){element=total-1;valid=threadIdx.x==0 && slot==0;}")
    source=_replace(source,"temp[element0]=make_double2(j0,ji0);",
                     "row_temp[(row-m)*d+col]=make_double2(j0,ji0);")
    source=_replace(source,"temp[element1]=make_double2(j1,ji1);",
                     "row_temp[(row-m)*d+col+1]=make_double2(j1,ji1);")
    source=_replace(source,"        }else{\n            double trace=0.;",
                     "        }else if(warp==0){\n            double trace=0.;")
    # Every warp in the CTA reaches this barrier. Its row-local intermediate
    # is complete independently of other channels or row groups.
    source=_replace(source,"        grid.sync();\n        if(!scalar){",
                     "        __syncthreads();\n        if(!scalar){")
    source=_replace(source,"loadz(temp,channel,ar,kp,d)",
                     "(ar<d && kp<d ? row_temp[(ar-m)*d+kp] : zeroz())")
    source=_replace(source,
        "    if(lane==0){partials[2*blockIdx.x]=sum5;partials[2*blockIdx.x+1]=sum3;}\n"
        "    grid.sync();\n    if(blockIdx.x==0 && lane==0)",
        "    if(lane==0){warp_norms[2*warp]=sum5;warp_norms[2*warp+1]=sum3;}\n"
        "    __syncthreads();\n    if(threadIdx.x==0){double v5=0.,v3=0.;\n"
        "        for(int w=0;w<4;++w){v5+=warp_norms[2*w];v3+=warp_norms[2*w+1];}\n"
        "        partials[2*blockIdx.x]=v5;partials[2*blockIdx.x+1]=v3;}\n"
        "    grid.sync();\n    if(blockIdx.x==0 && threadIdx.x==0)")
    return source


class PersistentRowsAttempt(PersistentAttempt):
    def __init__(self,rhs,atol,rtol):
        if rhs.d>32:raise ValueError("shared row experiment requires packed dimension <=32")
        cp=rhs.xp
        if cp.__name__!="cupy":raise ValueError("persistent rows require CuPy")
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        if int(props["major"])<8:raise ValueError("FP64 MMA requires SM80 or later")
        self.cp,self.rhs,self.atol,self.rtol=cp,rhs,float(atol),float(rtol)
        self.blocks=10*((rhs.d+7)//8)+1
        self.base=cp.empty_like(rhs.initial)
        self.temp=cp.empty_like(rhs.initial)  # ABI-compatible unused legacy argument.
        self.partials=cp.empty(2*self.blocks,dtype=cp.float64)
        self.kernel=cp.RawKernel(row_source(),"persistent_rows_attempt",
            options=("--std=c++17",),enable_cooperative_groups=True)
        self.kernel.compile()
        resident=cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.kernel.kernel.ptr,128,0)
        sms=int(props["multiProcessorCount"]);self.capacity=int(resident)*sms
        if self.blocks>self.capacity:
            raise RuntimeError(f"unsafe cooperative row grid: {self.blocks}>{self.capacity}")
        self.occupancy=dict(blocks=self.blocks,threads_per_block=128,
            active_blocks_per_sm=int(resident),multiprocessors=sms,resident_capacity=self.capacity,
            registers=int(self.kernel.num_regs),local_bytes=int(self.kernel.local_size_bytes),
            shared_bytes=int(self.kernel.shared_size_bytes),
            synchronization="cooperative grid.sync plus row-local __syncthreads",
            grid_barriers_per_attempt=26,row_cta_barriers_per_attempt=13,
            matrix_arithmetic="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64",
            scalar_arithmetic="float64",storage="complex128",fast_math=False)

    def launch(self,attempt):
        rhs=self.rhs
        self.kernel((self.blocks,1,1),(128,1,1),
            (attempt.y,attempt.f,attempt.stages,attempt.intermediate,attempt.proposed,
             self.base,self.temp,rhs.ql,rhs.qright,rhs.rl,rhs.rright,rhs.ol,rhs.oright,
             rhs.fixed,rhs.coefficients,rhs.device_indices,rhs.device_adjoint,rhs.a,rhs.b,
             attempt.linear.a,attempt.linear.b,attempt.linear.e5,attempt.linear.e3,
             attempt.params,np.float64(self.atol),np.float64(self.rtol),np.int32(rhs.d),
             self.partials,attempt.norms))


class PersistentRowsDOP853(PersistentDeviceDOP853):
    def __init__(self,rhs,points,**options):
        self.persistent=PersistentRowsAttempt(rhs,options.get("atol",1e-13),options.get("rtol",1e-12))
        # Avoid the parent constructor's creation of the original trial kernel.
        DeviceAdaptiveDOP853.__init__(self,rhs,points,**options)

    def run(self):
        samples,stats=super().run()
        stats["persistent_rows"]=True
        return samples,stats


def evaluate_quartic_persistent_rows(R,Q,rho,g=1.,*,xp,half_window=30.,n_quad=512,
        atol=1e-13,rtol=1e-12,initial_step=None,max_steps=1_000_000,return_samples=False):
    """Same standard raw observable API as the validated persistent backend."""
    from .triton_backend import TritonPackedQuarticRHS
    start=time.perf_counter();r,q,rho=(xp.asarray(x,dtype=xp.complex128) for x in (R,Q,rho))
    points,weights=quadrature(half_window,n_quad);rhs=TritonPackedQuarticRHS(q,r,rho,xp)
    solver=PersistentRowsDOP853(rhs,np.r_[points,math.sqrt(half_window)],
        atol=atol,rtol=rtol,initial_step=initial_step,max_steps=max_steps)
    try:
        samples,stats=solver.run();begin=time.perf_counter()
        ((value,covector,environment),)=_packed_z2_pullback(q,r,rhs,samples,points,weights)
        covector,environment=float(g)*covector,float(g)*environment
        xp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-begin,total_seconds=time.perf_counter()-start)
        result=dict(value=float(g)*value,energy=float(g)*value,covector=covector,environment=environment,stats=stats)
        if return_samples:result["samples"]=samples
        return result
    finally:solver.close()
