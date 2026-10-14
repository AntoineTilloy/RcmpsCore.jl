"""Isolated cooperative FP64 RK variant: one CTA owns an 8-row group.

Four warps cover all columns of each packed block (D/2<=32). The complex jump
intermediate stays in 4KB shared memory, so its dependency uses __syncthreads.
RK stage and completed-RHS dependencies still use cooperative grid.sync.
The original persistent kernel/backend and all validated files are unchanged.
Only the source rewrite row_source remains; persistent_rows_cached builds on it.
"""
from __future__ import annotations
from .persistent_backend import CUDA


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
