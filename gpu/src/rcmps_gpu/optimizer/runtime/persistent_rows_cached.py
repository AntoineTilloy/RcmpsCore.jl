"""Isolated FP64 row-panel cache candidate; validated row source is unchanged.

Each cooperative CTA preloads its constant ql, rl and ol 8-by-d complex panels
into shared memory once, then reuses them in all twelve new DOP853 stages.
All MMA products, gathering, adaptive control and dense output are unchanged.
Only the source rewrite cached_row_source remains: persistent_trajectory_backend
extracts the RHS and error reduction of its trajectory kernel from it.
"""
from __future__ import annotations
from .persistent_rows import row_source


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"cache source anchor is not unique: {old!r}")
    return source.replace(old, new, 1)


def cached_row_source():
    source = _replace(row_source(), "void persistent_rows_attempt(",
                      "void persistent_rows_cached_attempt(")
    source = _replace(source, "    __shared__ double2 row_temp[8*32];",
                      "    __shared__ double2 row_temp[8*32];\n"
                      "    __shared__ double2 cached_ql[8*32],cached_rl[8*32],cached_ol[8*32];")
    source = _replace(source, "    if(valid0)k[element0]=f[element0];", r'''
    if(!scalar){
        for(int offset=threadIdx.x;offset<8*d;offset+=blockDim.x){
            int cache_row=m+offset/d,cache_col=offset%d;
            cached_ql[offset]=loadz(ql,channel,cache_row,cache_col,d);
            cached_rl[offset]=loadz(rl,channel,cache_row,cache_col,d);
            cached_ol[offset]=loadz(ol,channel,cache_row,cache_col,d);
        }
    }
    __syncthreads();
    if(valid0)k[element0]=f[element0];''')
    for operand in ("ql", "rl", "ol"):
        source = _replace(source, f"loadz({operand},channel,ar,kp,d)",
                          f"(ar<d && kp<d ? cached_{operand}[(ar-m)*d+kp] : zeroz())")
    return source
