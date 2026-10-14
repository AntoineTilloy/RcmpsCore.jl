"""Isolated FP64 row-panel cache candidate; validated row source is unchanged.

Each cooperative CTA preloads its constant ql, rl and ol 8-by-d complex panels
into shared memory once, then reuses them in all twelve new DOP853 stages.
All MMA products, gathering, adaptive control and dense output are unchanged.
"""
from __future__ import annotations
import math
import time
import numpy as np
from .persistent_rows import PersistentRowsAttempt, PersistentRowsDOP853, row_source
from .device_control_backend import DeviceAdaptiveDOP853
from .ode_backend import _packed_z2_pullback, quadrature


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


class PersistentRowsCachedAttempt(PersistentRowsAttempt):
    def __init__(self, rhs, atol, rtol):
        super().__init__(rhs, atol, rtol)
        cp = self.cp
        self.kernel = cp.RawKernel(cached_row_source(), "persistent_rows_cached_attempt",
                                   options=("--std=c++17",), enable_cooperative_groups=True)
        self.kernel.compile()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        resident = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
            self.kernel.kernel.ptr, 128, 0)
        self.capacity = int(resident) * int(props["multiProcessorCount"])
        if self.blocks > self.capacity:
            raise RuntimeError(f"unsafe cached row grid: {self.blocks}>{self.capacity}")
        self.occupancy.update(active_blocks_per_sm=int(resident), resident_capacity=self.capacity,
                              registers=int(self.kernel.num_regs),
                              local_bytes=int(self.kernel.local_size_bytes),
                              shared_bytes=int(self.kernel.shared_size_bytes),
                              row_cta_barriers_per_attempt=14, cached_constant_panels=("ql", "rl", "ol"))


class PersistentRowsCachedDOP853(PersistentRowsDOP853):
    def __init__(self, rhs, points, **options):
        self.persistent = PersistentRowsCachedAttempt(rhs, options.get("atol", 1e-13),
                                                       options.get("rtol", 1e-12))
        DeviceAdaptiveDOP853.__init__(self, rhs, points, **options)

    def run(self):
        samples, stats = super().run()
        stats["persistent_rows_cached"] = True
        return samples, stats


def evaluate_quartic_persistent_rows_cached(R, Q, rho, g=1., *, xp, half_window=30.,
        n_quad=512, atol=1e-13, rtol=1e-12, initial_step=None, max_steps=1_000_000,
        return_samples=False):
    """Standard raw quartic value, covector and boundary environment API."""
    from .triton_backend import TritonPackedQuarticRHS
    start = time.perf_counter()
    r, q, rho = (xp.asarray(x, dtype=xp.complex128) for x in (R, Q, rho))
    points, weights = quadrature(half_window, n_quad)
    rhs = TritonPackedQuarticRHS(q, r, rho, xp)
    solver = PersistentRowsCachedDOP853(rhs, np.r_[points, math.sqrt(half_window)],
        atol=atol, rtol=rtol, initial_step=initial_step, max_steps=max_steps)
    try:
        samples, stats = solver.run()
        begin = time.perf_counter()
        ((value, covector, environment),) = _packed_z2_pullback(q, r, rhs, samples, points, weights)
        covector, environment = float(g) * covector, float(g) * environment
        xp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-begin,
                     total_seconds=time.perf_counter()-start)
        result = dict(value=float(g)*value, energy=float(g)*value, covector=covector,
                      environment=environment, stats=stats)
        if return_samples:
            result["samples"] = samples
        return result
    finally:
        solver.close()
