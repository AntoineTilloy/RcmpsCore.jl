"""Generalized isolated FP64 cached-row RK backend for D32/64/96/128.

The validated D64 evaluator and trial are delegated to unchanged existing code.
Other packed sizes use one warp per eight columns, shared panels sized to d,
and an exact cooperative-residency guard. The non-power-of-two Triton adapter
pads only the trace reduction and k extent; masks retain the same equations.
Importing this module does not initialize CUDA or launch GPU work.
"""
from __future__ import annotations
import math
import re
import time
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
from triton_backend import (TritonPackedQuarticRHS, _Pointer, _load_z, _load_gathered,
                            _complex_dot, _second, use_cupy_driver)
from persistent_backend import PersistentDeviceDOP853
from persistent_rows_cached import (cached_row_source, PersistentRowsCachedAttempt,
                                    evaluate_quartic_persistent_rows_cached)
from device_control_backend import DeviceAdaptiveDOP853
from ode_backend import _PackedZ2QuarticRHS, _packed_z2_pullback, quadrature

SUPPORTED_PACKED = tuple(range(1, 257))
from generic_layout import (packed_geometry, dynamic_shared_bytes,
    compilation_options, configure_shared_memory)


def layout(d):
    return packed_geometry(d)







def _replace(source,old,new):
    if source.count(old)!=1:
        raise RuntimeError(f"general source anchor is not unique: {old!r}")
    return source.replace(old,new,1)


def general_row_source(d):
    info=layout(d)
    if d==32:return cached_row_source()
    source=_replace(cached_row_source(),"void persistent_rows_cached_attempt(",
                    f"void persistent_rows_general_{d}(")
    if source.count("[8*32]")!=4:raise RuntimeError("unexpected shared-panel definitions")
    source=source.replace("[8*32]",f"[8*{d}]")
    source=_replace(source,"__shared__ double warp_norms[8];",
                    f"__shared__ double warp_norms[{2*info['warps']}];")
    source=_replace(source,"for(int w=0;w<4;++w)",f"for(int w=0;w<{info['warps']};++w)")
    if dynamic_shared_bytes(d):
        declaration=(f"    __shared__ double2 row_temp[8*{d}];\n"
            f"    __shared__ double2 cached_ql[8*{d}],cached_rl[8*{d}],cached_ol[8*{d}];\n"
            f"    __shared__ double warp_norms[{2*info['warps']}];")
        source=_replace(source,declaration,'    extern __shared__ double2 dynamic_panels[];\n    double2* row_temp=dynamic_panels;\n    double2* cached_ql=row_temp+8*d;\n    double2* cached_rl=cached_ql+8*d;\n    double2* cached_ol=cached_rl+8*d;\n    double* warp_norms=reinterpret_cast<double*>(cached_ol+8*d);')
    return source


# The matrix equations below are copied exactly from triton_backend._first.
# Only a masked power-of-two extent is added to its scalar trace reduction.
@triton.jit
def _first_general(Y, QL, QR, RL, OL, OR, Fixed, Coefficients, Indices, Adjoints,
           A, B, Jac, Source, Base, Temp, Energy,
           D: tl.constexpr, TK: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, TRACE: tl.constexpr):
    pm, pn, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    jac, source = tl.load(Jac).to(tl.float64), tl.load(Source).to(tl.float64)
    if channel == 10:
        if (pm == 0) & (pn == 0):
            idx = tl.arange(0, TRACE)
            rows, columns = idx // D, idx % D
            ar, ai = _load_z(A, idx, idx < D * D)
            br, bi = _load_z(B, idx, idx < D * D)
            fr, fi = _load_z(Y, 3 * D * D + idx, idx < D * D)
            ftr, fti = _load_z(Y, 3 * D * D + columns * D + rows, idx < D * D)
            trace = tl.sum(ar * fr + ai * fi + br * ftr - bi * fti, 0)
            tl.store(Energy, 8 * source * trace)
            tl.store(Energy + 1, tl.full((), 0, tl.float64))
    else:
        rows = pm * TM + tl.arange(0, TM)
        columns = pn * TN + tl.arange(0, TN)
        k = tl.arange(0, TK)
        offset = channel * D * D
        row_k = rows[:, None] * D + k[None, :]
        k_col = k[:, None] * D + columns[None, :]
        mask_left = (rows[:, None] < D) & (k[None, :] < D)
        mask_right = (k[:, None] < D) & (columns[None, :] < D)

        qlr, qli = _load_z(QL, offset + row_k, mask_left)
        yr, yi = _load_z(Y, offset + k_col, mask_right)
        lr, li = _complex_dot(qlr, qli, yr, yi)
        yr, yi = _load_z(Y, offset + row_k, mask_left)
        qrr, qri = _load_z(QR, offset + k_col, mask_right)
        rr, ri = _complex_dot(yr, yi, qrr, qri)

        ji = tl.load(Indices + channel)
        ja = tl.load(Adjoints + channel)
        gjr, gji = _load_gathered(Y, ji, ja, k[:, None], columns[None, :], D)
        rlr, rli = _load_z(RL, offset + row_k, mask_left)
        jr, j_im = _complex_dot(rlr, rli, gjr, gji)

        li_channel = tl.load(Indices + 10 + channel)
        la = tl.load(Adjoints + 10 + channel)
        glr, gli = _load_gathered(Y, li_channel, la, k[:, None], columns[None, :], D)
        olr, oli = _load_z(OL, offset + row_k, mask_left)
        ilr, ili = _complex_dot(olr, oli, glr, gli)
        ri_channel = tl.load(Indices + 20 + channel)
        ra = tl.load(Adjoints + 20 + channel)
        grr, gri = _load_gathered(Y, ri_channel, ra, rows[:, None], k[None, :], D)
        orr, ori = _load_z(OR, offset + k_col, mask_right)
        irr, iri = _complex_dot(grr, gri, orr, ori)

        elements = offset + rows[:, None] * D + columns[None, :]
        mask = (rows[:, None] < D) & (columns[None, :] < D)
        fixed_r, fixed_i = _load_z(Fixed, elements, mask)
        coefficient = tl.load(Coefficients + channel).to(tl.float64)
        base_r = jac * (lr + rr) + source * (coefficient * (ilr + irr) + fixed_r)
        base_i = jac * (li + ri) + source * (coefficient * (ili + iri) + fixed_i)
        tl.store(Base + 2 * elements, base_r, mask)
        tl.store(Base + 2 * elements + 1, base_i, mask)
        tl.store(Temp + 2 * elements, jr, mask)
        tl.store(Temp + 2 * elements + 1, j_im, mask)


class GeneralTritonPackedRHS(TritonPackedQuarticRHS):
    """Non-power-of-two extension without editing the validated Triton module."""
    def __init__(self,q,r,rho,cp):
        d=len(q)//2
        if d not in SUPPORTED_PACKED:raise ValueError("unsupported packed dimension")
        version=tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)",triton.__version__).groups())
        if version<(3,5,1):raise RuntimeError("native FP64 dot requires Triton>=3.5.1")
        if cp.__name__!="cupy":raise ValueError("generalized GPU backend requires CuPy")
        if any(x.dtype!=cp.complex128 for x in (q,r,rho)):raise TypeError("operators require complex128")
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        if (int(props['major']),int(props['minor'])) not in ((8,0),(9,0)):
            raise ValueError("FP64 MMA requires A100 SM80 or H100 SM90")
        self.triton_driver=use_cupy_driver(cp)
        _PackedZ2QuarticRHS.__init__(self,q,r,rho,cp,fused_rhs=False)
        self.tile_m,self.tile_n,self.num_warps=16,8,4
        self.triton_base=cp.empty((10,self.d,self.d),dtype=cp.complex128)
        self.triton_temp=cp.empty_like(self.triton_base)
        self.triton_parameters=cp.empty(2,dtype=cp.float64)
        self.triton_jac=self.triton_parameters[:1].reshape(())
        self.triton_source=self.triton_parameters[1:].reshape(())
        self._operator_pointers={name:_Pointer(cp.ascontiguousarray(getattr(self,name)),tl.float64)
            for name in ('ql','qright','rl','rright','ol','oright','fixed','a','b')}
        self._coefficients=_Pointer(self.coefficients,tl.float64)
        self._indices=_Pointer(self.device_indices,tl.int32)
        self._adjoints=_Pointer(self.device_adjoint,tl.int32)
        self._base,self._temp=_Pointer(self.triton_base,tl.float64),_Pointer(self.triton_temp,tl.float64)
        self.compiled_first=self.compiled_second=None

    def rhs_into(self,y,out,jac,source):
        p,d=self._operator_pointers,self.d
        grid=(triton.cdiv(d,self.tile_m),triton.cdiv(d,self.tile_n))
        tk=triton.next_power_of_2(max(16,d))
        self.compiled_first=_first_general[grid+(11,)](
            _Pointer(y,tl.float64),p['ql'],p['qright'],p['rl'],p['ol'],p['oright'],p['fixed'],
            self._coefficients,self._indices,self._adjoints,p['a'],p['b'],
            _Pointer(jac,tl.float64),_Pointer(source,tl.float64),self._base,self._temp,
            _Pointer(out[-1:],tl.float64),D=d,TK=tk,TM=self.tile_m,TN=self.tile_n,
            TRACE=triton.next_power_of_2(d*d),num_warps=self.num_warps,num_stages=1,enable_fp_fusion=True)
        self.compiled_second=_second[grid+(10,)](
            self._base,self._temp,p['rright'],_Pointer(jac,tl.float64),_Pointer(out,tl.float64),
            D=d,TK=tk,TM=self.tile_m,TN=self.tile_n,num_warps=self.num_warps,num_stages=1,enable_fp_fusion=True)


def make_rhs(q,r,rho,cp):
    d=len(q)//2
    layout(d)
    constructor=TritonPackedQuarticRHS if d>=2 and not(d&(d-1)) else GeneralTritonPackedRHS
    return constructor(q,r,rho,cp)


class GeneralCachedAttempt:
    def __new__(cls,rhs,atol,rtol):
        if rhs.d==32:return PersistentRowsCachedAttempt(rhs,atol,rtol)
        return super().__new__(cls)

    def __init__(self,rhs,atol,rtol):
        cp=rhs.xp
        if cp.__name__!='cupy':raise ValueError('generalized persistent trial requires CuPy')
        info=layout(rhs.d)
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        if int(props['major'])<8:raise ValueError('native FP64 MMA requires SM80 or later')
        self.cp,self.rhs,self.atol,self.rtol=cp,rhs,float(atol),float(rtol)
        self.blocks,self.threads=info['blocks'],info['threads']
        self.base,self.temp=cp.empty_like(rhs.initial),cp.empty_like(rhs.initial)
        self.partials=cp.empty(2*self.blocks,dtype=cp.float64)
        self.kernel=cp.RawKernel(general_row_source(rhs.d),f'persistent_rows_general_{rhs.d}',
                                options=compilation_options(rhs.d),enable_cooperative_groups=True)
        self.kernel.compile()
        self.dynamic_shared_bytes=configure_shared_memory(self.kernel,props,info)
        resident=cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.kernel.kernel.ptr,self.threads,self.dynamic_shared_bytes)
        sms=int(props['multiProcessorCount']);self.capacity=int(resident)*sms
        if self.blocks>self.capacity:raise RuntimeError(f'unsafe cooperative grid: {self.blocks}>{self.capacity}')
        self.occupancy=dict(blocks=self.blocks,threads_per_block=self.threads,active_blocks_per_sm=int(resident),
            multiprocessors=sms,resident_capacity=self.capacity,registers=int(self.kernel.num_regs),
            local_bytes=int(self.kernel.local_size_bytes),shared_bytes=int(self.kernel.shared_size_bytes)+self.dynamic_shared_bytes,
            static_shared_bytes=int(self.kernel.shared_size_bytes),dynamic_shared_bytes=self.dynamic_shared_bytes,
            requested_register_cap=64 if rhs.d>64 else None,compiler_options=list(compilation_options(rhs.d)),
            matrix_arithmetic='mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64',
            scalar_arithmetic='float64',storage='complex128',fast_math=False,
            grid_barriers_per_attempt=26,row_cta_barriers_per_attempt=14)
        if self.occupancy['shared_bytes']!=info['shared_bytes']:raise RuntimeError('unexpected shared-memory layout')

    def launch(self,attempt):
        rhs=self.rhs
        self.kernel((self.blocks,1,1),(self.threads,1,1),
            (attempt.y,attempt.f,attempt.stages,attempt.intermediate,attempt.proposed,
             self.base,self.temp,rhs.ql,rhs.qright,rhs.rl,rhs.rright,rhs.ol,rhs.oright,
             rhs.fixed,rhs.coefficients,rhs.device_indices,rhs.device_adjoint,rhs.a,rhs.b,
             attempt.linear.a,attempt.linear.b,attempt.linear.e5,attempt.linear.e3,
             attempt.params,np.float64(self.atol),np.float64(self.rtol),np.int32(rhs.d),self.partials,attempt.norms),
            shared_mem=self.dynamic_shared_bytes)


class GeneralCachedDOP853(PersistentDeviceDOP853):
    def __init__(self,rhs,points,**options):
        self.persistent=GeneralCachedAttempt(rhs,options.get('atol',1e-13),options.get('rtol',1e-12))
        DeviceAdaptiveDOP853.__init__(self,rhs,points,**options)

    def run(self):
        samples,stats=super().run()
        stats['persistent_rows_general']=True
        return samples,stats


def evaluate_quartic_persistent_rows_general(R,Q,rho,g=1.,*,xp,half_window=30.,n_quad=512,
        atol=1e-13,rtol=1e-12,initial_step=None,max_steps=1_000_000,return_samples=False):
    options=dict(xp=xp,half_window=half_window,n_quad=n_quad,atol=atol,rtol=rtol,
                 initial_step=initial_step,max_steps=max_steps,return_samples=return_samples)
    if len(Q)==64:
        return evaluate_quartic_persistent_rows_cached(R,Q,rho,g,**options)
    start=time.perf_counter()
    r,q,rho=(xp.asarray(x,dtype=xp.complex128) for x in (R,Q,rho))
    points,weights=quadrature(half_window,n_quad)
    rhs=make_rhs(q,r,rho,xp)
    solver=GeneralCachedDOP853(rhs,np.r_[points,math.sqrt(half_window)],
                             atol=atol,rtol=rtol,initial_step=initial_step,max_steps=max_steps)
    try:
        samples,stats=solver.run();begin=time.perf_counter()
        ((value,covector,environment),)=_packed_z2_pullback(q,r,rhs,samples,points,weights)
        xp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-begin,total_seconds=time.perf_counter()-start,
                     triton_ptx=rhs.ptx_report())
        result=dict(value=float(g)*value,energy=float(g)*value,covector=float(g)*covector,
                    environment=float(g)*environment,stats=stats)
        if return_samples:result['samples']=samples
        return result
    finally:solver.close()


def cuda_ptx_report(d,cp,directory=None):
    """Compile inspection-only PTX via NVRTC without changing CuPy global flags.

    Actual launched-kernel registers/local/shared and occupancy are reported by
    GeneralCachedAttempt; PTX arithmetic is separately inspected here.
    """
    import importlib.util
    from cupy.cuda import compiler
    from cupy._core import core
    source=general_row_source(d)
    cuda_path=cp.cuda.get_cuda_path()
    header_dirs=[] if cuda_path is None else [Path(cuda_path)/'include']
    # Wheels may provide runtime headers without a system CUDA toolkit.
    package=importlib.util.find_spec('nvidia.cuda_runtime')
    if package and package.submodule_search_locations:
        header_dirs.extend(Path(path)/'include' for path in package.submodule_search_locations)
    includes=tuple('-I'+str(path) for path in header_dirs if path.is_dir())
    includes+=tuple(core._get_cccl_include_options())
    arch=cp.cuda.Device().compute_capability
    program=compiler._NVRTCProgram(source,'general_rows_audit.cu',method='ptx')
    binary,_=program.compile(('--std=c++17','--device-c','--device-as-default-execution-space',
                              '-arch=compute_'+arch)+compilation_options(d)[1:]+includes)
    ptx=binary.decode()
    mma=sorted(set(re.findall(r'mma\.sync[^\s;]*\.f64\.f64\.f64\.f64',ptx)))
    fp32=sorted(set(re.findall(r'\b(?:add|sub|mul|fma|div|mad|sqrt|rcp|ex2|lg2)\.(?:[a-z0-9]+\.)*f32\b',ptx)))
    if not mma or fp32:raise RuntimeError(f'FP64 PTX audit failed: mma={mma},fp32={fp32}')
    if directory:
        target=Path(directory);target.mkdir(parents=True,exist_ok=True)
        (target/f'persistent_rows_general_d{d}.ptx').write_text(ptx)
    return dict(f64_mma_instructions=mma,fp32_arithmetic_instructions=fp32,arch=arch)
