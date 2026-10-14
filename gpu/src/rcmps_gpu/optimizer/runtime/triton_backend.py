"""Isolated FP64 Triton experiment for the existing packed quartic equations.

Requires Triton 3.5.1, CuPy, NumPy/SciPy and a C compiler on Linux. No Torch
dependency: the small driver adapter below directs Triton to CuPy's context and
current stream. This is deliberately an experiment using Triton's driver API.
V100 is excluded; use the existing CUDA/CuPy backend there.

Complex128 arrays stay interleaved in memory. Kernel arithmetic uses explicit
real/imaginary float64 tensors and four native FP64 dots per complex product.
The original packed backend supplies all operator and parity-gather tables.
"""
from __future__ import annotations

from pathlib import Path
import math
import re
import time

import numpy as np
import triton
import triton.language as tl

from ode_backend import (_PackedZ2QuarticRHS, _integrate, _packed_z2_pullback,
                         quadrature, transformed_source)


def use_cupy_driver(cp):
    """Install a CuPy stream/context adapter for Triton's NVIDIA driver.

    Triton's default GPUDriver imports torch and obtains Torch's current stream.
    Using that stream during a CuPy capture would miss the captured graph.
    Only device/stream discovery changes; compilation and launching use Triton's
    unmodified NVIDIA backend. This affects this process's Triton driver only.
    """
    from triton.backends.nvidia.driver import CudaDriver, CudaLauncher, CudaUtils
    from triton.runtime.driver import driver

    class CupyDriver(CudaDriver):
        def __init__(self):
            # Do not call GPUDriver.__init__, which imports torch.
            self.utils = CudaUtils()
            self.launcher_cls = CudaLauncher
            self.get_current_device = cp.cuda.runtime.getDevice
            self.set_current_device = cp.cuda.runtime.setDevice
            self.get_current_stream = lambda device: cp.cuda.get_current_stream().ptr

        def get_device_capability(self, device):
            props = cp.cuda.runtime.getDeviceProperties(device)
            return int(props["major"]), int(props["minor"])

        @staticmethod
        def is_active():
            return cp.cuda.runtime.getDeviceCount() > 0

    active = CupyDriver()
    driver.set_active(active)
    return active


class _Pointer:
    """Triton's documented data_ptr/dtype protocol without a tensor framework."""
    def __init__(self, array, dtype):
        self.array, self.dtype = array, dtype
        self.shape = array.shape

    def data_ptr(self):
        return int(self.array.data.ptr)


@triton.jit
def _load_z(pointer, offsets, mask):
    real = tl.load(pointer + 2 * offsets, mask, other=0).to(tl.float64)
    imag = tl.load(pointer + 2 * offsets + 1, mask, other=0).to(tl.float64)
    return real, imag


@triton.jit
def _load_gathered(pointer, channel, adjoint, rows, columns, D: tl.constexpr):
    offsets = channel * D * D + tl.where(adjoint != 0,
        columns * D + rows, rows * D + columns)
    real, imag = _load_z(pointer, offsets, (rows < D) & (columns < D))
    imag = tl.where(adjoint != 0, -imag, imag)
    return real, imag


@triton.jit
def _complex_dot(ar, ai, br, bi):
    # Do not use the three-product complex identity: its extra cancellation
    # complicates a strict accuracy comparison. All four accumulators are FP64.
    real = tl.dot(ar, br, input_precision="ieee", out_dtype=tl.float64)
    real -= tl.dot(ai, bi, input_precision="ieee", out_dtype=tl.float64)
    imag = tl.dot(ar, bi, input_precision="ieee", out_dtype=tl.float64)
    imag += tl.dot(ai, br, input_precision="ieee", out_dtype=tl.float64)
    return real, imag


@triton.jit
def _first(Y, QL, QR, RL, OL, OR, Fixed, Coefficients, Indices, Adjoints,
           A, B, Jac, Source, Base, Temp, Energy,
           D: tl.constexpr, TK: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    pm, pn, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    jac, source = tl.load(Jac).to(tl.float64), tl.load(Source).to(tl.float64)
    if channel == 10:
        if (pm == 0) & (pn == 0):
            idx = tl.arange(0, D * D)
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


@triton.jit
def _second(Base, Temp, RR, Jac, Out,
            D: tl.constexpr, TK: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    columns = tl.program_id(1) * TN + tl.arange(0, TN)
    channel = tl.program_id(2)
    k = tl.arange(0, TK)
    offset = channel * D * D
    tr, ti = _load_z(Temp, offset + rows[:, None] * D + k[None, :],
        (rows[:, None] < D) & (k[None, :] < D))
    rr, ri = _load_z(RR, offset + k[:, None] * D + columns[None, :],
        (k[:, None] < D) & (columns[None, :] < D))
    jr, ji = _complex_dot(tr, ti, rr, ri)
    elements = offset + rows[:, None] * D + columns[None, :]
    mask = (rows[:, None] < D) & (columns[None, :] < D)
    br, bi = _load_z(Base, elements, mask)
    jac = tl.load(Jac).to(tl.float64)
    tl.store(Out + 2 * elements, br + jac * jr, mask)
    tl.store(Out + 2 * elements + 1, bi + jac * ji, mask)


class TritonPackedQuarticRHS(_PackedZ2QuarticRHS):
    """Drop-in private RHS adapter, compatible with the existing RK graph.

    Buffers and compiled kernels belong to the instance and must remain alive
    with its graph. Inputs/operators are exact complex128 and scalars float64.
    """
    def __init__(self, q, r, rho, cp, *, tile_m=16, tile_n=8, num_warps=4):
        version = tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", triton.__version__).groups())
        if version < (3, 5, 1):
            raise RuntimeError("this experiment requires Triton >=3.5.1 for native fp64 dot")
        if cp.__name__ != "cupy":
            raise ValueError("TritonPackedQuarticRHS requires CuPy")
        if any(x.dtype != cp.complex128 for x in (q, r, rho)):
            raise TypeError("all operators must be complex128")
        major, minor = (int(cp.cuda.runtime.getDeviceProperties(
            cp.cuda.runtime.getDevice())[k]) for k in ("major", "minor"))
        if (major, minor) not in ((8, 0), (9, 0)):
            raise ValueError("this FP64 MMA experiment targets A100 SM80 or H100 SM90; use CUDA fallback")
        if tile_m not in (16, 32) or tile_n not in (8, 16, 32) or num_warps not in (4, 8):
            raise ValueError("supported tiles: M=16/32, N=8/16/32; warps=4/8")
        self.triton_driver = use_cupy_driver(cp)
        super().__init__(q, r, rho, cp, fused_rhs=False)
        if self.d < 2 or self.d & (self.d - 1):
            raise ValueError("Triton experiment requires power-of-two packed dimension >=2")
        self.tile_m, self.tile_n, self.num_warps = tile_m, tile_n, num_warps
        self.triton_base = cp.empty((10, self.d, self.d), dtype=cp.complex128)
        self.triton_temp = cp.empty_like(self.triton_base)
        self.triton_parameters = cp.empty(2, dtype=cp.float64)
        self.triton_jac = self.triton_parameters[:1].reshape(())
        self.triton_source = self.triton_parameters[1:].reshape(())
        self._operator_pointers = {name: _Pointer(cp.ascontiguousarray(getattr(self, name)), tl.float64)
            for name in ("ql", "qright", "rl", "rright", "ol", "oright", "fixed", "a", "b")}
        self._coefficients = _Pointer(self.coefficients, tl.float64)
        self._indices = _Pointer(self.device_indices, tl.int32)
        self._adjoints = _Pointer(self.device_adjoint, tl.int32)
        self._base = _Pointer(self.triton_base, tl.float64)
        self._temp = _Pointer(self.triton_temp, tl.float64)
        self.compiled_first = self.compiled_second = None

    def rhs_into(self, y, out, jac, source):
        p, d = self._operator_pointers, self.d
        grid = (triton.cdiv(d, self.tile_m), triton.cdiv(d, self.tile_n))
        self.compiled_first = _first[grid + (11,)](
            _Pointer(y, tl.float64), p["ql"], p["qright"], p["rl"], p["ol"], p["oright"],
            p["fixed"], self._coefficients, self._indices, self._adjoints, p["a"], p["b"],
            _Pointer(jac, tl.float64), _Pointer(source, tl.float64), self._base, self._temp,
            _Pointer(out[-1:], tl.float64), D=d, TK=max(16, d), TM=self.tile_m, TN=self.tile_n,
            num_warps=self.num_warps, num_stages=1, enable_fp_fusion=True)
        self.compiled_second = _second[grid + (10,)](
            self._base, self._temp, p["rright"], _Pointer(jac, tl.float64),
            _Pointer(out, tl.float64), D=d, TK=max(16, d), TM=self.tile_m, TN=self.tile_n,
            num_warps=self.num_warps, num_stages=1, enable_fp_fusion=True)

    def __call__(self, t, y):
        self.calls += 1
        self.triton_parameters.set(np.asarray([2 * abs(t), transformed_source(t)], dtype=np.float64))
        out = self.xp.empty_like(y)
        self.rhs_into(y, out, self.triton_jac, self.triton_source)
        return out

    def graph_rhs_into(self, s, y, out, work, jac, source, matmul):
        self.rhs_into(y, out, jac, source)

    def ptx_report(self, directory=None, *, require_mma=True):
        if self.compiled_first is None:
            raise RuntimeError("launch the RHS once before inspecting PTX")
        report = {"triton": triton.__version__, "arithmetic": "float64 real/imag",
                  "storage": "complex128", "kernels": {}}
        for name, kernel in (("first", self.compiled_first), ("second", self.compiled_second)):
            ptx = kernel.asm["ptx"]
            mma = sorted(set(re.findall(r"mma\.sync[^\s;]*\.f64\.f64\.f64\.f64", ptx)))
            fp32 = re.findall(r"\b(?:add|sub|mul|fma|div|mad|sqrt|rcp|ex2|lg2)\.(?:[a-z0-9]+\.)*f32\b", ptx)
            report["kernels"][name] = {"f64_mma_instructions": mma,
                "fp32_arithmetic_instructions": sorted(set(fp32)),
                "registers": int(kernel.n_regs), "spills": int(kernel.n_spills),
                "shared_bytes": int(kernel.metadata.shared)}
            if directory:
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"quartic_triton_{name}.ptx").write_text(ptx)
                (directory / f"quartic_triton_{name}.ttir").write_text(kernel.asm["ttir"])
            if fp32:
                raise RuntimeError(f"unexpected FP32 arithmetic in {name}: {sorted(set(fp32))}")
            if require_mma and not mma:
                raise RuntimeError(f"{name} did not lower to native FP64 MMA; inspect its PTX")
        return report


def evaluate_quartic_triton(R, Q, rho, g=1.0, *, xp, half_window=30.0,
                           n_quad=512, atol=1e-13, rtol=1e-12,
                           initial_step=None, max_steps=1_000_000,
                           use_graph=True, use_dense_graph=False, progress=None,
                           tile_m=16, tile_n=8, num_warps=4):
    """Same integration/pullback as evaluate_quartic, changing only its RHS."""
    r, q, rho = (xp.asarray(x, dtype=xp.complex128) for x in (R, Q, rho))
    points, weights = quadrature(half_window, n_quad)
    save_points = np.r_[points, math.sqrt(half_window)]
    xp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    rhs = TritonPackedQuarticRHS(q, r, rho, xp,
        tile_m=tile_m, tile_n=tile_n, num_warps=num_warps)
    samples, stats = _integrate(rhs, save_points, atol=atol, rtol=rtol,
        initial_step=initial_step, max_steps=max_steps,
        use_graph=use_graph, use_dense_graph=use_dense_graph, progress=progress)
    xp.cuda.get_current_stream().synchronize()
    propagated = time.perf_counter()
    ((value, covector, environment),) = _packed_z2_pullback(q, r, rhs, samples, points, weights)
    covector, environment = float(g) * covector, float(g) * environment
    xp.cuda.get_current_stream().synchronize()
    finished = time.perf_counter()
    stats.update(propagation_seconds=propagated - started,
        contraction_seconds=finished - propagated, total_seconds=finished - started,
        backend="triton", dtype="complex128", half_window=half_window,
        n_quad=n_quad, atol=atol, rtol=rtol, packed_z2=True,
        matrix_dimension=rhs.d, tile_m=tile_m, tile_n=tile_n,
        num_warps=num_warps, ptx=rhs.ptx_report())
    return dict(value=float(g) * value, covector=covector,
        environment=environment, stats=stats, component_values=[value])
