"""Independent FP64 NumPy/CuPy RCMPS observable prototype.

The CPU Julia implementation is unchanged.  Public routines return the raw
covector and stationary-state environment; the caller must add the pinned
Poisson response and apply the Riemannian metric.  No automatic differentiation
or differentiation of an adaptive numerical solver is used.

The adaptive order-eight method uses SciPy's DOP853 coefficient tables (SciPy
is BSD licensed), with its order-seven dense interpolant.  The transformed
source and the finite window are identical to src/observables.jl.  All forward
and adjoint channels share an adaptive step.  This deliberately favors simple,
auditable batched matrix operations over a complete production GPU optimizer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dop
from scipy.special import gamma, kv, roots_legendre


def transformed_source(t):
    """2|t| J(t²), including its exact finite value at zero."""
    a = abs(float(t))
    if a == 0:
        return 1 / math.sqrt(math.pi)
    return 2 * a * kv(0.25, a * a) / (
        2**2.25 * math.sqrt(math.pi) * gamma(1.25) * math.sqrt(a)
    )


def quadrature(half_window=20.0, n_quad=192):
    if half_window <= 0 or n_quad < 8:
        raise ValueError("positive half_window and n_quad >= 8 required")
    x, w = roots_legendre(n_quad)
    a = math.sqrt(half_window)
    points = (x + 1) * a / 2
    weights = w * a / 2
    return np.r_[-points[::-1], points], np.r_[weights[::-1], weights]


def _adj(a):
    return a.swapaxes(-2, -1).conj()


def _scalar(a):
    return float(a.item()) if hasattr(a, "item") else float(a)


def _sync(xp):
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


@dataclass(frozen=True)
class _Spec:
    channels: int
    edges: tuple  # (source, destination, left, coefficient), zero based
    commutator: bool = False


_KINETIC = (
    (0, 1, True, math.sqrt(2)),
    (0, 2, False, math.sqrt(2)),
    (2, 3, True, math.sqrt(2)),
    (1, 3, False, math.sqrt(2)),
)
_QUARTIC = _Spec(5, tuple((k, k + 1, left, k + 1) for k in range(4) for left in (True, False)))


class _BatchedRHS:
    def __init__(self, q, r, rho, specs, xp):
        self.xp, self.specs = xp, specs
        self.d, self.calls = q.shape[0], 0
        self.offsets = np.cumsum([0] + [s.channels for s in specs])[:-1].tolist()
        self.half = sum(s.channels for s in specs)
        self.channels = 2 * self.half
        ops = [q @ r - r @ q if s.commutator else r for s in specs]
        forward_ops = xp.stack([op for op, s in zip(ops, specs) for _ in range(s.channels)])
        self.ops = ops
        self.ol = xp.concatenate([forward_ops, _adj(forward_ops)])
        self.oright = _adj(self.ol)
        self.ql = xp.concatenate([
            xp.broadcast_to(q, (self.half, self.d, self.d)),
            xp.broadcast_to(_adj(q), (self.half, self.d, self.d)),
        ])
        self.qright = _adj(self.ql)
        self.rl = xp.concatenate([
            xp.broadcast_to(r, (self.half, self.d, self.d)),
            xp.broadcast_to(_adj(r), (self.half, self.d, self.d)),
        ])
        self.rright = _adj(self.rl)
        self.initial = xp.zeros((self.channels, self.d, self.d), dtype=xp.complex128)
        self.fixed = xp.zeros_like(self.initial)
        identity = xp.eye(self.d, dtype=xp.complex128)
        left_edges, right_edges = [[] for _ in range(self.channels)], [[] for _ in range(self.channels)]
        for offset, s in zip(self.offsets, specs):
            self.initial[offset] = rho
            self.initial[self.half + offset + s.channels - 1] = identity
            for src, dst, left, coefficient in s.edges:
                (left_edges if left else right_edges)[offset + dst].append((offset + src, coefficient))
                # Trace-dual adjoint uses right O for a left insertion.
                (right_edges if left else left_edges)[self.half + offset + src].append(
                    (self.half + offset + dst, coefficient)
                )
        self.edges = (left_edges, right_edges)
        if xp.__name__ == "cupy":
            self._prepare_kernels()

    def _prepare_kernels(self):
        xp = self.xp
        width = max(1, max(len(e) for group in self.edges for e in group))
        indices, coefficients = [], []
        for group in self.edges:
            idx = np.full((self.channels, width), -1, dtype=np.int32)
            coeff = np.zeros((self.channels, width))
            for k, edges in enumerate(group):
                for j, (src, c) in enumerate(edges):
                    idx[k, j], coeff[k, j] = src, c
            indices.append(xp.asarray(idx))
            coefficients.append(xp.asarray(coeff))
        self.edge_width = width
        self.edge_indices, self.edge_coefficients = indices, coefficients
        self._combine_rhs = xp.ElementwiseKernel(
            "complex128 l, complex128 qr, complex128 jump, complex128 fixed, raw complex128 il, raw complex128 ir, "
            "raw int32 li, raw int32 ri, raw float64 lc, raw float64 rc, int64 matrix_size, "
            "int32 width, float64 jac, float64 source",
            "complex128 out",
            """
            long channel = i / matrix_size;
            long element = i % matrix_size;
            complex<double> insertion = 0.;
            for (int e = 0; e < width; ++e) {
                int a = li[channel * width + e];
                int b = ri[channel * width + e];
                if (a >= 0) insertion += lc[channel * width + e] * il[a * matrix_size + element];
                if (b >= 0) insertion += rc[channel * width + e] * ir[b * matrix_size + element];
            }
            out = jac * (l + qr + jump) + source * (insertion + fixed);
            """,
            "rcmps_batched_rhs_combine",
        )

    def __call__(self, t, y):
        xp = self.xp
        self.calls += 1
        left = self.ql @ y
        right = y @ self.qright
        jump = (self.rl @ y) @ self.rright
        il, ir = self.ol @ y, y @ self.oright
        jac, source = 2 * abs(t), transformed_source(t)
        if xp.__name__ == "cupy":
            return self._combine_rhs(left, right, jump, self.fixed, il, ir,
                self.edge_indices[0], self.edge_indices[1],
                self.edge_coefficients[0], self.edge_coefficients[1],
                self.d**2, self.edge_width, jac, source)
        out = jac * (left + right + jump)
        for product, group in zip((il, ir), self.edges):
            for k, edges in enumerate(group):
                for src, coefficient in edges:
                    out[k] += (source * coefficient) * product[src]
        return out


class _ReducedQuarticRHS(_BatchedRHS):
    """Seven live matrix channels and one scalar, with rho and I fixed.

    The terminal forward matrix is unnecessary: its trace is the observable,
    and its bulk covector contractions cancel because its adjoint equals I.
    The scalar accumulator retains the same adaptive integration accuracy.
    """
    reduced_quartic = True

    def __init__(self, q, r, rho, xp):
        self.xp, self.d, self.calls = xp, len(q), 0
        self.channels, self.matrix_size = 7, 7 * self.d**2
        self.initial = xp.zeros(self.matrix_size + 1, dtype=xp.complex128)
        self.ql = xp.concatenate([xp.broadcast_to(q, (3, self.d, self.d)),
                                  xp.broadcast_to(_adj(q), (4, self.d, self.d))])
        self.qright = _adj(self.ql)
        self.rl = xp.concatenate([xp.broadcast_to(r, (3, self.d, self.d)),
                                  xp.broadcast_to(_adj(r), (4, self.d, self.d))])
        self.rright = _adj(self.rl)
        self.ol, self.oright = self.rl, self.rright
        self.fixed = xp.zeros((7, self.d, self.d), dtype=xp.complex128)
        self.fixed[0] = r @ rho + rho @ _adj(r)
        self.fixed[6] = 4 * (r + _adj(r))
        group = [[], [(0, 2.)], [(1, 3.)], [(4, 1.)], [(5, 2.)], [(6, 3.)], []]
        self.edges = (group, group)
        self.q, self.r, self.rho = q, r, rho
        if xp.__name__ == "cupy":
            self._prepare_kernels()
            self.trace_kernel = xp.ElementwiseKernel(
                "raw complex128 il, raw complex128 ir, int32 d, float64 source",
                "complex128 out",
                """
                complex<double> value = 0.;
                for (int j = 0; j < d; ++j) {
                    long index = 2 * d * d + j * d + j;
                    value += il[index] + ir[index];
                }
                out = 4 * source * value;
                """, "rcmps_quartic_trace_accumulator")

    def __call__(self, t, y):
        xp = self.xp
        self.calls += 1
        matrices = y[:self.matrix_size].reshape(7, self.d, self.d)
        left, right = self.ql @ matrices, matrices @ self.qright
        jump = (self.rl @ matrices) @ self.rright
        il, ir = self.ol @ matrices, matrices @ self.oright
        jac, source = 2 * abs(t), transformed_source(t)
        out = xp.empty_like(y)
        matrix_out = out[:self.matrix_size].reshape(7, self.d, self.d)
        if xp.__name__ == "cupy":
            self._combine_rhs(left, right, jump, self.fixed, il, ir,
                self.edge_indices[0], self.edge_indices[1],
                self.edge_coefficients[0], self.edge_coefficients[1],
                self.d**2, self.edge_width, jac, source, matrix_out)
            self.trace_kernel(il, ir, self.d, source, out[-1:])
        else:
            matrix_out[:] = jac * (left + right + jump) + source * self.fixed
            for product, group in zip((il, ir), self.edges):
                for k, edges in enumerate(group):
                    for src, coefficient in edges:
                        matrix_out[k] += source * coefficient * product[src]
            out[-1] = 4 * source * np.trace(il[2] + ir[2])
        return out


class _PackedZ2QuarticRHS:
    """Seven Hermitian D×D channels packed as ten (D/2)² blocks.

    Odd channels store their upper off-diagonal block; its adjoint gives the
    lower block. Even channels store their two diagonal blocks. Structure is
    checked strictly, so this mode never projects an arbitrary state onto Z2.
    """
    packed_z2 = True

    def __init__(self, q, r, rho, xp, fused_rhs=False):
        full_d = len(q)
        if full_d % 2:
            raise ValueError("packed_z2 requires a positive even bond dimension")
        d = full_d // 2
        forbidden = (r[:d, :d], r[d:, d:], q[:d, d:], q[d:, :d], rho[:d, d:], rho[d:, :d])
        if any(bool(xp.any(block != 0).item()) for block in forbidden):
            raise ValueError("packed_z2 requires exact balanced block structure; no projection is performed")
        hermiticity = _scalar(xp.linalg.norm(rho - _adj(rho)))
        if hermiticity > 1e-13 * max(1., _scalar(xp.linalg.norm(rho))):
            raise ValueError("packed_z2 requires a Hermitian initial density")
        self.xp, self.d, self.full_d, self.calls = xp, d, full_d, 0
        self.fused_rhs = bool(fused_rhs)
        self.q, self.r, self.rho = q, r, rho
        self.channels, self.matrix_size = 10, 10 * d*d
        self.initial = xp.zeros(self.matrix_size + 1, dtype=xp.complex128)
        q0, q1 = q[:d, :d], q[d:, d:]
        a, b = xp.ascontiguousarray(r[:d, d:]), xp.ascontiguousarray(r[d:, :d])
        ah, bh, q0h, q1h = _adj(a), _adj(b), _adj(q0), _adj(q1)
        z = xp.zeros_like(a)
        self.a, self.b = a, b
        # F1 odd; F2 even+even; F3 odd; A1 even+even; A2 odd;
        # A3 even+even; A4 odd.
        self.ql = xp.stack([q0, q0, q1, q0, q0h, q1h, q0h, q0h, q1h, q0h])
        self.qright = xp.stack([q1h, q0h, q1h, q1h, q0, q1, q1, q0, q1, q1])
        self.rl = xp.stack([a, a, b, a, bh, ah, bh, bh, ah, bh])
        self.rright = xp.stack([bh, ah, bh, bh, b, a, a, b, a, a])
        self.ol = xp.stack([z, a, b, a, bh, ah, bh, bh, ah, z])
        self.oright = xp.stack([z, ah, bh, bh, b, a, a, b, a, z])
        self.coefficients = xp.asarray([0., 2., 2., 3., 1., 1., 2., 3., 3., 0.])
        self.fixed = xp.zeros((10, d, d), dtype=xp.complex128)
        self.fixed[0] = a @ rho[d:, d:] + rho[:d, :d] @ bh
        self.fixed[9] = 4 * (a + bh)
        self.gather_indices = np.asarray([
            [0, 2, 1, 3, 5, 4, 6, 8, 7, 9],
            [0, 0, 0, 2, 6, 6, 8, 9, 9, 0],
            [0, 0, 0, 1, 6, 6, 7, 9, 9, 0],
        ], dtype=np.int32)
        self.gather_adjoint = np.asarray([
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        ], dtype=np.int32)
        self.gather_buffers = [xp.empty((10, d, d), dtype=xp.complex128) for _ in range(3)]
        self.trace_terms = xp.empty(d*d, dtype=xp.complex128)
        if xp.__name__ == "cupy":
            self.device_indices = xp.asarray(self.gather_indices)
            self.device_adjoint = xp.asarray(self.gather_adjoint)
            self.gather_kernel = xp.ElementwiseKernel(
                "raw complex128 y, raw int32 indices, raw int32 adjoints, int32 d",
                "complex128 jump, complex128 left, complex128 right",
                """
                long channel = i / (d * d);
                long element = i % (d * d);
                long row = element / d, column = element % d;
                complex<double> values[3];
                for (int g = 0; g < 3; ++g) {
                    bool trans = adjoints[g * 10 + channel];
                    long source = indices[g * 10 + channel] * d * d;
                    complex<double> v = y[source + (trans ? column * d + row : element)];
                    values[g] = trans ? conj(v) : v;
                }
                jump = values[0]; left = values[1]; right = values[2];
                """, "rcmps_z2_block_gather")
            self.combine_kernel = xp.ElementwiseKernel(
                "complex128 l, complex128 qr, complex128 jump, complex128 il, complex128 ir, "
                "complex128 fixed, raw float64 coefficients, int32 d, float64 jac, float64 source",
                "complex128 out",
                "out = jac * (l + qr + jump) + source * (coefficients[i / (d * d)] * (il + ir) + fixed);",
                "rcmps_z2_rhs_combine")
            self.trace_kernel = xp.ElementwiseKernel(
                "raw complex128 y, raw complex128 a, raw complex128 b, int32 d, float64 source",
                "complex128 out",
                """
                long row = i / d, column = i % d;
                complex<double> value = a[i] * conj(y[3 * d * d + i]);
                value += b[i] * y[3 * d * d + column * d + row];
                out = complex<double>(8 * source * value.real(), 0.);
                """, "rcmps_z2_trace_accumulator")
            if self.fused_rhs:
                self._prepare_fused_kernels()

    def _prepare_fused_kernels(self):
        xp = self.xp
        self.fused_base = xp.empty((10, self.d, self.d), dtype=xp.complex128)
        self.fused_temp = xp.empty_like(self.fused_base)
        self.fused_parameters = xp.empty(2, dtype=xp.float64)
        self.fused_jac = self.fused_parameters[:1].reshape(())
        self.fused_source = self.fused_parameters[1:].reshape(())
        source = r'''
        __device__ __forceinline__ double2 zadd(double2 a, double2 b) {
            double2 c; c.x = a.x + b.x; c.y = a.y + b.y; return c;
        }
        __device__ __forceinline__ double2 zmul(double2 a, double2 b) {
            double2 c; c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x; return c;
        }
        __device__ __forceinline__ double2 zscale(double a, double2 b) {
            double2 c; c.x = a * b.x; c.y = a * b.y; return c;
        }
        __device__ __forceinline__ double2 gathered(const double2* y, int channel,
                int adjoint, int row, int column, int d) {
            double2 v = y[channel * d * d + (adjoint ? column * d + row : row * d + column)];
            if (adjoint) v.y = -v.y;
            return v;
        }
        extern "C" __global__ void rcmps_z2_first(
                const double2* y, const double2* ql, const double2* qr,
                const double2* rl, const double2* ol, const double2* op_right,
                const double2* fixed, const double* coefficients,
                const int* indices, const int* adjoints,
                const double2* a, const double2* b, int d,
                const double* jac_ptr, const double* source_ptr,
                double2* base, double2* temp, double2* energy) {
            int channel = blockIdx.z;
            double jac = jac_ptr[0], source = source_ptr[0];
            if (channel == 10) {
                if (blockIdx.x || blockIdx.y) return;
                int tid = threadIdx.y * 8 + threadIdx.x;
                double value = 0.;
                for (int i = tid; i < d*d; i += 64) {
                    int row = i / d, column = i % d;
                    double2 f = y[3*d*d + i];
                    double2 ft = y[3*d*d + column*d + row];
                    value += a[i].x*f.x + a[i].y*f.y + b[i].x*ft.x - b[i].y*ft.y;
                }
                for (int delta = 16; delta; delta /= 2)
                    value += __shfl_down_sync(0xffffffff, value, delta);
                __shared__ double partials[2];
                if ((tid & 31) == 0) partials[tid >> 5] = value;
                __syncthreads();
                if (tid == 0) {
                    double2 out; out.x = 8*source*(partials[0]+partials[1]); out.y = 0.;
                    energy[0] = out;
                }
                return;
            }
            int row = blockIdx.y * 8 + threadIdx.y;
            int column = blockIdx.x * 8 + threadIdx.x;
            if (row >= d || column >= d) return;
            int offset = channel*d*d;
            double2 left = {0., 0.}, right = {0., 0.}, jump = {0., 0.};
            double2 insert_left = {0., 0.}, insert_right = {0., 0.};
            int ji = indices[channel], ja = adjoints[channel];
            int li = indices[10+channel], la = adjoints[10+channel];
            int ri = indices[20+channel], ra = adjoints[20+channel];
            for (int k = 0; k < d; ++k) {
                left = zadd(left, zmul(ql[offset+row*d+k], y[offset+k*d+column]));
                right = zadd(right, zmul(y[offset+row*d+k], qr[offset+k*d+column]));
                jump = zadd(jump, zmul(rl[offset+row*d+k], gathered(y, ji, ja, k, column, d)));
                insert_left = zadd(insert_left, zmul(ol[offset+row*d+k], gathered(y, li, la, k, column, d)));
                insert_right = zadd(insert_right, zmul(gathered(y, ri, ra, row, k, d), op_right[offset+k*d+column]));
            }
            int element = offset+row*d+column;
            double2 insert = zadd(zscale(coefficients[channel], zadd(insert_left, insert_right)), fixed[element]);
            base[element] = zadd(zscale(jac, zadd(left, right)), zscale(source, insert));
            temp[element] = jump;
        }
        extern "C" __global__ void rcmps_z2_second(const double2* base,
                const double2* temp, const double2* rr, int d,
                const double* jac_ptr, double2* out) {
            int channel = blockIdx.z;
            int row = blockIdx.y * 8 + threadIdx.y;
            int column = blockIdx.x * 8 + threadIdx.x;
            if (row >= d || column >= d) return;
            int offset = channel*d*d;
            double2 value = {0., 0.};
            for (int k = 0; k < d; ++k)
                value = zadd(value, zmul(temp[offset+row*d+k], rr[offset+k*d+column]));
            int element = offset+row*d+column;
            out[element] = zadd(base[element], zscale(jac_ptr[0], value));
        }
        '''
        self.fused_module = xp.RawModule(code=source, options=("--std=c++11",),
            name_expressions=("rcmps_z2_first", "rcmps_z2_second"))
        self.fused_first = self.fused_module.get_function("rcmps_z2_first")
        self.fused_second = self.fused_module.get_function("rcmps_z2_second")

    def _fused_rhs_into(self, y, out, jac, source):
        d, n = self.d, self.matrix_size
        tiles = (d+7)//8
        self.fused_first((tiles, tiles, 11), (8, 8, 1),
            (y, self.ql, self.qright, self.rl, self.ol, self.oright,
             self.fixed, self.coefficients, self.device_indices, self.device_adjoint,
             self.a, self.b, np.int32(d), jac, source,
             self.fused_base, self.fused_temp, out[-1:]))
        self.fused_second((tiles, tiles, 10), (8, 8, 1),
            (self.fused_base, self.fused_temp, self.rright, np.int32(d), jac, out[:n]))

    def _gather(self, y):
        xp, d = self.xp, self.d
        if xp.__name__ == "cupy":
            self.gather_kernel(y, self.device_indices, self.device_adjoint, d,
                               *self.gather_buffers)
        else:
            matrices = y[:self.matrix_size].reshape(10, d, d)
            for out, indices, flags in zip(self.gather_buffers, self.gather_indices, self.gather_adjoint):
                out[:] = matrices[indices]
                for k in np.flatnonzero(flags):
                    out[k] = _adj(matrices[indices[k]])
        return self.gather_buffers

    def __call__(self, t, y):
        xp, d = self.xp, self.d
        self.calls += 1
        if self.fused_rhs:
            self.fused_parameters.set(np.asarray([2*abs(t), transformed_source(t)]))
            out = xp.empty_like(y)
            self._fused_rhs_into(y, out, self.fused_jac, self.fused_source)
            return out
        matrices = y[:self.matrix_size].reshape(10, d, d)
        gj, gl, gr = self._gather(y)
        left, right = self.ql @ matrices, matrices @ self.qright
        jump = (self.rl @ gj) @ self.rright
        il, ir = self.ol @ gl, gr @ self.oright
        jac, source = 2 * abs(t), transformed_source(t)
        out = xp.empty_like(y)
        matrix_out = out[:self.matrix_size].reshape(10, d, d)
        if xp.__name__ == "cupy":
            self.combine_kernel(left, right, jump, il, ir, self.fixed, self.coefficients,
                                d, jac, source, matrix_out)
            self.trace_kernel(y, self.a, self.b, d, source, self.trace_terms)
            xp.sum(self.trace_terms, out=out[-1:].reshape(()))
        else:
            matrix_out[:] = jac * (left + right + jump) + source * (
                self.coefficients[:, None, None] * (il + ir) + self.fixed)
            f3 = matrices[3]
            out[-1] = 8 * source * (np.sum(self.a * f3.conj()) + np.sum(self.b * f3.T)).real
        return out


class _LinearCombination:
    """Fuse RK stage sums on GPU; NumPy uses small BLAS contractions."""
    def __init__(self, xp, shape):
        self.xp, self.shape = xp, shape
        self.a = xp.asarray(_dop.A)
        self.b = xp.asarray(_dop.B)
        self.e3, self.e5 = xp.asarray(_dop.E3), xp.asarray(_dop.E5)
        self.d = xp.asarray(_dop.D)
        if xp.__name__ == "cupy":
            self.kernel = xp.ElementwiseKernel(
                "complex128 y, raw complex128 k, raw float64 a, int64 n, int32 stages, float64 h",
                "complex128 out",
                """
                complex<double> value = y;
                for (int s = 0; s < stages; ++s) value += h * a[s] * k[s * n + i];
                out = value;
                """, "rcmps_rk_linear_combination")

    def __call__(self, y, k, coefficients, stages, h):
        if self.xp.__name__ == "cupy":
            return self.kernel(y, k, coefficients, y.size, stages, h)
        return y + h * np.tensordot(coefficients[:stages], k[:stages], axes=(0, 0))


def _integrate(rhs, points, *, atol, rtol, initial_step, max_steps, progress=None):
    xp, y = rhs.xp, rhs.initial.copy()
    a = abs(points[-1])
    # points[-1] includes the actual endpoint, unlike the interior Gauss nodes.
    t, h = -a, min(initial_step or 0.01, 2 * a)
    f = rhs(t, y)
    stages = xp.empty((_dop.N_STAGES_EXTENDED,) + y.shape, dtype=xp.complex128)
    samples = xp.empty((len(points),) + y.shape, dtype=xp.complex128)
    linear = _LinearCombination(xp, y.shape)
    zero = xp.zeros_like(y)
    next_point, accepted, rejected, attempts = 0, 0, 0, 0
    dense_steps, dense_nodes_max = 0, 0
    rejected_since_accept = False
    progress_time = time.perf_counter()
    while t < a:
        # The transformed source has a cusp: no accepted step crosses zero.
        endpoint = 0.0 if t < 0 else a
        h = min(h, endpoint - t)
        if h <= 10 * abs(np.nextafter(t, math.inf) - t):
            raise RuntimeError(f"ODE step underflow at t={t}, h={h}")
        if attempts >= max_steps:
            raise RuntimeError("ODE exceeded max_steps")
        attempts += 1
        stages[0] = f
        for s in range(1, _dop.N_STAGES):
            intermediate = linear(y, stages, linear.a[s], s, h)
            stages[s] = rhs(t + _dop.C[s] * h, intermediate)
        proposed = linear(y, stages, linear.b, _dop.N_STAGES, h)
        stages[_dop.N_STAGES] = rhs(t + h, proposed)
        scale = atol + rtol * xp.maximum(xp.abs(y), xp.abs(proposed))
        err5 = linear(zero, stages, linear.e5, _dop.N_STAGES + 1, 1) / scale
        err3 = linear(zero, stages, linear.e3, _dop.N_STAGES + 1, 1) / scale
        norms = xp.stack([xp.sum(xp.abs(err5)**2), xp.sum(xp.abs(err3)**2)])
        # One device-to-host synchronization per attempted adaptive step.
        if xp.__name__ == "cupy":
            e5, e3 = xp.asnumpy(norms)
        else:
            e5, e3 = norms
        denominator = e5 + 0.01 * e3
        error = h * e5 / math.sqrt(denominator * y.size) if denominator else 0.0
        if not math.isfinite(error):
            raise RuntimeError("nonfinite ODE error estimate")
        factor = min(10.0, 0.9 * error**(-1 / 8)) if error else 10.0
        if error > 1:
            rejected += 1
            rejected_since_accept = True
            h *= max(0.2, factor)
            continue
        accepted += 1
        t_new = endpoint if h == endpoint - t else t + h
        # Build dense interpolation only for steps containing requested nodes.
        if next_point < len(points) and points[next_point] <= t_new:
            first_point = next_point
            dense_steps += 1
            for s in range(_dop.N_STAGES + 1, _dop.N_STAGES_EXTENDED):
                intermediate = linear(y, stages, linear.a[s], s, h)
                stages[s] = rhs(t + _dop.C[s] * h, intermediate)
            delta = proposed - y
            dense = [delta, h * f - delta, 2 * delta - h * (stages[_dop.N_STAGES] + f)]
            dense += [linear(zero, stages, row, _dop.N_STAGES_EXTENDED, h) for row in linear.d]
            while next_point < len(points) and points[next_point] <= t_new:
                point = points[next_point]
                if point == t_new:
                    samples[next_point] = proposed
                else:
                    x = (point - t) / h
                    result = xp.zeros_like(y)
                    for i, coefficient in enumerate(reversed(dense)):
                        result += coefficient
                        result *= x if i % 2 == 0 else 1 - x
                    samples[next_point] = result + y
                next_point += 1
            dense_nodes_max = max(dense_nodes_max, next_point-first_point)
        y, f, t = proposed, stages[_dop.N_STAGES].copy(), t_new
        h *= min(1, factor) if rejected_since_accept else factor
        rejected_since_accept = False
        if progress is not None and time.perf_counter() - progress_time >= 10:
            progress(dict(t=t, endpoint=a, accepted_steps=accepted, rejected_steps=rejected,
                          rhs_calls=rhs.calls, next_step=h))
            progress_time = time.perf_counter()
    if next_point != len(points):
        raise RuntimeError("ODE did not save all quadrature nodes")
    return samples, {"accepted_steps": accepted, "rejected_steps": rejected,
                     "rhs_calls": rhs.calls, "matrix_channels": rhs.channels,
                     "algorithm": "batched DOP853", "gpu_graph": False,
                     "dense_graph": False, "dense_steps": dense_steps,
                     "saved_nodes": next_point, "dense_nodes_max": dense_nodes_max}


def _pullback(q, r, specs, rhs, samples, points, weights):
    xp = rhs.xp
    n = len(points)
    w = xp.asarray(weights * 2 * np.abs(points))[:, None, None, None]
    j = xp.asarray(weights * np.asarray([transformed_source(t) for t in points]))[:, None, None]
    components = []
    for offset, spec in zip(rhs.offsets, specs):
        f = samples[:n, offset:offset + spec.channels]
        b = samples[n - 1::-1, rhs.half + offset:rhs.half + offset + spec.channels]
        m = xp.sum(w * (f @ (_adj(r) @ b - b @ _adj(r))), axis=(0, 1))
        nn = xp.sum(w * ((b @ r - r @ b) @ f), axis=(0, 1))
        u, v = xp.zeros_like(r), xp.zeros_like(r)
        for src, dst, left, coefficient in spec.edges:
            product = f[:, src] @ b[:, dst] if left else b[:, dst] @ f[:, src]
            contribution = xp.sum((coefficient * j) * product, axis=0)
            if left:
                u += contribution
            else:
                v += contribution
        if spec.commutator:
            m += u @ q - q @ u - r @ u @ _adj(r) + u @ r @ _adj(r)
            nn += _adj(q) @ v - v @ _adj(q) - r @ v @ _adj(r) + r @ _adj(r) @ v
        else:
            m += u
            nn += v
        covector = (m + _adj(nn)) / 2
        environment = samples[-1, rhs.half + offset]
        environment = (environment + _adj(environment)) / 2
        value = _scalar(xp.trace(samples[-1, offset + spec.channels - 1]).real)
        components.append((value, covector, environment))
    return components


def _reduced_quartic_pullback(q, r, rhs, samples, points, weights):
    xp, n, d = rhs.xp, len(points), rhs.d
    matrices = samples[:, :rhs.matrix_size].reshape(len(samples), 7, d, d)
    # Reconstruct the channel interfaces used by the independently validated
    # full pullback.  The unused terminal forward matrix is zero; its paired
    # bulk terms are identically zero when the corresponding adjoint is I.
    full = xp.empty((len(samples), 10, d, d), dtype=xp.complex128)
    full[:, 0] = rhs.rho
    full[:, 1:4] = matrices[:, :3]
    full[:, 4] = 0
    full[:, 5:9] = matrices[:, 3:]
    full[:, 9] = xp.eye(d, dtype=xp.complex128)
    interface = type("QuarticChannelInterface", (), {"xp": xp, "offsets": [0], "half": 5})()
    _, covector, environment = _pullback(q, r, [_QUARTIC], interface, full, points, weights)[0]
    value = _scalar(samples[-1, -1].real)
    return [(value, covector, environment)]


def _unpack_z2_matrices(packed, xp):
    """Restore seven Hermitian channels solely for physical contractions."""
    n, _, d, _ = packed.shape
    full = xp.zeros((n, 7, 2*d, 2*d), dtype=xp.complex128)
    for full_channel, packed_channel in ((0, 0), (2, 3), (4, 6), (6, 9)):
        full[:, full_channel, :d, d:] = packed[:, packed_channel]
        full[:, full_channel, d:, :d] = _adj(packed[:, packed_channel])
    for full_channel, first, second in ((1, 1, 2), (3, 4, 5), (5, 7, 8)):
        full[:, full_channel, :d, :d] = packed[:, first]
        full[:, full_channel, d:, d:] = packed[:, second]
    return full



def packed_pullback_memory_estimate(full_d,n_quad,chunk_size=64):
    """Explicit buffer estimates; not a bound on allocator/library peak memory."""
    if full_d%2 or full_d<2 or n_quad<8 or chunk_size<1:
        raise ValueError('invalid packed pullback dimensions')
    nodes=2*int(n_quad);d=full_d//2;active=min(nodes,int(chunk_size))
    return dict(packed_sample_bytes=(nodes+1)*(10*d*d+1)*16,
        old_global_reconstruction_bytes=(nodes+1)*24*full_d*full_d*16,
        chunk_reconstruction_bytes=(active*14+(active+1)*10)*full_d*full_d*16,
        largest_chunk_five_channel_product_bytes=active*5*full_d*full_d*16,
        chunk_size=int(chunk_size),includes_kinetic_or_allocator_cached_bytes=False)


def _packed_z2_pullback_chunked(q,r,rhs,samples,points,weights,chunk_size=64):
    """Same physical contraction, fixed-memory chunks and global adjoint pairing.

    The reverse-time adjoint slice uses the full quadrature count, never a
    chunk-local approximation. Endpoint energy/environment are read once.
    Chunking changes only FP64 reduction association; no samples are omitted.
    """
    if not isinstance(chunk_size,(int,np.integer)) or chunk_size<1:
        raise ValueError('pullback chunk_size must be a positive integer')
    xp,d,n=rhs.xp,rhs.full_d,len(points)
    if len(weights)!=n or len(samples)!=n+1:
        raise ValueError('packed pullback requires all quadrature nodes and endpoint')
    blocks=samples[:,:rhs.matrix_size].reshape(len(samples),10,rhs.d,rhs.d)
    covector=xp.zeros_like(r);correction=xp.zeros_like(r)
    identity=xp.eye(d,dtype=xp.complex128)
    interface=type('QuarticChannelInterface',(),{'xp':xp,'offsets':[0],'half':5})()
    endpoint=_unpack_z2_matrices(blocks[-1:],xp)[0,3]
    for lo in range(0,n,int(chunk_size)):
        hi=min(n,lo+int(chunk_size));width=hi-lo
        forward=_unpack_z2_matrices(blocks[lo:hi],xp)
        backward=_unpack_z2_matrices(blocks[n-hi:n-lo],xp)
        full=xp.zeros((width+1,10,d,d),dtype=xp.complex128)
        full[:width,0]=rhs.rho;full[:width,1:4]=forward[:,:3]
        full[:width,5:9]=backward[:,3:];full[:width,9]=identity
        full[-1,5]=endpoint
        del forward,backward
        (_,chunk_covector,_),=_pullback(q,r,[_QUARTIC],interface,full,points[lo:hi],weights[lo:hi])
        adjusted=chunk_covector-correction;updated=covector+adjusted
        correction=(updated-covector)-adjusted;covector=updated
    environment=(endpoint+_adj(endpoint))/2
    value=_scalar(samples[-1,-1].real)
    return [(value,covector,environment)]


def _packed_z2_pullback(q,r,rhs,samples,points,weights):
    if rhs.full_d>=192:
        return _packed_z2_pullback_chunked(q,r,rhs,samples,points,weights)
    return _packed_z2_pullback_unchunked(q,r,rhs,samples,points,weights)


def _packed_z2_pullback_unchunked(q, r, rhs, samples, points, weights):
    xp, d = rhs.xp, rhs.full_d
    blocks = samples[:, :rhs.matrix_size].reshape(len(samples), 10, rhs.d, rhs.d)
    full = _unpack_z2_matrices(blocks, xp)
    expanded = xp.empty((len(samples), 7*d*d + 1), dtype=xp.complex128)
    expanded[:, :-1] = full.reshape(len(samples), -1)
    expanded[:, -1] = samples[:, -1]
    interface = type("ReducedQuarticInterface", (),
        {"xp": xp, "d": d, "matrix_size": 7*d*d, "rho": rhs.rho})()
    return _reduced_quartic_pullback(q, r, interface, expanded, points, weights)


def _evaluate(r, q, rho, specs, coefficients, *, xp, half_window, n_quad,
              atol, rtol, initial_step, max_steps, use_graph, progress, reduced=False,
              packed_z2=False, fused_rhs=False, use_dense_graph=False):
    if xp is None:
        xp = np
    if use_graph or use_dense_graph:
        # Accepted as keywords so that callers stay valid; only the host-driven
        # integrator remains.
        raise ValueError("the CUDA-graph integrator was removed: use_graph and use_dense_graph must be False")
    if fused_rhs and (xp.__name__ != "cupy" or not packed_z2):
        raise ValueError("fused_rhs requires CuPy and packed_z2=True")
    r, q, rho = (xp.asarray(x, dtype=xp.complex128) for x in (r, q, rho))
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[0] != r.shape[1] or q.shape != r.shape or rho.shape != r.shape:
        raise ValueError("R, Q, and rho must be equally sized square matrices")
    if atol <= 0 or rtol < 100 * np.finfo(float).eps or max_steps < 1:
        raise ValueError("invalid ODE controls")
    points, weights = quadrature(half_window, n_quad)
    save_points = np.r_[points, math.sqrt(half_window)]
    _sync(xp)
    start = time.perf_counter()
    if packed_z2:
        rhs = _PackedZ2QuarticRHS(q, r, rho, xp, fused_rhs=fused_rhs)
    else:
        rhs = _ReducedQuarticRHS(q, r, rho, xp) if reduced else _BatchedRHS(q, r, rho, specs, xp)
    samples, stats = _integrate(rhs, save_points, atol=atol, rtol=rtol,
        initial_step=initial_step, max_steps=max_steps, progress=progress)
    _sync(xp)
    propagated = time.perf_counter()
    if packed_z2:
        components = _packed_z2_pullback(q, r, rhs, samples, points, weights)
    elif reduced:
        components = _reduced_quartic_pullback(q, r, rhs, samples, points, weights)
    else:
        components = _pullback(q, r, specs, rhs, samples, points, weights)
    value = sum(c * result[0] for c, result in zip(coefficients, components))
    covector, environment = xp.zeros_like(r), xp.zeros_like(r)
    for coefficient, (_, cov, env) in zip(coefficients, components):
        covector += coefficient * cov
        environment += coefficient * env
    _sync(xp)
    end = time.perf_counter()
    stats.update(propagation_seconds=propagated - start, contraction_seconds=end - propagated,
        total_seconds=end - start, backend=xp.__name__, dtype="complex128",
        half_window=half_window, n_quad=n_quad, atol=atol, rtol=rtol,
        finite_window=True)
    stats.update(reduced_channels=bool(reduced or packed_z2), packed_z2=bool(packed_z2),
                 matrix_dimension=rhs.d, fused_rhs=bool(fused_rhs))
    return dict(value=value, covector=covector, environment=environment,
                stats=stats, component_values=[x[0] for x in components])


def evaluate_phi4(R, Q, rho, g, *, xp=None, half_window=20.0, n_quad=192,
                  atol=1e-13, rtol=1e-12, initial_step=None, max_steps=1_000_000,
                  use_graph=False, use_dense_graph=False, progress=None):
    """Full finite-window phi4 value, raw covector, and boundary environment."""
    return _evaluate(R, Q, rho, [_Spec(4, _KINETIC), _Spec(4, _KINETIC, True), _QUARTIC],
        [1.0, 1.0, float(g)], xp=xp, half_window=half_window, n_quad=n_quad,
        atol=atol, rtol=rtol, initial_step=initial_step, max_steps=max_steps,
        use_graph=use_graph, use_dense_graph=use_dense_graph, progress=progress)


def evaluate_quartic(R, Q, rho, g=1.0, *, xp=None, half_window=20.0, n_quad=192,
                     atol=1e-13, rtol=1e-12, initial_step=None, max_steps=1_000_000,
                     use_graph=False, use_dense_graph=False, progress=None, reduced=False, packed_z2=False,
                     fused_rhs=False):
    """The g*phi^4 interaction, for use alongside the kinetic resolvent backend."""
    return _evaluate(R, Q, rho, [_QUARTIC], [float(g)], xp=xp,
        half_window=half_window, n_quad=n_quad, atol=atol, rtol=rtol,
        initial_step=initial_step, max_steps=max_steps, use_graph=use_graph,
        use_dense_graph=use_dense_graph, progress=progress,
        reduced=reduced, packed_z2=packed_z2, fused_rhs=fused_rhs)

