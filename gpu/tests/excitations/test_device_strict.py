"""Host/device mixing checks of the production modules on the strict stand-in fakecupy.

No GPU or CuPy exists on the development machine, and NumPy hides every host/device
mixing bug of the xp-generic modules. Each routine is run twice on the same host input,
with xp = numpy and with xp = fakecupy (NumPy-backed, with CuPy's rules for operand
mixing, implicit conversion, 0-d results, uploads and downloads; see fakecupy.py). The
device result, downloaded with .get(), must agree with the NumPy one to 1e-12 of its
largest entry. The stand-in computes with NumPy, so the two usually agree to the last
bit: what a pass proves is that the device run completes, i.e. no host array reaches a
device operation, no device array reaches NumPy, SciPy or host algebra, and every
result lives where it should (device arrays for matrices, host objects for the white
coefficient vectors, statistics and energies). It proves nothing about CUDA.

State: random_state(4, seed=11, scaling=0.5, z2=True) in the density gauge (both parity
sectors), window max_point = 14, n_quad = 32, atol = 1e-13, rtol = 1e-12, 16 kinetic
nodes, p = 0.6; S = 1, sqrt(diag rho) and exp(linspace(-2.5, 1.5, D)).

StandIn   fakecupy refuses what CuPy refuses, and its documented stricter cases.
Solver    ScaledState, typed maps, resolvent and Poisson solves of every type, GMRES branches.
Actions   KineticAction, PotentialAction, HamiltonianAction on ActionSystem and PackedActionSystem.
Bilinear  Pair, BilinearElement, the packed diagonal pair of the fused validation.
Driver    WhiteBasis, kinetic preconditioner, field seed, excitation_eigsolve (validate=True).
Mutations (RCMPS_EXCITATION_SLOW=1, about a minute) the harness has teeth: one host/device
          bug planted at a time in a temporary copy of the modules fails the named test.

RCMPS_EXCITATION_REPORT=1 prints the largest measured differences at the end. About
90 s on one thread, most of it the three excitation_eigsolve runs on both backends.
"""
from __future__ import annotations

import functools
import math
import operator
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
import os
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.linalg

from rcmps_gpu.excitations import bilinear
from rcmps_gpu.excitations import bilinear_fused
from rcmps_gpu.excitations import coordinates as tc
from rcmps_gpu.excitations import eigsolve as es
import fakecupy as cp
from rcmps_gpu.excitations import packed
from rcmps_gpu.excitations import scaled
from rcmps_gpu.excitations.davidson import DavidsonOptions
from rcmps_gpu.excitations.hamiltonian import HamiltonianAction
from rcmps_gpu.excitations.kinetic import KineticAction, TangentFrame, kinetic_terms
from rcmps_gpu.excitations.models import kinetic_spec, monomial_spec, phi4_terms, quadrature
from rcmps_gpu.excitations.ode import integrate_host
from rcmps_gpu.excitations.potential import ActionSystem, PotentialAction
from rcmps_gpu.excitations.states import IntegratorParameters, adj, random_state

TOL, P, NODES = 1e-12, 0.6, 16
PHI4 = phi4_terms(1.0)
STATE = random_state(4, seed=11, scaling=0.5, z2=True)
STATE.coupling = 1.0
STATE.integrator = IntegratorParameters(n_quad=32, max_point=14, atol=1e-13, rtol=1e-12)
D = STATE.D
SCALINGS = {"none": "none", "density": "density", "skew": np.exp(np.linspace(-2.5, 1.5, D))}
TYPES = [(a, b, dual) for a in "LH" for b in "LH" for dual in (False, True)]
BACKENDS = (np, cp)
WORST = {}
slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1",
                           "set RCMPS_EXCITATION_SLOW=1 (about a minute)")
# (module, exact source text, planted bug, test of this file that must then fail)
MUTATIONS = (
    ("scaled.py", "safe = xp.asarray(np.where(beta_host > 0, beta_host, 1.0))",
     "safe = np.where(beta_host > 0, beta_host, 1.0)", "Solver.test_solves_of_every_type"),
    ("scaled.py", "column[:, :j + 1] += _host(h).T", "column[:, :j + 1] += h.T", "Solver.test_solves_of_every_type"),
    ("scaled.py", "before = _host(xp.sqrt(xp.sum(w.real * w.real + w.imag * w.imag, axis=1)))",
     "before = np.sqrt(np.sum(w.real * w.real + w.imag * w.imag, axis=1))", "Solver.test_solves_of_every_type"),
    ("scaled.py", "self.s = xp.asarray(s)", "self.s = s", "Solver.test_scaled_state"),
    ("kinetic.py", "self.z, self.weights = xp.asarray(self.z_host), xp.asarray(weights)",
     "self.z, self.weights = self.z_host, weights", "Actions.test_kinetic_action"),
    ("potential.py", "w = xp.asarray(jacobian[first:last])[:, None, None, None]",
     "w = jacobian[first:last][:, None, None, None]", "Actions.test_potential_action"),
    ("ode.py", "e5 = float(xp.sum(xp.abs(err5 / scale) ** 2))", "e5 = float(np.sum(np.abs(err5 / scale) ** 2))",
     "Actions.test_trajectory"),
    ("hamiltonian.py", "out, a0 = xp.zeros_like(frame.wh), xp.zeros_like(frame.wh)",
     "out, a0 = xp.empty_like(frame.wh), xp.empty_like(frame.wh)", "Actions.test_hamiltonian_action"),
    ("packed.py", "self._coefficient = xp.asarray(t[\"coefficient\"])[:, None, None]",
     "self._coefficient = t[\"coefficient\"][:, None, None]", "Actions.test_packed_action"),
    ("bilinear.py", "z = xp.asarray(z_host)[None, :, None]", "z = z_host[None, :, None]", "Bilinear.test_element"),
    ("bilinear.py", "return complex(pair.contract(final[:, :, -1:, :])[0])",
     "return complex(np.asarray(pair.contract(final[:, :, -1:, :]))[0])", "Bilinear.test_element"),
    ("eigsolve.py", "return to_host(self.xp.moveaxis(g, 1, 0)[:, self.rows, self.cols])",
     "return self.xp.moveaxis(g, 1, 0)[:, self.rows, self.cols]", "Driver.test_preconditioner_and_field_seed"),
    ("eigsolve.py", "m[:, self.rows, self.cols] = xp.asarray(xs)", "m[:, self.rows, self.cols] = xs",
     "Driver.test_white_basis"),
    ("eigsolve.py", "return u @ (to_host(wh) / self.ss.s_host[None, :]) @ adj(u)",
     "return u @ (wh / self.ss.s[None, :]) @ adj(u)", "Driver.test_white_basis"),
)


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for name, value in sorted(WORST.items()):
            print(f"{name}: {value:.3e}", file=sys.stderr)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


@functools.lru_cache(maxsize=None)
def chart(sector="odd"):
    return tc.make_coordinates(STATE, sector=sector)


def fresh(xp, sector="odd", scaling="density", **controls):
    """A new SolverState and WhiteBasis on the density-gauge state, on backend xp."""
    ss = es.SolverState(chart(sector).state, SCALINGS[scaling], xp=xp, **controls)
    return ss, es.WhiteBasis(chart(sector), ss)


@functools.lru_cache(maxsize=None)
def shared(sector="odd", scaling="density"):
    return {xp: fresh(xp, sector, scaling) for xp in BACKENDS}


def both(function, sector="odd", scaling="density"):
    """(NumPy result, device result) of function(xp, ss, basis) on fresh solver states."""
    return tuple(function(xp, *fresh(xp, sector, scaling)) for xp in BACKENDS)


def coefficients(sector="odd", count=2):
    """White coefficient vectors (count, n) of a sector, on the host."""
    return gaussian(9, count, len(chart(sector)))


class Case(unittest.TestCase):
    def measure(self, value, reference, name, tol):
        self.assertEqual((value.shape, value.dtype), (reference.shape, reference.dtype), name)
        if value.dtype.kind not in "fc" or not value.size:
            return self.assertTrue(np.array_equal(value, reference), name)
        error = float(np.abs(value - reference).max() / max(1.0, np.abs(reference).max()))
        WORST[name.split()[0]] = max(WORST.get(name.split()[0], 0.0), error)
        self.assertTrue(error <= tol, f"{name}: device - numpy = {error:.3e} exceeds {tol:.1e}")

    def same(self, device, reference, name, tol=TOL):
        """A device array whose download agrees with the NumPy array or scalar `reference`."""
        self.assertIsInstance(device, cp.ndarray, name)
        self.assertIsInstance(reference, (np.ndarray, np.generic), name)
        self.measure(device.get(), np.asarray(reference), name, tol)

    def same_host(self, value, reference, name, tol=TOL):
        """Host objects on both backends (arrays, numbers, containers of them), equal to tol."""
        self.assertNotIsInstance(value, cp.ndarray, name)
        self.assertIs(type(value), type(reference), name)
        if isinstance(reference, dict):
            self.assertEqual(list(value), list(reference), name)
            for key in reference:
                self.same_host(value[key], reference[key], f"{name} {key}", tol)
        elif isinstance(reference, (list, tuple)):
            self.assertEqual(len(value), len(reference), name)
            for k, (x, y) in enumerate(zip(value, reference)):
                self.same_host(x, y, f"{name} {k}", tol)
        elif isinstance(reference, (np.ndarray, np.generic, float, complex)) and not isinstance(reference, np.bool_):
            x, y = np.asarray(value), np.asarray(reference)
            if y.dtype.kind in "fc":
                self.assertTrue(np.array_equal(np.isnan(x), np.isnan(y)), name)
                x, y = np.nan_to_num(x), np.nan_to_num(y)
            self.measure(x, y, name, tol)
        else:
            self.assertEqual(value, reference, name)

    def same_info(self, device, reference, name):
        self.same_host({k: device[k] for k in ("iterations", "residuals", "steps")},
                       {k: reference[k] for k in ("iterations", "residuals", "steps")}, name)


class StandIn(unittest.TestCase):
    """fakecupy itself: the rules that make a pass of the other classes meaningful."""

    def setUp(self):
        self.h = gaussian(1, 3, 3)
        self.d = cp.asarray(self.h)

    def test_name_and_gpu_only(self):
        self.assertEqual(cp.__name__, "cupy")
        self.assertIsNot(sys.modules.get("cupy"), cp)       # the stand-in never registers as cupy
        self.assertTrue(hasattr(cp, "ReductionKernel") and hasattr(cp, "RawKernel"))
        for call in (lambda: cp.RawKernel("", "f"), lambda: cp.RawModule(code=""),
                     lambda: cp.ElementwiseKernel("", "", ""), lambda: cp.cuda.Device(),
                     lambda: cp.cuda.runtime.getDeviceProperties(0), lambda: cp.cuda.get_current_stream(),
                     lambda: cp.get_default_memory_pool()):
            with self.assertRaises(NotImplementedError):
                call()
        for call in (lambda: cp.einsum, lambda: cp.linalg.eig, lambda: cp.linalg.cond, lambda: cp.linalg.eigh,
                     lambda: self.d.device, lambda: self.d._missing):
            with self.assertRaises(AttributeError):
                call()

    def test_operand_mixing(self):
        """A numpy.ndarray operand raises on either side, also 0-d and in place; scalars pass."""
        h, d = self.h.real.copy(), cp.asarray(self.h.real)
        binary = (operator.add, operator.sub, operator.mul, operator.truediv, operator.matmul, operator.pow,
                  operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge,
                  operator.iadd, operator.isub, operator.imul, operator.itruediv)
        for f in binary:
            for x, y in ((d, h), (h, d), (d, np.array(2.0)), (np.array(2.0), d), (d, [1.0, 2.0, 3.0]), (d, None)):
                with self.subTest(f=f.__name__, x=type(x).__name__, y=type(y).__name__), self.assertRaises(TypeError):
                    f(x.copy() if hasattr(x, "copy") else x, y)
        self.assertTrue(np.array_equal(d.get(), h) and np.array_equal(h, self.h.real))
        for scalar in (2, 2.5, 2.5j, np.float64(2.5), np.int64(2), np.complex128(2.5j)):
            for f in (operator.add, operator.sub, operator.mul, operator.truediv):
                for x, y, hx, hy in ((d, scalar, h, scalar), (scalar, d, scalar, h)):
                    out = f(x, y)
                    self.assertIsInstance(out, cp.ndarray)
                    self.assertTrue(np.array_equal(out.get(), f(hx, hy)))
        self.assertTrue(np.array_equal((np.float64(0.1) < d).get(), 0.1 < h))
        self.assertTrue(np.array_equal((-d @ d + abs(d) ** 2).get(), -h @ h + abs(h) ** 2))
        total = d.copy()
        total += d
        total *= 0.5
        self.assertTrue(np.array_equal(total.get(), h))

    def test_no_implicit_conversion(self):
        """NumPy, SciPy and Python containers cannot take a device array."""
        h, d = self.h.copy(), self.d
        calls = dict(
            asarray=lambda: np.asarray(d), array=lambda: np.array(d), nested=lambda: np.array([d, d]),
            typed=lambda: np.asarray(d, dtype=complex), sum=lambda: np.sum(d), sqrt=lambda: np.sqrt(d),
            isfinite=lambda: np.isfinite(d), norm=lambda: np.linalg.norm(d), solve=lambda: np.linalg.solve(d, d),
            einsum=lambda: np.einsum("ij,ij->", d, d), concatenate=lambda: np.concatenate([h, d]),
            shape=lambda: np.shape(d), vdot=lambda: np.vdot(h, d), index=lambda: h[cp.arange(2)],
            element=lambda: h[cp.asarray(1)], assign=lambda: h.__setitem__(0, d[0]),
            scipy=lambda: scipy.linalg.cholesky(d), range=lambda: range(cp.asarray(3)),
            vector=lambda: float(d[0, :1].real), text=lambda: f"{cp.sum(d.real):.3e}", key=lambda: {cp.sum(d): 1})
        for name, call in calls.items():
            with self.subTest(name), self.assertRaises(TypeError):
                call()
        self.assertTrue(np.array_equal(h, self.h))

    def test_uploads_and_downloads(self):
        """Only asarray and array take a host array; uploads and downloads copy."""
        h, d = self.h, self.d
        calls = dict(
            sum=lambda: cp.sum(h), abs=lambda: cp.abs(h), conj=lambda: cp.conj(h), matmul=lambda: cp.matmul(d, h),
            maximum=lambda: cp.maximum(h.real, d.real), concatenate=lambda: cp.concatenate([d, h]),
            stack=lambda: cp.stack([h, h]), norm=lambda: cp.linalg.norm(h), zeros_like=lambda: cp.zeros_like(h),
            moveaxis=lambda: cp.moveaxis(h, 0, 1), contiguous=lambda: cp.ascontiguousarray(h),
            full=lambda: cp.full(2, np.ones(())), assign=lambda: d.__setitem__(0, h[0]),
            out=lambda: cp.sum(d, out=np.zeros((), dtype=complex)), sequence=lambda: cp.asarray([d, d]),
            listed=lambda: cp.sum([1.0, 2.0]), single=lambda: cp.concatenate(d))
        for name, call in calls.items():
            with self.subTest(name), self.assertRaises(TypeError):
                call()
        source = h.copy()
        up = cp.asarray(source)
        source[0, 0] = 99
        down = up.get()
        down[1, 1] = 77
        for value in (up.get(), cp.asnumpy(up), cp.array(h).get(), cp.asarray(up, dtype=cp.complex128).get()):
            self.assertIsInstance(value, np.ndarray)
            self.assertTrue(np.array_equal(value, h))
        self.assertIs(cp.asarray(up), up)
        self.assertIs(cp.asnumpy(h), h)
        self.assertEqual(cp.asarray([1, 2]).dtype, np.int64)
        self.assertEqual(cp.asarray(2.5).shape, ())

    def test_indexing_and_views(self):
        h, d = self.h, self.d
        rows, mask = np.array([0, 2]), np.array([True, False, True])
        for key, device_key in ((1, 1), ((slice(None), 1),) * 2, (rows, rows), (rows, cp.asarray(rows)),
                                ((rows, slice(None), ), (cp.asarray(rows), slice(None))),
                                ((rows, rows), (cp.asarray(rows), cp.asarray(rows))), (mask, mask),
                                (mask, cp.asarray(mask)), ((Ellipsis, 0),) * 2, (slice(None, None, -1),) * 2,
                                ([0, 1], [0, 1]), ((None, 1), (None, 1))):
            out = d[device_key]
            self.assertIsInstance(out, cp.ndarray)
            self.assertTrue(np.array_equal(out.get(), h[key]))
        batch, hbatch = cp.asarray(gaussian(2, 3, 4, 3)), gaussian(2, 3, 4, 3)
        index = cp.arange(3)
        self.assertTrue(np.array_equal(batch[index, :, index].get(), hbatch[np.arange(3), :, np.arange(3)]))
        with self.assertRaises(IndexError):
            d[np.array([0.0, 1.0])]
        element = d[0, 0]                     # CuPy: a 0-d view, not a scalar
        self.assertEqual((type(element), element.shape), (cp.ndarray, ()))
        element += 1
        row = d[1]
        row *= 2
        fancy = d[cp.asarray(rows)]
        fancy += 5                            # advanced indexing copies
        expected = h.copy()
        expected[0, 0] += 1
        expected[1] *= 2
        self.assertTrue(np.array_equal(d.get(), expected))
        d[cp.asarray(rows), 0] = cp.asarray(np.array([7.0, 8.0]))
        d[:, 2] = 0.5
        expected[rows, 0] = [7.0, 8.0]
        expected[:, 2] = 0.5
        self.assertTrue(np.array_equal(d.get(), expected))
        self.assertEqual([type(r) for r in d], [cp.ndarray] * 3)
        self.assertEqual(len(d), 3)

    def test_zero_dimensional_results(self):
        h, d = self.h, self.d
        for value in (cp.sum(d), d.sum(), cp.linalg.norm(d), d[1, 2], cp.abs(d).max(), cp.sqrt(2.0), d.trace()):
            self.assertIsInstance(value, cp.ndarray)
            self.assertEqual(value.shape, ())
            self.assertNotIsInstance(value, (float, complex))
        self.assertEqual(complex(cp.sum(d)), complex(h.sum()))
        self.assertEqual(float(cp.linalg.norm(d)), float(np.linalg.norm(h)))
        self.assertEqual(int(cp.arange(4).sum()), 6)
        self.assertIs(bool(cp.isfinite(d).all()), True)
        self.assertEqual(cp.sum(d).item(), h.sum())
        self.assertTrue(math.isfinite(cp.linalg.norm(d)))
        self.assertEqual(float(max(1.0, cp.linalg.norm(d))), max(1.0, float(np.linalg.norm(h))))
        with self.assertRaises(ValueError):
            bool(d == d)

    def test_attributes(self):
        h, d = self.h, self.d
        self.assertEqual((d.shape, d.ndim, d.dtype, d.size), (h.shape, h.ndim, h.dtype, h.size))
        for name, value, reference in (
                ("real", d.real, h.real), ("imag", d.imag, h.imag), ("T", d.T, h.T), ("conj", d.conj(), h.conj()),
                ("reshape", d.reshape(1, 9), h.reshape(1, 9)), ("copy", d.copy(), h),
                ("sum", d.sum(axis=(0, 1)), h.sum(axis=(0, 1))), ("axis", d.sum(axis=1), h.sum(axis=1)),
                ("astype", d.real.astype(cp.complex128), h.real.astype(complex)),
                ("swapaxes", d.conj().swapaxes(-1, -2), h.conj().T), ("all", (d == d).all(), np.True_),
                ("any", (d != d).any(), np.False_), ("diag", cp.diag(d), np.diag(h)),
                ("moveaxis", cp.moveaxis(d, 0, 1), h.T), ("broadcast", cp.broadcast_to(d[:, None, :], (3, 2, 3)),
                                                         np.broadcast_to(h[:, None, :], (3, 2, 3)))):
            self.assertIsInstance(value, cp.ndarray, name)
            self.assertTrue(np.array_equal(value.get(), reference), name)
        self.assertTrue(d.flags.c_contiguous and not d.T.flags.c_contiguous)
        self.assertTrue(cp.ascontiguousarray(d.T).flags.c_contiguous)
        view = d.real                        # a view, as in CuPy and NumPy
        view[0, 0] = 5.0
        self.assertEqual(complex(d[0, 0]), 5.0 + 1j * h[0, 0].imag)
        real = cp.ones(3)
        self.assertIs(real.conj(), real)     # CuPy returns the array itself for a real dtype
        self.assertIs(real.real, real)
        self.assertEqual(cp.eye(2, dtype=cp.complex128).dtype, np.complex128)
        self.assertTrue(np.array_equal(cp.full(2, 1j, dtype=cp.complex128).get(), np.full(2, 1j)))

    def test_empty_is_poisoned(self):
        for a in (cp.empty((2, 2), dtype=cp.complex128), cp.empty_like(self.d), cp.empty(3)):
            self.assertTrue(np.isnan(a.get()).all())

    def test_reduction_kernel(self):
        dot = cp.ReductionKernel("complex128 v, complex128 w", "complex128 h", "conj(v) * w", "a + b", "h = a", "0",
                                 "dot")
        v, w = gaussian(2, 3, 2, 5), gaussian(3, 2, 5)
        out = dot(cp.asarray(v), cp.asarray(w)[None], axis=2)
        self.assertIsInstance(out, cp.ndarray)
        self.assertTrue(np.array_equal(out.get(), np.sum(np.conj(v) * w[None], axis=2)))
        for arguments in ((cp.asarray(v), w[None]), (cp.asarray(v), cp.asarray(w.real)[None]), (cp.asarray(v),)):
            with self.assertRaises(TypeError):
                dot(*arguments, axis=2)
        for arguments in (("complex128 v", "complex128 h", "v", "a * b", "h = a", "1"),
                          ("complex128 v", "complex128 h", "v.real() > 0 ? v : -v", "a + b", "h = a", "0")):
            with self.assertRaises(NotImplementedError):
                cp.ReductionKernel(*arguments)


class Solver(Case):
    """scaled.py: ScaledState, the typed maps and the batched solves."""

    def test_scaled_state(self):
        """Generators, eigenbases, weights and the hat fixed point are device arrays equal
        to the NumPy ones; S and the solve reports stay on the host."""
        for scaling in SCALINGS:
            reference, device = (shared("odd", scaling)[xp][0] for xp in BACKENDS)
            for name in ("Q", "R", "Qd", "Rd", "lam", "V", "Vi", "Vd", "Vid", "g", "w"):
                for gauge in "LH":
                    self.same(getattr(device, name)[gauge], getattr(reference, name)[gauge], f"state {name}")
            for name in ("s", "eye", "rh"):
                self.same(getattr(device, name), getattr(reference, name), f"state {name}")
            self.assertIs(device.xp, cp)
            self.same_host(device.s_host, reference.s_host, "state s_host")
            self.same_host(device.fixed_point, reference.fixed_point, "state fixed_point")
            self.same_host(device.eigenbasis_condition, reference.eigenbasis_condition, "state condition")
            reference, device = both(lambda xp, ss, basis: es.solver_statistics(ss), "odd", scaling)
            self.same_host(device, reference, "state statistics")

    def test_fixed_point_refinement(self):
        """A stationary density wrong by 1e-6 is refined on the device by the preconditioned
        GMRES of _fixed_point (it does not iterate for a converged density)."""
        state = chart().state
        noise = gaussian(21, D, D)
        rough = replace(state, rho=state.rho + 1e-6 * (noise + adj(noise)))
        reference, device = (scaled.ScaledState(rough, "density", xp=xp) for xp in BACKENDS)
        self.same(device.rh, reference.rh, "fixed point rh")
        self.same_host(device.fixed_point, reference.fixed_point, "fixed point report")
        self.assertGreater(device.fixed_point["steps"], 0)
        s = device.s_host
        self.assertLess(np.abs(s[:, None] * device.rh.get() * s[None, :] - state.rho).max(), 1e-10)

    def test_typed_maps(self):
        x = gaussian(4, D, 3, D)
        sigma, z = 1 + 0.3j * np.arange(3), np.array([0.2, 0.6, 0.9])
        for scaling in SCALINGS:
            reference, device = (shared("odd", scaling)[xp][0] for xp in BACKENDS)
            for a, b, dual in TYPES:
                name = f"maps {a}{b}{'*' if dual else ''}"
                self.same(device.lmap(cp.asarray(x), a, b, dual), reference.lmap(x, a, b, dual), name)
                expected = reference.nojump_inverse(x, a, b, sigma, z, dual)
                for arguments in ((cp.asarray(sigma), cp.asarray(z)), (sigma, z)):   # host node values are uploaded
                    self.same(device.nojump_inverse(cp.asarray(x), a, b, *arguments, dual), expected, name)
                self.same(device.zero_mode(a, b, dual), reference.zero_mode(a, b, dual), name)
                self.same(device.pairing(cp.asarray(x), a, b, dual), reference.pairing(x, a, b, dual), name)
                self.same(device.project(cp.asarray(x), a, b, dual), reference.project(x, a, b, dual), name)

    def test_solves_of_every_type(self):
        """Resolvent and pinned Poisson solves of the eight typed maps; the node values z and
        sigma may be scalars, host arrays or device arrays, and the reports are host arrays."""
        rhs = gaussian(5, D, 3, D)
        z = np.array([0.2, 0.6, 0.9])
        sigma = 1 + 1j * z * P
        for scaling in ("density", "skew"):
            reference, device = (shared("odd", scaling)[xp][0] for xp in BACKENDS)
            for a, b, dual in TYPES:
                name = f"solve {a}{b}{'*' if dual else ''}"
                for nodes in ((0.5, 1.0), (z, sigma), (z, 1.0)):
                    expected, report = reference.resolvent(rhs.copy(), a, b, *nodes, dual=dual)
                    forms = [nodes] if np.isscalar(nodes[0]) else [nodes, tuple(
                        cp.asarray(v) if isinstance(v, np.ndarray) else v for v in nodes)]
                    for form in forms:
                        x, info = device.resolvent(cp.asarray(rhs), a, b, *form, dual=dual)
                        self.same(x, expected, name)
                        self.same_info(info, report, name)
                for shift in (0.0, 1j * P, -1j * P):
                    expected, report = reference.poisson(rhs.copy(), a, b, shift, dual=dual)
                    x, info = device.poisson(cp.asarray(rhs), a, b, shift, dual=dual)
                    self.same(x, expected, name)
                    self.same_info(info, report, name)
                    self.assertIsInstance(info["residuals"], np.ndarray)

    def test_chunks_restarts_and_statistics(self):
        """A bounded Krylov basis (chunks of two systems), restarts every three steps, and
        the counters of SolverState."""
        rhs, z = gaussian(6, D, 5, D), np.linspace(0.1, 0.9, 5)

        def solve(xp, ss, basis):
            x, info = ss.resolvent(xp.asarray(rhs), "L", "H", z, 1 + 1j * z * P, restart=3)
            y, _ = ss.poisson(xp.asarray(rhs), "H", "L", -1j * P, dual=True, restart=5)
            return x, y, info, dict(ss.stats)

        chunk = 2 * (3 + 1) * D * D * 16
        (x0, y0, info0, stats0), (x1, y1, info1, stats1) = (
            solve(xp, *fresh(xp, max_basis_bytes=chunk)) for xp in BACKENDS)
        self.same(x1, x0, "chunks resolvent")
        self.same(y1, y0, "chunks poisson")
        self.same_info(info1, info0, "chunks report")
        self.same_host(stats1, stats0, "chunks statistics")
        self.assertEqual(stats1["solves"], 10)
        self.assertGreater(info1["steps"], 3 * 3)       # three chunks, each restarted

    def test_gram_schmidt_branches(self):
        """xp.__name__ == 'cupy' selects the ReductionKernel projections; without
        ReductionKernel the generic sums are used. Both give the NumPy result."""
        rhs, z = gaussian(7, D, 3, D), np.array([0.2, 0.6, 0.9])
        reference, device = (shared()[xp][0] for xp in BACKENDS)
        expected, _ = reference.resolvent(rhs.copy(), "L", "H", z, 1 + 1j * z * P)
        calls = []

        class Counting(cp.ReductionKernel):
            def __call__(self, *args, **kw):
                calls.append(self.name)
                return super().__call__(*args, **kw)

        with mock.patch.dict(scaled._PROJECTIONS, clear=True), mock.patch.object(cp, "ReductionKernel", Counting):
            x, _ = device.resolvent(cp.asarray(rhs), "L", "H", z, 1 + 1j * z * P)
        self.same(x, expected, "kernel projections")
        self.assertEqual(set(calls), {"rcmps_gmres_dot", "rcmps_gmres_combine"})
        kernel = cp.ReductionKernel
        with mock.patch.dict(scaled._PROJECTIONS, clear=True):
            del cp.ReductionKernel
            try:
                self.assertFalse(hasattr(cp, "ReductionKernel"))
                x, _ = device.resolvent(cp.asarray(rhs), "L", "H", z, 1 + 1j * z * P)
            finally:
                cp.ReductionKernel = kernel
        self.same(x, expected, "generic projections")

    def test_roundoff_floor(self):
        """A tolerance below the attainable accuracy: the iteration ends on the roundoff-floor
        rules, which compare downloaded true residual norms on the host."""
        rhs, z = gaussian(8, D, 3, D), np.array([0.2, 0.6, 0.9])
        reference, device = (shared()[xp][0] for xp in BACKENDS)
        expected, report = reference.resolvent(rhs.copy(), "L", "H", z, 1 + 1j * z * P, tol=1e-17, accept=1e4)
        x, info = device.resolvent(cp.asarray(rhs), "L", "H", z, 1 + 1j * z * P, tol=1e-17, accept=1e4)
        self.same(x, expected, "floor resolvent")
        self.same_info(info, report, "floor report")
        self.assertTrue((info["residuals"] > 1e-17).any())


class Actions(Case):
    """kinetic.py, potential.py, ode.py, hamiltonian.py, packed.py."""

    def test_kinetic_action(self):
        """Both kinetic specs and a quadratic monomial on a batch of two tangents."""
        terms = kinetic_terms(PHI4 + [(0.3, monomial_spec(2))])
        xs = coefficients()

        def run(xp, ss, basis):
            action = KineticAction(ss, P, terms, NODES)
            frame = TangentFrame(ss, P, basis.tangents(xs))
            out, a0 = action.bulk(frame)
            return action, frame, out, a0, action.action(frame)

        (a0, f0, *rest0), (a1, f1, *rest1) = both(run)
        for name in ("wh", "vh", "rw"):
            self.same(getattr(f1, name), getattr(f0, name), f"kinetic frame {name}")
        self.same_info(f1.rw_info, f0.rw_info, "kinetic frame report")
        for name, x, y in zip(("bulk", "a0", "action"), rest1, rest0):
            self.same(x, y, f"kinetic {name}")
        self.same(a1.boundary.d, a0.boundary.d, "kinetic boundary d")
        self.same(a1.boundary.g, a0.boundary.g, "kinetic boundary g")
        for c in a0.sources:
            for key in (False, True):
                self.same(a1.x0[c][key], a0.x0[c][key], "kinetic x0")
                self.same(a1.a1[c][key], a0.a1[c][key], "kinetic a1")
        for key, value in a0.kappa.items():
            self.same(a1.kappa[key], value, "kinetic kappa")
        self.same_host(a1.stats, a0.stats, "kinetic statistics")
        self.same_host((a1.z_host, a1.weights_host), (a0.z_host, a0.weights_host), "kinetic grid")

    def test_potential_action(self):
        """One ODE term through ode.integrate_host: a cubic monomial and, for the commutator
        lift and its pullback, a kinetic spec."""
        xs = coefficients(count=1)
        for label, spec in (("cubic", monomial_spec(3)), ("commutator", kinetic_spec(True))):
            def run(xp, ss, basis):
                potential = PotentialAction(ss, P, spec, STATE.integrator, integrate=integrate_host)
                frame = TangentFrame(ss, P, basis.tangents(xs))
                return (potential, *potential.bulk(frame))

            (p0, out0, a0), (p1, out1, a1) = both(run)
            self.same(out1, out0, f"potential {label} bulk")
            self.same(a1, a0, f"potential {label} a0")
            self.same(p1.boundary.d, p0.boundary.d, f"potential {label} boundary d")
            self.same(p1.boundary.g, p0.boundary.g, f"potential {label} boundary g")
            self.assertEqual((p1.trajectories, p0.trajectories), (2, 2))

    def test_trajectory(self):
        """integrate_host on the device: samples at given nodes and at +a, step statistics."""
        wh = gaussian(10, D, D)
        nodes = np.array([-3.0, -0.5, 0.0, 0.25, 2.0])      # t = 0 ends a step: no dense output there
        results = []
        for xp in BACKENDS:
            ss = shared()[xp][0]
            frame = TangentFrame(ss, P, xp.asarray(wh[:, None, :]))
            tangent = tuple(xp.ascontiguousarray(m[:, 0, :]) for m in (frame.wh, frame.vh, frame.rw))
            statistics = {}
            samples = integrate_host(ActionSystem(ss, P, monomial_spec(2), tangent), nodes,
                                     math.sqrt(STATE.integrator.max_point), STATE.integrator, statistics=statistics)
            results.append((samples, statistics))
        self.same(results[1][0], results[0][0], "trajectory samples")
        self.same_host(results[1][1], results[0][1], "trajectory statistics")
        self.assertEqual(results[1][0].shape, (6, 4, 3, D, D))

    def test_hamiltonian_action(self):
        """The complete phi^4 action on ActionSystem for one tangent (D, D) and a batch (D, 2, D)."""
        xs = coefficients()

        def run(xp, ss, basis):
            action = HamiltonianAction(ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES,
                                       integrate=integrate_host)
            wh = basis.tangents(xs)
            return action, action(xp.ascontiguousarray(wh[:, 0, :])), action(wh)

        (h0, single0, batch0), (h1, single1, batch1) = both(run)
        self.assertEqual((single1.shape, batch1.shape, h1.calls, h0.calls), ((D, D), (D, 2, D), 3, 3))
        self.same(single1, single0, "hamiltonian single")
        self.same(batch1, batch0, "hamiltonian batch")
        self.same(h1.boundary.d, h0.boundary.d, "hamiltonian boundary d")
        self.same(h1.boundary.g, h0.boundary.g, "hamiltonian boundary g")
        with self.assertRaises(ValueError):
            h1(cp.asarray(np.full((D, D), np.nan + 0j)))
        with self.assertRaises(TypeError):
            h1(gaussian(11, D, D))            # a host tangent is not uploaded silently

    def test_packed_action(self):
        """phi^4 + phi^3 on packed.PackedActionSystem in both sectors, and the packed
        system itself (right-hand side, unpack, pack, and the pullback in chunks of
        3 mirrored node pairs of 16 nodes) on the device."""
        terms = PHI4 + [(0.7, monomial_spec(3))]
        points, weights = quadrature(IntegratorParameters(n_quad=8, max_point=4.0))
        for sector in ("odd", "even"):
            xs = coefficients(sector, 1)
            y, samples = gaussian(12, 26, D // 2, D // 2), gaussian(13, 17, 26, D // 2, D // 2)

            def run(xp, ss, basis):
                action = HamiltonianAction(ss, P, terms, integrator=STATE.integrator, kinetic_nodes=NODES,
                                           integrate=integrate_host, system=packed.PackedActionSystem)
                wh = xp.ascontiguousarray(basis.tangents(xs)[:, 0, :])
                frame = action.frame(wh)
                tangent = tuple(xp.ascontiguousarray(m[:, 0, :]) for m in (frame.wh, frame.vh, frame.rw))
                system = packed.PackedActionSystem(ss, P, monomial_spec(4), tangent)
                state = xp.asarray(y)
                unpacked = system.unpack(state)
                bulk, a0, (a1, xout) = system.pullback(xp.asarray(samples), points, weights, chunk=3)
                return (action(wh), system, system.rhs(state, 0.7, 0.3), unpacked, system.pack(unpacked),
                        system.constants, system.initial, bulk, a0, a1, xout)

            (f0, s0, *arrays0), (f1, s1, *arrays1) = both(run, sector)
            self.same(f1, f0, f"packed action {sector}")
            names = ("rhs", "unpack", "pack", "constants", "initial", "pullback", "pullback-a0", "pullback-a1",
                     "pullback-xout")
            for name, x, y0 in zip(names, arrays1, arrays0, strict=True):
                self.same(x, y0, f"packed {name}")
            self.assertEqual((s1.sector, s0.sector, len(s1.blocks)), (sector, sector, 26))
            self.same_host(s1.tables(), s0.tables(), "packed tables")
            self.same_host(s1.live, s0.live, "packed live")
            self.assertTrue(np.array_equal(arrays1[2].get(), y))      # pack(unpack(y)) = y

    def test_fused_backend_needs_a_gpu(self):
        """The fused trajectory kernel is CUDA: on the stand-in it stops at the first device
        query with NotImplementedError, before any array work."""
        from rcmps_gpu.excitations import fused
        ss, basis = shared()[cp]
        with self.assertRaisesRegex(NotImplementedError, "needs a real GPU"):
            HamiltonianAction(ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES,
                              integrate=fused.fused_integrator(cp), system=packed.PackedActionSystem)

    def test_scipy_integrators_are_refused(self):
        """The SciPy reference integrators (the defaults of the two low-level classes) stop
        with a ValueError on a device state instead of handing device arrays to SciPy."""
        ss, basis = shared()[cp]
        wh = basis.tangent(coefficients(count=1)[0])
        with self.assertRaisesRegex(ValueError, "CPU only"):
            HamiltonianAction(ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES)
        element = bilinear.BilinearElement(ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES)
        with self.assertRaisesRegex(ValueError, "CPU only"):
            element.element(wh, wh)


class Bilinear(Case):
    """bilinear.py and the packed diagonal pair (packed_pair.py, bilinear_fused.py)."""

    def test_pair(self):
        """Environments and the unit maps of ExcitationPair for two different tangents."""
        uh, wh = gaussian(13, D, D), gaussian(14, D, D)
        x = gaussian(15, 4, D, 2, D)

        def run(xp, ss, basis):
            pair = bilinear.Pair(ss, P, xp.asarray(uh), xp.asarray(wh))
            state = xp.asarray(x)
            out = dict(overlap=pair.overlap, right=pair.right, left=pair.left, vh=pair.vh, vbrah=pair.vbrah,
                       transfer=pair.transfer(state), contract=pair.contract(state),
                       range=pair.contract(pair.transfer(pair.right)))
            for commutator in (False, True):
                operators = pair.operators(commutator)
                for left in (False, True):
                    out[f"source{commutator:d}{left:d}"] = pair.source(state, operators, left)
                    for k, m in enumerate(operators[left]):
                        out[f"operator{commutator:d}{left:d}{k}"] = m
            return out, pair.infos

        (out0, infos0), (out1, infos1) = both(run)
        for name, value in out0.items():
            self.same(out1[name], value, f"pair {name}")
        for info1, info0 in zip(infos1, infos0):
            self.same_info(info1, info0, "pair report")

    def test_element(self):
        """Matrix element and energy through ode.integrate_host: Python complex numbers.
        Tangents may be given on the device or on the host (Pair uploads them)."""
        uh, wh = gaussian(13, D, D), gaussian(14, D, D)

        def run(xp, ss, basis):
            element = bilinear.BilinearElement(ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES,
                                               integrate=integrate_host)
            values = (element.element(xp.asarray(uh), xp.asarray(wh)), element.energy(wh))
            return values, element.stats, (element.trajectories, element.elements), element.ground

        (v0, stats0, counts0, ground0), (v1, stats1, counts1, ground1) = both(run)
        self.same_host(v1, v0, "element values")
        self.assertEqual([type(v) for v in v1], [complex, complex])
        self.same_host(stats1, stats0, "element statistics")
        self.assertEqual((counts1, counts0), ((2, 2), (2, 2)))
        self.same(ground1, ground0, "element ground")

    def test_packed_diagonal_pair(self):
        """The validation energy of the fused backend: FusedBilinearElement with the packed
        four-block system, here on the host-controlled integrator; an off-diagonal element
        falls back to the unpacked route."""
        x = coefficients(count=1)[0]

        uh = gaussian(13, D, D)

        def run(xp, ss, basis):
            element = bilinear_fused.bound(integrate_host)(
                ss, P, PHI4, integrator=STATE.integrator, kinetic_nodes=NODES, integrate=integrate_host)
            wh = basis.tangent(x)
            values = (element.energy(wh), element.element(xp.asarray(uh), wh))    # packed, then the fallback
            return values, element.fused_trajectories, element.trajectories

        (v0, *counts0), (v1, *counts1) = both(run)
        self.same_host(v1, v0, "packed pair values")
        self.assertEqual((counts1, counts0), ([1, 2], [1, 2]))


class Driver(Case):
    """eigsolve.py: white coordinates on the host, tangents and actions on the device."""

    def test_white_basis(self):
        for sector in ("odd", "even"):
            n = len(chart(sector))
            xs, fh, w = gaussian(16, 3, n), gaussian(17, D, 3, D), gaussian(18, D, D)
            for scaling in SCALINGS:
                (ss0, b0), (ss1, b1) = (shared(sector, scaling)[xp] for xp in BACKENDS)
                name = f"basis {sector}"
                for attribute in ("bh", "bhd", "metric", "rows", "cols"):
                    self.same(getattr(b1, attribute), getattr(b0, attribute), name)
                self.same_host((b1.defect, b1.kept, b1.scaling, len(b1)),
                               (b0.defect, b0.kept, b0.scaling, len(b0)), name)
                self.same(b1.tangents(xs), b0.tangents(xs), name)
                self.same(b1.tangent(xs[0]), b0.tangent(xs[0]), name)
                self.same(b1.hat(w), b0.hat(w), name)
                self.same_host(b1.covectors(cp.asarray(fh)), b0.covectors(fh), name)
                self.same_host(b1.covector(cp.asarray(fh[:, 0, :])), b0.covector(fh[:, 0, :]), name)
                self.same_host(b1.coordinates_batch(cp.asarray(fh)), b0.coordinates_batch(fh), name)
                self.same_host(b1.coordinates(cp.asarray(w)), b0.coordinates(w), name)
                self.same_host(b1.physical_tangent(cp.asarray(w)), b0.physical_tangent(w), name)
                self.same_host(b1.physical(xs[0]), b0.physical(xs[0]), name)
                self.assertIsInstance(b1.covector(cp.asarray(w)), np.ndarray)

    def test_preconditioner_and_field_seed(self):
        """Coarse kinetic matrix built three tangents at a time, and the field insertion:
        host arrays, equal to the NumPy ones."""
        r = gaussian(19, len(chart()))

        def run(xp, ss, basis):
            pre = es.kinetic_preconditioner(basis, ss, P, PHI4, nodes=8, chunk=3)
            return ({k: pre[k] for k in ("matrix", "values", "vectors", "skew", "floor", "nodes")}, pre["apply"](r),
                    es.field_seed(basis, ss, P), es.field_seed(basis, ss, 0.0))

        reference, device = both(run)
        self.same_host(device, reference, "preconditioner")
        self.assertGreater(np.linalg.norm(device[2]), 0.5)        # the field insertion is odd

    def test_eigsolve_with_validation(self):
        """excitation_eigsolve at p = 0.5 with validate=True: Davidson search, refined action
        and both bilinear energies on the device; every reported number is a host object."""
        lines = {np: [], cp: []}
        reference, device = (es.excitation_eigsolve(STATE, 0.5, PHI4, kinetic_nodes=NODES, xp=xp,
                                                    log=lines[xp].append) for xp in BACKENDS)
        self.assertIs(device["basis"].ss.xp, cp)
        self.assertTrue(device["accepted"] and device["converged"])
        skipped = ("timings", "coordinates", "basis")
        self.same_host({k: v for k, v in device.items() if k not in skipped},
                       {k: v for k, v in reference.items() if k not in skipped}, "eigsolve")
        self.assertEqual([line.split()[0] for line in lines[cp]], [line.split()[0] for line in lines[np]])
        self.assertLess(device["validation"][0]["energy_spread"], 1e-8)

    def test_eigsolve_supplied_tangent_and_refinement(self):
        """A start tangent in the gauge of the state goes host -> device -> white coordinates;
        three Davidson operations, then one correction under a second operator."""
        request = dict(kinetic_nodes=NODES, validate=False, start=gaussian(20, D, D), refine_steps=1,
                       refined=replace(STATE.integrator), refined_kinetic_nodes=NODES,
                       davidson_options=DavidsonOptions(maxops=3, stall_window=0))
        reference, device = (es.excitation_eigsolve(STATE, 0.5, PHI4, xp=xp, **request) for xp in BACKENDS)
        self.assertEqual((device["start"], device["stop_reason"], len(device["corrections"])),
                         ("supplied", "maxops", 2))
        skipped = ("timings", "coordinates", "basis")
        self.same_host({k: v for k, v in device.items() if k not in skipped},
                       {k: v for k, v in reference.items() if k not in skipped}, "eigsolve supplied")

    def test_eigsolve_packed_even_sector(self):
        """The even sector at p = 0 on the packed system: the field insertion vanishes, so
        the search starts from the coarse kinetic modes; two levels, no validation."""
        request = dict(nev=2, sector="even", kinetic_nodes=NODES, validate=False, system=packed.PackedActionSystem)
        reference, device = (es.excitation_eigsolve(STATE, 0.0, PHI4, xp=xp, **request) for xp in BACKENDS)
        self.assertEqual((device["start"], device["sector"], device["converged"]), ("kinetic", "even", True))
        skipped = ("timings", "coordinates", "basis")
        self.same_host({k: v for k, v in device.items() if k not in skipped},
                       {k: v for k, v in reference.items() if k not in skipped}, "eigsolve packed")


@slow
class Mutations(unittest.TestCase):
    """The modules pass the other classes unchanged, so this shows that those classes can
    fail: each entry of MUTATIONS removes one upload or download (or leaves memory
    uninitialized) in excitations/<module> of a temporary copy of the rcmps_gpu package,
    and the named test, run in a subprocess that imports rcmps_gpu from that copy (checked),
    must fail with the stand-in's TypeError or a nonfinite result."""

    # Run in the child: report where rcmps_gpu comes from, then run the named test.
    CHILD = ("import os, sys, unittest, rcmps_gpu\n"
             "print('rcmps_gpu imported from', os.path.dirname(os.path.realpath(rcmps_gpu.__file__)),"
             " file=sys.stderr, flush=True)\n"
             "unittest.main(module=None, argv=['unittest', sys.argv[1]])\n")

    def test_planted_bugs_are_detected(self):
        tests = os.path.dirname(os.path.abspath(__file__))
        package = os.path.dirname(os.path.abspath(es.__file__))       # .../rcmps_gpu/excitations
        root = os.path.dirname(package)                                # .../rcmps_gpu
        for module, old, new, test in MUTATIONS:
            with self.subTest(module=module, bug=new):
                with open(os.path.join(package, module)) as stream:
                    source = stream.read()
                if source.count(old) != 1:
                    self.skipTest(f"excitations/{module} no longer has the line {old!r}")
                with tempfile.TemporaryDirectory() as work:
                    copy = os.path.join(work, "src", "rcmps_gpu")
                    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    with open(os.path.join(copy, "excitations", module), "w") as stream:
                        stream.write(source.replace(old, new))
                    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                                       PYTHONPATH=os.pathsep.join([os.path.join(work, "src"), tests]))
                    run = subprocess.run([sys.executable, "-c", self.CHILD, f"test_device_strict.{test}"], cwd=work,
                                         capture_output=True, text=True, env=environment)
                    origin = re.search(r"^rcmps_gpu imported from (.*)$", run.stderr, re.M)
                    self.assertIsNotNone(origin, run.stderr[-2000:])
                    self.assertEqual(os.path.realpath(origin.group(1)), os.path.realpath(copy))
                self.assertNotEqual(run.returncode, 0, "the planted bug was not detected")
                self.assertRegex(run.stderr, "TypeError|nonfinite")


if __name__ == "__main__":
    unittest.main()
