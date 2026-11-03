"""Tests of the excitation systems of the sector all (any state): packed.UnpackedActionSystem
and packed_pair.UnpackedDiagonalPair, the trivial blocking (one D x D block per live matrix)
of the Z2-packed systems, which the fused kernel integrates as it does those.

States: random_state(D) without symmetry at D = 3 and 5 (odd D included), and the Z2 state
random_state(4, z2=True) with a tangent of no definite class. S = 1 and S = sqrt(diag rho).

Tables       (1) shapes: 7, 11, 15 blocks and 43, 74, 105 products for :phi^2:, :phi^3:,
             :phi^4:, d = D, sector 'all'; the widths of fused.PHI4_UNPACKED_KERNELS; the
             kernel's reading of the tables (fused.emulate_rhs) equals rhs.
RightHandSide (2) unpack(rhs(y)) equals ActionSystem.rhs(unpack(y)) on every live matrix.
Action       (3) the bulk action and boundary covector equal those of ActionSystem
             (ode.integrate_host, 1e-10); on the Z2 state, for a tangent of one class, those
             of PackedActionSystem; the Hamiltonian action equals the dense oracle (1e-10).
Pair         (4) diagonal_observable on UnpackedDiagonalPair equals BilinearElement.observable,
             and the pair kernel shape is that of fused.PHI4_UNPACKED_KERNELS.
Solve        (5) excitation_eigsolve of the D = 3 state at p = 0.5 with the unpacked systems
             (host-controlled integrator) gives the dense spectrum (1e-8), and the four
             validation energies, the bilinear ones through the unpacked pair system, are
             those of the reference route (1e-10). Window 14 is too small for this state
             (window 24 moves the level by 2.2e-7): neither route accepts the level.
Compile      (6) with NVRTC (test_fused.nvrtc_include), the kernels of the shapes of
             fused.PHI4_UNPACKED_KERNELS at d = 5 and 64 compile and pass the PTX audit with
             either tile path, and assemble to cubins (scalar tiles sm_70, tensor cores sm_80).
"""
from __future__ import annotations

import functools
import itertools
import math
import unittest

import numpy as np

import dense_reference as ref
from rcmps_gpu.excitations import bilinear_fused, fused
from rcmps_gpu.excitations import eigsolve as es
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations.bilinear import BilinearElement
from rcmps_gpu.excitations.hamiltonian import HamiltonianAction
from rcmps_gpu.excitations.kinetic import TangentFrame
from rcmps_gpu.excitations.models import monomial_spec
from rcmps_gpu.excitations.ode import integrate_host
from rcmps_gpu.excitations.packed import PackedActionSystem, UnpackedActionSystem
from rcmps_gpu.excitations.packed_pair import UnpackedDiagonalPair, diagonal_observable
from rcmps_gpu.excitations.potential import ActionSystem, PotentialAction
from rcmps_gpu.excitations.scaled import ScaledState, stack
from rcmps_gpu.excitations.states import IntegratorParameters, adj, random_state
from test_fused import compile_cubin, nvrtc_include

norm = np.linalg.norm
INTEGRATOR = IntegratorParameters(n_quad=24, max_point=12.0, atol=1e-13, rtol=1e-12)
TOL = 1e-13
NODES = 16
CASES = [("plain", 3), ("plain", 5), ("z2", 4)]
SCALINGS, MOMENTA, DEGREES = ("none", "density"), (0.0, 0.5), (2, 4)
SHAPES = {2: (7, 43), 3: (11, 74), 4: (15, 105)}


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


@functools.lru_cache(maxsize=None)
def scaled(kind, dim, scaling):
    state = random_state(dim, seed=7, scaling=0.6, z2=kind == "z2").with_integrator(INTEGRATOR)
    return ScaledState(state, scaling)


@functools.lru_cache(maxsize=None)
def frame(kind, dim, scaling, p):
    """A tangent of no definite class: for the Z2 state it mixes both sectors."""
    return TangentFrame(scaled(kind, dim, scaling), p, stack([gaussian(5, dim, dim)]), tol=TOL)


def tangent(kind, dim, scaling, p):
    f = frame(kind, dim, scaling, p)
    return tuple(np.ascontiguousarray(m[:, 0, :]) for m in (f.wh, f.vh, f.rw))


def random_blocks(system, seed):
    """Random blocks, the Hermitian live matrices (x_k, a1_k) Hermitian."""
    y = gaussian(seed, *system.shape)
    for b, (kind, *_) in enumerate(system.layout):
        if kind in (0, 2):
            y[b] = (y[b] + adj(y[b])) / 2
    return y


class Case(unittest.TestCase):
    def assertClose(self, x, y, rtol, msg="", scale=None):
        x, y = np.asarray(x), np.asarray(y)
        scale = max(1.0, norm(y)) if scale is None else scale
        self.assertLessEqual(norm(x - y), rtol * scale, msg)


class Tables(Case):
    def test_shapes_and_kernel_reading(self):
        for (kind, dim), n in itertools.product(CASES, (2, 3, 4)):
            ss = scaled(kind, dim, "density")
            system = UnpackedActionSystem(ss, 0.5, monomial_spec(n), tangent(kind, dim, "density", 0.5))
            t = system.tables()
            msg = f"{kind} D={dim} n={n}"
            self.assertEqual((system.d, system.sector, t["n_blocks"], t["products"]), (dim, "all", *SHAPES[n]), msg)
            self.assertEqual(system.shape, (SHAPES[n][0], dim, dim), msg)
            self.assertFalse(any(partner for *_, partner in system.layout), msg)
            device = fused.kernel_tables(t)
            if n == 4:
                self.assertEqual((len(device["fixed"]), device["left_constants"].shape[1],
                                  device["temp_right"].shape[1]), fused.PHI4_UNPACKED_KERNELS["action"], msg)
            y = random_blocks(system, 3)
            for jac, source in ((1.0, 0.0), (0.4, 0.9)):
                self.assertClose(fused.emulate_rhs(device, system.constants, y, jac, source),
                                 system.rhs(y, jac, source), 1e-13, msg)

    def test_no_symmetry_is_required(self):
        """Odd D, no zero blocks: nothing is refused; the gauge of the tangent still is."""
        ss = scaled("plain", 5, "none")
        wh, vh, rw = tangent("plain", 5, "none", 0.0)
        UnpackedActionSystem(ss, 0.0, monomial_spec(4), (wh, vh, rw))
        with self.assertRaisesRegex(ValueError, "the packed action system needs an even bond dimension"):
            PackedActionSystem(ss, 0.0, monomial_spec(4), (wh, vh, rw))
        with self.assertRaisesRegex(ValueError, "gauge-fixed tangent"):
            UnpackedActionSystem(ss, 0.0, monomial_spec(4), (wh, vh + 1e-6, rw))


class RightHandSide(Case):
    def test_matches_unpacked(self):
        for (kind, dim), scaling, p, n in itertools.product(CASES, SCALINGS, MOMENTA, DEGREES):
            ss, spec, w = scaled(kind, dim, scaling), monomial_spec(n), tangent(kind, dim, scaling, p)
            system, full = UnpackedActionSystem(ss, p, spec, w), ActionSystem(ss, p, spec, w)
            y = random_blocks(system, 11)
            unpacked = system.unpack(y)
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                want = full.rhs(unpacked, jac, source)
                got = system.unpack(system.rhs(y, jac, source), constants=False)
                self.assertFalse(got[~system.live].any())
                for kd, k in zip(*np.nonzero(system.live)):
                    self.assertClose(got[kd, k], want[kd, k], 1e-12,
                                     f"{kind} D={dim} S={scaling} p={p} n={n} block {kd},{k} jac={jac}")


class Action(Case):
    def test_bulk_equals_the_reference_system(self):
        for kind, dim in CASES:
            for scaling, p, n in (("density", 0.5, 4), ("none", 0.0, 2)):
                ss, spec, f = scaled(kind, dim, scaling), monomial_spec(n), frame(kind, dim, scaling, p)
                results = [PotentialAction(ss, p, spec, INTEGRATOR, integrate=integrate_host, system=system,
                                           tol=TOL).bulk(f) for system in (None, UnpackedActionSystem)]
                (f0, a0), (f1, a1) = results
                msg = f"{kind} D={dim} S={scaling} p={p} n={n}"
                self.assertClose(f1, f0, 1e-10, msg)
                self.assertClose(a1, a0, 1e-10, msg, max(norm(a0), norm(f0)))

    def test_a_class_tangent_of_a_Z2_state_gives_the_packed_action(self):
        ss, d = scaled("z2", 4, "density"), 2
        w = gaussian(9, 4, 4)
        odd = w.copy()
        odd[:d, d:] = odd[d:, :d] = 0
        f = TangentFrame(ss, 0.5, stack([odd, w - odd]), tol=TOL)
        packed, unpacked = (PotentialAction(ss, 0.5, monomial_spec(4), INTEGRATOR, integrate=integrate_host,
                                            system=system, tol=TOL).bulk(f)
                            for system in (PackedActionSystem, UnpackedActionSystem))
        self.assertClose(unpacked[0], packed[0], 1e-10)
        self.assertClose(unpacked[1], packed[1], 1e-10, scale=max(norm(packed[1]), norm(packed[0])))

    def test_hamiltonian_action_equals_the_dense_oracle(self):
        terms = models.phi4_terms(3.0)
        for kind, dim in (("plain", 3), ("z2", 4)):
            ss = scaled(kind, dim, "density")
            action = HamiltonianAction(ss, 0.5, terms, integrator=INTEGRATOR, kinetic_nodes=NODES, tol=TOL,
                                       integrate=integrate_host, system=UnpackedActionSystem)
            w = gaussian(5, dim, dim)
            fh = action(stack([w]))[:, 0, :]
            ctx, s = ref.make_context(ss.state, 0.5), ss.s_host
            want = ref.hamiltonian_action(ctx, w / s[None, :], terms, kinetic_nodes=NODES)
            self.assertClose(fh * s[None, :], want, 1e-10, f"{kind} D={dim}")


class Pair(Case):
    def test_diagonal_observable(self):
        for (kind, dim), p, n in itertools.product(CASES, MOMENTA, DEGREES):
            ss = scaled(kind, dim, "density")
            element = BilinearElement(ss, p, [], integrator=INTEGRATOR, integrate=integrate_host, tol=TOL)
            wh = gaussian(5, dim, dim) * ss.s_host[None, :]
            q = element.pair(wh, wh)
            got = diagonal_observable(q, monomial_spec(n), INTEGRATOR, integrate_host, UnpackedDiagonalPair)
            want = element.observable(q, monomial_spec(n))
            self.assertLess(abs(got - want), 1e-10 * max(1.0, abs(want)), f"{kind} D={dim} p={p} n={n}")
            system = UnpackedDiagonalPair(q, monomial_spec(n))
            self.assertEqual((system.d, system.sector), (dim, "all"))
            if n == 4:
                device = fused.kernel_tables(system.tables())
                self.assertEqual((len(device["fixed"]), device["left_constants"].shape[1],
                                  device["temp_right"].shape[1]), fused.PHI4_UNPACKED_KERNELS["pair"])


class Solve(Case):
    def test_levels_without_symmetry(self):
        """The D = 3 state of test_eigsolve.test_no_parity (no parity, full space of dimension
        9) at p = 0.5: the unpacked systems on the host-controlled integrator give the dense
        spectrum and, with the validation through UnpackedDiagonalPair, the four energies of
        the reference route (ActionSystem, bilinear.BilinearElement)."""
        window = dict(max_point=14, n_quad=32, atol=1e-13, rtol=1e-12)
        state = random_state(3, seed=5, scaling=0.5).with_integrator(IntegratorParameters(**window))
        terms = models.phi4_terms(1.0)
        dense = ref.spectrum(state, 0.5, terms, nev=1, assembly="action", kinetic_nodes=32)
        results = [es.excitation_eigsolve(state, 0.5, terms, kinetic_nodes=32, **options) for options in (
            {}, dict(integrate=integrate_host, system=UnpackedActionSystem,
                     bilinear_element=bilinear_fused.bound(integrate_host, UnpackedDiagonalPair)))]
        reference, unpacked = results
        self.assertEqual((unpacked["sector"], unpacked["dimension"], unpacked["converged"]), ("all", 9, True))
        self.assertLess(abs(unpacked["energies"][0] - dense["energies"][0]), 1e-8)
        names = ("standard_energy", "refined_energy", "bilinear_energy", "refined_bilinear_energy")
        (v0,), (v1,) = reference["validation"], unpacked["validation"]
        for name in names:
            self.assertLess(abs(v1[name] - v0[name]), 1e-10, name)
        self.assertLess(abs(v1["bilinear_energy"] - v1["standard_energy"]), 1e-9)
        self.assertEqual((v0["accepted"], v1["accepted"]), (False, False))


class Compile(Case):
    def test_unpacked_kernels_compile_and_assemble(self):
        include, reason = nvrtc_include()
        if include is None:
            self.skipTest(reason)
        for (name, (n_blocks, panels, temps)), d in itertools.product(fused.PHI4_UNPACKED_KERNELS.items(), (5, 64)):
            for use_mma, arch in ((False, "compute_70"), (True, "compute_80"), (True, "compute_90")):
                with self.subTest(name=name, d=d, arch=arch):
                    try:
                        ptx, _ = fused.compile_ptx(d, n_blocks, use_mma=use_mma, arch=arch, include_dirs=(include,),
                                                   panels=panels, temps=temps)
                    except OSError as error:
                        self.skipTest(f"libnvrtc cannot be loaded: {error}")
                    self.assertEqual(bool(fused.audit_ptx(ptx, use_mma=use_mma)["f64_mma_instructions"]), use_mma)
            for use_mma, arch in ((False, "sm_70"), (True, "sm_80")):
                status, cubin, log = compile_cubin(fused.kernel_source(d, n_blocks, use_mma=use_mma, panels=panels,
                                                                       temps=temps), arch, include)
                self.assertEqual(status, 0, log)
                self.assertEqual(cubin[:4], b"\x7fELF")


if __name__ == "__main__":
    unittest.main()
