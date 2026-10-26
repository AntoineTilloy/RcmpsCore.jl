"""The optimizer on states without Z2 symmetry (--symmetry none, the unpacked channels), on the CPU.

The kernel source is the packed one rewritten at its anchors, pinned by golden_cuda_unpacked.json;
the kernel's tables (per channel: Lindblad products, jump, insertions of the gathered channel,
fixed sources, scalar trace) reproduce the host equations of _UnpackedQuarticRHS, which are those
of _ReducedQuarticRHS; the integrated energy and covectors equal those of the full ten-channel
system built from the observable specification (the route of RcmpsCore's evaluate) and, on a Z2
state, those of the packed channels; the chunked pullback equals the unchunked one; <phi>
(phi_expectation) equals the ODE of the first monomial (RcmpsCore's phi_monomial(s, 1)); the
planner and the worker's random state follow --symmetry; and, with NVRTC (test_tiles.nvrtc),
the kernel compiles, passes the PTX audit and assembles for the targets of either tile path.
No CuPy, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rcmps_gpu import io
from rcmps_gpu.optimizer import cli, layout, worker
from rcmps_gpu.optimizer.runtime import generic_layout, numerics
from rcmps_gpu.optimizer.runtime import ode_backend as ob
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as backend

from test_tiles import compile_cuda, nvrtc

GOLDEN = json.loads(Path(__file__).with_name("golden_cuda_unpacked.json").read_bytes())


def random_state(D, seed, scale=1.0, z2=False):
    rng = np.random.default_rng(seed)
    n = D // 2 if z2 else D

    def draw():
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2)
    if z2:
        z = np.zeros((n, n), complex)
        k1, k2 = draw(), draw()
        r = np.block([[z, draw()], [draw(), z]])
        k = np.block([[k1 + k1.conj().T, z], [z, k2 + k2.conj().T]])
    else:
        k1 = draw()
        r, k = draw(), k1 + k1.conj().T
    r, k = scale * r / math.sqrt(D), scale * k / math.sqrt(D)
    q = -1j * k - r.conj().T @ r / 2
    rho, _ = numerics.stationary_state(q, r, tolerance=1e-13)
    return q, r, k, rho


def kernel_rhs(rhs, t, y):
    """The right-hand side as the CUDA kernel computes it from the tables of `rhs`."""
    d, jac, source = rhs.d, 2 * abs(t), ob.transformed_source(t)
    matrices = y[:rhs.matrix_size].reshape(7, d, d)
    indices, adjoints = rhs.gather_indices.reshape(-1), rhs.gather_adjoint.reshape(-1)

    def gather(slot):
        x = matrices[indices[slot]]
        return x.conj().T if adjoints[slot] else x
    out = np.empty_like(y)
    for c in range(7):
        x = matrices[c]
        base = jac * (rhs.ql[c] @ x + x @ rhs.qright[c]) + source * (
            rhs.coefficients[c] * (rhs.ol[c] @ gather(7 + c) + gather(14 + c) @ rhs.oright[c]) + rhs.fixed[c])
        out[c * d * d:(c + 1) * d * d] = (base + jac * ((rhs.rl[c] @ gather(c)) @ rhs.rright[c])).reshape(-1)
    f3 = matrices[2]
    trace = np.sum(rhs.a.real * f3.real + rhs.a.imag * f3.imag + rhs.b.real * f3.T.real - rhs.b.imag * f3.T.imag)
    out[-1] = 4 * source * trace
    return out


def trajectory(rhs, half_window=0.8, n_quad=16):
    points, weights = ob.quadrature(half_window, n_quad)
    samples, _ = ob._integrate(rhs, np.r_[points, math.sqrt(half_window)], atol=1e-13, rtol=1e-12,
                               initial_step=None, max_steps=100000)
    return samples, points, weights


class Source(unittest.TestCase):
    def test_the_unpacked_source_is_the_packed_one_rewritten_at_its_anchors(self):
        for D in (1, 3, 5, 8, 9, 17, 64, 65, 129, 192, 256):
            for tiles in ("mma", "scalar"):
                with self.subTest(D=D, tiles=tiles):
                    source, packed = backend.unpacked_trajectory_source(D, tiles), backend.trajectory_source(D, tiles)
                    expected = packed
                    for old, new, count in backend.UNPACKED_REWRITES:
                        self.assertEqual(packed.count(old), count)
                        expected = expected.replace(old, new)
                    self.assertEqual(source.split("\n", 1)[1], expected)
                    self.assertTrue(source.startswith("// Unpacked layout"))
                    for old, _, _ in backend.UNPACKED_REWRITES:
                        self.assertNotIn(old, source)
                    self.assertNotIn("10*", source.replace("10*fabs", ""))
        for D in (0, 257, 2.0):
            with self.assertRaises(ValueError):
                backend.unpacked_trajectory_source(D)

    def test_unpacked_CUDA_options_and_geometry_equal_the_golden_hashes(self):
        actual = {str(D): dict(trajectory_sha256=hashlib.sha256(backend.unpacked_trajectory_source(D).encode()).hexdigest(),
                               scalar_trajectory_sha256=hashlib.sha256(
                                   backend.unpacked_trajectory_source(D, "scalar").encode()).hexdigest(),
                               options=list(generic_layout.compilation_options(D)),
                               geometry=generic_layout.unpacked_geometry(D)) for D in map(int, GOLDEN["sizes"])}
        self.assertEqual(actual, GOLDEN["sizes"])

    def test_geometry_of_every_D_and_its_plan(self):
        for D in range(1, 257):
            info = layout.bond_geometry(D, "none")
            self.assertEqual((info["D"], info["d"], info["layout"]), (D, D, "unpacked"))
            self.assertEqual(info["blocks"], 7 * info["warps"] + 1)
            self.assertEqual({k: v for k, v in info.items() if k not in ("D", "blocks", "layout")},
                             {k: v for k, v in layout.packed_geometry(D).items() if k != "blocks"})
        for D in (0, 257, True, 4.0):
            with self.subTest(D=D), self.assertRaises(ValueError):
                layout.bond_geometry(D, "none")
        with self.assertRaises(ValueError):
            layout.bond_geometry(8, "Z2")
        plan = layout.describe(64, 512, "auto", "none")
        self.assertEqual((plan["blocks"], plan["threads"], plan["symmetry"], plan["historical_shape"]),
                         (57, 256, "none", False))
        self.assertEqual(plan["packed_sample_bytes"], 1025 * (7 * 64 * 64 + 1) * 16)
        self.assertNotIn("symmetry", layout.describe(64))   # the Z2 plan is unchanged

    def test_configure_shared_memory_accepts_the_unpacked_grid(self):
        class Kernel:
            max_threads_per_block, shared_size_bytes = 1024, 0
            max_dynamic_shared_size_bytes = preferred_shared_memory_carveout = 0
        props = dict(major=8, minor=0, cooperativeLaunch=True, sharedMemPerBlock=49152,
                     sharedMemPerBlockOptin=166912, maxThreadsPerBlock=1024)
        for D in (16, 128):
            info = generic_layout.unpacked_geometry(D)
            kernel = Kernel()
            kernel.shared_size_bytes = 0 if generic_layout.dynamic_shared_bytes(D) else info["shared_bytes"]
            self.assertEqual(generic_layout.configure_shared_memory(kernel, props, info),
                             generic_layout.dynamic_shared_bytes(D))
            with self.assertRaisesRegex(ValueError, "geometry differs"):
                generic_layout.configure_shared_memory(kernel, props, dict(info, blocks=info["blocks"] + 1))


class Equations(unittest.TestCase):
    def test_the_kernel_tables_reproduce_the_host_equations(self):
        q, r, _, rho = random_state(5, 1)
        rhs = ob._UnpackedQuarticRHS(q, r, rho, np)
        rng = np.random.default_rng(2)
        y = rng.standard_normal(rhs.matrix_size + 1) + 1j * rng.standard_normal(rhs.matrix_size + 1)
        for t in (-1.3, -0.2, 0.0, 0.7):
            expected, actual = rhs(t, y), kernel_rhs(rhs, t, y)
            self.assertEqual(expected[-1].imag, 0.0)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13 * np.abs(expected).max())
            # The parent's host call, which keeps the imaginary part of the accumulator.
            parent = ob._ReducedQuarticRHS.__call__(rhs, t, y)
            np.testing.assert_array_equal(parent[:-1], expected[:-1])
            self.assertEqual(parent[-1].real, expected[-1].real)

    def test_energy_and_covectors_equal_the_ten_channel_system_and_the_packed_channels(self):
        cases = [("none", *random_state(5, 3, 1.5)), ("none", *random_state(6, 4)), ("z2", *random_state(6, 5, z2=True))]
        for symmetry, q, r, _, rho in cases:
            with self.subTest(symmetry=symmetry, D=len(r)):
                rhs = ob._UnpackedQuarticRHS(q, r, rho, np)
                samples, points, weights = trajectory(rhs)
                ((value, covector, environment),) = ob._unpacked_pullback(q, r, rhs, samples, points, weights)
                full = ob.evaluate_quartic(r, q, rho, 1., xp=np, half_window=0.8, n_quad=16)
                scale = max(1., abs(full["value"]))
                self.assertLess(abs(value - full["value"]), 1e-12 * scale)
                self.assertLess(np.abs(covector - full["covector"]).max(), 1e-11 * scale)
                self.assertLess(np.abs(environment - full["environment"]).max(), 1e-11 * scale)
                if symmetry == "z2":
                    packed = ob.evaluate_quartic(r, q, rho, 1., xp=np, half_window=0.8, n_quad=16, packed_z2=True)
                    self.assertLess(abs(value - packed["value"]), 1e-12 * scale)
                    self.assertLess(np.abs(covector - packed["covector"]).max(), 1e-11 * scale)

    def test_the_kernel_oracle_integrates_the_unpacked_channels(self):
        q, r, _, rho = random_state(7, 6)
        rhs = ob._UnpackedQuarticRHS(q, r, rho, np)
        points, _ = ob.quadrature(0.8, 16)
        nodes = np.r_[points, math.sqrt(0.8)]
        oracle, stats = backend.integrate_trajectory_numpy(rhs, nodes, atol=1e-13, rtol=1e-12)
        host, _ = ob._integrate(ob._UnpackedQuarticRHS(q, r, rho, np), nodes, atol=1e-13, rtol=1e-12,
                                initial_step=None, max_steps=100000)
        np.testing.assert_allclose(oracle, host, rtol=2e-10, atol=1e-11)
        q, r, _, rho = random_state(9, 7)
        with self.assertRaisesRegex(ValueError, "physical D<=8"):
            backend.integrate_trajectory_numpy(ob._UnpackedQuarticRHS(q, r, rho, np), nodes)

    def test_the_chunked_pullback_equals_the_unchunked_one(self):
        q, r, _, rho = random_state(6, 8)
        rhs = ob._UnpackedQuarticRHS(q, r, rho, np)
        samples, points, weights = trajectory(rhs)
        ((e, c, h),) = ob._reduced_quartic_pullback(q, r, rhs, samples, points, weights)
        gradient = numerics.finish_gradient(q, r, rho, c, h)[0]
        for chunk in (1, 3, 7, 64):
            ((ec, cc, hc),) = ob._unpacked_pullback_chunked(q, r, rhs, samples, points, weights, chunk)
            delta = numerics.finish_gradient(q, r, rho, cc, hc)[0] - gradient
            self.assertEqual(ec, e)
            self.assertLess(math.sqrt(numerics.metric(rho, delta, delta)), 1e-11)
            np.testing.assert_array_equal(hc, h)


class OnePointFunction(unittest.TestCase):
    def test_the_source_integral(self):
        self.assertEqual(ob.source_integral(40.), 2**-0.5)
        # 2^-1/2 minus twice the tail (mpmath, 40 digits): 4: 1.801910921847185e-3, 30: 2.3308e-15.
        self.assertAlmostEqual(ob.source_integral(4.), 2**-0.5 - 1.801910921847185e-3, delta=1e-16)
        self.assertAlmostEqual(ob.source_integral(30.), 2**-0.5 - 2.3308091814782017e-15, delta=2e-16)
        with self.assertRaises(ValueError):
            ob.source_integral(0.)

    def test_phi_is_the_ode_of_the_first_monomial(self):
        spec = ob._Spec(2, ((0, 1, True, 1.), (0, 1, False, 1.)))
        for (q, r, k, rho), window in ((random_state(5, 9), 2.), (random_state(4, 10, 2.), 30.)):
            ode = ob._evaluate(r, q, rho, [spec], [1.], xp=np, half_window=window, n_quad=32, atol=1e-13, rtol=1e-12,
                               initial_step=None, max_steps=100000, use_graph=False, progress=None)
            phi = ob.phi_expectation(r, rho, window)
            self.assertNotEqual(phi, 0.)
            self.assertLess(abs(ode["value"] - phi), 1e-12 * max(1., abs(phi)))
            self.assertAlmostEqual(phi, 2 * ob.source_integral(window) * np.trace(rho @ r).real, delta=1e-15)
        q, r, k, rho = random_state(6, 11, z2=True)
        self.assertEqual(ob.phi_expectation(r, rho, 30.), 0.)   # odd R, even rho


class Compile(unittest.TestCase):
    def setUp(self):
        lib, reason = nvrtc()
        if lib is None:
            self.skipTest(reason)

    def test_the_unpacked_kernel_compiles_passes_the_audit_and_assembles(self):
        """D = 5, 64 and 128 (static and dynamic panels), with the options of
        trajectory_ptx_report, for compute_60 to 90 (scalar tiles) and 80 to 90 (tensor cores);
        cubins for sm_70 (scalar tiles, D = 8 and 128) and sm_80 (tensor cores, D = 8)."""
        for D in (5, 64, 128):
            options = ("--std=c++17", "--device-c") + generic_layout.compilation_options(D)[1:]
            for tiles, targets in (("scalar", ("compute_60", "compute_70", "compute_75", "compute_80", "compute_86",
                                               "compute_90")), ("mma", ("compute_80", "compute_86", "compute_90"))):
                for arch in targets:
                    with self.subTest(D=D, tiles=tiles, arch=arch):
                        status, ptx, log = compile_cuda(backend.unpacked_trajectory_source(D, tiles), arch, options)
                        self.assertEqual(status, 0, log)
                        self.assertEqual(backend.audit_trajectory_ptx(ptx, tiles)["ptx_target"], arch.replace("compute", "sm"))
        for D, tiles, arch in ((8, "scalar", "sm_70"), (128, "scalar", "sm_70"), (8, "mma", "sm_80")):
            with self.subTest(D=D, tiles=tiles, arch=arch):
                options = ("--std=c++17", "--device-c") + generic_layout.compilation_options(D)[1:]
                status, cubin, log = compile_cuda(backend.unpacked_trajectory_source(D, tiles), arch, options, cubin=True)
                self.assertEqual(status, 0, log)
                self.assertEqual(cubin[:4], b"\x7fELF")


class Planning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, *arguments):
        return cli.prepare(cli.parser().parse_args(list(arguments)), cli.runtime_source_sha256())

    def state(self, name, D, z2, symmetry=None):
        q, r, k, rho = random_state(D, 12, z2=z2)
        return io.write_portable(self.root / name, io.StateRecord(D=D, model="phi4", coupling=3.0, R=r, K=k, rho=rho,
                                                                  symmetry=symmetry))

    def test_plans_with_and_without_symmetry(self):
        plan = self.plan("--D", "5", "--symmetry", "none")
        self.assertEqual((plan["schema_version"], plan["symmetry"], plan["layout"]["layout"], plan["layout"]["blocks"]),
                         (5, "none", "unpacked", 8))
        self.assertEqual(self.plan("--D", "8")["symmetry"], "z2")
        with self.assertRaisesRegex(ValueError, "even integer D >= 2"):
            layout.describe(5)
        with self.assertRaisesRegex(ValueError, r"integer D in \[1, 256\]"):
            layout.describe(257, symmetry="none")

    def test_a_Z2_run_needs_a_Z2_initial_state_and_a_resume_keeps_its_symmetry(self):
        general, symmetric = self.state("general", 6, False, "none"), self.state("symmetric", 6, True)
        with self.assertRaisesRegex(ValueError, r"not Z2 symmetric in the packed basis .*\(its metadata states "
                                                r"symmetry = 'none'\): use --symmetry none"):
            self.plan("--D", "6", "--initial", str(general))
        self.assertEqual(self.plan("--D", "6", "--initial", str(general), "--symmetry", "none")["symmetry"], "none")
        self.assertEqual(self.plan("--D", "6", "--initial", str(symmetric))["symmetry"], "z2")
        self.assertEqual(self.plan("--D", "6", "--initial", str(symmetric), "--symmetry", "none")["symmetry"], "none")
        for reduced, requested, recorded in ((True, "z2", "none"), (False, "none", "z2")):
            checkpoint = self.root / f"checkpoint_{recorded}.json"
            checkpoint.write_text(json.dumps(dict(stop_reason="iteration limit", reference=dict(D=6),
                                                  controls=dict(reduced=reduced, packed_z2=not reduced))))
            with self.subTest(recorded=recorded), self.assertRaisesRegex(
                    ValueError, f"cannot resume an optimizer with another --symmetry \\(the checkpoint's is {recorded}\\)"):
                self.plan("--D", "6", "--resume-from", str(checkpoint), "--symmetry", requested)

    def test_the_emitted_CUDA_of_a_run_without_symmetry(self):
        import contextlib
        import io as text
        target = self.root / "D24.cu"
        with contextlib.redirect_stdout(text.StringIO()) as printed:
            self.assertEqual(cli.main(["--D", "24", "--symmetry", "none", "--emit-cuda", str(target)]), 0)
        self.assertEqual(target.read_text(), backend.unpacked_trajectory_source(24))
        self.assertEqual(json.loads(printed.getvalue())["layout"]["layout"], "unpacked")

    def test_the_random_state_of_a_run_without_symmetry(self):
        plan = self.plan("--D", "5", "--symmetry", "none", "--seed", "7", "--coupling", "3.0")
        worker.random_reference(plan, self.root / "initial")
        record = io.read_portable(self.root / "initial")
        self.assertEqual((record.D, record.symmetry, record.source, record.coupling), (5, "none", "random state, seed 7", 3.0))
        self.assertTrue(record.R.all() and record.K.all())
        np.testing.assert_array_equal(record.K, record.K.conj().T)
        rng = np.random.default_rng(7)

        def draw():
            return (rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))) / math.sqrt(2)
        k1 = draw()
        np.testing.assert_array_equal(record.K, (k1 + k1.conj().T) / math.sqrt(5))
        np.testing.assert_array_equal(record.R, draw() / math.sqrt(5))
        self.assertLess(np.linalg.norm(numerics.lindblad(-1j * record.K - record.R.conj().T @ record.R / 2,
                                                         record.R, record.rho)), 1e-12)
        z2 = self.plan("--D", "6", "--seed", "7")
        worker.random_reference(z2, self.root / "z2")
        self.assertEqual(io.read_portable(self.root / "z2").symmetry, "Z2")


if __name__ == "__main__":
    unittest.main()
