"""State files (rcmps_gpu.io): the portable format 1, the research dialects it also reads,
the JLD2 schema of RcmpsCore, and the excitation loaders built on them. CPU only."""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import struct
import tempfile
import tomllib
import unittest

import numpy as np

from rcmps_gpu import io
from rcmps_gpu.excitations import states

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = Path(os.environ.get("RCMPS_TEST_FIXTURES") or REPOSITORY / "test" / "fixtures")


def complex_matrix(rng, d):
    return rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))


def full_record(d=6, seed=3):
    rng = np.random.default_rng(seed)
    return io.StateRecord(
        D=d, model="phi4", coupling=2.7, R=complex_matrix(rng, d), K=complex_matrix(rng, d),
        rho=complex_matrix(rng, d), gradient=complex_matrix(rng, d), energy=-0.30228143199435653,
        gradient_norm=2.900325420717565e-05, validation_passed=True,
        integration=io.Integration(1024, 40.0, 1e-14, 1e-13, 1_000_000),
        manifold=io.Manifold(1e-3, 8.41190161973086e-12, 1e-2),
        producer="test", source='a "quoted" \\ source\nwith a newline, a tab\t and é', info="info")


def write_bin(path, matrix):
    np.asarray(matrix, dtype=np.complex128).ravel(order="F").astype("<c16").tofile(path)


def write_legacy(directory, metadata, **matrices):
    """A research-dialect directory: metadata.toml as given (TOML text) and .bin files."""
    directory.mkdir()
    (directory / "metadata.toml").write_text(metadata)
    for name, matrix in matrices.items():
        write_bin(directory / f"{name}.bin", matrix)
    return directory


def write_jld2(path, R, K, *, model="phi4", coupling=2.0 + 0j, e0=-0.25, grad=None, mass=None,
               info="a state", integration=(192, 20.0, 1e-13, 1e-12, 1_000_000),
               manifold=(1e-3, 1e-2, 7e-14)):
    """An HDF5 file laid out as JLD2 writes an rcmps_store checkpoint (schema 1)."""
    import h5py
    c = np.dtype([("re", "<f8"), ("im", "<f8")])

    def julia(x):   # JLD2 stores column-major Julia arrays: the HDF5 view is the transpose
        x = np.asarray(x, dtype=np.complex128).T
        out = np.empty(x.shape, dtype=c)
        out["re"], out["im"] = x.real, x.imag
        return out
    with h5py.File(path, "w") as f:
        f["schema_version"] = np.int64(1)
        f["R"], f["K"] = julia(R), julia(K)
        f["model"] = np.bytes_(model.encode())
        f["coupling"] = np.array((coupling.real, coupling.imag), dtype=c)
        if mass is not None:
            f["mass"] = np.float64(mass)
        f["e0"] = np.float64(e0)
        f["grad"] = julia(np.zeros_like(R) if grad is None else grad)
        f["info"] = np.bytes_(info.encode())
        f["integration"] = np.array(integration, dtype=[("n_quad", "<i8"), ("max_point", "<f8"), ("atol", "<f8"),
                                                        ("rtol", "<f8"), ("maxiters", "<i8")])
        f["manifold"] = np.array(manifold, dtype=[("metric_reg", "<f8"), ("precond_reg_prefactor", "<f8"),
                                                  ("precond_reg", "<f8")])
    return path


class Format1(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_is_exact(self):
        record = full_record()
        directory = io.write_portable(self.root / "state", record, extra=dict(gradient_target=3e-5, note="x"),
                                      extra_matrices=dict(extra=record.R))
        back = io.read_portable(directory)
        for name in io.MATRICES:
            self.assertEqual(getattr(back, name).tobytes(), getattr(record, name).tobytes(), name)
            self.assertTrue(getattr(back, name).flags.c_contiguous)
        for name in ("D", "model", "coupling", "mass", "energy", "gradient_norm", "validation_passed",
                     "integration", "manifold", "producer", "source", "info"):
            self.assertEqual(getattr(back, name), getattr(record, name), name)
        self.assertEqual(back.origin, "portable")
        self.assertEqual(back.files, ("metadata.toml", "R.bin", "K.bin", "rho.bin", "gradient.bin"))
        self.assertEqual((back.metadata["gradient_target"], back.metadata["note"]), (3e-5, "x"))
        text = (directory / "metadata.toml").read_text(encoding="utf-8")
        self.assertTrue(text.startswith('format = "rcmps-portable-state"\nformat_version = 1\nD = 6\n'))
        self.assertEqual(sorted(p.name for p in directory.iterdir()),
                         ["K.bin", "R.bin", "extra.bin", "gradient.bin", "metadata.toml", "rho.bin"])
        again = io.write_portable(self.root / "again", back)
        self.assertEqual((again / "metadata.toml").read_bytes(),
                         text.replace("gradient_target = 3e-05\nnote = \"x\"\n", "").encode())

    def test_minimal_state_and_absent_values(self):
        record = full_record()
        minimal = io.StateRecord(D=6, model="phi4", coupling=2, R=record.R, K=record.K)
        back = io.read_portable(io.write_portable(self.root / "state", minimal))
        self.assertEqual((back.rho, back.gradient, back.energy, back.gradient_norm, back.validation_passed,
                          back.integration, back.manifold, back.producer, back.source, back.info),
                         (None,) * 10)
        self.assertEqual((back.mass, back.coupling, type(back.coupling)), (1.0, 2.0, float))
        self.assertEqual(back.files, ("metadata.toml", "R.bin", "K.bin"))
        metadata = io.read_portable(self.root / "state", arrays=False)
        self.assertIsNone(metadata.R)
        self.assertEqual((metadata.D, metadata.files), (6, back.files))

    def test_an_existing_directory_is_never_overwritten(self):
        target = self.root / "state"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            io.write_portable(target, full_record())
        self.assertEqual(list(target.iterdir()), [])
        io.write_portable(self.root / "other", full_record())
        before = {p.name: p.read_bytes() for p in (self.root / "other").iterdir()}
        with self.assertRaises(FileExistsError):
            io.write_portable(self.root / "other", full_record(seed=4))
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / "other").iterdir()}, before)

    def test_files_are_little_endian_column_major_complex128(self):
        d = 3
        R = np.arange(d * d).reshape(d, d) + 1j * (100 + np.arange(d * d).reshape(d, d))
        K = R.astype(">c16")   # a big-endian input is written little-endian all the same
        directory = io.write_portable(self.root / "state", io.StateRecord(D=d, model="phi4", coupling=1.0, R=R, K=K))
        expected = b"".join(struct.pack("<dd", R[i, j].real, R[i, j].imag) for j in range(d) for i in range(d))
        self.assertEqual((directory / "R.bin").read_bytes(), expected)
        self.assertEqual((directory / "K.bin").read_bytes(), expected)
        self.assertEqual((directory / "R.bin").stat().st_size, 16 * d * d)
        back = io.read_portable(directory)
        np.testing.assert_array_equal(back.R, R)
        self.assertEqual(back.R.dtype, np.dtype(np.complex128))
        self.assertTrue(back.R.dtype.isnative)

    def test_matrix_sizes_and_values_are_checked(self):
        directory = io.write_portable(self.root / "state", full_record())
        raw = (directory / "K.bin").read_bytes()
        for wrong in (raw[:-1], raw + b"\0", raw[:16 * 25]):
            (directory / "K.bin").write_bytes(wrong)
            with self.assertRaisesRegex(ValueError, f"K.bin has {len(wrong)} bytes, not the 576 of a 6x6"):
                io.read_portable(directory)
        bad = np.frombuffer(raw, dtype="<c16").copy()
        bad[7] = complex(math.nan, 0)
        (directory / "K.bin").write_bytes(bad.tobytes())
        with self.assertRaisesRegex(ValueError, "K.bin holds nonfinite values"):
            io.read_portable(directory)
        (directory / "K.bin").write_bytes(raw)
        (directory / "rho.bin").unlink()
        self.assertIsNone(io.read_portable(directory).rho)
        (directory / "R.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            io.read_portable(directory)

    def test_metadata_is_checked(self):
        directory = io.write_portable(self.root / "state", full_record())
        text = (directory / "metadata.toml").read_text(encoding="utf-8")
        cases = [
            ('format = "rcmps-portable-state"', 'format = "something-else"', "is not 'rcmps-portable-state'"),
            ("format_version = 1", "format_version = 2", "unsupported rcmps-portable-state format_version 2"),
            ("format_version = 1", "format_version = true", "format_version True"),
            ("format_version = 1", "format_version = 1.0", "format_version 1.0"),   # an integer, as Julia reads it
            ("D = 6", "D = 0", "D must be an integer >= 1"),
            ("D = 6", "D = 6.0", "D must be an integer"),
            ("D = 6", "D = 5", "R.bin has 576 bytes, not the 400 of a 5x5"),
            ('model = "phi4"\n', "", "required key model is missing"),
            ('model = "phi4"', 'model = ""', "the model name is empty"),
            ("coupling = 2.7", 'coupling = "2.7"', "coupling must be a real number"),
            ("energy = -0.30228143199435653", "energy = nan", "energy must be finite"),
            ("gradient_norm = 2.900325420717565e-05", "gradient_norm = -1.0", "gradient_norm must be finite and nonnegative"),
            ("validation_passed = true", "validation_passed = 1", "validation_passed must be true or false"),
            ("atol = 1e-14\n", "", r"\[integration\] has no atol"),
            ("n_quad = 1024", "n_quad = 0", r"\[integration\].n_quad must be an integer >= 1"),
            ("metric_reg = 0.001\n", "", r"\[manifold\] has no metric_reg"),
            ("mass = 1.0", "mass = 0.5", "model phi4 has no mass parameter"),
        ]
        for old, new, message in cases:
            with self.subTest(new=new):
                self.assertEqual(text.count(old), 1, old)
                (directory / "metadata.toml").write_text(text.replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    io.read_portable(directory)

    def test_writer_checks_its_record(self):
        record = full_record()
        cases = [
            (replace(record, energy=math.nan), {}, "energy must be finite"),
            (replace(record, rho=record.rho[:5, :5]), {}, "rho must be a 6x6 matrix"),
            (replace(record, K=None), {}, "R and K are required"),
            (replace(record, gradient=np.full((6, 6), np.inf)), {}, "gradient holds nonfinite values"),
            (replace(record, mass=0.5), {}, "model phi4 has no mass parameter"),
            (replace(record, model=""), {}, "the model name is empty"),
            (replace(record, integration=io.Integration(0, 40.0, 1e-14, 1e-13)), {}, "integration.n_quad"),
            (record, dict(extra=dict(energy=1.0)), "extra metadata key 'energy' is not allowed"),
            (record, dict(extra=dict(integration=1.0)), "extra metadata key 'integration' is not allowed"),
            (record, dict(extra_matrices=dict(rho=record.R)), "extra matrix name 'rho' is not allowed"),
            (record, dict(extra_matrices=dict(Q=record.R[:2, :2])), "Q must be a 6x6 matrix"),
        ]
        for n, (bad, options, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                io.write_portable(self.root / f"state{n}", bad, **options)
            self.assertFalse((self.root / f"state{n}").exists())
        with self.assertRaises(TypeError):
            io.write_portable(self.root / "list", record, extra=dict(values=[1, 2]))

    def test_a_mass_is_kept_for_models_that_have_one(self):
        record = replace(full_record(), model="schwinger", mass=0.5, coupling=0.3)
        back = io.read_portable(io.write_portable(self.root / "state", record))
        self.assertEqual((back.model, back.mass, back.coupling), ("schwinger", 0.5, 0.3))


class ResearchDialects(unittest.TestCase):
    """The unversioned directories of the research code (no format key: phi4, mass 1)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = states.random_state(4, z2=True, seed=5, scaling=0.6)
        rng = np.random.default_rng(9)
        self.gradient = 1e-3 * complex_matrix(rng, 4)
        self.matrices = dict(R=self.state.R, K=self.state.K, Q=self.state.Q, rho=self.state.rho)

    def tearDown(self):
        self.tmp.cleanup()

    def check_matrices(self, record):
        for name in ("R", "K", "rho"):
            self.assertEqual(getattr(record, name).tobytes(), getattr(self.state, name).tobytes(), name)

    def test_initial_of_the_research_gpu_worker(self):
        directory = write_legacy(self.root / "initial", (
            'D = 4\ncoupling = 2.7\nseed = 0\nsymmetry = "Z2"\nrandom_generator = "NumPy PCG64; complex standard normal"\n'
            "saved_energy = 0.0\nsaved_gradient_norm = 0.0\nsaved_energy_available = false\n"
            "saved_gradient_available = false\nsaved_gradient_norm_available = false\n"
            "CPU_objective_evaluated = false\nCPU_optimizer_run = false\n"),
            saved_gradient=np.zeros((4, 4)), **self.matrices)
        record = io.read_portable(directory)
        self.check_matrices(record)
        self.assertEqual((record.origin, record.D, record.model, record.coupling, record.mass),
                         ("portable-legacy", 4, "phi4", 2.7, 1.0))
        self.assertEqual((record.energy, record.gradient, record.gradient_norm, record.integration, record.manifold),
                         (None,) * 5)
        self.assertEqual(record.files, ("metadata.toml", "R.bin", "K.bin", "rho.bin"))

    def test_julia_export_of_a_saved_state(self):
        """research prepare_initial.jl / export_reference.jl of an evaluated JLD2: the saved_*
        values hold at [saved_integration]; the top-level settings are those of the GPU run."""
        directory = write_legacy(self.root / "reference", (
            'D = 4\ncoupling = 2.7\nsource = "Exact saved R,K from somewhere"\nsaved_energy = -0.30190930265518967\n'
            "saved_energy_available = true\nsaved_gradient_available = true\nsaved_gradient_norm = 2.98e-6\n"
            "saved_gradient_norm_available = true\nhalf_window = 30.0\nn_quad = 512\natol = 1.0e-13\nrtol = 1.0e-12\n"
            "metric_reg = 0.001\nprecond_reg = 8.935056507103287e-14\n\n[saved_integration]\nmaxiters = 1000000\n"
            "n_quad = 192\nrtol = 1.0e-12\natol = 1.0e-13\nmax_point = 20.0\n"),
            saved_gradient=self.gradient, **self.matrices)
        record = io.read_portable(directory)
        self.check_matrices(record)
        self.assertEqual(record.gradient.tobytes(), self.gradient.tobytes())
        self.assertEqual((record.energy, record.gradient_norm, record.source),
                         (-0.30190930265518967, 2.98e-6, "Exact saved R,K from somewhere"))
        self.assertEqual(record.integration, io.Integration(192, 20.0, 1e-13, 1e-12, 1_000_000))
        self.assertEqual(record.manifold, io.Manifold(0.001, 8.935056507103287e-14))
        self.assertEqual(record.files, ("metadata.toml", "R.bin", "K.bin", "rho.bin", "saved_gradient.bin"))
        # The earlier exports have no [saved_integration]: the settings of the saved values are
        # unknown, not the top-level ones (as in results/reference_D64 of the research tree).
        text = (directory / "metadata.toml").read_text()
        older = write_legacy(self.root / "older", text[:text.index("\n[saved_integration]")] + "\n",
                             saved_gradient=self.gradient, **self.matrices)
        record = io.read_portable(older)
        self.assertEqual((record.energy, record.gradient_norm, record.integration), (-0.30190930265518967, 2.98e-6, None))
        self.assertEqual(record.gradient.tobytes(), self.gradient.tobytes())
        self.assertEqual(states.load_portable(older).integrator, states.IntegratorParameters())

    def test_julia_export_of_a_fresh_or_scalar_only_state(self):
        text = ("D = 4\ncoupling = 2.7\nsaved_energy = 0.0\nsaved_energy_available = false\n"
                "saved_gradient_available = false\nsaved_gradient_norm = 0.0\nsaved_gradient_norm_available = false\n"
                "half_window = 30.0\nn_quad = 512\natol = 1.0e-13\nrtol = 1.0e-12\nmetric_reg = 0.001\nprecond_reg = 0.0001\n"
                "\n[saved_integration]\nmaxiters = 1000000\nn_quad = 128\nrtol = 1.0e-10\natol = 1.0e-11\nmax_point = 20.0\n")
        record = io.read_portable(write_legacy(self.root / "fresh", text, saved_gradient=np.zeros((4, 4)),
                                               **self.matrices))
        self.assertEqual((record.energy, record.gradient, record.gradient_norm), (None, None, None))
        self.assertEqual(record.integration, io.Integration(512, 30.0, 1e-13, 1e-12))
        # The initial row of a saved trajectory: energy and norm without a gradient.
        text = text.replace("saved_energy = 0.0\nsaved_energy_available = false",
                            "saved_energy = 0.8549733696608923\nsaved_energy_available = true")
        text = text.replace("saved_gradient_norm = 0.0\nsaved_gradient_norm_available = false",
                            "saved_gradient_norm = 3.9464877191463175\nsaved_gradient_norm_available = true")
        record = io.read_portable(write_legacy(self.root / "scalars", text, saved_gradient=np.zeros((4, 4)),
                                               **self.matrices))
        self.assertEqual((record.energy, record.gradient, record.gradient_norm),
                         (0.8549733696608923, None, 3.9464877191463175))
        self.assertEqual(record.integration, io.Integration(128, 20.0, 1e-11, 1e-10, 1_000_000))

    def test_optimized_state_of_the_research_validator(self):
        directory = write_legacy(self.root / "optimized_state", (
            "validation_passed = true\nD = 4\ncoupling = 2.7\nenergy = -0.30228143199435653\n"
            "gradient_norm = 2.900325420717565e-05\ngradient_target = 3e-05\ngradient_target_reached = true\n"
            "metric_reg = 0.001\nprecond_reg = 8.41190161973086e-12\nwindow = 40.0\nquadrature = 1024\n"
            "atol = 1e-14\nrtol = 1e-13\n"),
            R=self.state.R, K=self.state.K, rho=self.state.rho, gradient=self.gradient)
        record = io.read_portable(directory)
        self.check_matrices(record)
        self.assertEqual(record.gradient.tobytes(), self.gradient.tobytes())
        self.assertEqual((record.energy, record.gradient_norm, record.validation_passed),
                         (-0.30228143199435653, 2.900325420717565e-05, True))
        self.assertEqual(record.integration, io.Integration(1024, 40.0, 1e-14, 1e-13))
        self.assertEqual(record.manifold, io.Manifold(0.001, 8.41190161973086e-12))
        self.assertEqual(record.metadata["gradient_target_reached"], True)
        state = states.load_portable(directory)
        self.assertEqual((state.e0, state.coupling, state.model), (-0.30228143199435653, 2.7, "phi4"))
        self.assertEqual(state.integrator, states.IntegratorParameters(1024, 40.0, 1e-14, 1e-13))

    def test_minimal_directory_and_required_keys(self):
        directory = write_legacy(self.root / "minimal", "D = 4\ncoupling = 1.5\n", **self.matrices)
        state = states.load_portable(directory)
        self.assertEqual(state.R.tobytes(), self.state.R.tobytes())
        self.assertLess(np.abs(state.rho - self.state.rho).max(), 1e-15)
        self.assertTrue(math.isnan(state.e0))
        self.assertEqual(state.integrator, states.IntegratorParameters())
        (directory / "rho.bin").unlink()   # recomputed
        self.assertLess(np.abs(states.load_portable(directory).rho - self.state.rho).max(), 1e-12)
        (directory / "metadata.toml").write_text("D = 4\n")
        with self.assertRaisesRegex(ValueError, "required key coupling is missing"):
            io.read_portable(directory)


class JLD2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = states.random_state(4, z2=True, seed=8, scaling=0.6)
        self.gradient = 1e-4 * complex_matrix(np.random.default_rng(2), 4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_1_layout(self):
        path = write_jld2(self.root / "s.jld2", self.state.R, self.state.K, grad=self.gradient, e0=-0.25)
        record = io.read_jld2(path)
        self.assertEqual(record.R.tobytes(), self.state.R.tobytes())
        R = self.state.R.copy()
        R[0, 0] = complex(-0.0, -0.0)   # in a zero block: read bit for bit, signed zeros included
        self.assertEqual(io.read_jld2(write_jld2(self.root / "z.jld2", R, self.state.K)).R.tobytes(), R.tobytes())
        self.assertEqual(record.K.tobytes(), self.state.K.tobytes())
        self.assertEqual(record.gradient.tobytes(), self.gradient.tobytes())
        self.assertEqual((record.D, record.model, record.coupling, record.mass, record.energy, record.info,
                          record.origin, record.rho), (4, "phi4", 2.0, 1.0, -0.25, "a state", "jld2", None))
        self.assertEqual(record.integration, io.Integration(192, 20.0, 1e-13, 1e-12, 1_000_000))
        self.assertEqual(record.manifold, io.Manifold(1e-3, 7e-14, 1e-2))
        state = states.load_jld2(path)
        self.assertEqual((state.e0, state.coupling, state.model, state.source), (-0.25, 2.0, "phi4", str(path)))
        self.assertEqual(state.integrator, states.IntegratorParameters(192, 20.0, 1e-13, 1e-12, 1_000_000))
        self.assertLess(np.abs(state.rho - self.state.rho).max(), 1e-12)

    def test_an_unevaluated_state_has_no_energy_or_gradient(self):
        record = io.read_jld2(write_jld2(self.root / "s.jld2", self.state.R, self.state.K, e0=math.nan))
        self.assertEqual((record.energy, record.gradient), (None, None))
        self.assertTrue(math.isnan(states.load_jld2(self.root / "s.jld2").e0))

    def test_mass(self):
        R, K = self.state.R, self.state.K
        self.assertEqual(io.read_jld2(write_jld2(self.root / "a.jld2", R, K, mass=1.0)).mass, 1.0)
        with self.assertRaisesRegex(ValueError, "model phi4 has no mass parameter .* mass = 0.5, refused"):
            io.read_jld2(write_jld2(self.root / "b.jld2", R, K, mass=0.5))
        with self.assertRaisesRegex(ValueError, "model phi4 has no mass parameter"):
            states.load_jld2(self.root / "b.jld2")
        record = io.read_jld2(write_jld2(self.root / "c.jld2", R, K, model="schwinger", mass=0.5))
        self.assertEqual((record.model, record.mass), ("schwinger", 0.5))
        directory = io.write_portable(self.root / "c", record)
        self.assertEqual(io.read_portable(directory).mass, 0.5)

    def test_an_empty_model_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, "the model name is empty"):
            io.read_jld2(write_jld2(self.root / "a.jld2", self.state.R, self.state.K, model=""))

        class Unopenable(dict):   # JLD2 writes "" with a datatype of size zero, which HDF5 cannot open
            def __getitem__(self, name):
                raise KeyError("Unable to synchronously open object (invalid datatype size)")
        self.assertEqual(io._string(Unopenable(model=None), "model"), "")
        with self.assertRaises(KeyError):   # a missing dataset is not an empty string
            io._string(Unopenable(), "model")

    def test_complex_coupling_and_other_schemas_are_refused(self):
        with self.assertRaisesRegex(ValueError, "model coupling must be real"):
            io.read_jld2(write_jld2(self.root / "a.jld2", self.state.R, self.state.K, coupling=2 + 1e-3j))
        import h5py
        path = write_jld2(self.root / "b.jld2", self.state.R, self.state.K)
        with h5py.File(path, "r+") as f:
            f["schema_version"][()] = 2
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            io.read_jld2(path)


@unittest.skipUnless((FIXTURES / "phi4_g2_D16_z2_portable").is_dir(), "the shared fixtures are not available")
class SharedFixture(unittest.TestCase):
    """test/fixtures/phi4_g2_D16_z2_portable is gpu/julia/export_state.jl applied to
    test/fixtures/phi4_g2_D16_z2_state.jld2 (gpu/julia/check_portable.jl checks it in Julia)."""

    def setUp(self):
        self.portable = FIXTURES / "phi4_g2_D16_z2_portable"
        self.jld2 = FIXTURES / "phi4_g2_D16_z2_state.jld2"

    def test_the_export_holds_the_checkpoint(self):
        exported, saved = io.read_portable(self.portable), io.read_jld2(self.jld2)
        self.assertEqual((exported.origin, exported.files),
                         ("portable", ("metadata.toml", "R.bin", "K.bin", "rho.bin", "gradient.bin")))
        for name in ("R", "K", "gradient"):
            self.assertEqual(getattr(exported, name).tobytes(), getattr(saved, name).tobytes(), name)
        for name in ("D", "model", "coupling", "mass", "energy", "integration", "manifold", "info"):
            self.assertEqual(getattr(exported, name), getattr(saved, name), name)
        self.assertEqual((exported.D, exported.coupling, exported.energy), (16, 2.0, -0.15648780475169466))
        from rcmps_gpu.optimizer.runtime.numerics import metric
        norm = math.sqrt(metric(exported.rho, exported.gradient, exported.gradient, exported.manifold.metric_reg))
        self.assertAlmostEqual(exported.gradient_norm / norm, 1.0, delta=1e-12)
        self.assertLess(np.abs(exported.rho - states.load_jld2(self.jld2).rho).max(), 1e-12)

    def test_the_excitation_loaders_agree(self):
        portable, saved = states.load_portable(self.portable), states.load_jld2(self.jld2)
        self.assertEqual(portable.R.tobytes(), saved.R.tobytes())
        self.assertEqual(portable.K.tobytes(), saved.K.tobytes())
        self.assertLess(np.abs(portable.rho - saved.rho).max(), 1e-12)
        self.assertEqual((portable.integrator, portable.e0, portable.coupling, portable.model),
                         (saved.integrator, saved.e0, saved.coupling, saved.model))

    def test_the_files_are_portable(self):
        for name in ("R.bin", "K.bin", "rho.bin", "gradient.bin"):
            self.assertEqual((self.portable / name).stat().st_size, 16 * 16 * 16, name)
        raw = (self.portable / "metadata.toml").read_bytes()
        metadata = tomllib.loads(raw.decode("utf-8"))
        self.assertEqual((metadata["format"], metadata["format_version"], metadata["source"]),
                         ("rcmps-portable-state", 1, "phi4_g2_D16_z2_state.jld2"))
        self.assertTrue(metadata["producer"].startswith("RcmpsCore 0.2"))
        for marker in (b"/home", b"/tmp", b"\\", b"Users"):
            self.assertNotIn(marker, raw)
        self.assertEqual(sorted(path.name for path in self.portable.iterdir()),
                         ["K.bin", "R.bin", "gradient.bin", "metadata.toml", "rho.bin"])


if __name__ == "__main__":
    unittest.main()
