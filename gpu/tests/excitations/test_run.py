"""Tests of the command line run.py that need no GPU.

The fused backend integrates packed.PackedActionSystem, which takes tangents of one Z2
class: `--sector all` cannot work there (the second Davidson vector mixes the classes) and
must be refused when the arguments are parsed, before the output directory exists. The
control supports sector=:all (src/excitation_eigsolve.jl:113), and so do the host and
numpy backends: one short CPU run on a portable export written here (random_state(4,
seed=11, scaling=0.5, z2=True), window 14, n_quad 32, 16 theta nodes, two actions).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run
from states import random_state


def export(directory, state, coupling):
    """A portable export as states.load_portable reads it."""
    directory.mkdir()
    for name in ("R", "K", "rho"):
        getattr(state, name).astype("<c16").ravel(order="F").tofile(directory / f"{name}.bin")
    (directory / "metadata.toml").write_text(f"D = {state.D}\ncoupling = {coupling}\n")


def main(*arguments):
    """run.main() on the given arguments; returns what it printed."""
    printed = io.StringIO()
    with mock.patch.object(sys, "argv", ["run.py", *map(str, arguments)]), \
            contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        run.main()
    return printed.getvalue()


class Arguments(unittest.TestCase):
    def test_fused_backend_needs_a_parity_sector(self):
        """`--sector all` with the fused backend (the default one, or named) is a usage
        error: exit status 2, nothing created, nothing imported from the GPU stack."""
        loaded = "cupy" in sys.modules
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "out"
            for backend in ((), ("--backend", "fused")):
                with self.assertRaises(SystemExit) as failure:
                    main(Path(scratch) / "state", "--output", output, "--sector", "all", *backend)
                self.assertEqual(failure.exception.code, 2)
                self.assertFalse(output.exists())
        self.assertEqual("cupy" in sys.modules, loaded)

    def test_numpy_backend_runs_every_sector(self):
        """The CPU backend on a Z2 state with --sector all (dimension 16): two actions, then
        result.json, vectors.npz and the checkpoint of the last Davidson row."""
        with tempfile.TemporaryDirectory() as scratch:
            source, output = Path(scratch) / "state", Path(scratch) / "out"
            export(source, random_state(4, seed=11, scaling=0.5, z2=True), 1.0)
            printed = main(source, "--output", output, "--backend", "numpy", "--sector", "all", "--momentum", 0.3,
                           "--window", 14, "--n-quad", 32, "--kinetic-nodes", 16, "--no-validate", "--maxops", 2)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual((result["sector"], result["dimension"], result["numops"], result["stop_reason"]),
                             ("all", 16, 2, "maxops"))
            self.assertEqual((result["settings"]["backend"], result["settings"]["D"], result["scaling"]),
                             ("numpy", 4, "density"))
            self.assertIn("davidson    2 ", printed)
            with np.load(output / "vectors.npz") as stored:
                self.assertEqual((stored["coefficients"].shape, stored["vectors"].shape), ((1, 16), (1, 4, 4)))
            with np.load(output / "checkpoint.npz") as stored:
                self.assertEqual(int(stored["count"]), 2)


if __name__ == "__main__":
    unittest.main()
