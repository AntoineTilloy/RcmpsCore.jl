"""Tests of the command line cli.py that need no GPU.

The fused backend integrates packed.PackedActionSystem, which takes tangents of one Z2
class: `--sector all` cannot work there (the second Davidson vector mixes the classes) and
must be refused when the arguments are parsed, before the output directory exists. The
control supports sector=:all (src/excitation_eigsolve.jl:113), and so do the host and
numpy backends: one short CPU run on a portable export written here (random_state(4,
seed=11, scaling=0.5, z2=True), window 14, n_quad 32, 16 theta nodes, two actions).

Defaults resolved by cli.py (Defaults): the research Davidson schedule (maxspace
min(160, n), retain min(max(12, nev), maxspace - 1), thick restarts), --fresh-restarts,
explicit --maxspace/--retain, the clamp in small sectors, nev > 12, and the kinetic nodes
of eigsolve.default_kinetic_nodes, on random Z2 states of D = 4, 16, 72 and 130 that are
never solved, and one end-to-end CPU run of the D = 4 state on every default (odd sector,
n = 8: maxspace 8, retain 7, 256 kinetic nodes; eight actions, about 5 s). result.json
`settings` records the values used (integer kinetic nodes, maxspace, retain, thick_restart),
as the A100 result files did, and the parsed arguments under `settings.arguments`.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rcmps_gpu.excitations import eigsolve
from rcmps_gpu.excitations import cli as run
from rcmps_gpu.excitations.davidson import DavidsonOptions
from rcmps_gpu.excitations.states import random_state


def export(directory, state, coupling):
    """A portable export as states.load_portable reads it."""
    directory.mkdir()
    for name in ("R", "K", "rho"):
        getattr(state, name).astype("<c16").ravel(order="F").tofile(directory / f"{name}.bin")
    (directory / "metadata.toml").write_text(f"D = {state.D}\ncoupling = {coupling}\n")


def parse(*arguments):
    """Parsed arguments of `rcmps-gpu-excitations STATE --output OUT *arguments`."""
    return run.make_parser().parse_args(["STATE", "--output", "OUT", *map(str, arguments)])


def main(*arguments):
    """run.main() on the given arguments; returns what it printed."""
    printed = io.StringIO()
    with mock.patch.object(sys, "argv", ["rcmps-gpu-excitations", *map(str, arguments)]), \
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
            self.assertIn("warning: kinetic_nodes=16 is below the default 256 for D = 4", printed)
            self.assertIn("davidson schedule research: maxspace=16 retain=12 refresh=False", printed)
            settings = result["settings"]
            self.assertEqual([settings[k] for k in ("kinetic_nodes", "maxspace", "retain", "thick_restart",
                                                    "fresh_restarts")], [16, 16, 12, True, False])
            self.assertEqual([settings["arguments"][k] for k in ("kinetic_nodes", "maxspace", "retain",
                                                                 "thick_restart")], [16, None, None, False])


class Defaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Odd-sector dimensions 8, 128, 2592 and 8450; nothing is solved on these states.
        cls.states = {d: random_state(d, seed=11 if d == 4 else 3, scaling=0.5, z2=True) for d in (4, 16, 72, 130)}

    def resolve(self, d, *arguments):
        return run.resolve_defaults(parse(*arguments), self.states[d])

    def schedule(self, d, *arguments):
        """(maxspace, retain, refresh) of the DavidsonOptions that cli.py passes."""
        o = self.resolve(d, *arguments)["davidson"]
        return o.maxspace, o.retain, o.refresh

    def test_research_schedule_is_the_default(self):
        """No Davidson option: maxspace min(160, n), retain 12, thick restarts, with the
        unchanged tolerance, maxops and stall controls. --thick-restart, kept for older
        command lines, changes nothing; the flags of the A100 runs give the same options
        whenever n >= 160."""
        for d, n, maxspace in ((16, 128, 128), (72, 2592, 160)):
            resolved = self.resolve(d)
            self.assertEqual((resolved["schedule"], resolved["dimension"]), ("research", n))
            self.assertEqual(n, eigsolve.sector_dimension(self.states[d]))
            self.assertEqual(resolved["davidson"],
                             DavidsonOptions(tol=1e-7, maxops=200, maxspace=maxspace, retain=12, refresh=False,
                                             stall_residual_limit=None, stall_energy_atol=1e-9))
            self.assertEqual(resolved["davidson"], self.resolve(d, "--thick-restart")["davidson"])
            self.assertIn(f"davidson schedule research: maxspace={maxspace} retain=12 refresh=False tol=1e-07 "
                          f"maxops=200 (sector dimension {n})", resolved["messages"])
        self.assertEqual(self.resolve(72, "--maxspace", 160, "--retain", 12, "--thick-restart")["davidson"],
                         self.resolve(72)["davidson"])
        self.assertEqual(self.schedule(16, "--maxspace", 160, "--retain", 12, "--thick-restart"), (160, 12, False))
        tuned = self.resolve(16, "--tol", 1e-9, "--maxops", 50, "--stall-residual-limit", 1e-6,
                             "--stall-energy-atol", 1e-10)["davidson"]
        self.assertEqual((tuned.tol, tuned.maxops, tuned.stall_residual_limit, tuned.stall_energy_atol),
                         (1e-9, 50, 1e-6, 1e-10))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse("--thick-restart", "--fresh-restarts")

    def test_fresh_restarts(self):
        """--fresh-restarts is the library schedule of the CPU code, as excitation_eigsolve
        uses for davidson_options=None."""
        for nev, maxspace in ((1, 8), (5, 12), (13, 28)):
            resolved = self.resolve(16, "--fresh-restarts", "--nev", nev)
            self.assertEqual(resolved["schedule"], "fresh-restarts")
            self.assertEqual(resolved["davidson"],
                             DavidsonOptions(tol=1e-7, maxops=200, maxspace=maxspace, retain=None, refresh=True,
                                             stall_residual_limit=None, stall_energy_atol=1e-9))
            self.assertIn(f"davidson schedule fresh-restarts: maxspace={maxspace} retain={nev} refresh=True "
                          f"tol=1e-07 maxops=200 (sector dimension 128)", resolved["messages"])
        self.assertEqual(self.resolve(4, "--fresh-restarts")["davidson"].maxspace, 8)

    def test_explicit_overrides(self):
        """--maxspace and --retain replace the respective rule of either schedule; with
        --maxspace alone the research retain follows it."""
        self.assertEqual(self.schedule(72, "--maxspace", 40), (40, 12, False))
        self.assertEqual(self.schedule(72, "--maxspace", 10), (10, 9, False))
        self.assertEqual(self.schedule(16, "--retain", 5), (128, 5, False))
        self.assertEqual(self.schedule(72, "--maxspace", 400, "--retain", 30), (400, 30, False))
        self.assertEqual(self.schedule(4, "--maxspace", 20), (20, 12, False))
        self.assertEqual(self.schedule(16, "--fresh-restarts", "--maxspace", 30), (30, None, True))
        self.assertEqual(self.schedule(16, "--fresh-restarts", "--maxspace", 20, "--retain", 3), (20, 3, True))
        self.assertEqual(self.schedule(16, "--fresh-restarts", "--retain", 4), (8, 4, True))
        for arguments in (("--maxspace", 5, "--retain", 5), ("--retain", 128), ("--fresh-restarts", "--retain", 8),
                          ("--nev", 3, "--retain", 2)):
            with self.assertRaises(ValueError, msg=str(arguments)):
                self.resolve(16, *arguments)

    def test_small_sectors(self):
        """The D = 4 state: odd sector n = 8 gives maxspace 8 and retain 7; --sector all
        n = 16, maxspace 16 and retain 12, or nev for 12 < nev < 16; --metric-tol drops a
        column (n = 6) exactly as the solver does. nev >= maxspace and nev > n are errors."""
        self.assertEqual(self.schedule(4), (8, 7, False))
        self.assertEqual(self.schedule(4, "--nev", 7), (8, 7, False))
        self.assertEqual(self.schedule(4, "--sector", "even", "--nev", 3), (8, 7, False))
        self.assertEqual(self.schedule(4, "--sector", "all"), (16, 12, False))
        self.assertEqual(self.schedule(4, "--sector", "all", "--nev", 13), (16, 13, False))
        self.assertEqual(self.schedule(4, "--sector", "all", "--nev", 15), (16, 15, False))
        dropped = self.resolve(4, "--metric-tol", 0.0828)
        self.assertEqual((dropped["dimension"], dropped["davidson"].maxspace, dropped["davidson"].retain), (6, 6, 5))
        self.assertEqual(dropped["dimension"], eigsolve.sector_dimension(self.states[4], metric_tol=0.0828))
        for arguments in (("--nev", 8), ("--nev", 9), ("--sector", "all", "--nev", 16), ("--nev", 0),
                          ("--metric-tol", 0.0828, "--nev", 6), ("--kinetic-nodes", 1)):
            with self.assertRaises(ValueError, msg=str(arguments)):
                self.resolve(4, *arguments)
        self.assertEqual(self.schedule(4, "--nev", 8, "--fresh-restarts"), (18, None, True))

    def test_more_than_twelve_levels(self):
        """nev > 12: retain is nev in the research schedule, so nev < maxspace is needed."""
        self.assertEqual(self.schedule(16, "--nev", 13), (128, 13, False))
        self.assertEqual(self.schedule(72, "--nev", 40), (160, 40, False))
        self.assertEqual(self.schedule(72, "--nev", 159), (160, 159, False))
        self.assertEqual(self.schedule(72, "--nev", 20, "--maxspace", 21), (21, 20, False))
        for arguments in (("--nev", 160), ("--nev", 13, "--maxspace", 13), ("--nev", 13, "--retain", 12)):
            with self.assertRaises(ValueError, msg=str(arguments)):
                self.resolve(72, *arguments)

    def test_kinetic_nodes(self):
        """256 nodes for D <= 64 and 512 beyond; an explicit value below that default is
        warned about, one above is not; beyond D = 128 one note says that the count was not
        measured there and names the nodes of the refined check."""
        def warnings(resolved):
            return [m for m in resolved["messages"] if m.startswith(("warning", "note"))]

        for d, nodes in ((4, 256), (16, 256), (72, 512), (130, 512)):
            resolved = self.resolve(d)
            self.assertEqual((resolved["kinetic_nodes"], resolved["default_kinetic_nodes"]), (nodes, nodes))
            self.assertEqual(nodes, eigsolve.default_kinetic_nodes(d))
            self.assertIn(f"kinetic_nodes={nodes} (default for D = {d})", resolved["messages"])
            self.assertEqual(len(warnings(resolved)), int(d > 128))
        for d, given, warned in ((16, 128, True), (16, 512, False), (72, 256, True), (72, 512, False),
                                 (72, 1024, False), (4, 16, True)):
            resolved = self.resolve(d, "--kinetic-nodes", given)
            self.assertEqual(resolved["kinetic_nodes"], given)
            self.assertEqual(any(m.startswith(f"warning: kinetic_nodes={given} is below the default")
                                 for m in resolved["messages"]), warned, f"D = {d}, {given} nodes")
        note = warnings(self.resolve(130))
        self.assertEqual(len(note), 1)
        self.assertTrue(note[0].startswith("note: kinetic node counts are measured only up to D = 128; at D = 130 "
                                           "the refined check of the validation (1024 nodes)"), note[0])
        self.assertIn("(2048 nodes)", warnings(self.resolve(130, "--refined-kinetic-nodes", 2048))[0])
        self.assertTrue(warnings(self.resolve(130, "--no-validate"))[0].endswith("--no-validate skips it"))
        self.assertEqual(len(warnings(self.resolve(130, "--kinetic-nodes", 256))), 2)

    def test_defaults_end_to_end(self):
        """The D = 4 state on every default of the CPU backend (odd sector n = 8; window 14
        and n_quad 32 for speed): cli.py passes maxspace 8, retain 7, thick restarts and 256
        kinetic nodes, the solver works in the same dimension and its eight actions span the
        sector. --nev 8 is a usage error before the output directory exists."""
        with tempfile.TemporaryDirectory() as scratch:
            source, output = Path(scratch) / "state", Path(scratch) / "out"
            export(source, self.states[4], 1.0)
            printed = main(source, "--output", output, "--backend", "numpy", "--window", 14, "--n-quad", 32,
                           "--no-validate")
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["davidson"],
                             dict(tol=1e-7, maxops=200, maxspace=8, retain=7, refresh=False, hermitian_tol=1e-7,
                                  stall_window=20, stall_minops=40, stall_rtol=0.1, stall_residual_limit=None,
                                  stall_energy_atol=1e-9))
            self.assertEqual((result["sector"], result["dimension"], result["kinetic_nodes"]), ("odd", 8, 256))
            self.assertEqual((result["converged"], result["numops"]), (True, 8))
            settings = result["settings"]
            self.assertEqual((settings["davidson_schedule"], settings["sector_dimension"],
                              settings["default_kinetic_nodes"]), ("research", 8, 256))
            # The values used, as the A100 result.json files recorded them; the raw arguments apart.
            self.assertEqual([settings[k] for k in ("kinetic_nodes", "maxspace", "retain", "thick_restart",
                                                    "fresh_restarts")], [256, 8, 7, True, False])
            self.assertEqual([settings["arguments"][k] for k in ("kinetic_nodes", "maxspace", "retain",
                                                                 "thick_restart", "fresh_restarts")],
                             [None, None, None, False, False])
            self.assertEqual(settings["arguments"]["nev"], settings["nev"])
            # Sources: the 16 modules and the package files on their import path, and their provenance.
            self.assertEqual(sorted(settings["source_sha256"]), sorted(run.SOURCES + run.SHARED_SOURCES))
            self.assertTrue(settings["source_provenance"]["verified"])
            self.assertIn("kinetic_nodes=256 (default for D = 4)", printed)
            self.assertIn("davidson schedule research: maxspace=8 retain=7 refresh=False", printed)
            self.assertIn("operator p=0 dimension=8 kinetic_nodes=256", printed)
            self.assertNotIn("warning", printed)
            refused = Path(scratch) / "refused"
            with self.assertRaises(SystemExit) as failure:
                main(source, "--output", refused, "--backend", "numpy", "--nev", 8)
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse(refused.exists())

    def test_fresh_restart_settings(self):
        """With --fresh-restarts result.json records retain = nev and thick_restart false in
        `settings` (the DavidsonOptions at the top level keep retain None, meaning nev, as in
        the A100 runs on the CPU schedule). Two actions of the D = 4 state, 16 nodes."""
        with tempfile.TemporaryDirectory() as scratch:
            source, output = Path(scratch) / "state", Path(scratch) / "out"
            export(source, self.states[4], 1.0)
            main(source, "--output", output, "--backend", "numpy", "--window", 14, "--n-quad", 32, "--no-validate",
                 "--fresh-restarts", "--nev", 2, "--maxops", 2, "--kinetic-nodes", 16)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual([result["davidson"][k] for k in ("maxspace", "retain", "refresh")], [8, None, True])
            settings = result["settings"]
            self.assertEqual([settings[k] for k in ("kinetic_nodes", "maxspace", "retain", "thick_restart",
                                                    "fresh_restarts", "davidson_schedule")],
                             [16, 8, 2, False, True, "fresh-restarts"])
            self.assertEqual([settings["arguments"][k] for k in ("maxspace", "retain", "thick_restart",
                                                                 "fresh_restarts")], [None, None, False, True])


if __name__ == "__main__":
    unittest.main()
