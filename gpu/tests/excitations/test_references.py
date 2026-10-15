"""Cross-language references: the levels of test/fixtures/excitation_references.toml.

The file is shared with the Julia package, whose benchmark/excitation_references.jl computes
and checks its regenerable entries with RcmpsCore's excitation_eigsolve. Here the same solves
run through eigsolve.excitation_eigsolve on the CPU (NumPy backend), with the settings recorded
in the file (_references.solve), and every level must agree with the Julia one: the validated
energy and the four validation energies within the level's `tolerance`, the Ritz energy within
its `ritz_tolerance`, with the search converged and the level accepted.

Default: the four solves (odd and even sector, p = 0 and 0.4, nev = 2) on the random D = 4 Z2
state of the file, about a minute on one thread. Measured on 2026-10-15 (NumPy 2.2.6, SciPy
1.16.2, OPENBLAS_NUM_THREADS=1): every validated energy within 5.5e-14 of Julia's, every
validation energy within 8.2e-13, every Ritz energy within 9.7e-13; the file's tolerance for
these levels is 1e-11.

RCMPS_EXCITATION_SLOW=1 adds the regenerable D = 16 levels (g = 2 ground state, the settings
of the A100 runs; one test per solve, 460 to 560 s each on one thread), at the file's
tolerance 1e-10 between implementations. Measured on 2026-10-15: validated energies within
2.4e-12 of Julia's, validation energies within 2.8e-11, Ritz energies within 8.4e-11, with
Julia's operation counts. The D = 32 levels (about 40 minutes each in Julia) and every GPU
recomputation are left to gpu/qualification/excitations/check_references.py.
RCMPS_EXCITATION_REPORT=1 prints every measured difference.
"""
from __future__ import annotations

import math
import os
import sys
import time
import unittest

from rcmps_gpu.excitations import states
import _references as refs

REFERENCES = refs.REFERENCES
REPORT = os.environ.get("RCMPS_EXCITATION_REPORT") == "1"
SLOW = os.environ.get("RCMPS_EXCITATION_SLOW") == "1"
available = REFERENCES is not None and all(refs.state_path(name).exists() for name in REFERENCES["states"])
requires_file = unittest.skipUnless(available, "the reference file or its states are not available "
                                               "(RCMPS_TEST_FIXTURES)")
slow = unittest.skipUnless(SLOW, "set RCMPS_EXCITATION_SLOW=1 (about nine minutes per D = 16 solve)")


def solve_and_compare(case, key, entries, **controls):
    state_name, settings_name, sector, momentum, nev = key
    tick = time.time()
    log = (lambda line: print(f"[{time.time() - tick:7.1f} s] {line}", flush=True)) if SLOW and nev == 1 else None
    r = refs.solve(refs.load_state(state_name), REFERENCES["settings"][settings_name], sector, momentum, nev,
                   log=log, **controls)
    if REPORT or log:
        print(f"{state_name} {sector} p={momentum}: numops={r['numops']} action_calls={r['action_calls']} "
              f"start={r['start']} {time.time() - tick:.1f} s", file=sys.stderr, flush=True)
    case.assertTrue(r["converged"], r["stop_reason"])
    case.assertEqual((r["sector"], r["dimension"]), (sector, entries[0]["dimension"]))
    for entry in entries:
        with case.subTest(level=entry["name"]):
            case.assertTrue(r["validation"][entry["level"] - 1]["accepted"])
            case.assertEqual(r["start"], entry["start"])
            for quantity, (difference, tolerance) in refs.differences(entry, r).items():
                if REPORT or log:
                    print(f"  {entry['name']} {quantity}: {difference:+.3e} (tolerance {tolerance:.0e})",
                          file=sys.stderr, flush=True)
                case.assertLessEqual(abs(difference), tolerance, f"{entry['name']} {quantity}")


@requires_file
class File(unittest.TestCase):
    def test_format_and_cross_references(self):
        self.assertEqual((REFERENCES["format"], REFERENCES["format_version"]), ("rcmps-excitation-references", 1))
        names = [entry["name"] for entry in REFERENCES["levels"]]
        self.assertEqual(len(names), len(set(names)))
        for entry in REFERENCES["levels"]:
            self.assertIn(entry["state"], REFERENCES["states"])
            self.assertIn(entry["settings"], REFERENCES["settings"])
            self.assertIn(entry["sector"], ("odd", "even"))
            self.assertTrue(1 <= entry["level"] <= entry["nev"])
            self.assertTrue(math.isfinite(entry["validated_energy"]) and entry["tolerance"] > 0)
            if entry["regenerate"]:
                self.assertEqual(len(entry["four_energies"]), 4)
                self.assertGreater(entry["ritz_tolerance"], 0)
        for key, entries in refs.solves(REFERENCES["levels"]).items():
            self.assertEqual(sorted(entry["level"] for entry in entries), list(range(1, len(entries) + 1)), key)

    def test_state_files(self):
        for name, entry in REFERENCES["states"].items():
            with self.subTest(state=name):
                self.assertEqual(refs.sha256(refs.state_path(name)), entry["sha256"])
                state = refs.load_state(name)
                self.assertEqual(state.D, entry["D"])

    def test_the_small_state_is_the_julia_recipe(self):
        """The D = 4 file holds what its recipe describes: a Z2 state (R odd, K even under
        diag(1, 1, -1, -1)) with the recipe's integration settings."""
        entry = REFERENCES["states"]["phi4_g1_D4"]
        state = states.load_jld2(refs.state_path("phi4_g1_D4"))
        recipe = entry["recipe"]
        self.assertEqual(state.integrator, refs.integrator(recipe["integration"]))
        self.assertLess(states.parity_error(state, [1, 1, -1, -1]), 1e-14)


@requires_file
class SmallLevels(unittest.TestCase):
    """The D = 4 levels: four solves with nev = 2 at the [settings.small] of the file."""

    def test_small_levels(self):
        groups = refs.solves(entry for entry in REFERENCES["levels"] if entry["state"] == "phi4_g1_D4")
        self.assertEqual(len(groups), 4)
        for key, entries in groups.items():
            with self.subTest(solve=key):
                solve_and_compare(self, key, entries)


@requires_file
class LargeLevels(unittest.TestCase):
    """The regenerable D = 16 levels at [settings.gpu_comparison]: one test per solve."""


def add_large_tests():
    if REFERENCES is None:
        return
    entries = [entry for entry in REFERENCES["levels"] if entry["regenerate"] and entry["state"] == "phi4_g2_D16"]
    for key, group in refs.solves(entries).items():
        def test(self, key=key, group=group):
            solve_and_compare(self, key, group)
        name = "test_" + "_".join(entry["name"] for entry in group).replace(".", "p")
        test.__doc__ = f"{key[0]}, {key[2]} sector, p = {key[3]} (about nine minutes on one thread)"
        setattr(LargeLevels, name, slow(test))


add_large_tests()


if __name__ == "__main__":
    unittest.main()
