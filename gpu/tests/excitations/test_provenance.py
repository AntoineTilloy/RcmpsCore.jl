"""The excitation sources are the A100-qualified bytes with their recorded edits.

excitations/provenance.json holds settings.source_sha256 of the A100 final2 runs (the
qualified sources, imported byte for byte) and every edit made since; cli.source_provenance()
reverts the edits and compares, and every result.json records the outcome.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rcmps_gpu import provenance
from rcmps_gpu.excitations import cli

RECORD = json.loads((cli.HERE / "provenance.json").read_bytes())
# sha256 of json.dumps(settings.source_sha256 of the final2 result.json files, indent=2, sort_keys=True) + "\n"
QUALIFIED_SHA = "68c84931cca9aad270078af8e4a46ef7e8c4b680673a80b2fc2080429d5baa35"
IMPORT_LIST = Path(__file__).resolve().parents[2] / "IMPORTED_FROM_RESEARCH.sha256"


class Provenance(unittest.TestCase):
    def test_the_qualified_table_is_pinned(self):
        table = RECORD["qualified"]["source_sha256"]
        raw = (json.dumps(table, indent=2, sort_keys=True) + "\n").encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), QUALIFIED_SHA)
        self.assertEqual(cli.QUALIFIED_SOURCES_SHA256, QUALIFIED_SHA)
        names = sorted(RECORD["qualified"]["renamed"].get(name, name) for name in table)
        self.assertEqual(names, sorted(cli.SOURCES))

    @unittest.skipUnless(IMPORT_LIST.exists(), "IMPORTED_FROM_RESEARCH.sha256 is not available")
    def test_the_import_commit_holds_the_qualified_bytes(self):
        listed = {}
        for line in IMPORT_LIST.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                digest, path, _ = line.split()
                listed[path] = digest
        for name, digest in RECORD["qualified"]["source_sha256"].items():
            name = RECORD["qualified"]["renamed"].get(name, name)
            self.assertEqual(listed["gpu/src/rcmps_gpu/excitations/" + name], digest, name)

    def test_reverting_the_recorded_edits_gives_the_qualified_bytes(self):
        result = cli.source_provenance()
        self.assertTrue(result["verified"], result.get("error"))
        self.assertEqual(result["edits"], [entry["commit_subject"] for entry in RECORD["edits"]])
        self.assertNotIn("models.py", result["edited_files"])   # unchanged since the A100 runs
        self.assertIn("eigsolve.py", result["edited_files"])

    def test_an_unrecorded_edit_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "excitations"
            shutil.copytree(cli.HERE, copy, ignore=shutil.ignore_patterns("__pycache__"))
            imported = {RECORD["qualified"]["renamed"].get(name, name): digest
                        for name, digest in RECORD["qualified"]["source_sha256"].items()}
            self.assertEqual(provenance.verify(copy, imported, RECORD["edits"], cli.SOURCES),
                             cli.source_provenance()["edited_files"])
            target = copy / "kinetic.py"
            target.write_bytes(target.read_bytes() + b"x = 0\n")
            with self.assertRaisesRegex(ValueError, "kinetic.py with its recorded edits reverted is not the imported"):
                provenance.verify(copy, imported, RECORD["edits"], cli.SOURCES)

    def test_every_file_that_runs_with_the_solver_is_hashed(self):
        package = cli.HERE.parent
        files = {path.relative_to(package).as_posix() for path in package.rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and (path.parent == package or path.relative_to(package).parts[0] == "excitations")}
        hashed = {"excitations/" + name for name in cli.SOURCES} | set(cli.SHARED_SOURCES)
        self.assertEqual(files - hashed, set())
        for name in ("__init__.py", "excitations/__init__.py", "optimizer/__init__.py", "optimizer/runtime/__init__.py"):
            self.assertTrue(provenance.inert(package / name), name)


if __name__ == "__main__":
    unittest.main()
