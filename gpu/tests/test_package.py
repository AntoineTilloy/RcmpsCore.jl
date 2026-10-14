"""Package-level checks: the provenance record format (rcmps_gpu.provenance), the version
and the license."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import tempfile
import tomllib
import unittest

import rcmps_gpu
from rcmps_gpu import provenance

PROJECT = Path(__file__).resolve().parents[1]   # gpu/ of the repository, or an unpacked sdist


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ProvenanceRecords(unittest.TestCase):
    """verify() on small trees: every edit kind, in commit order, on bytes."""

    ORIGINAL = {
        "a.py": "import b\nfrom c import d\n\nx = 1  # é\n\x0cy = 2\n",
        "b.py": "one\ntwo\nthree\nfour\n",
        "gone.py": "removed later\n",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.imported = {name: sha(text.encode()) for name, text in self.ORIGINAL.items()}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, files):
        for path in self.root.iterdir():
            path.unlink()
        for name, text in files.items():
            (self.root / name).write_bytes(text.encode())
        return sorted(files)

    def verify(self, files, edits):
        return provenance.verify(self.root, self.imported, edits, self.write(files))

    def test_unedited_tree(self):
        self.assertEqual(self.verify(self.ORIGINAL, []), [])
        with self.assertRaisesRegex(ValueError, "b.py with its recorded edits reverted is not the imported file"):
            self.verify({**self.ORIGINAL, "b.py": "one\ntwo\nthree\nfive\n"}, [])
        with self.assertRaisesRegex(ValueError, "not accounted for"):
            self.verify({**self.ORIGINAL, "new.py": ""}, [])

    def test_every_kind_in_commit_order(self):
        a = "from . import b\nfrom .c import d\n\nx = 1  # é\n\x0cy = 2\n"
        b = "zero\none\nTWO\nfour\nfive\n"
        edits = [
            dict(commit_subject="relative imports",
                 imports={"a.py": [[1, "import b", "from . import b"], [2, "from c import d", "from .c import d"]]}),
            dict(commit_subject="insert, replace and delete lines",
                 hunks={"b.py": [[1, [], ["zero"]], [3, ["two", "three"], ["TWO"]], [5, [], ["five"]]]}),
            dict(commit_subject="drop a file and add one", removed={"gone.py": self.imported["gone.py"]},
                 added={"new.py": sha(b"print()\n")}),
        ]
        self.assertEqual(self.verify({"a.py": a, "b.py": b, "new.py": "print()\n"}, edits), ["a.py", "b.py", "new.py"])
        edits[1]["hunks"]["b.py"][1] = [2, ["two", "three"], ["TWO"]]
        with self.assertRaisesRegex(ValueError, "b.py differs from the recorded edit of line 2"):
            self.verify({"a.py": a, "b.py": b, "new.py": "print()\n"}, edits)
        edits[1]["hunks"]["b.py"][1] = [1, ["two", "three"], ["one", "TWO"]]
        with self.assertRaisesRegex(ValueError, "overlapping edits of b.py"):
            self.verify({"a.py": a, "b.py": b, "new.py": "print()\n"}, edits)

    def test_a_deletion_next_to_an_edited_line(self):
        # 'import b' made relative and the line before it deleted, as when a sys.path line goes.
        edits = [dict(commit_subject="c", imports={"a.py": [[1, "from c import d", "from .c import d"]]},
                      hunks={"a.py": [[1, ["import b"], []]]})]
        self.assertEqual(self.verify({**self.ORIGINAL, "a.py": "from .c import d\n\nx = 1  # é\n\x0cy = 2\n"}, edits),
                         ["a.py"])
        edits[0]["hunks"]["a.py"].append([1, ["gone"], []])
        with self.assertRaisesRegex(ValueError, "overlapping edits of a.py at lines 1 and 1"):
            self.verify({**self.ORIGINAL, "a.py": "from .c import d\n\nx = 1  # é\n\x0cy = 2\n"}, edits)

    def test_later_edits_of_a_rewritten_file(self):
        rewritten = "completely\nnew\n"
        edits = [dict(commit_subject="rewrite", replaced={"b.py": dict(before=self.imported["b.py"],
                                                                         after=sha(rewritten.encode()))}),
                 dict(commit_subject="touch up", hunks={"b.py": [[2, ["new"], ["newer"]]]})]
        self.assertEqual(self.verify({**self.ORIGINAL, "b.py": "completely\nnewer\n"}, edits), ["b.py"])
        with self.assertRaisesRegex(ValueError, "b.py differs from the file written by 'rewrite'"):
            self.verify({**self.ORIGINAL, "b.py": "completely\nnew!\n"}, edits[:1])
        early = dict(commit_subject="before the rewrite", hunks={"b.py": [[1, ["one"], ["ONE"]]]})
        with self.assertRaisesRegex(ValueError, "no file b.py to revert"):
            self.verify({**self.ORIGINAL, "b.py": "completely\nnewer\n"}, [early] + edits)

    def test_refusals(self):
        not_import = [dict(commit_subject="c", imports={"b.py": [[2, "two", "2"]]})]
        with self.assertRaisesRegex(ValueError, "line 2 is not a relative import"):
            self.verify({**self.ORIGINAL, "b.py": "one\n2\nthree\nfour\n"}, not_import)
        with self.assertRaisesRegex(ValueError, "unknown edit kinds"):
            self.verify(self.ORIGINAL, [dict(commit_subject="c", lines={})])
        twice = dict(commit_subject="c", hunks={"b.py": []}, replaced={"b.py": dict(before="", after="")})
        with self.assertRaisesRegex(ValueError, "both edited by lines and added, removed or replaced"):
            self.verify(self.ORIGINAL, [twice])
        with self.assertRaisesRegex(ValueError, "gone.py was removed by 'c' but exists"):
            self.verify(self.ORIGINAL, [dict(commit_subject="c", removed={"gone.py": self.imported["gone.py"]})])

    def test_flat_import(self):
        for new, old in (("from . import x", "import x"), ("    from .x import y as z  # noqa", "    from x import y as z  # noqa"),
                         ("v=__import__('m',globals(),level=1).f(t)", "v=__import__('m').f(t)"),
                         ("from ..optimizer import x", "from ..optimizer import x"), ("s = 'from .x import y'", "s = 'from .x import y'")):
            self.assertEqual(provenance.flat_import(new), old)

    def test_inert(self):
        for text, expected in (('"""Docstring."""\n__version__ = "1"\n', True), ('"""Docstring."""\nimport os\n', False),
                               ("def f():\n    pass\n", False), ("x = y = 1\n", True), ("x.y = 1\n", False)):
            path = self.root / "module.py"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(provenance.inert(path), expected, text)


class Packaging(unittest.TestCase):
    @unittest.skipUnless((PROJECT / "pyproject.toml").exists(), "no pyproject.toml next to the tests")
    def test_the_version_is_written_once(self):
        project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertNotIn("version", project["project"])
        self.assertIn("version", project["project"]["dynamic"])
        self.assertEqual(project["tool"]["hatch"]["version"], dict(path="src/rcmps_gpu/__init__.py"))
        self.assertEqual(project["project"]["license-files"], ["LICENSE"])

    def test_an_installed_distribution_has_the_version_of_its_package(self):
        try:
            distribution = importlib.metadata.distribution("rcmps-gpu")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("rcmps-gpu is not installed")
        if Path(distribution.locate_file("rcmps_gpu/__init__.py")).resolve() != Path(rcmps_gpu.__file__).resolve():
            self.skipTest("the installed rcmps-gpu is not the copy under test")
        self.assertEqual(distribution.version, rcmps_gpu.__version__)
        self.assertTrue(any(Path(str(f)).name == "LICENSE" for f in distribution.files))

    @unittest.skipUnless((PROJECT.parent / "Project.toml").exists(), "not in the RcmpsCore.jl repository")
    def test_the_license_is_that_of_the_repository(self):
        self.assertEqual((PROJECT / "LICENSE").read_bytes(), (PROJECT.parent / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
