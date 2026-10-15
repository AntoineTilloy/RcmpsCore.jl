"""The GPU qualification batch qualification/run_all.sh, on the CPU: shell syntax, and every
option it gives a script is one that script accepts (its --help lists it). The scripts run
only on a GPU; this catches a renamed option or script before a rental does.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

import rcmps_gpu

QUALIFICATION = Path(__file__).resolve().parents[1] / "qualification"
RUN_ALL = QUALIFICATION / "run_all.sh"


def commands():
    """{script: options} of the commands of run_all.sh, continuation lines joined."""
    text = RUN_ALL.read_text().replace("\\\n", " ")
    found = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for match in re.finditer(r'(\$OPT|"\$EXC/(\w+\.py)"|-m rcmps_gpu\.excitations)(.*)', line):
            script = ("optimizer/gpu_optimizer_check.py" if match.group(1) == "$OPT" else
                      f"excitations/{match.group(2)}" if match.group(2) else "-m rcmps_gpu.excitations")
            found.setdefault(script, set()).update(re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", match.group(3)))
    return found


@unittest.skipUnless(RUN_ALL.is_file(), "qualification/ is not here")
class RunAll(unittest.TestCase):
    def test_shell_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        result = subprocess.run([bash, "-n", str(RUN_ALL)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_options_exist(self):
        found = commands()
        self.assertEqual(set(found), {"optimizer/gpu_optimizer_check.py", "excitations/gpu_bringup.py",
                                      "excitations/gpu_fused_check.py", "excitations/gpu_pair_check.py",
                                      "excitations/check_references.py", "-m rcmps_gpu.excitations"})
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1",
                           PYTHONPATH=os.pathsep.join([str(Path(rcmps_gpu.__file__).parents[1]),
                                                       *filter(None, [os.environ.get("PYTHONPATH")])]))
        for script, options in sorted(found.items()):
            command = [sys.executable, *(script.split() if script.startswith("-m") else [str(QUALIFICATION / script)])]
            with self.subTest(script=script):
                result = subprocess.run([*command, "--help"], capture_output=True, text=True, env=environment,
                                        timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("cupy", result.stderr)
                for option in sorted(options):
                    self.assertRegex(result.stdout, rf"(?<![\w-]){re.escape(option)}\b", f"{script}: {option}")


if __name__ == "__main__":
    unittest.main()
