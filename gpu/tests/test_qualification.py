"""The GPU qualification, on the CPU.

RunAll       qualification/run_all.sh: shell syntax, and every option it gives a script is
             one that script accepts (its --help lists it). The scripts run only on a GPU;
             this catches a renamed option or script before a rental does.
Optimizer    qualification/optimizer/gpu_optimizer_check.py starts every rcmps-gpu-optimize
             run with its own model, integration, seed, device and watchdog settings.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import rcmps_gpu
from rcmps_gpu.optimizer import cli

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


@unittest.skipUnless(RUN_ALL.is_file(), "qualification/ is not here")
class Optimizer(unittest.TestCase):
    def test_runs_get_the_settings_of_the_check(self):
        """The command of a compile or run stage, as rcmps-gpu-optimize parses it, has the
        coupling, window, quadrature, tolerances, seed, device and watchdog of the check
        (the trajectory stage uses the same), with the defaults of the command line when
        none is given."""
        spec = importlib.util.spec_from_file_location("gpu_optimizer_check",
                                                      QUALIFICATION / "optimizer" / "gpu_optimizer_check.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        given = ["--coupling", "1.25", "--window", "20.5", "--quadrature", "256", "--atol", "3e-14", "--rtol", "2e-13",
                 "--seed", "7", "--device", "1", "--max-device-seconds", "120"]
        for options in ([], given):
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(check.subprocess, "run") as run:
                run.return_value = SimpleNamespace(returncode=0)
                args = check.parser().parse_args(["--output", tmp, *options])
                (Path(tmp) / "compile").mkdir()
                check.optimize(args, "compile", 16, "scalar", "--compile-only")
                command = run.call_args.args[0]
            self.assertEqual(command[1:3], ["-m", "rcmps_gpu.optimizer"])
            parsed = cli.parser().parse_args(command[3:])
            self.assertEqual((parsed.D, parsed.tiles, parsed.run_gpu, parsed.compile_only), (16, "scalar", True, True))
            self.assertEqual((parsed.coupling, parsed.window, parsed.quadrature, parsed.atol, parsed.rtol, parsed.seed,
                              parsed.device, parsed.max_device_seconds),
                             (args.coupling, args.window, args.quadrature, args.atol, args.rtol, args.seed, args.device,
                              args.max_device_seconds))
            if not options:
                self.assertEqual((parsed.window, parsed.quadrature, parsed.atol, parsed.rtol),
                                 tuple(cli.DEFAULTS[name] for name in ("window", "quadrature", "atol", "rtol")))
                self.assertEqual((parsed.coupling, parsed.max_device_seconds),
                                 (2.7, cli.parser().get_default("max_device_seconds")))
            else:
                self.assertEqual((parsed.coupling, parsed.quadrature, parsed.atol), (1.25, 256, 3e-14))





if __name__ == "__main__":
    unittest.main()
