"""The GPU qualification, on the CPU.

RunAll       qualification/run_all.sh: shell syntax, and every option it gives a script is
             one that script accepts (its --help lists it). The scripts run only on a GPU;
             this catches a renamed option or script before a rental does. With a stand-in
             interpreter, the excitation solves at D = 64 are recorded as failed in
             status.tsv, and in the exit status, unless their level is accepted.
Optimizer    qualification/optimizer/gpu_optimizer_check.py starts every rcmps-gpu-optimize
             run with its own model, integration, seed, device and watchdog settings; its
             trajectory stage reports a grid the device cannot hold as refused, and fails on
             any other error.
Excitations  the scripts of qualification/excitations that run_all.sh calls, run as written
             on a stand-in device (qualification_on_cpu.py: NumPy arrays, a fused kernel
             whose launch is the host integrator): their exit status is 0 when every check
             passes or the device refuses a size, and 1 when a kernel fails to compile, when
             its PTX fails the audit, when a repeated launch is not the same bytes, or when a
             result is off by more than the tolerance; each failed check is named.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
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

import numpy as np

import rcmps_gpu
from rcmps_gpu.optimizer import cli
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend

TESTS = Path(__file__).resolve().parent
QUALIFICATION = TESTS.parent / "qualification"
RUN_ALL = QUALIFICATION / "run_all.sh"
REPOSITORY = TESTS.parents[1]
FIXTURES = Path(os.environ.get("RCMPS_TEST_FIXTURES") or REPOSITORY / "test" / "fixtures")
D4 = FIXTURES / "phi4_g1_D4_z2_state.jld2"


def commands():
    """{script: options} of the commands of run_all.sh, continuation lines joined."""
    text = RUN_ALL.read_text().replace("\\\n", " ")
    found = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        # solve "DIR" STATE [options] runs rcmps-gpu-excitations with these options.
        for match in re.finditer(r'(\$OPT|"\$EXC/(\w+\.py)"|-m rcmps_gpu\.excitations|\bsolve ")(.*)', line):
            script = ("optimizer/gpu_optimizer_check.py" if match.group(1) == "$OPT" else
                      f"excitations/{match.group(2)}" if match.group(2) else "-m rcmps_gpu.excitations")
            found.setdefault(script, set()).update(re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", match.group(3)))
    return found


ENVIRONMENT = dict(os.environ, OPENBLAS_NUM_THREADS="1",
                   PYTHONPATH=os.pathsep.join([str(Path(rcmps_gpu.__file__).parents[1]),
                                               *filter(None, [os.environ.get("PYTHONPATH")])]))


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
        self.assertTrue({"--tiles", "--output"} <= found["-m rcmps_gpu.excitations"])
        self.assertEqual(set(found), {"optimizer/gpu_optimizer_check.py", "excitations/gpu_bringup.py",
                                      "excitations/gpu_fused_check.py", "excitations/gpu_pair_check.py",
                                      "excitations/gpu_action_check.py", "excitations/check_references.py",
                                      "-m rcmps_gpu.excitations"})
        for script, options in sorted(found.items()):
            command = [sys.executable, *(script.split() if script.startswith("-m") else [str(QUALIFICATION / script)])]
            with self.subTest(script=script):
                result = subprocess.run([*command, "--help"], capture_output=True, text=True, env=ENVIRONMENT,
                                        timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("cupy", result.stderr)
                for option in sorted(options):
                    self.assertRegex(result.stdout, rf"(?<![\w-]){re.escape(option)}\b", f"{script}: {option}")

    def test_a_level_that_is_not_accepted_fails_its_step(self):
        """The steps excitation_d64 and excitation_d64_scalar, on a stand-in interpreter
        (PYTHON) that reports compute capability 8.0 and whose rcmps-gpu-excitations writes a
        result.json with "accepted" true or false: status.tsv records exit status 0 or 1 for
        each, and so does run_all.sh."""
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "python"
            fake.write_text('#!/usr/bin/env bash\n'
                            'if [ "$1" = "-c" ]; then echo 80; exit 0; fi\n'
                            'if [ "$1" = "-m" ] && [ "$2" = "rcmps_gpu.excitations" ]; then\n'
                            '    while [ $# -gt 0 ]; do [ "$1" = "--output" ] && out=$2; shift; done\n'
                            '    mkdir -p "$out" && echo "{\\"accepted\\": $ACCEPTED}" > "$out/result.json"; exit 0\n'
                            'fi\n'
                            f'exec "{sys.executable}" "$@"\n')
            fake.chmod(0o755)
            for accepted, status in (("true", 0), ("false", 1)):
                output = Path(tmp) / f"qualification_{accepted}"
                environment = dict(ENVIRONMENT, PYTHON=str(fake), ACCEPTED=accepted, RCMPS_STATE_D64=tmp,
                                   RCMPS_STEPS="excitation_d64 excitation_d64_scalar")
                result = subprocess.run([bash, str(RUN_ALL), str(output)], capture_output=True, text=True,
                                        env=environment, timeout=120)
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                rows = [line.split("\t") for line in (output / "status.tsv").read_text().splitlines()[1:]]
                self.assertEqual([(row[0], row[1]) for row in rows],
                                 [("13_excitation_d64", str(status)), ("14_excitation_d64_scalar", str(status))])
                log = (output / "logs" / "13_excitation_d64.log").read_text()
                self.assertIn(f'"accepted": {accepted}', log)


def optimizer_check():
    """qualification/optimizer/gpu_optimizer_check.py as a module (a fresh one: FAILED is empty)."""
    spec = importlib.util.spec_from_file_location("gpu_optimizer_check",
                                                  QUALIFICATION / "optimizer" / "gpu_optimizer_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    return check


@unittest.skipUnless(RUN_ALL.is_file(), "qualification/ is not here")
class Optimizer(unittest.TestCase):
    def test_a_grid_the_device_cannot_hold_is_refused_in_the_trajectory_stage(self):
        """stage_trajectory at D = 8 on a stand-in compute capability 7.0 (NumPy for CuPy)
        whose PersistentTrajectoryDOP853 raises: the residency error of the backend at the
        last register cap is "refused" with ok, a compilation error fails (ok false, in
        FAILED)."""
        for error, refused in (("unsafe cooperative trajectory grid 11>10", True),
                               ("CompileException: stand-in compilation failure", False)):
            check = optimizer_check()

            def solver(*arguments, **options):
                raise RuntimeError(error)

            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out, \
                    mock.patch.object(persistent_trajectory_backend, "PersistentTrajectoryDOP853", solver):
                args = check.parser().parse_args(["--output", tmp])
                (Path(tmp) / "trajectory").mkdir()      # as main() does
                check.stage_trajectory(np, args, (7, 0), 8, ["auto"])
            row = json.loads(out.getvalue().splitlines()[-1])
            self.assertEqual((row["stage"], row["D"], row["tiles"], row["ok"]), ("trajectory", 8, "scalar", refused))
            self.assertIn(error, row["refused" if refused else "error"])
            self.assertEqual(len(check.FAILED), 0 if refused else 1)

    def test_runs_get_the_settings_of_the_check(self):
        """The command of a compile or run stage, as rcmps-gpu-optimize parses it, has the
        coupling, window, quadrature, tolerances, seed, device and watchdog of the check
        (the trajectory stage uses the same), with the defaults of the command line when
        none is given."""
        check = optimizer_check()
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




def on_cpu(mode, script, *options, capability="8.0"):
    """(exit status, JSON rows by stage, stderr) of qualification/excitations/SCRIPT run by
    qualification_on_cpu.py in MODE on a stand-in device of this compute capability."""
    environment = dict(ENVIRONMENT, RCMPS_CPU_COMPUTE_CAPABILITY=capability)
    result = subprocess.run([sys.executable, str(TESTS / "qualification_on_cpu.py"), mode,
                             str(QUALIFICATION / "excitations" / script), *options],
                            capture_output=True, text=True, env=environment, timeout=600)
    rows = {}
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            rows.setdefault(row["stage"], []).append(row)
    return result.returncode, rows, result.stderr


@unittest.skipUnless(RUN_ALL.is_file(), "qualification/ is not here")
class Excitations(unittest.TestCase):
    def outcome(self, mode, script, *options, status, capability="8.0"):
        code, rows, stderr = on_cpu(mode, script, *options, capability=capability)
        self.assertEqual(code, status, stderr[-3000:])
        failed = sum(row.get("ok") is False for stage in rows.values() for row in stage)
        self.assertEqual(rows["summary"], [dict(stage="summary", failed=failed)])
        self.assertEqual(failed > 0, status == 1)
        return rows

    def test_fused_check(self):
        """gpu_fused_check.py (random D = 4, both paths at 8.0): equal samples pass, and the
        paths are compared bitwise; a second launch that differs in the last bit fails
        repeat_bitwise, though it is within the tolerance (the check copies the samples
        before relaunching, since a launch overwrites the kernel's buffer); samples off by
        1e-6 fail against the host integrator; a compilation error fails; a size the device
        cannot hold is refused, not failed."""
        options = ("gpu_fused_check.py", "--random", "4", "--quick")
        row = self.outcome("exact", *options, status=0)["fused"][0]
        for path in ("mma", "scalar"):
            self.assertEqual((row[path]["ok"], row[path]["repeat_bitwise"], row[path]["relative"],
                              row[path]["failed_checks"]), (True, True, 0.0, []))
        self.assertEqual((row["mma_vs_scalar"], row["mma_vs_scalar_bitwise"], row["mma_vs_scalar_same_steps"]),
                         (0.0, True, True))
        row = self.outcome("unstable", *options, status=1)["fused"][0]
        self.assertEqual(row["mma"]["failed_checks"], ["repeat_bitwise"])
        self.assertLess(row["mma"]["relative"], 1e-15)
        row = self.outcome("perturbed", *options, status=1)["fused"][0]
        self.assertEqual(row["scalar"]["failed_checks"], ["relative", "action_relative_difference"])
        row = self.outcome("broken", *options, status=1)["fused"][0]
        self.assertIn("stand-in compilation failure", row["mma"]["error"])
        row = self.outcome("refused", *options, status=0)["fused"][0]
        self.assertIn("unsafe cooperative grid", row["scalar"]["refused"])
        row = self.outcome("exact", *options, "--tiles", "mma", status=0, capability="7.0")["fused"][0]
        self.assertIn("--tiles mma needs", row["mma"]["refused"])

    def test_compile_stage(self):
        """gpu_bringup.py --stages compile (d = 2, both kernels): resident kernels whose PTX
        passes the audit pass; an FP32 instruction in the PTX fails the audit; a compilation
        error fails; a grid that is not resident and --tiles mma below 8.0 are refused."""
        options = ("gpu_bringup.py", "--stages", "compile", "--packed", "2")
        rows = self.outcome("exact", *options, status=0)["compile"]
        self.assertEqual([(row["kernel"], row["tiles"]) for row in rows],
                         [("action", "mma"), ("action", "scalar"), ("pair", "mma"), ("pair", "scalar")])
        for row in rows:
            self.assertEqual((row["resident"], row["ptx_audit"]["ptx_target"], row["ptx_audit"]["tiles"]),
                             (True, "sm_80", row["tiles"]))
        rows = self.outcome("fp32", *options, status=1)["compile"]
        self.assertTrue(all("PTX audit failed" in row["error"] and "add.f32" in row["error"] for row in rows), rows)
        rows = self.outcome("broken", *options, status=1)["compile"]
        self.assertTrue(all(row["resident"] is False and "stand-in compilation failure" in row["error"] for row in rows))
        rows = self.outcome("refused", *options, status=0)["compile"]
        self.assertTrue(all(row["resident"] is False and "unsafe cooperative grid" in row["refused"] for row in rows))
        rows = self.outcome("exact", *options, "--tiles", "mma", "scalar", status=0, capability="7.0")["compile"]
        self.assertEqual([(row["tiles"], row["resident"], "refused" in row) for row in rows],
                         [("mma", False, True), ("scalar", True, False)] * 2)
        self.assertEqual(rows[1]["ptx_audit"]["ptx_target"], "sm_70")

    @unittest.skipUnless(D4.is_file(), f"the fixture {D4} is not here")
    def test_pair_and_action_checks(self):
        """gpu_pair_check.py and gpu_action_check.py --reference on the D = 4 fixture: the
        host-integrator launch passes; a pair value that changes between launches, and
        launches off by 1e-6, fail against the host references."""
        options = ("--state", str(D4), "--reference", "--n-quad", "32", "--window", "10")
        row = self.outcome("exact", "gpu_pair_check.py", *options, status=0)["pair"][0]
        for path in ("mma", "scalar"):
            self.assertEqual((row[path]["ok"], row[path]["repeat_bitwise"], row[path]["relative_difference_from_host_packed"]),
                             (True, True, 0.0))
            self.assertLess(row[path]["relative_difference_from_host_unpacked"], 1e-11)
        self.assertTrue(row["mma_vs_scalar_bitwise"])
        row = self.outcome("unstable", "gpu_pair_check.py", *options, status=1)["pair"][0]
        self.assertEqual(row["mma"]["failed_checks"], ["repeat_bitwise"])
        row = self.outcome("perturbed", "gpu_pair_check.py", *options, status=1)["pair"][0]
        self.assertEqual(row["scalar"]["failed_checks"], ["relative_difference_from_host_packed",
                                                          "relative_difference_from_host_unpacked"])
        rows = self.outcome("exact", "gpu_action_check.py", *options, "--nodes", "32", status=0)
        self.assertEqual((rows["action"][0]["ok"], rows["action"][0]["tiles"], rows["action_reference"][0]["ok"]),
                         (True, "mma", True))
        rows = self.outcome("perturbed", "gpu_action_check.py", *options, "--nodes", "32", "--tiles", "scalar", status=1)
        self.assertEqual((rows["action"][0]["ok"], rows["action"][0]["tiles"], rows["action_reference"][0]["ok"]),
                         (True, "scalar", False))


if __name__ == "__main__":
    unittest.main()
