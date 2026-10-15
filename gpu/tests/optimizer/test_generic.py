"""CPU regressions for generated dimensions, physics, provenance and limits.

These tests never import CuPy, query a GPU or evaluate a D64 CPU objective.
They do not establish actual-GPU correctness for newly generated dimensions.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import rcmps_gpu
from rcmps_gpu import provenance
from rcmps_gpu.optimizer import cli as run
from rcmps_gpu.optimizer import layout

RUNTIME = run.RUNTIME
PROVENANCE = json.loads((run.HERE / "provenance.json").read_bytes())
GOLDEN = json.loads(Path(__file__).with_name("golden_cuda.json").read_bytes())
GOLDEN_SCALAR = json.loads(Path(__file__).with_name("golden_cuda_scalar.json").read_bytes())
BASE_MANIFEST_SHA = "4b68b5fc76de4cfa3b049364ad1bc9ab7be2f61ea56a2eec0fe9cf9450efb21e"
# sha256 of research generic_gpu/runtime/manifest.json, imported by "Import the GPU code verbatim"
IMPORTED_MANIFEST_SHA = "dda1fdbfa9f435f04958e418379f269b2381cc3e58d9b09bb81c7f42bdd6beea"


def child(code):
    """Run code in a fresh interpreter, which must import this same copy of rcmps_gpu."""
    prefix = ("import sys,json,rcmps_gpu\nassert rcmps_gpu.__file__==" + repr(rcmps_gpu.__file__)
              + ",'child imported another rcmps_gpu: '+rcmps_gpu.__file__\n")
    result = subprocess.run([sys.executable, "-c", prefix + code],
                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stdout + "\n" + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def package_copy(root):
    """A copy of this rcmps_gpu source tree under root/src, for tests that edit it."""
    shutil.copytree(run.PACKAGE, root / "src" / "rcmps_gpu", ignore=shutil.ignore_patterns("__pycache__"))
    return root / "src" / "rcmps_gpu"


def run_copy(root, *arguments, code=None):
    """python -m rcmps_gpu.optimizer *arguments (or -c code) on the copy under root/src."""
    command = [sys.executable, "-c", code] if code else [sys.executable, "-m", "rcmps_gpu.optimizer", *arguments]
    return subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONPATH=str(root / "src"), OPENBLAS_NUM_THREADS="1"))


class GenericTests(unittest.TestCase):
    def test_provenance_lists_the_runtime_and_reverting_its_edits_gives_the_imported_bytes(self):
        manifest = PROVENANCE["manifest"]
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), IMPORTED_MANIFEST_SHA)
        self.assertEqual(PROVENANCE["imported"]["manifest_sha256"], IMPORTED_MANIFEST_SHA)
        self.assertEqual(run.IMPORTED_MANIFEST_SHA256, IMPORTED_MANIFEST_SHA)
        self.assertEqual(manifest["base_manifest_sha256"], BASE_MANIFEST_SHA)
        self.assertFalse(manifest["actual_GPU_qualified"])
        files = sorted(path.name for path in RUNTIME.glob("*.py") if path.name != "__init__.py")
        # 24 modules were imported; the pruning commit deleted two, recorded with their imported sha256.
        removed = {name: digest for entry in PROVENANCE["edits"] for name, digest in entry.get("removed", {}).items()}
        self.assertEqual(len(removed), 2)
        self.assertEqual(len(manifest["runtime_source_sha256"]), 24)
        self.assertEqual({name: manifest["runtime_source_sha256"][name] for name in removed}, removed)
        self.assertEqual(len(files), 22)
        self.assertEqual(sorted(set(manifest["runtime_source_sha256"]) - set(removed)), files)
        self.assertEqual(sorted(run.runtime_source_sha256()), files)
        record = run.verify_runtime()   # reverts the recorded edits on the bytes of the files
        self.assertEqual(record["edits"], [entry["commit_subject"] for entry in PROVENANCE["edits"]])
        self.assertEqual(len(record["edited_files"]), 19)

    def test_planner_and_worker_refuse_a_runtime_that_differs_from_its_provenance(self):
        line = "        factor = min(10.0, 0.9 * error**(-1 / 8)) if error else 10.0\n"
        edited = line.replace("0.9 *", "0.8 *")   # the DOP853 step-size safety factor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = package_copy(root)
            target = package / "optimizer" / "runtime" / "ode_backend.py"
            record_path = package / "optimizer" / "provenance.json"
            source = target.read_text(encoding="utf-8")
            self.assertEqual(source.count(line), 1)
            number = source[:source.index(line)].count("\n") + 1
            planned = run_copy(root, code="import json\nfrom rcmps_gpu.optimizer import cli\n"
                               "print(json.dumps(cli.prepare(cli.parser().parse_args(['--D','8','--output','run']),"
                               "cli.runtime_source_sha256())))")
            self.assertEqual(planned.returncode, 0, planned.stderr)
            target.write_text(source.replace(line, edited), encoding="utf-8")
            refused = run_copy(root, "--D", "8")
            self.assertEqual(refused.returncode, 2)
            self.assertIn("runtime differs from its recorded provenance: ode_backend.py with its recorded edits "
                          "reverted is not the imported file", refused.stderr)
            # The worker refuses it too, even with the runtime hashes of the plan made to match.
            plan = json.loads(planned.stdout)
            plan["runtime_source_sha256"]["ode_backend.py"] = hashlib.sha256(target.read_bytes()).hexdigest()
            (root / "plan.json").write_text(json.dumps(plan))
            worker = run_copy(root, code="from rcmps_gpu.optimizer import worker\nworker.main(['plan.json'])")
            self.assertIn("ValueError: runtime differs from its recorded provenance", worker.stderr)
            self.assertFalse((root / "run").exists())
            # Rewriting the imported manifest to match the edited file is refused as well.
            record = json.loads(record_path.read_bytes())
            lines = target.read_bytes().splitlines(keepends=True)
            for n, old, new in record["edits"][0]["imports"].get("ode_backend.py", []):
                lines[n - 1] = (old + "\n").encode()
            manifest = record["manifest"]
            manifest["runtime_source_sha256"]["ode_backend.py"] = hashlib.sha256(b"".join(lines)).hexdigest()
            record["imported"]["manifest_sha256"] = hashlib.sha256(
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
            record_path.write_text(json.dumps(record, indent=1) + "\n")
            self.assertIn("the imported runtime manifest changed", run_copy(root, "--D", "8").stderr)
            # A recorded edit is accepted, and the plan lists it.
            record = json.loads((run.HERE / "provenance.json").read_bytes())
            record["edits"].append(dict(commit_subject="test", hunks={"ode_backend.py": [
                [number, [line.rstrip("\n")], [edited.rstrip("\n")]]]}))
            record_path.write_text(json.dumps(record, indent=1) + "\n")
            accepted = run_copy(root, "--D", "8")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads(accepted.stdout)["runtime_provenance"]["edits"][-1], "test")
            # Code in a package __init__ runs with every runtime import: refused.
            target.write_text(source, encoding="utf-8")
            record_path.write_bytes((run.HERE / "provenance.json").read_bytes())
            self.assertEqual(run_copy(root, "--D", "8").returncode, 0)
            with (package / "optimizer" / "runtime" / "__init__.py").open("a", encoding="utf-8") as stream:
                stream.write("from . import ode_backend\n")
            self.assertIn("rcmps_gpu/optimizer/runtime/__init__.py must hold only a docstring and constants",
                          run_copy(root, "--D", "8").stderr)

    def test_provenance_check_reads_bytes_whatever_the_locale(self):
        result = subprocess.run([sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-c",
                                 "import json\nfrom rcmps_gpu.optimizer import cli\n"
                                 "print(json.dumps(cli.verify_runtime()))"],
                                env=dict(os.environ, PYTHONUTF8="0", LC_ALL="C", OPENBLAS_NUM_THREADS="1"),
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), run.verify_runtime())

    def test_every_file_that_runs_with_the_optimizer_is_hashed(self):
        package = run.PACKAGE
        files = {path.relative_to(package).as_posix() for path in package.rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts
                 and (path.parent == package or path.relative_to(package).parts[0] == "optimizer")}
        runtime = {"optimizer/runtime/" + name for name in run.runtime_source_sha256()}
        self.assertEqual(files, runtime | set(run.LAUNCHER_SOURCES))
        for name in run.INERT_SOURCES:
            self.assertTrue(provenance.inert(package / name), name)

    def test_generated_CUDA_compiler_options_and_geometry_equal_the_golden_hashes(self):
        sizes = sorted(map(int, GOLDEN["sizes"]))
        self.assertLessEqual(set(GOLDEN["historical_sizes"]) | {80, 96}, set(sizes))
        code = '''
import hashlib
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as t
print(json.dumps({str(d):dict(trajectory_sha256=hashlib.sha256(t.trajectory_source(d).encode()).hexdigest(),
    options=list(t.compilation_options(d)),geometry=t.geometry(d)) for d in SIZES}))
assert 'cupy' not in sys.modules
'''.replace("SIZES", repr(tuple(sizes)))
        self.assertEqual(child(code), GOLDEN["sizes"])

    def test_scalar_tile_CUDA_equals_its_golden_hashes_and_differs_only_by_the_tile_product(self):
        # The tensor-core path is the validated source byte for byte (tiles="mma" is the
        # default); the scalar path replaces the one definition of mma().
        sizes = sorted(map(int, GOLDEN["sizes"]))
        self.assertEqual(sorted(map(int, GOLDEN_SCALAR["sizes"])), sizes)
        code = '''
import hashlib
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as t
from rcmps_gpu.excitations import fused
rows={}
for d in SIZES:
    mma,scalar=t.trajectory_source(d,'mma'),t.trajectory_source(d,'scalar')
    assert mma==t.trajectory_source(d) and mma.count(t.MMA_TILE)==1 and scalar.count(t.SCALAR_TILE)==1
    assert scalar.replace(t.SCALAR_TILE,t.MMA_TILE)==mma
    assert 'mma.sync' not in scalar and 'asm' not in scalar and '__shfl_sync' not in mma
    rows[str(d)]=dict(trajectory_sha256=hashlib.sha256(scalar.encode()).hexdigest())
# The scalar tile product is the one of the excitation kernel ('#else' branch of fused.CUDA).
assert t.SCALAR_TILE==fused.CUDA.split('#else\\n',1)[1].split('#endif\\n',1)[0]
for tiles in ('auto','MMA',None):
    try:t.trajectory_source(8,tiles)
    except ValueError:pass
    else:raise AssertionError(tiles)
print(json.dumps(rows))
assert 'cupy' not in sys.modules
'''.replace("SIZES", repr(tuple(sizes)))
        self.assertEqual(child(code), GOLDEN_SCALAR["sizes"])

    def test_emitted_CUDA_equals_the_golden_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "D16.cu"
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                self.assertEqual(run.main(["--D", "16", "--emit-cuda", str(target)]), 0)
            plan = json.loads(printed.getvalue())
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(digest, GOLDEN["sizes"]["8"]["trajectory_sha256"])
            self.assertEqual(plan["emitted_cuda_sha256"], digest)

    def test_children_import_this_copy_whatever_the_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            decoy = Path(tmp) / "rcmps_gpu"
            (decoy / "optimizer" / "runtime").mkdir(parents=True)
            for directory in (decoy, decoy / "optimizer", decoy / "optimizer" / "runtime"):
                (directory / "__init__.py").write_text("raise ImportError('decoy rcmps_gpu')\n", encoding="utf-8")
            command, environment = run.child_process("-c", "import rcmps_gpu\nprint(rcmps_gpu.__file__)")
            self.assertEqual(command[:2], [sys.executable, "-P"])
            found = subprocess.run(command, env=environment, cwd=tmp, capture_output=True, text=True, timeout=60)
            self.assertEqual(Path(found.stdout.strip()).resolve(), Path(rcmps_gpu.__file__).resolve(), found.stderr)
            # Without -P the working directory comes first and the decoy is imported.
            shadowed = subprocess.run([sys.executable, *command[2:]], env=environment, cwd=tmp,
                                      capture_output=True, text=True, timeout=60)
            self.assertIn("decoy rcmps_gpu", shadowed.stderr)
            target, cwd = Path(tmp) / "D16.cu", os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(["--D", "16", "--emit-cuda", str(target)]), 0)
                # The worker is started the same way (not run here).
                with mock.patch.object(run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as spawn:
                    self.assertEqual(run.main(["--D", "8", "--run-gpu", "--output", "run"]), 0)
            finally:
                os.chdir(cwd)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), GOLDEN["sizes"]["8"]["trajectory_sha256"])
            worker = spawn.call_args_list[-1]
            self.assertEqual(worker.args[0], [sys.executable, "-P", "-m", "rcmps_gpu.optimizer.worker",
                                              str((Path(tmp) / "run" / "plan.json").resolve())])
            self.assertEqual(worker.kwargs["env"], environment)

    def test_a_failed_CUDA_generation_reports_the_error_of_the_child(self):
        failed = subprocess.CompletedProcess([], 1, "", "Traceback (most recent call last):\nValueError: mock failure\n")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "D16.cu"
            with mock.patch.object(run.subprocess, "run", return_value=failed), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                    self.assertRaises(SystemExit) as stopped:
                run.main(["--D", "16", "--emit-cuda", str(target)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("CUDA source generation failed: ValueError: mock failure", stderr.getvalue())
            self.assertFalse(target.exists())

    def test_every_even_D_has_geometry_and_non_integer_odd_oversize_rejected(self):
        for D in range(2, 513, 2):
            info = layout.bond_geometry(D)
            self.assertEqual(info["threads"] % 32, 0)
            self.assertLessEqual(info["threads"], 1024)
            self.assertEqual(info["blocks"], 10 * info["warps"] + 1)
        for D in (-2, 0, 1, 3, 159, 513, 514, True, 160., "160"):
            with self.subTest(D=D), self.assertRaises(ValueError):
                layout.bond_geometry(D)

    def test_masked_row_mapping_and_panel_loading_cover_each_component_once(self):
        result = child('''
import numpy as np
from rcmps_gpu.optimizer.runtime.generic_layout import packed_geometry
for d in (1,5,7,9,10,11,17,24,33,40,65,80,95,96,97,127,128,129,160,192,255,256):
    info=packed_geometry(d);threads=np.arange(info['threads'])
    writes=np.zeros((10,d,d),dtype=np.uint8)
    for block in range(info['blocks']-1):
        channel=block//info['warps'];m=8*(block%info['warps'])
        row=m+(threads%32//4);col=8*(threads//32)+2*(threads%4)
        for slot in (0,1):
            mask=(row<d)&(col+slot<d)
            np.add.at(writes[channel],(row[mask],(col+slot)[mask]),1)
        panel=np.zeros(8*d,dtype=np.uint8)
        for thread in threads:panel[thread::info['threads']]+=1
        assert np.all(panel==1)
    assert np.all(writes==1),d
assert 'cupy' not in sys.modules
print(json.dumps(dict(all_components_once=True,CUDA_imported=False)))
''')
        self.assertTrue(result["all_components_once"])

    def test_new_partial_tiles_and_dynamic_panels_generate_without_CUDA(self):
        result = child('''
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as t
from rcmps_gpu.optimizer.runtime.generic_layout import dynamic_shared_bytes
for d in (1,5,7,9,17,33,65,80,95,96,97,129,160,255,256):
    source=t.trajectory_source(d)
    assert f'#define PACKED_D {d}\\n' in source
    assert f'for(int w=0;w<{t.geometry(d)["warps"]};++w)' in source
    assert 'row<d && col+1<d' in source and 'ar<d && kp<d' in source
    assert 'mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64' in source
    assert 'float ' not in source and '--use_fast_math' not in t.compilation_options(d)
    assert ('extern __shared__ double2 dynamic_panels[];' in source)==bool(dynamic_shared_bytes(d))
assert 'cupy' not in sys.modules
print(json.dumps(dict(new_shapes_generated=True,CUDA_imported=False)))
''')
        self.assertTrue(result["new_shapes_generated"])

    def test_removed_backends_and_graph_integrator_are_refused(self):
        result = child('''
import argparse,numpy as np
from rcmps_gpu.optimizer.runtime import sweep_driver,ode_backend,overlap_benchmark
def refused(call,message,kind=ValueError):
    try:call()
    except kind as error:assert message in str(error),error
    else:raise AssertionError('accepted: '+message)
def parse(arguments):
    parser=sweep_driver.build_parser();parser.exit_on_error=False
    return parser.parse_args(['reference','--backend','cupy','--mode','hybrid','--packed-z2']+arguments)
args=parse(['--quartic-backend','persistent_trajectory'])
sweep_driver.validate_arguments(args);assert args.device_controller
for label in ('persistent_rows_general','persistent_rows_cached','persistent_rows','persistent'):
    refused(lambda:parse(['--quartic-backend',label]),'invalid choice',argparse.ArgumentError)
for flag in ('--graph','--dense-graph'):
    refused(lambda:sweep_driver.validate_arguments(parse([flag])),'the CUDA-graph integrator was removed')
refused(lambda:sweep_driver.objective(None,None,None,2.7,args,np),'runs only through trajectory_sweep_driver')
z=np.zeros((4,4),complex)
refused(lambda:ode_backend.evaluate_quartic(z,z,z,xp=np,packed_z2=True,use_graph=True),'the CUDA-graph integrator was removed')
refused(lambda:overlap_benchmark.PreparedQuartic(np,(z,z,z),2.7,None),'prepares no solver itself',TypeError)
assert 'cupy' not in sys.modules
print(json.dumps(dict(refused=True)))
''')
        self.assertTrue(result["refused"])

    def test_overlap_dispatch_logic_self_tests(self):
        # Bounded D4 NumPy tests of the kinetic/quartic dispatch that the overlap objective
        # uses; the research benchmark command line ran them with --self-test.
        result = child('''
import contextlib,io
from rcmps_gpu.optimizer.runtime import overlap_benchmark
log=io.StringIO()
with contextlib.redirect_stderr(log):status=overlap_benchmark.logic_self_tests()
assert status==0,log.getvalue()
assert 'cupy' not in sys.modules
print(json.dumps(dict(passed=True,tests=log.getvalue().count(' ... ok'))))
''')
        self.assertEqual(result, dict(passed=True, tests=3))

    def test_occupancy_check_precedes_large_sample_buffers(self):
        source = (RUNTIME / "persistent_trajectory_backend.py").read_text()
        self.assertLess(source.index("unsafe cooperative trajectory grid"), source.index("self.samples=cp.empty"))
        self.assertIn("self.info['threads'],self.dynamic_shared_bytes", source)

    def test_device_memory_optin_thread_limits_and_architecture_guards(self):
        class Kernel:
            shared_size_bytes = 0
            max_dynamic_shared_size_bytes = 0
            max_threads_per_block = 1024
        props = dict(major=9, minor=0, cooperativeLaunch=True, sharedMemPerBlockOptin=227328,
                     sharedMemPerBlock=49152, maxThreadsPerBlock=1024)
        kernel = Kernel()
        self.assertEqual(layout.configure_shared_memory(kernel, props, layout.packed_geometry(128)), 65792)
        self.assertEqual(kernel.max_dynamic_shared_size_bytes, 65792)
        # Any compute capability from 8.0 on runs the tensor-core tiles, any one the scalar tiles.
        for major, minor in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
            self.assertEqual(layout.configure_shared_memory(Kernel(), dict(props, major=major, minor=minor),
                                                            layout.packed_geometry(128)), 65792)
        for major, minor in ((6, 0), (7, 0), (7, 5), (8, 6)):
            self.assertEqual(layout.configure_shared_memory(Kernel(), dict(props, major=major, minor=minor),
                                                            layout.packed_geometry(128), "scalar"), 65792)
        for changes, message in ((dict(major=7, minor=5), r"--tiles mma needs FP64 tensor-core instructions .*"
                                  r"compute capability 8.0 or later\); this GPU is 7.5: use --tiles auto or scalar"),
                                 (dict(major=6, minor=0), "this GPU is 6.0"),
                                 (dict(cooperativeLaunch=False), "cooperative"),
                                 (dict(sharedMemPerBlockOptin=49152), "exceeds"),
                                 (dict(maxThreadsPerBlock=256), "thread-block")):
            with self.subTest(changes=changes), self.assertRaisesRegex((ValueError, RuntimeError), message):
                layout.configure_shared_memory(Kernel(), dict(props, **changes), layout.packed_geometry(128))
        for tiles in ("auto", None, "MMA"):
            with self.subTest(tiles=tiles), self.assertRaisesRegex(ValueError, "tiles"):
                layout.configure_shared_memory(Kernel(), props, layout.packed_geometry(128), tiles)

    def test_plan_has_no_hidden_run_time_cap_and_explicit_cap_is_preserved(self):
        hashes = run.runtime_source_sha256()
        args = run.parser().parse_args(["--D", "160"])
        plan = run.prepare(args, hashes)
        self.assertIsNone(plan["max_seconds"])
        self.assertFalse(plan["optimization_wall_time_limit_is_explicit"])
        self.assertEqual((plan["package"]["version"], plan["runtime"], plan["runtime_source_sha256"]),
                         (rcmps_gpu.__version__, str(RUNTIME), hashes))
        self.assertEqual(sorted(plan["launcher_source_sha256"]), sorted(run.LAUNCHER_SOURCES))
        self.assertEqual(plan["runtime_provenance"], run.verify_runtime())
        args = run.parser().parse_args(["--D", "160", "--max-seconds", "7200"])
        self.assertEqual(run.prepare(args, hashes)["max_seconds"], 7200.)
        for value in ("nan", "inf", "0", "-1"):
            args = run.parser().parse_args(["--D", "160", "--max-seconds", value])
            with self.assertRaises(ValueError):
                run.prepare(args, hashes)

    def test_launcher_refuses_output_overwrite_and_initial_D_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            with self.assertRaises(SystemExit), unittest.mock.patch("sys.stderr"):
                run.main(["--D", "160", "--output", str(target)])
            export = target / "initial"
            export.mkdir()
            (export / "metadata.toml").write_text("D = 4\ncoupling = 2.7\n")
            with self.assertRaisesRegex(ValueError, "D/coupling"):
                run.input_ledger(export, 160, 2.7)

    def test_unsafe_actual_residency_rejected_without_numerical_launch(self):
        result = child('''
from types import SimpleNamespace
from rcmps_gpu.optimizer import worker
from rcmps_gpu.optimizer.runtime.persistent_trajectory_backend import geometry
from unittest.mock import patch
info=geometry(80)
class Kernel:
    kernel=SimpleNamespace(ptr=1);max_threads_per_block=1024;shared_size_bytes=info['shared_bytes']
    num_regs=64;local_size_bytes=0
    def compile(self):pass
    def __call__(self,*args):raise AssertionError('numerical launch forbidden')
props=dict(major=9,minor=0,cooperativeLaunch=True,sharedMemPerBlock=49152,
    sharedMemPerBlockOptin=227328,maxThreadsPerBlock=1024,multiProcessorCount=80,name='mock')
cp=SimpleNamespace(RawKernel=lambda *a,**k:Kernel(),cuda=SimpleNamespace(
    Device=lambda:SimpleNamespace(id=0),runtime=SimpleNamespace(getDeviceProperties=lambda d:props),
    driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda *v:1)))
with patch('rcmps_gpu.optimizer.runtime.persistent_trajectory_backend.trajectory_ptx_report',side_effect=AssertionError('unsafe compile must reject before PTX audit')):
    try:worker.compile_dimension(160,cp)
    except RuntimeError as e:assert 'requires 101 resident blocks' in str(e)
    else:raise AssertionError('unsafe grid accepted')
print(json.dumps(dict(unsafe_residency_rejected=True)))
''')
        self.assertTrue(result["unsafe_residency_rejected"])

    def test_register_caps_ladder(self):
        # The first entry of the ladder is the historical options (golden_cuda.json) for every
        # d; packed d <= 64, which compiles without a cap, also tries 128, 96 and 64.
        for d in range(1, 257):
            caps = layout.register_caps(d)
            self.assertEqual(caps, (None, 128, 96, 64) if d <= 64 else (None,))
            self.assertEqual(layout.compilation_options(d, caps[0]), layout.compilation_options(d))
            for cap in caps[1:]:
                self.assertEqual(layout.compilation_options(d, cap), ("--std=c++17", f"--maxrregcount={cap}"))
            for cap in (32, 100, 255) + (() if d <= 64 else (128, 64)):
                with self.assertRaisesRegex(ValueError, "register cap"):
                    layout.compilation_options(d, cap)
        for d, entry in GOLDEN["sizes"].items():
            self.assertEqual(list(layout.compilation_options(int(d), None)), entry["options"])
        result = child('''
from types import SimpleNamespace
from unittest.mock import patch
from rcmps_gpu.optimizer import worker
from rcmps_gpu.optimizer.runtime.persistent_trajectory_backend import geometry
compiled=[]
class Kernel:
    def __init__(self,code,name,options=(),enable_cooperative_groups=False):
        caps=[int(o.split('=')[1]) for o in options if o.startswith('--maxrregcount=')]
        self.options=options;self.num_regs=min([170,*caps]);self.local_size_bytes=0 if not caps else 52
        self.kernel=SimpleNamespace(ptr=len(compiled));compiled.append(self)
        self.max_threads_per_block=1024;self.shared_size_bytes=geometry(int(code.split('#define PACKED_D ')[1].split()[0]))['shared_bytes']
    def compile(self):pass
    def __call__(self,*args):raise AssertionError('numerical launch forbidden')
def device(sms,limit):
    props=dict(major=7,minor=0,cooperativeLaunch=True,sharedMemPerBlock=49152,sharedMemPerBlockOptin=98304,
        maxThreadsPerBlock=1024,multiProcessorCount=sms,name='mock V100')
    # 65536 registers per multiprocessor and at most `limit` blocks
    occupancy=lambda pointer,threads,shared:min(limit,65536//(compiled[pointer].num_regs*threads))
    return SimpleNamespace(RawKernel=Kernel,cuda=SimpleNamespace(Device=lambda:SimpleNamespace(id=0),
        runtime=SimpleNamespace(getDeviceProperties=lambda d:props),
        driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=occupancy)))
rows={}
with patch('rcmps_gpu.optimizer.runtime.persistent_trajectory_backend.trajectory_ptx_report',return_value={}) as audit:
    # D = 128: 81 blocks of 256 threads; 170 registers hold 1 block per multiprocessor (80), cap 128 holds 2.
    compiled.clear();receipt=worker.compile_dimension(128,device(80,32))
    rows['V100']=[receipt['register_cap'],receipt['compiler_options'],[r['register_cap'] for r in receipt['register_caps_tried']],
                  receipt['resident_capacity'],receipt['tiles']['resolved'],audit.call_args.args[3:]]
    # A device that holds the grid at once never moves: the historical options.
    compiled.clear();receipt=worker.compile_dimension(128,device(108,32))
    rows['wide']=[receipt['register_cap'],receipt['compiler_options'],len(receipt['register_caps_tried'])]
    compiled.clear()
    try:worker.compile_dimension(128,device(40,2))
    except RuntimeError as error:rows['refused']=str(error)
print(json.dumps(rows))
''')
        self.assertEqual(result["V100"], [128, ["--std=c++17", "--maxrregcount=128"], [None, 128], 160, "scalar",
                                          ["scalar", 128]])
        self.assertEqual(result["wide"], [None, ["--std=c++17"], 1])
        self.assertEqual(result["refused"], "D=128 requires 81 resident blocks; this GPU can hold 80 at most (register "
                                            "caps [None, 128, 96, 64] tried). The current one-kernel row layout cannot "
                                            "run safely at this size.")

    def test_worker_refuses_changed_runtime_other_installation_and_mixed_modules(self):
        result = child('''
import tempfile,types
from pathlib import Path
from rcmps_gpu.optimizer import cli,worker
with tempfile.TemporaryDirectory() as tmp:
    plan=cli.prepare(cli.parser().parse_args(['--D','8','--output',str(Path(tmp)/'run')]),cli.runtime_source_sha256())
    def refused(changes,message):
        path=Path(tmp)/'plan.json';path.write_text(json.dumps({**plan,**changes}))
        try:worker.main([str(path)])
        except ValueError as e:assert message in str(e),e
        else:raise AssertionError('worker accepted '+message)
    refused(dict(runtime='/elsewhere/runtime'),'another launcher version or installation')
    refused(dict(schema_version=1),'another launcher version or installation')
    refused(dict(runtime_source_sha256={**plan['runtime_source_sha256'],'numerics.py':'0'*64}),'runtime changed after planning')
    refused(dict(launcher_source_sha256={**plan['launcher_source_sha256'],'optimizer/worker.py':'0'*64}),'launcher source changed after planning: optimizer/worker.py')
    refused(dict(launcher_source_sha256={k:v for k,v in plan['launcher_source_sha256'].items() if k!='provenance.py'}),'launcher source changed after planning: provenance.py')
    refused(dict(runtime_provenance={**plan['runtime_provenance'],'edits':[]}),'runtime provenance changed after planning')
    fake=types.ModuleType('rcmps_gpu.optimizer.runtime.cooperative_arnoldi');fake.__file__='/elsewhere/cooperative_arnoldi.py'
    sys.modules[fake.__name__]=fake
    refused({},'mixed numerical runtimes: cooperative_arnoldi.py')
    assert not (Path(tmp)/'run').exists()
from rcmps_gpu.optimizer import validate_trajectory_sweep as v
assert v.validate_sweep.driver is v.wrapper.overlap_driver.driver
assert 'cupy' not in sys.modules
print(json.dumps(dict(refused=True)))
''')
        self.assertTrue(result["refused"])

    def test_complete_oracle_uses_CuPy_factory_and_restores_fast_binding(self):
        result = child('''
import tempfile,numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rcmps_gpu.optimizer import worker
from rcmps_gpu.optimizer.runtime import overlap_sweep_driver as driver,persistent_trajectory_overlap as fast
original=fast.ORIGINAL_PREPARED;calls=[]
class Coordinator:
    def __enter__(self):return self
    def __exit__(self,*args):pass
def evaluate(*args,mode,kinetic,**options):
    calls.append((mode,kinetic))
    if mode=='cached':assert fast.ORIGINAL_PREPARED.__name__=='CuPyQuartic'
    else:assert fast.ORIGINAL_PREPARED is original
    z=np.zeros((4,4),np.complex128)
    return dict(value=-.1,gradient=z,covector=z,environment=z,stats=dict(
        kinetic=dict(maximum_relative_residual=1e-13,fallback_solves=0)))
z=np.zeros((4,4),np.complex128);rho=np.eye(4,dtype=np.complex128)/4
args=SimpleNamespace(metric_reg=.001,krylov_tolerance=1e-12)
with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'OverlapObjective',Coordinator),patch.object(fast,'evaluate_complete',evaluate):
    receipt=worker.qualify(z,z,rho,2.7,args,None,300.,Path(tmp))
    assert receipt['passed']
assert calls==[('trajectory','cooperative'),('cached','baseline')]
assert fast.ORIGINAL_PREPARED is original and 'cupy' not in sys.modules
print(json.dumps(dict(independent_factory_used=True,restored=True)))
''')
        self.assertTrue(result["independent_factory_used"])

    def test_safe_compile_audits_same_device_without_numerical_launch(self):
        result = child('''
from types import SimpleNamespace
from unittest.mock import patch
from rcmps_gpu.optimizer import worker
from rcmps_gpu.optimizer.runtime.persistent_trajectory_backend import geometry
info=geometry(80)
class Kernel:
    kernel=SimpleNamespace(ptr=1);max_threads_per_block=1024;shared_size_bytes=info['shared_bytes']
    num_regs=64;local_size_bytes=120
    def compile(self):pass
    def __call__(self,*args):raise AssertionError('numerical launch forbidden')
props=dict(major=9,minor=0,cooperativeLaunch=True,sharedMemPerBlock=49152,
    sharedMemPerBlockOptin=227328,maxThreadsPerBlock=1024,multiProcessorCount=132,name=b'mock')
cp=SimpleNamespace(RawKernel=lambda *a,**k:Kernel(),cuda=SimpleNamespace(
    Device=lambda:SimpleNamespace(id=0),runtime=SimpleNamespace(getDeviceProperties=lambda d:props),
    driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda *v:1)))
with patch('rcmps_gpu.optimizer.runtime.persistent_trajectory_backend.trajectory_ptx_report',return_value={'FP64':True}) as audit:
    receipt=worker.compile_dimension(160,cp,'mock_output')
    audit.assert_called_once_with(80,cp,'mock_output','mma',None)
    assert receipt['full_grid_residency_verified'] and receipt['resident_capacity']==132
    assert receipt['tiles']['requested']=='auto' and receipt['tiles']['resolved']=='mma'
    assert receipt['register_cap'] is None and receipt['compiler_options']==['--std=c++17','--maxrregcount=64']
    assert receipt['PTX']['FP64'] and not receipt['numerical_launch_performed']
    assert not receipt['actual_GPU_qualified'] and 'cupy' not in sys.modules
print(json.dumps(dict(device_audit_forwarded=True)))
''')
        self.assertTrue(result["device_audit_forwarded"])

    def test_failed_oracle_retains_fast_raw_arrays_and_restores_binding(self):
        result = child('''
import tempfile,numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rcmps_gpu.optimizer import worker
from rcmps_gpu.optimizer.runtime import overlap_sweep_driver as driver,persistent_trajectory_overlap as fast
original=fast.ORIGINAL_PREPARED
class Coordinator:
    def __enter__(self):return self
    def __exit__(self,*args):pass
z=np.zeros((4,4),np.complex128);rho=np.eye(4,dtype=np.complex128)/4
def evaluate(*args,mode,**options):
    if mode=='cached':raise RuntimeError('mock strict oracle failure')
    return dict(value=-.1,gradient=z,covector=z,environment=z,stats=dict(
        kinetic=dict(maximum_relative_residual=1e-13,fallback_solves=0)))
args=SimpleNamespace(metric_reg=.001,krylov_tolerance=1e-12)
with tempfile.TemporaryDirectory() as tmp,patch.object(driver,'OverlapObjective',Coordinator),patch.object(fast,'evaluate_complete',evaluate):
    directory=Path(tmp)
    try:worker.qualify(z,z,rho,2.7,args,None,300.,directory)
    except RuntimeError as error:assert 'mock strict oracle' in str(error)
    else:raise AssertionError('failed oracle accepted')
    report=json.loads((directory/'initial_qualification.json').read_bytes())
    assert not report['passed'] and report['fast_energy']==-.1 and 'mock strict oracle' in report['error']
    with np.load(directory/'initial_qualification.npz') as saved:
        for name in ('fast_gradient','fast_covector','fast_environment'):
            assert saved[name].tobytes()==z.tobytes() and saved[name].dtype==np.complex128
        assert 'oracle_gradient' not in saved.files
assert fast.ORIGINAL_PREPARED is original and 'cupy' not in sys.modules
print(json.dumps(dict(failed_partial_raw_preserved=True)))
''')
        self.assertTrue(result["failed_partial_raw_preserved"])

    def test_generic_checkpoint_no_reset_fork_and_resume_controls(self):
        result = child('''
import tempfile,hashlib,numpy as np
from pathlib import Path
from rcmps_gpu.optimizer import cli as run
from rcmps_gpu.optimizer.runtime import trajectory_sweep_driver as t,robust_sweep_driver as robust
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);ref=root/'initial';ref.mkdir();D=10
    rng=np.random.default_rng(415)
    r=np.asarray(rng.normal(size=(D,D))+1j*rng.normal(size=(D,D)),dtype=np.complex128)
    k=np.zeros_like(r);rho=np.eye(D,dtype=np.complex128)/D;gradient=.01*r
    for name,value in (('R',r),('K',k),('rho',rho),('saved_gradient',gradient)):
        value.ravel(order='F').astype('<c16').tofile(ref/(name+'.bin'))
    (ref/'metadata.toml').write_text('D = 10\\ncoupling = 2.7\\n')
    args=t.ORIGINAL_PARSE([str(ref),'--backend','cupy','--mode','hybrid','--packed-z2',
        '--quartic-backend','persistent_trajectory','--arnoldi','gpu','--projection','hessenberg_gpu','--window','20'])
    args.trajectory_controls=dict(enabled=True,whole_trajectory_cooperative_kernel=True,kinetic='cooperative',max_device_seconds=300.)
    args.overlap_controls=dict(enabled=True,controller='robust')
    hashes=t.source_hashes(args);controls=vars(args)
    state=dict(R=r,K=k,rho=rho,gradient=gradient,energy=np.asarray(-.1),iteration=np.asarray(2),
        lbfgs_steps=np.stack([r*.001,r*.002]),lbfgs_changes=np.stack([r*.003,r*.004]),
        lbfgs_inverse_curvatures=np.asarray([1.,2.]),next_step=np.asarray(.5),precond_reg=np.asarray(1e-4),
        optimizer_seconds=np.asarray(17.),optimizer_evaluations=np.asarray(3))
    old=root/'old.json';np.savez(old.with_suffix('.npz'),**state)
    report=dict(reference=dict(D=D,coupling=2.7),controls=controls,source_sha256=hashes,
        runs=[dict(source_sha256=hashes,controls=controls)],history=[{},{}],final_energy=-.1,
        stop_reason='iteration limit',reference_sha256={p.name:sha(p) for p in ref.iterdir()})
    old.write_text(json.dumps(report));before=(old.read_bytes(),old.with_suffix('.npz').read_bytes())
    snapshot=t.verify_checkpoint(old)
    new=root/'new.json';robust.fork_checkpoint(snapshot,new,args,reset_lbfgs=False)
    with np.load(new.with_suffix('.npz')) as copied:
        for name,value in state.items():
            assert copied[name].dtype==value.dtype and copied[name].shape==value.shape
            assert copied[name].tobytes()==value.tobytes(),name
    assert before==(old.read_bytes(),old.with_suffix('.npz').read_bytes())
    plan=run.prepare(run.parser().parse_args(['--D','10','--resume-from',str(old)]),run.runtime_source_sha256())
    assert plan['controls']['window']==20. and plan['max_seconds'] is None
    try:run.prepare(run.parser().parse_args(['--D','10','--resume-from',str(old),'--window','30']),run.runtime_source_sha256())
    except ValueError as e:assert 'cannot change objective' in str(e)
    else:raise AssertionError('changed resume physics accepted')
    assert 'generic_layout.py' in hashes and 'cupy' not in sys.modules
print(json.dumps(dict(no_reset_fork_bitwise=True,old_checkpoint_unchanged=True,control_change_rejected=True)))
''')
        self.assertTrue(result["no_reset_fork_bitwise"])

    def test_real_small_trajectory_and_new_D10_D14_chunked_pullback(self):
        result = child('''
import math,numpy as np
from rcmps_gpu.optimizer.runtime.numerics import stationary_state,finish_gradient,metric
from rcmps_gpu.optimizer.runtime.ode_backend import _PackedZ2QuarticRHS,_integrate,quadrature,_packed_z2_pullback_unchunked,_packed_z2_pullback_chunked
from rcmps_gpu.optimizer.runtime.persistent_trajectory_backend import integrate_trajectory_numpy
rows=[]
for D in (6,10,14):
    rng=np.random.default_rng(817+D);n=D//2
    def draw():return .12*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    z=np.zeros((n,n),complex);k1,k2=draw(),draw()
    r=np.block([[z,draw()],[draw(),z]]);k=np.block([[k1+k1.conj().T,z],[z,k2+k2.conj().T]])
    q=-1j*k-r.conj().T@r/2;rho,_=stationary_state(q,r,tolerance=1e-13)
    points,weights=quadrature(.7,16);nodes=np.r_[points,math.sqrt(.7)]
    rhs=_PackedZ2QuarticRHS(q,r,rho,np)
    expected,_=_integrate(_PackedZ2QuarticRHS(q,r,rho,np),nodes,atol=1e-13,rtol=1e-12,initial_step=None,max_steps=100000)
    if D<=8:
        actual,stats=integrate_trajectory_numpy(rhs,nodes,atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=1e-11)
    else:actual=expected
    ((e,c,h),)=_packed_z2_pullback_unchunked(q,r,rhs,actual,points,weights)
    gradient=finish_gradient(q,r,rho,c,h)[0]
    for chunk in (1,3,7,64):
        ((ec,cc,hc),)=_packed_z2_pullback_chunked(q,r,rhs,actual,points,weights,chunk)
        delta=finish_gradient(q,r,rho,cc,hc)[0]-gradient
        assert abs(ec-e)<1e-13 and math.sqrt(metric(rho,delta,delta))<1e-10
    rows.append(dict(D=D,source_masks_and_pullback_checked=True))
assert 'cupy' not in sys.modules
print(json.dumps(dict(rows=rows,CUDA_imported=False)))
''')
        self.assertEqual([r["D"] for r in result["rows"]], [6, 10, 14])


if __name__ == "__main__":
    unittest.main()
