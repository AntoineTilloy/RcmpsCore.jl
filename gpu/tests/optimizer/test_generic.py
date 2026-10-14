"""CPU regressions for generated dimensions, physics, provenance and limits.

These tests never import CuPy, query a GPU or evaluate a D64 CPU objective.
They do not establish actual-GPU correctness for newly generated dimensions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generic_gpu import build_runtime as builder
from generic_gpu import layout
from generic_gpu import run


def child(code, directory=None):
    directory = directory or builder.TARGET
    prefix = "import sys,json;sys.path.insert(0," + repr(str(directory)) + ")\n"
    result = subprocess.run([sys.executable, "-c", prefix + code], cwd=builder.PARENT,
                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stdout + "\n" + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class GenericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder.build()

    def test_builder_verified_exact_original_bytes_and_only_four_derived_changes(self):
        manifest = builder.verify()
        self.assertFalse(manifest["actual_GPU_qualified"])
        self.assertEqual(len(manifest["runtime_source_sha256"]), 24)
        for name, expected in manifest["base_source_sha256"].items():
            self.assertEqual(builder.sha((builder.BASE / name).read_bytes()), expected)
            if name not in builder.CHANGED:
                self.assertEqual((builder.BASE / name).read_bytes(), (builder.TARGET / name).read_bytes())

    def test_idempotent_builder_and_mutated_runtime_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runtime"
            builder.build(target)
            self.assertEqual(builder.build(target), builder.verify(target))
            path = target / "generic_layout.py"
            path.write_text(path.read_text() + "\n# changed\n")
            old = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "source changed"):
                builder.build(target)
            self.assertEqual(path.read_bytes(), old)

    def test_all_old_sizes_generate_bitwise_identical_CUDA_and_compiler_options(self):
        code = '''
import hashlib
import persistent_rows_general as g,persistent_trajectory_backend as t
print(json.dumps({str(d):dict(row=hashlib.sha256(g.general_row_source(d).encode()).hexdigest(),
    trajectory=hashlib.sha256(t.trajectory_source(d).encode()).hexdigest(),
    options=list(t.compilation_options(d)),geometry=t.geometry(d))
    for d in (2,3,4,8,16,32,48,64,128)}))
'''
        self.assertEqual(child(code), child(code, builder.BASE))

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
from generic_layout import packed_geometry
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
import persistent_rows_general as g,persistent_trajectory_backend as t
from generic_layout import dynamic_shared_bytes
for d in (1,5,7,9,17,33,65,80,95,96,97,129,160,255,256):
    source=t.trajectory_source(d);row=g.general_row_source(d)
    assert f'#define PACKED_D {d}\\n' in source
    assert f'for(int w=0;w<{t.geometry(d)["warps"]};++w)' in source
    assert 'row<d && col+1<d' in source and 'ar<d && kp<d' in source
    assert 'mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64' in source
    assert 'float ' not in source and '--use_fast_math' not in t.compilation_options(d)
    for s in (source,row):
        assert ('extern __shared__ double2 dynamic_panels[];' in s)==bool(dynamic_shared_bytes(d))
assert 'cupy' not in sys.modules
print(json.dumps(dict(new_shapes_generated=True,CUDA_imported=False)))
''')
        self.assertTrue(result["new_shapes_generated"])

    def test_occupancy_check_precedes_large_sample_buffers(self):
        source = (builder.TARGET / "persistent_trajectory_backend.py").read_text()
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
        for changes, message in ((dict(major=8, minor=6), "SM80"),
                                 (dict(cooperativeLaunch=False), "cooperative"),
                                 (dict(sharedMemPerBlockOptin=49152), "exceeds"),
                                 (dict(maxThreadsPerBlock=256), "thread-block")):
            with self.subTest(changes=changes), self.assertRaisesRegex((ValueError, RuntimeError), message):
                layout.configure_shared_memory(Kernel(), dict(props, **changes), layout.packed_geometry(128))

    def test_plan_has_no_hidden_run_time_cap_and_explicit_cap_is_preserved(self):
        manifest = builder.verify()
        args = run.parser().parse_args(["--D", "160"])
        plan = run.prepare(args, manifest)
        self.assertIsNone(plan["max_seconds"])
        self.assertFalse(plan["optimization_wall_time_limit_is_explicit"])
        args = run.parser().parse_args(["--D", "160", "--max-seconds", "7200"])
        self.assertEqual(run.prepare(args, manifest)["max_seconds"], 7200.)
        for value in ("nan", "inf", "0", "-1"):
            args = run.parser().parse_args(["--D", "160", "--max-seconds", value])
            with self.assertRaises(ValueError):
                run.prepare(args, manifest)

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
from generic_gpu import worker
sys.path.insert(0,''' + repr(str(builder.TARGET)) + ''')
from persistent_trajectory_backend import geometry
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
with patch('persistent_trajectory_backend.trajectory_ptx_report',side_effect=AssertionError('unsafe compile must reject before PTX audit')):
    try:worker.compile_dimension(160,cp)
    except RuntimeError as e:assert 'requires 101 resident blocks' in str(e)
    else:raise AssertionError('unsafe grid accepted')
print(json.dumps(dict(unsafe_residency_rejected=True)))
''')
        self.assertTrue(result["unsafe_residency_rejected"])

    def test_complete_oracle_uses_CuPy_factory_and_restores_fast_binding(self):
        result = child('''
import tempfile,numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from generic_gpu import worker
sys.path.insert(0,''' + repr(str(builder.TARGET)) + ''')
import overlap_sweep_driver as driver,persistent_trajectory_overlap as fast
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
from generic_gpu import worker
sys.path.insert(0,''' + repr(str(builder.TARGET)) + ''')
from persistent_trajectory_backend import geometry
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
with patch('persistent_trajectory_backend.trajectory_ptx_report',return_value={'FP64':True}) as audit:
    receipt=worker.compile_dimension(160,cp,'mock_output')
    audit.assert_called_once_with(80,cp,'mock_output')
    assert receipt['full_grid_residency_verified'] and receipt['resident_capacity']==132
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
from generic_gpu import worker
sys.path.insert(0,''' + repr(str(builder.TARGET)) + ''')
import overlap_sweep_driver as driver,persistent_trajectory_overlap as fast
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
from generic_gpu import run,build_runtime
sys.path.insert(0,''' + repr(str(builder.TARGET)) + ''')
import trajectory_sweep_driver as t,robust_sweep_driver as robust
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
        '--quartic-backend','persistent_rows_general','--arnoldi','gpu','--projection','hessenberg_gpu','--window','20'])
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
    plan=run.prepare(run.parser().parse_args(['--D','10','--resume-from',str(old)]),build_runtime.verify())
    assert plan['controls']['window']==20. and plan['max_seconds'] is None
    try:run.prepare(run.parser().parse_args(['--D','10','--resume-from',str(old),'--window','30']),build_runtime.verify())
    except ValueError as e:assert 'cannot change objective' in str(e)
    else:raise AssertionError('changed resume physics accepted')
    assert 'generic_layout.py' in hashes and 'cupy' not in sys.modules
print(json.dumps(dict(no_reset_fork_bitwise=True,old_checkpoint_unchanged=True,control_change_rejected=True)))
''')
        self.assertTrue(result["no_reset_fork_bitwise"])

    def test_real_small_trajectory_and_new_D10_D14_chunked_pullback(self):
        result = child('''
import math,numpy as np
from numerics import stationary_state,finish_gradient,metric
from ode_backend import _PackedZ2QuarticRHS,_integrate,quadrature,_packed_z2_pullback_unchunked,_packed_z2_pullback_chunked
from persistent_trajectory_backend import integrate_trajectory_numpy
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
