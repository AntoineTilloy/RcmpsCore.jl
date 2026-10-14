"""Complete E/gradient overlap comparison, including preparation and boundary.

New isolated harness; no optimizer or backend edits. One persistent coordinator
owns two workers and two component streams plus the combine stream throughout
all variants/repeats. Its unchanged cached-graph setup is serialized before
dispatch; the experimental trajectory is prepared at the same boundary. Both
workers join before scratch/kernel owners close, including component failure.

Only --run-gpu performs CUDA work. Baseline kinetic is the default. Optional
cooperative Arnoldi is an additive experimental case, never a silent fallback.
No large CPU objective or checkpoint mutation is performed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import overlap_sweep_driver as overlap_driver
import overlap_benchmark as overlap
from warm_sweep_driver import binding
from numerics import adj,finish_gradient,load_reference,metric
from ode_backend import _PackedZ2QuarticRHS,_packed_z2_pullback,quadrature
from persistent_trajectory_backend import PersistentTrajectoryDOP853,trajectory_ptx_report

ROOT=Path(__file__).parent
ORIGINAL_PREPARED=overlap.PreparedQuartic
ORIGINAL_KINETIC=overlap.kinetic_raw
EXTRA_SOURCES=('persistent_trajectory_backend.py','persistent_trajectory_overlap.py')


class PreparedTrajectoryQuartic(overlap.PreparedQuartic):
    """Same run/pullback/scaling as PreparedQuartic; new one-kernel solver."""
    def __init__(self,cp,inputs,coupling,args,*,max_device_seconds=30.):
        q,r,rho=inputs;self.cp,self.q,self.r,self.coupling=cp,q,r,float(coupling)
        self.pullback=_packed_z2_pullback;started=time.perf_counter()
        self.points,self.weights=quadrature(args.window,args.quadrature)
        self.rhs=_PackedZ2QuarticRHS(q,r,rho,cp)
        self.solver=PersistentTrajectoryDOP853(self.rhs,np.r_[self.points,math.sqrt(args.window)],
            atol=args.atol,rtol=args.rtol,max_steps=args.max_steps,max_device_seconds=max_device_seconds)
        self.setup_seconds=time.perf_counter()-started


def cooperative_kinetic(q,r,rho,nodes=128,*,xp,tolerance,maxdim,arnoldi_backend,
                        projection_backend,fallback_tolerance=None,**options):
    if arnoldi_backend!='gpu' or fallback_tolerance is not None:
        raise ValueError('cooperative experiment requires GPU Arnoldi with no fallback')
    from cooperative_arnoldi import kinetic_raw_cooperative
    return kinetic_raw_cooperative(q,r,rho,nodes,xp=xp,tolerance=tolerance,maxdim=maxdim,
        projection_backend=projection_backend,fma=True,**options)


def source_hashes(args,kinetic='baseline'):
    hashes=overlap_driver.source_hashes(args)
    names=EXTRA_SOURCES+(('cooperative_arnoldi.py',) if kinetic=='cooperative' else ())
    hashes.update({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
    return hashes


@contextmanager
def candidate_bindings(mode,kinetic,max_device_seconds):
    if mode not in ('cached','trajectory') or kinetic not in ('baseline','cooperative'):
        raise ValueError('unknown explicit candidate selection')
    factory=ORIGINAL_PREPARED if mode=='cached' else lambda cp,inputs,coupling,args: \
        PreparedTrajectoryQuartic(cp,inputs,coupling,args,max_device_seconds=max_device_seconds)
    selected=ORIGINAL_KINETIC if kinetic=='baseline' else cooperative_kinetic
    with binding(overlap,'PreparedQuartic',factory),binding(overlap,'kinetic_raw',selected):yield


def evaluate_complete(coordinator,r,k,rho,coupling,args,xp,*,mode,kinetic='baseline',max_device_seconds=30.):
    """Return complete result; capture already-summed raw terms without rerun."""
    raw={};original_finish=overlap_driver.driver.finish_gradient
    def capture_finish(q,r,rho,covector,environment,**options):
        raw['covector']=covector.copy();raw['environment']=environment.copy()
        return original_finish(q,r,rho,covector,environment,**options)
    started=time.perf_counter()
    with candidate_bindings(mode,kinetic,max_device_seconds), \
            binding(overlap_driver.driver,'finish_gradient',capture_finish):
        energy,gradient,stats=coordinator(r,k,rho,coupling,args,xp)
    if set(raw)!= {'covector','environment'}:
        raise RuntimeError('complete objective did not use its ordinary summed boundary conversion')
    if gradient.dtype!=np.complex128 or not np.isfinite(gradient).all():
        raise RuntimeError('complete gradient must be finite complex128')
    return dict(value=float(energy),gradient=gradient,stats=stats,
                measured_complete_seconds=time.perf_counter()-started,**raw)


def compare(reference,result,rho,metric_reg):
    delta=result['gradient']-reference['gradient']
    return dict(energy_difference=abs(result['value']-reference['value']),
        metric_gradient_difference=math.sqrt(max(0.,metric(rho,delta,delta,metric_reg))),
        covector_relative_frobenius=float(np.linalg.norm(result['covector']-reference['covector']) /
            max(1.,np.linalg.norm(reference['covector']))),
        environment_relative_frobenius=float(np.linalg.norm(result['environment']-reference['environment']) /
            max(1.,np.linalg.norm(reference['environment']))))


def passed(comparison,energy_tolerance=1e-9,gradient_tolerance=1e-8):
    return bool(physical_passed(comparison,energy_tolerance,gradient_tolerance) and
        all(math.isfinite(value) for value in comparison.values()) and
        comparison['covector_relative_frobenius']<=1e-9 and comparison['environment_relative_frobenius']<=1e-9)


def physical_passed(comparison,energy_tolerance=1e-9,gradient_tolerance=1e-8):
    # Raw covector/environment include representation-dependent boundary
    # terms. Across windows only the complete physical derivative is invariant.
    return bool(math.isfinite(comparison['energy_difference']) and
        math.isfinite(comparison['metric_gradient_difference']) and
        comparison['energy_difference']<=energy_tolerance and
        comparison['metric_gradient_difference']<=gradient_tolerance)


def atomic_save(output,report):
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)


def reference_oracle(matrices,metadata):
    energy_available=metadata.get('saved_energy_available',True)
    # An explicitly unavailable saved energy marks random-state exports whose
    # saved_gradient.bin may be a zeros sentinel. An explicit gradient flag
    # overrides that convention; legacy optimized D64 exports omit both flags.
    gradient_available=metadata.get('saved_gradient_available',energy_available)
    gradient=matrices.get('saved_gradient') if gradient_available else None
    energy=metadata.get('saved_energy') if energy_available else None
    kind='saved_CPU' if gradient is not None or energy is not None else 'unevaluated_export'
    if gradient is None and 'ode_energy' in metadata:
        julia_gradient=matrices.get('ode_gradient',matrices.get('gradient'))
        if julia_gradient is not None:
            gradient=julia_gradient;energy=metadata['ode_energy'];kind='Julia_full_reference'
    return dict(value=energy,gradient=gradient,kind=kind)


def load_case(cli):
    matrices,metadata=load_reference(cli.reference)
    snapshot=None
    if cli.checkpoint:
        snapshot=overlap_driver.verify_checkpoint(cli.checkpoint)
        if Path(snapshot['args'].reference).resolve()!=cli.reference.resolve():
            raise ValueError('initial reference differs from stopped checkpoint controls')
        args=argparse.Namespace(**vars(snapshot['args']));state=snapshot['state']
        r,k,rho=(state[name].copy() for name in ('R','K','rho'))
        expected=dict(value=float(state['energy']),gradient=state['gradient'].copy(),kind='stopped_GPU_checkpoint')
        for key in ('window','quadrature','nodes','atol','rtol','maxdim','krylov_tolerance','max_steps','metric_reg'):
            value=getattr(cli,key)
            if value is not None and value!=getattr(args,key):
                raise ValueError('standard checkpoint comparison must inherit exact objective control: '+key)
        overlap_driver.validate_overlap_arguments(args)
    else:
        r,k,rho=(matrices[name].copy() for name in ('R','K','rho'))
        family='persistent_rows_cached' if len(r)==64 else 'persistent_rows_general'
        args=overlap_driver.BASE_PARSE_ARGS([str(cli.reference),'--backend','cupy','--mode','hybrid',
            '--packed-z2','--quartic-backend',family,'--arnoldi','gpu','--projection','hessenberg_gpu',
            '--window',str(cli.window if cli.window is not None else 20.)])
        for key in ('quadrature','nodes','atol','rtol','maxdim','krylov_tolerance','max_steps','metric_reg'):
            value=getattr(cli,key)
            if value is not None:setattr(args,key,value)
        expected=reference_oracle(matrices,metadata)
        overlap_driver.validate_overlap_arguments(args)
    if (r.ndim!=2 or r.shape[0]!=r.shape[1] or any(x.shape!=r.shape or x.dtype!=np.complex128 or
            not np.isfinite(x).all() for x in (r,k,rho))):
        raise ValueError('state requires equally shaped finite square complex128 matrices')
    return args,r,k,rho,float(metadata['coupling']),expected,snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--repeat',type=int,default=3);parser.add_argument('--refine',action='store_true')
    parser.add_argument('--kinetic',choices=('baseline','cooperative'),default='baseline')
    parser.add_argument('--max-device-seconds',type=float,default=30.)
    parser.add_argument('--device',type=int,default=0);parser.add_argument('--ptx-directory',type=Path)
    for name in ('window','atol','rtol','krylov-tolerance','metric-reg'):parser.add_argument('--'+name,type=float)
    for name in ('quadrature','nodes','maxdim','max-steps'):parser.add_argument('--'+name,type=int)
    cli=parser.parse_args(argv)
    if (not 1<=cli.repeat<=10 or cli.device<0 or not math.isfinite(cli.max_device_seconds) or
            cli.max_device_seconds<=0):raise ValueError('invalid repeat/device/watchdog controls')
    if cli.output and (cli.output.exists() or cli.output.with_suffix('.npz').exists()):
        raise ValueError('complete comparison output must be new')
    if cli.run_gpu and cli.output is None:raise ValueError('GPU experiment requires a new output report')
    args,r,k,rho,g,expected,snapshot=load_case(cli)
    references={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cli.reference.glob('*')) if p.is_file()}
    sources=source_hashes(args,cli.kinetic)
    report=dict(D=len(r),coupling=g,controls=vars(args),source_sha256=sources,reference_sha256=references,
        checkpoint_provenance=snapshot['checkpoint_sha256'] if snapshot else None,reference_kind=expected['kind'],
        candidate='persistent full trajectory',kinetic_candidate=cli.kinetic,cpu_objective_rerun=False,
        implicit_fallbacks=0,persistent_workers_and_streams=True,setup_included_in_each_complete_objective=True,
        preparation_serial_before_worker_dispatch=True,stream_intervals_do_not_prove_kernel_concurrency=True,
        evaluations=[],refinement=[],passed=False)
    if not cli.run_gpu:
        report['gpu_execution']=False;print(json.dumps(report,indent=2));return 0
    import cupy as cp
    try:
        with cp.cuda.Device(cli.device),overlap_driver.OverlapObjective() as coordinator:
            props=cp.cuda.runtime.getDeviceProperties(cli.device);name=props['name']
            report['hardware']=dict(gpu=name.decode() if isinstance(name,bytes) else name,cupy=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
            report['ptx']=trajectory_ptx_report(len(r)//2,cp,cli.ptx_directory)
            # First complete evaluations initialize per-worker libraries/kernels.
            # Both warmups are explicitly excluded; their graph preparation is
            # still serial, exactly as for all subsequent measured evaluations.
            started=time.perf_counter();warm={}
            for mode in ('cached','trajectory'):
                warm[mode]=evaluate_complete(coordinator,r,k,rho,g,args,cp,mode=mode,
                    kinetic=cli.kinetic,max_device_seconds=cli.max_device_seconds)
            baseline=warm['cached'];report['excluded_first_objectives_seconds']=time.perf_counter()-started
            report['excluded_first_objectives_stats']={mode:value['stats'] for mode,value in warm.items()}
            if not passed(compare(baseline,warm['trajectory'],rho,args.metric_reg)):
                raise overlap.OverlapFailure('warm complete energy/gradient/raw comparison failed')
            for repeat in range(cli.repeat):
                for mode in (('cached','trajectory') if repeat%2==0 else ('trajectory','cached')):
                    result=evaluate_complete(coordinator,r,k,rho,g,args,cp,mode=mode,
                        kinetic=cli.kinetic,max_device_seconds=cli.max_device_seconds)
                    comparison=compare(baseline,result,rho,args.metric_reg)
                    row=dict(repeat=repeat,mode=mode,complete_seconds=result['measured_complete_seconds'],
                        energy=result['value'],gradient_norm=math.sqrt(max(0.,metric(rho,result['gradient'],result['gradient'],args.metric_reg))),
                        comparison=comparison,stats=result['stats'],passed=passed(comparison))
                    if expected['value'] is not None:
                        row['energy_difference_reference']=abs(result['value']-expected['value'])
                        row['passed']=bool(row['passed'] and row['energy_difference_reference']<=1e-9)
                    if expected['gradient'] is not None:
                        delta=result['gradient']-expected['gradient']
                        row['gradient_metric_difference_reference']=math.sqrt(max(0.,metric(rho,delta,delta,args.metric_reg)))
                        row['passed']=bool(row['passed'] and row['gradient_metric_difference_reference']<=1e-8)
                    report['evaluations'].append(row);atomic_save(cli.output,report)
                    print(json.dumps({key:row[key] for key in ('repeat','mode','complete_seconds','passed','comparison')}),flush=True)
                    if not row['passed']:raise overlap.OverlapFailure('complete comparison failed')
            if cli.refine:
                # Established accuracy controls; keep the same strict kinetic
                # residual target so basis growth does not confound this test.
                for nodes in (512,1024):
                    fine_args=argparse.Namespace(**vars(args));fine_args.window=40.;fine_args.quadrature=1024
                    fine_args.nodes=nodes;fine_args.atol=1e-14;fine_args.rtol=1e-13
                    fine_cached=None
                    for mode in ('cached','trajectory'):
                        result=evaluate_complete(coordinator,r,k,rho,g,fine_args,cp,mode=mode,
                            kinetic=cli.kinetic,max_device_seconds=cli.max_device_seconds)
                        comparison=compare(baseline,result,rho,args.metric_reg)
                        if mode=='cached':fine_cached=result
                        matching_comparison=compare(fine_cached,result,rho,args.metric_reg)
                        report['refinement'].append(dict(mode=mode,controls=dict(window=40.,quadrature=1024,
                            nodes=nodes,atol=1e-14,rtol=1e-13,krylov_tolerance=fine_args.krylov_tolerance),
                            complete_seconds=result['measured_complete_seconds'],energy=result['value'],
                            comparison=comparison,matching_cached_comparison=matching_comparison,
                            cross_window_raw_differences_are_diagnostic=True,stats=result['stats'],
                            passed=physical_passed(comparison) and passed(matching_comparison)))
                        atomic_save(cli.output,report)
                        if not report['refinement'][-1]['passed']:raise overlap.OverlapFailure('complete refinement failed')
            report['summary']={mode:dict(complete_median_seconds=statistics.median(
                row['complete_seconds'] for row in report['evaluations'] if row['mode']==mode)) for mode in ('cached','trajectory')}
            report['summary']['paired_complete_speedup']=report['summary']['cached']['complete_median_seconds']/report['summary']['trajectory']['complete_median_seconds']
            report['passed']=True
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));report['passed']=False
    try:
        report['sources_unchanged']=(sources==source_hashes(args,cli.kinetic))
        report['references_unchanged']=(references=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cli.reference.glob('*')) if p.is_file()})
        report['checkpoint_unchanged']=(not snapshot or overlap_driver.verify_checkpoint(cli.checkpoint)['checkpoint_sha256']==snapshot['checkpoint_sha256'])
    except Exception as error:
        report.update(sources_unchanged=False,references_unchanged=False,checkpoint_unchanged=False,
                      provenance_guard_error=f'{type(error).__name__}: {error}')
    report['passed']=bool(report['passed'] and report['sources_unchanged'] and report['references_unchanged'] and report['checkpoint_unchanged'])
    report['gpu_execution']=True;atomic_save(cli.output,report)
    if report.get('summary'):print(json.dumps(report['summary']),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
