"""Isolated FP64 kinetic/quartic stream-overlap comparison; no optimizer edits.

Actual GPU execution is opt-in by running this script after other GPU work has
finished. Example (the benchmark is not safe to run beside an active sweep):
  OPENBLAS_NUM_THREADS=1 python overlap_benchmark.py results/reference_D64 \
      --repeat 3 --output results/a100_overlap.json

Two host workers use Device(device) and separate nonblocking CuPy streams.
An input-ready event precedes both components, and completion events precede
their sum and the single CPU canonical boundary/metric conversion. Each backend
owns its scratch and graph; no context or solver is reused across evaluations.
Quartic construction/capture is serialized before worker dispatch and included
in every measured objective. The inherited legacy graphs use GLOBAL capture,
so creating them while the kinetic worker synchronizes would be unsafe. Only
run/pullback overlap kinetic execution. Kernel and library compilation is
prewarmed serially and explicitly excluded/recorded.

The stream timing intervals include host submission gaps, so overlapping event
intervals do not prove simultaneous kernel execution. Latency and numerical
agreement must both be measured before claiming a gain. Cooperative quartic
kernels and cuBLAS can compete for GPU residency/bandwidth.

`--self-test` runs only bounded D4 NumPy logic tests and never imports CuPy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from kinetic_backend import kinetic_raw
from numerics import adj, finish_gradient, load_reference, metric, stationary_state


class OverlapFailure(RuntimeError):
    """A strict component or comparison requirement failed; no fallback."""


@dataclass
class ComponentResult:
    raw: dict
    wall_seconds: float
    start_event: object = None
    end_event: object = None
    thread_id: int = 0
    stream_pointer: int = 0


class PreparedQuartic:
    """Build unchanged cached solver classes before concurrent GPU execution.

    This reproduces the selected evaluator's run/pullback/scaling sequence.
    The benchmark compares it to that unmodified evaluator's serial warmup.
    One object is used once; its graph is closed after both workers finish.
    """
    def __init__(self, cp, inputs, coupling, args):
        from ode_backend import _packed_z2_pullback, quadrature
        q, r, rho = inputs
        self.cp, self.q, self.r, self.coupling = cp, q, r, float(coupling)
        self.pullback = _packed_z2_pullback
        started = time.perf_counter()
        self.points, self.weights = quadrature(args.window, args.quadrature)
        if args.quartic == 'cached' or len(q) == 64:
            from triton_backend import TritonPackedQuarticRHS
            from persistent_rows_cached import PersistentRowsCachedDOP853
            self.rhs = TritonPackedQuarticRHS(q, r, rho, cp)
            solver_type = PersistentRowsCachedDOP853
        else:
            from persistent_rows_general import make_rhs, GeneralCachedDOP853
            self.rhs = make_rhs(q, r, rho, cp)
            solver_type = GeneralCachedDOP853
        self.solver = solver_type(self.rhs, np.r_[self.points, math.sqrt(args.window)],
            atol=args.atol, rtol=args.rtol, max_steps=args.max_steps)
        self.setup_seconds = time.perf_counter()-started

    def __call__(self):
        started = time.perf_counter()
        samples, stats = self.solver.run()
        begin = time.perf_counter()
        ((value, covector, environment),) = self.pullback(
            self.q, self.r, self.rhs, samples, self.points, self.weights)
        covector, environment = self.coupling*covector, self.coupling*environment
        self.cp.cuda.get_current_stream().synchronize()
        stats.update(contraction_seconds=time.perf_counter()-begin,
            total_seconds=self.setup_seconds+time.perf_counter()-started,
            benchmark_serial_preparation_seconds=self.setup_seconds,
            benchmark_serial_preparation_before_worker=True)
        return dict(value=self.coupling*float(value), covector=covector,
                    environment=environment, stats=stats)

    def close(self):
        self.solver.close()


def checked_component(name, raw, xp, tolerance):
    """Require native FP64 storage and actually checked kinetic convergence."""
    if not math.isfinite(float(raw['value'])):
        raise OverlapFailure(f'{name} returned a nonfinite energy')
    for field in ('covector', 'environment'):
        if raw[field].dtype != xp.complex128:
            raise OverlapFailure(f'{name} {field} must retain complex128')
    if name == 'kinetic':
        stats = raw['stats']
        residual = float(stats['maximum_relative_residual'])
        if (not math.isfinite(residual) or residual > tolerance
                or int(stats['fallback_solves']) != 0):
            raise OverlapFailure('kinetic true residual or no-fallback requirement failed')
        for solve in stats['solves'].values():
            if (bool(solve['fallback_used'])
                    or float(solve['true_relative_residual']) > tolerance):
                raise OverlapFailure('an individual kinetic solve failed its strict requirement')
    return raw


def dispatch_pair(calls, executor, *, parallel, worker):
    """Dispatch precisely two independent jobs and join both before returning.

    A failed job never abandons the other running graph. Exceptions are
    collected only after both futures finish, so stream/graph owners remain
    alive through their final synchronization.
    """
    if set(calls) != {'kinetic', 'quartic'}:
        raise ValueError('exactly one kinetic and one quartic component required')
    names = ('kinetic', 'quartic')
    if not parallel:
        return {name: executor.submit(worker, name, calls[name], None).result()
                for name in names}
    barrier = threading.Barrier(2)
    futures = {name: executor.submit(worker, name, calls[name], barrier) for name in names}
    wait(tuple(futures.values()))
    errors = {name: f.exception() for name, f in futures.items() if f.exception() is not None}
    if errors:
        message = '; '.join(f'{name}: {type(error).__name__}: {error}'
                            for name, error in errors.items())
        raise OverlapFailure(message) from next(iter(errors.values()))
    return {name: future.result() for name, future in futures.items()}


def gpu_worker(cp, device, streams, ready, tolerance):
    def worker(name, call, barrier):
        with cp.cuda.Device(device), streams[name]:
            stream = cp.cuda.get_current_stream()
            if int(stream.ptr) != int(streams[name].ptr):
                raise OverlapFailure('worker did not enter its declared current stream')
            stream.wait_event(ready)
            # Both jobs enter their own stream/device before the common start.
            if barrier is not None:
                barrier.wait(timeout=30.)
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(stream)
            start = time.perf_counter()
            try:
                raw = checked_component(name, call(), cp, tolerance)
                end.record(stream)
                # Covers any final coupling scaling enqueued after the
                # backend's own synchronization, including generalized RHS.
                end.synchronize()
            except BaseException:
                stream.synchronize()
                raise
            return ComponentResult(raw, time.perf_counter()-start, begin, end,
                                   threading.get_ident(), int(stream.ptr))
    return worker


def raw_sum(results, xp):
    kinetic, quartic = (results[name].raw for name in ('kinetic', 'quartic'))
    return dict(value=float(kinetic['value']+quartic['value']),
                covector=kinetic['covector']+quartic['covector'],
                environment=kinetic['environment']+quartic['environment'])


def require_immutable_inputs(original, actual):
    if any(not np.array_equal(a, b) for a, b in zip(original, actual)):
        raise OverlapFailure('a component mutated immutable Q/R/rho inputs')


def source_hashes(quartic):
    names = ('overlap_benchmark.py', 'kinetic_backend.py', 'numerics.py',
             'ode_backend.py', 'triton_backend.py', 'device_control_backend.py',
             'jax_source.py', 'persistent_backend.py', 'persistent_rows.py',
             'persistent_rows_cached.py')
    if quartic == 'general':
        names += ('persistent_rows_general.py',)
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}


def gpu_evaluation(cp, calls, executor, streams, combine, original, inputs,
                   q, r, rho, *, parallel, tolerance, metric_reg,
                   coupling, args):
    # Inputs were produced on this calling thread. Each worker waits for this
    # event even after repeated evaluations, making the dependency explicit.
    started = time.perf_counter()
    inputs_ready = cp.cuda.Event()
    inputs_ready.record(cp.cuda.get_current_stream())
    with combine:
        combine.wait_event(inputs_ready)
        # The legacy attempt/dense constructors use global capture. There are
        # no active worker jobs here; their graph creation stays serialized.
        prepared = PreparedQuartic(cp, inputs, coupling, args)
        ready = cp.cuda.Event()
        ready.record(combine)
    try:
        measured_calls = dict(kinetic=calls['kinetic'], quartic=prepared)
        results = dispatch_pair(measured_calls, executor, parallel=parallel,
            worker=gpu_worker(cp, cp.cuda.Device().id, streams, ready, tolerance))
    finally:
        prepared.close()
    with combine:
        for result in results.values():
            combine.wait_event(result.end_event)
        raw = raw_sum(results, cp)
        covector, environment = (cp.asnumpy(raw[key]) for key in ('covector', 'environment'))
        actual = tuple(cp.asnumpy(array) for array in inputs)
    require_immutable_inputs(original, actual)
    gradient, boundary = finish_gradient(q, r, rho, covector, environment,
                                         metric_reg=metric_reg)
    total_seconds = time.perf_counter()-started
    intervals = {name: dict(
        start_ms=float(cp.cuda.get_elapsed_time(ready, result.start_event)),
        end_ms=float(cp.cuda.get_elapsed_time(ready, result.end_event)))
        for name, result in results.items()}
    overlap_ms = max(0., min(row['end_ms'] for row in intervals.values())
                    - max(row['start_ms'] for row in intervals.values()))
    item = dict(parallel=bool(parallel), total_seconds=total_seconds,
        energy=raw['value'], gradient_norm=math.sqrt(max(0., metric(rho, gradient, gradient, metric_reg))),
        component_wall_seconds={name: float(result.wall_seconds) for name, result in results.items()},
        component_stats={name: result.raw['stats'] for name, result in results.items()},
        worker_thread_ids={name: int(result.thread_id) for name, result in results.items()},
        stream_pointers={name: int(result.stream_pointer) for name, result in results.items()},
        stream_intervals=intervals, stream_interval_overlap_ms=float(overlap_ms),
        boundary=boundary, implicit_fallbacks=0, immutable_inputs_verified=True,
        serial_quartic_preparation_seconds=float(prepared.setup_seconds))
    if parallel and len(set(item['worker_thread_ids'].values())) != 2:
        raise OverlapFailure('parallel evaluation did not use two host workers')
    if len(set(item['stream_pointers'].values())) != 2:
        raise OverlapFailure('components did not use separate CUDA streams')
    return item, dict(value=raw['value'], covector=covector,
                     environment=environment, gradient=gradient)


def compare_finished(first, second, rho, metric_reg):
    return dict(energy_difference=abs(first['value']-second['value']),
        covector_frobenius_difference=float(np.linalg.norm(first['covector']-second['covector'])),
        environment_frobenius_difference=float(np.linalg.norm(first['environment']-second['environment'])),
        gradient_metric_difference=math.sqrt(max(0.,metric(rho,
            first['gradient']-second['gradient'],first['gradient']-second['gradient'],metric_reg))))


def logic_self_tests():
    """Scientific D4/controller tests; no CuPy import or GPU initialization."""
    import unittest
    from ode_backend import evaluate_quartic

    class LogicTests(unittest.TestCase):
        def test_D4_parallel_raw_sum_and_stationary_response(self):
            rng = np.random.default_rng(8057)
            r = np.zeros((4,4), dtype=np.complex128)
            r[:2,2:] = .15*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
            r[2:,:2] = .15*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
            k = np.zeros_like(r)
            for block in (slice(0,2),slice(2,4)):
                a = .15*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
                k[block,block] = a+adj(a)
            q = -1j*k-adj(r)@r/2
            rho = stationary_state(q,r,tolerance=1e-13)[0]
            original = tuple(a.copy() for a in (q,r,rho))
            calls = dict(kinetic=lambda: kinetic_raw(q,r,rho,32,xp=np,
                                                   tolerance=1e-12,maxdim=16),
                quartic=lambda: evaluate_quartic(r,q,rho,2.7,xp=np,packed_z2=True,
                    half_window=2.,n_quad=32,atol=1e-12,rtol=1e-11))
            def worker(name,call,barrier):
                if barrier is not None:barrier.wait(timeout=5.)
                return ComponentResult(checked_component(name,call(),np,1e-12),0.,
                                       thread_id=threading.get_ident())
            values=[]
            with ThreadPoolExecutor(max_workers=2) as executor:
                for parallel in (False,True):
                    results=dispatch_pair(calls,executor,parallel=parallel,worker=worker)
                    if parallel:self.assertEqual(len({x.thread_id for x in results.values()}),2)
                    raw=raw_sum(results,np)
                    gradient=finish_gradient(q,r,rho,raw['covector'],raw['environment'],tolerance=1e-13)[0]
                    values.append(dict(raw,gradient=gradient))
            compare=compare_finished(*values,rho,1e-3)
            self.assertLess(compare['energy_difference'],1e-13)
            self.assertLess(compare['gradient_metric_difference'],1e-12)
            require_immutable_inputs(original,(q,r,rho))

        def test_strict_precision_residual_and_fallback_gates(self):
            raw=dict(value=1.,covector=np.zeros((4,4),complex),environment=np.zeros((4,4),complex),
                stats=dict(maximum_relative_residual=1e-13,fallback_solves=0,
                    solves={'a':dict(fallback_used=False,true_relative_residual=1e-13)}))
            checked_component('kinetic',raw,np,1e-12)
            raw['covector']=np.zeros((4,4),np.complex64)
            with self.assertRaises(OverlapFailure):checked_component('kinetic',raw,np,1e-12)
            raw['covector']=np.zeros((4,4),complex)
            raw['stats']['maximum_relative_residual']=2e-12
            with self.assertRaises(OverlapFailure):checked_component('kinetic',raw,np,1e-12)
            raw['stats']['maximum_relative_residual']=1e-13
            raw['stats']['fallback_solves']=1
            with self.assertRaises(OverlapFailure):checked_component('kinetic',raw,np,1e-12)
            with self.assertRaises(OverlapFailure):require_immutable_inputs((np.zeros(4),),(np.ones(4),))

        def test_failed_component_joins_other_worker(self):
            finished=threading.Event()
            def fail():raise ArithmeticError('explicit numerical failure')
            def succeed():time.sleep(.01);finished.set();return None
            def worker(name,call,barrier):barrier.wait(timeout=5.);return call()
            with ThreadPoolExecutor(max_workers=2) as executor:
                with self.assertRaisesRegex(OverlapFailure,'explicit numerical failure'):
                    dispatch_pair(dict(kinetic=fail,quartic=succeed),executor,parallel=True,worker=worker)
                self.assertTrue(finished.is_set())

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LogicTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',nargs='?')
    parser.add_argument('--checkpoint',help='Read an atomic GPU JSON/NPZ checkpoint; never modify it')
    parser.add_argument('--output')
    parser.add_argument('--quartic',choices=('cached','general'),default='cached')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--window',type=float,default=30.)
    parser.add_argument('--quadrature',type=int,default=512)
    parser.add_argument('--nodes',type=int,default=256)
    parser.add_argument('--atol',type=float,default=1e-13)
    parser.add_argument('--rtol',type=float,default=1e-12)
    parser.add_argument('--krylov-tolerance',type=float,default=1e-12)
    parser.add_argument('--maxdim',type=int,default=1024)
    parser.add_argument('--projection-chunk',type=int,default=64)
    parser.add_argument('--max-steps',type=int,default=1_000_000)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:return logic_self_tests()
    if (not args.reference or not args.output or args.repeat<2 or args.device<0
            or not 0<args.window<=40 or args.quadrature<2 or args.nodes<2
            or not 0<args.krylov_tolerance<=1e-12 or args.maxdim<16
            or args.projection_chunk<1 or args.max_steps<1):
        parser.error('GPU comparison requires reference/output, repeat>=2, and strict positive controls')
    matrices,metadata=load_reference(args.reference)
    r,k,rho=(matrices[key] for key in ('R','K','rho'))
    reference_energy=metadata.get('saved_energy') if metadata.get('saved_energy_available',True) else None
    reference_gradient=matrices.get('saved_gradient') if metadata.get('saved_gradient_available',True) else None
    reference_kind='saved_CPU' if reference_energy is not None else 'unevaluated_initial_state'
    checkpoint_provenance=None
    if args.checkpoint:
        path=Path(args.checkpoint)
        history=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as state:
            if int(state['iteration'])!=len(history['history']):
                raise OverlapFailure('checkpoint JSON/NPZ generations differ')
            r,k,rho,reference_gradient=(state[key].copy() for key in ('R','K','rho','gradient'))
            reference_energy=float(state['energy'])
        reference_kind='cached_GPU_checkpoint'
        checkpoint_provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (path,path.with_suffix('.npz'))}
    if len(r)!=metadata['D'] or any(x.dtype!=np.complex128 for x in (r,k,rho)):
        raise OverlapFailure('state dimensions/precision differ from the export')
    q=-1j*k-adj(r)@r/2
    original=tuple(x.copy() for x in (q,r,rho))
    import cupy as cp
    if args.quartic=='general':
        from persistent_rows_general import evaluate_quartic_persistent_rows_general as evaluate
    else:
        from persistent_rows_cached import evaluate_quartic_persistent_rows_cached as evaluate
    with cp.cuda.Device(args.device):
        inputs=tuple(cp.asarray(x,dtype=cp.complex128) for x in original)
        qg,rg,rhog=inputs
        cp.cuda.get_current_stream().synchronize()
        streams={name:cp.cuda.Stream(non_blocking=True) for name in ('kinetic','quartic')}
        combine=cp.cuda.Stream(non_blocking=True)
        calls=dict(kinetic=lambda:kinetic_raw(qg,rg,rhog,args.nodes,xp=cp,
            tolerance=args.krylov_tolerance,maxdim=args.maxdim,arnoldi_backend='gpu',
            projection_backend='hessenberg_gpu',projection_chunk=args.projection_chunk,
            fallback_tolerance=None),
            quartic=lambda:evaluate(rg,qg,rhog,metadata['coupling'],xp=cp,
                half_window=args.window,n_quad=args.quadrature,atol=args.atol,rtol=args.rtol,
                max_steps=args.max_steps))
        properties=cp.cuda.runtime.getDeviceProperties(args.device)
        gpu=properties['name']
        report=dict(controls=vars(args),D=len(r),coupling=metadata['coupling'],
            reference_kind=reference_kind,checkpoint_provenance=checkpoint_provenance,
            source_sha256=source_hashes(args.quartic),
            reference_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(args.reference).glob('*.bin')},
            hardware=dict(gpu=gpu.decode() if isinstance(gpu,bytes) else gpu,cupy=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
            cpu_objective_rerun=False,implicit_fallbacks=0,evaluations=[],
            setup_included_in_each_measured_objective=True,
            quartic_construction_capture_serialized_before_worker_dispatch=True,
            comparison_reference='unmodified serial component evaluators during excluded prewarm',
            stream_interval_overlap_does_not_prove_simultaneous_kernels=True)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        def save():
            temporary=output.with_suffix(output.suffix+'.tmp')
            temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)
        try:
            started=time.perf_counter()
            warm_stats={}
            warm_raw={}
            # Serialize first library/kernel compilation and process-wide
            # Triton adapter initialization. No graph/solver survives warmup.
            for name in ('kinetic','quartic'):
                with streams[name]:
                    raw=checked_component(name,calls[name](),cp,args.krylov_tolerance)
                    streams[name].synchronize();warm_stats[name]=raw['stats']
                    warm_raw[name]=dict(value=float(raw['value']),
                        covector=cp.asnumpy(raw['covector']),environment=cp.asnumpy(raw['environment']))
                    del raw
            report['excluded_serial_prewarm']=dict(seconds=time.perf_counter()-started,stats=warm_stats)
            baseline=dict(value=warm_raw['kinetic']['value']+warm_raw['quartic']['value'],
                covector=warm_raw['kinetic']['covector']+warm_raw['quartic']['covector'],
                environment=warm_raw['kinetic']['environment']+warm_raw['quartic']['environment'])
            baseline['gradient']=finish_gradient(q,r,rho,baseline['covector'],baseline['environment'],
                                                  metric_reg=metadata['metric_reg'])[0]
            with ThreadPoolExecutor(max_workers=2) as executor:
                for pair in range(args.repeat):
                    order=(False,True) if pair%2==0 else (True,False)
                    paired={}
                    for parallel in order:
                        item,value=gpu_evaluation(cp,calls,executor,streams,combine,original,inputs,q,r,rho,
                            parallel=parallel,tolerance=args.krylov_tolerance,metric_reg=metadata['metric_reg'],
                            coupling=metadata['coupling'],args=args)
                        item.update(pair=pair,mode='overlap' if parallel else 'sequential',
                                    difference_from_first=compare_finished(baseline,value,rho,metadata['metric_reg']))
                        if reference_energy is not None:item['energy_difference_reference']=abs(value['value']-reference_energy)
                        if reference_gradient is not None:
                            delta=value['gradient']-reference_gradient
                            item['gradient_metric_difference_reference']=math.sqrt(max(0.,metric(rho,delta,delta,metadata['metric_reg'])))
                        difference=item['difference_from_first']
                        item['passed']=bool(difference['energy_difference']<=1e-10
                            and difference['gradient_metric_difference']<=5e-9
                            and difference['covector_frobenius_difference']<=1e-10*max(1.,np.linalg.norm(baseline['covector']))
                            and difference['environment_frobenius_difference']<=1e-10*max(1.,np.linalg.norm(baseline['environment'])))
                        report['evaluations'].append(item);paired[parallel]=item;save()
                        print(json.dumps({key:item[key] for key in ('pair','mode','total_seconds','passed','difference_from_first')}),flush=True)
                        if not item['passed']:raise OverlapFailure('finished-gradient comparison failed')
                    paired[True]['paired_wall_speedup']=paired[False]['total_seconds']/paired[True]['total_seconds']
            sequential=[row['total_seconds'] for row in report['evaluations'] if not row['parallel']]
            overlapped=[row['total_seconds'] for row in report['evaluations'] if row['parallel']]
            report['summary']=dict(sequential_median_seconds=float(np.median(sequential)),
                overlap_median_seconds=float(np.median(overlapped)),
                median_wall_speedup=float(np.median(sequential)/np.median(overlapped)),
                all_numerical_checks_passed=bool(all(row['passed'] for row in report['evaluations'])))
        except Exception as error:
            report['failure']=dict(type=type(error).__name__,message=str(error))
        save()
        return 1 if 'failure' in report else 0


if __name__=='__main__':
    raise SystemExit(main())
