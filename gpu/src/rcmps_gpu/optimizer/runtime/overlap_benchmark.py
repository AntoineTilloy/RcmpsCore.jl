"""FP64 kinetic/quartic stream overlap of one objective, as overlap_sweep_driver runs it.

Two host workers use Device(device) and separate nonblocking CuPy streams.
An input-ready event precedes both components, and completion events precede
their sum and the single CPU canonical boundary/metric conversion. Each backend
owns its scratch; no context or solver is reused across evaluations.
Quartic preparation is serialized before worker dispatch and included in every
objective. Only run/pullback overlap kinetic execution.

The stream timing intervals include host submission gaps, so overlapping event
intervals do not prove simultaneous kernel execution. Cooperative quartic
kernels and cuBLAS can compete for GPU residency/bandwidth.

The benchmark command line of the research tree was removed with the cached
row solvers it compared. logic_self_tests() runs only bounded D4 NumPy logic
tests and never imports CuPy.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .kinetic_backend import kinetic_raw
from .numerics import adj, finish_gradient, metric, stationary_state


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
    """Run/pullback/scaling sequence of a quartic solver prepared before concurrent GPU execution.

    A subclass constructor prepares cp, q, r, coupling, pullback, points, weights,
    rhs, solver and setup_seconds (persistent_trajectory_overlap.PreparedTrajectoryQuartic).
    The cached row solvers that this constructor built were removed.
    One object is used once; its solver is closed after both workers finish.
    """
    def __init__(self, cp, inputs, coupling, args):
        raise TypeError("PreparedQuartic prepares no solver itself; use "
                        "persistent_trajectory_overlap.PreparedTrajectoryQuartic")

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

    A failed job never abandons the other running job. Exceptions are
    collected only after both futures finish, so stream and solver owners
    remain alive through their final synchronization.
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


def compare_finished(first, second, rho, metric_reg):
    return dict(energy_difference=abs(first['value']-second['value']),
        covector_frobenius_difference=float(np.linalg.norm(first['covector']-second['covector'])),
        environment_frobenius_difference=float(np.linalg.norm(first['environment']-second['environment'])),
        gradient_metric_difference=math.sqrt(max(0.,metric(rho,
            first['gradient']-second['gradient'],first['gradient']-second['gradient'],metric_reg))))


def logic_self_tests():
    """Scientific D4/controller tests; no CuPy import or GPU initialization."""
    import unittest
    from .ode_backend import evaluate_quartic

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
