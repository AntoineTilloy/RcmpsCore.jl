"""Complete E/gradient of the one-kernel trajectory, including preparation and boundary.

One persistent coordinator owns two workers and two component streams plus the
combine stream. The trajectory is prepared serially before dispatch. Both
workers join before scratch/kernel owners close, including component failure.
With mode 'cached', candidate_bindings selects instead the factory bound to
ORIGINAL_PREPARED, where the worker binds its independent CuPy oracle.

Baseline kinetic is the default. Optional cooperative Arnoldi is an additive
experimental case, never a silent fallback. The comparison command line of the
research tree was removed with the cached row solver it compared.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import time
import numpy as np
from . import overlap_sweep_driver as overlap_driver
from . import overlap_benchmark as overlap
from .warm_sweep_driver import binding
from .numerics import metric
from .ode_backend import _PackedZ2QuarticRHS,_packed_z2_pullback,quadrature
from .persistent_trajectory_backend import PersistentTrajectoryDOP853

ORIGINAL_PREPARED=overlap.PreparedQuartic
ORIGINAL_KINETIC=overlap.kinetic_raw


class PreparedTrajectoryQuartic(overlap.PreparedQuartic):
    """Same run/pullback/scaling as PreparedQuartic; new one-kernel solver."""
    def __init__(self,cp,inputs,coupling,args,*,max_device_seconds=30.,tiles='mma'):
        q,r,rho=inputs;self.cp,self.q,self.r,self.coupling=cp,q,r,float(coupling)
        self.pullback=_packed_z2_pullback;started=time.perf_counter()
        self.points,self.weights=quadrature(args.window,args.quadrature)
        self.rhs=_PackedZ2QuarticRHS(q,r,rho,cp)
        self.solver=PersistentTrajectoryDOP853(self.rhs,np.r_[self.points,math.sqrt(args.window)],
            atol=args.atol,rtol=args.rtol,max_steps=args.max_steps,max_device_seconds=max_device_seconds,tiles=tiles)
        self.setup_seconds=time.perf_counter()-started


def cooperative_kinetic(q,r,rho,nodes=128,*,xp,tolerance,maxdim,arnoldi_backend,
                        projection_backend,fallback_tolerance=None,**options):
    if arnoldi_backend!='gpu' or fallback_tolerance is not None:
        raise ValueError('cooperative experiment requires GPU Arnoldi with no fallback')
    from .cooperative_arnoldi import kinetic_raw_cooperative
    return kinetic_raw_cooperative(q,r,rho,nodes,xp=xp,tolerance=tolerance,maxdim=maxdim,
        projection_backend=projection_backend,fma=True,**options)


@contextmanager
def candidate_bindings(mode,kinetic,max_device_seconds,tiles='mma'):
    if mode not in ('cached','trajectory') or kinetic not in ('baseline','cooperative') or tiles not in ('mma','scalar'):
        raise ValueError('unknown explicit candidate selection')
    factory=ORIGINAL_PREPARED if mode=='cached' else lambda cp,inputs,coupling,args: \
        PreparedTrajectoryQuartic(cp,inputs,coupling,args,max_device_seconds=max_device_seconds,tiles=tiles)
    selected=ORIGINAL_KINETIC if kinetic=='baseline' else cooperative_kinetic
    with binding(overlap,'PreparedQuartic',factory),binding(overlap,'kinetic_raw',selected):yield


def evaluate_complete(coordinator,r,k,rho,coupling,args,xp,*,mode,kinetic='baseline',max_device_seconds=30.,
                      tiles='mma'):
    """Return complete result; capture already-summed raw terms without rerun."""
    raw={};original_finish=overlap_driver.driver.finish_gradient
    def capture_finish(q,r,rho,covector,environment,**options):
        raw['covector']=covector.copy();raw['environment']=environment.copy()
        return original_finish(q,r,rho,covector,environment,**options)
    started=time.perf_counter()
    with candidate_bindings(mode,kinetic,max_device_seconds,tiles), \
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
