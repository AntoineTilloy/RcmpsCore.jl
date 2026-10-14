"""CPU LBFGS controller for the separate FP64 GPU observable implementation.

The metric, canonical retraction, preconditioner, initial direction, inverse-
Hessian scaling, and cautious updates follow RcmpsCore/OptimKit 0.5.  The line
search is bounded Armijo backtracking; it deliberately does not implement the
CPU driver's Hager-Zhang Wolfe search. No objective is evaluated by this module
except through the supplied callable. No checkpoint file I/O is performed.

``optimize`` yields trial/accepted/stopped events. Every accepted or stopped
event carries a complete ``state`` suitable for serialization and exact LBFGS
resumption. ``max_seconds`` limits the current invocation; elapsed_seconds in
the returned state includes earlier resumed invocations.
"""

from dataclasses import dataclass, fields
from time import perf_counter

import numpy as np

from .numerics import SolverFailure, metric, precondition, retract, stationary_state


@dataclass(frozen=True)
class OptimizationControls:
    max_iterations: int = 3000
    gradient_tol: float = 3e-5
    max_seconds: float | None = None
    history: int = 100
    metric_reg: float = 1e-3
    stationary_tolerance: float = 1e-13
    initial_precond_reg: float = 1e-4
    precond_prefactor: float = 1e-2
    initial_step: float = 1.0
    initial_direction_norm: float = 0.01
    warm_start_line_search: bool = False
    max_relative_step: float = 0.5
    line_search_trials: int = 30
    armijo_c1: float = 1e-4
    shrink: float = 0.5
    energy_atol: float = 0.0
    refresh_curvatures: bool = False


def _controls(controls):
    if isinstance(controls, OptimizationControls):
        out = controls
    else:
        values = dict(controls) if isinstance(controls, dict) else vars(controls).copy()
        aliases = {"iterations": "max_iterations", "gradtol": "gradient_tol",
                   "gradient_tolerance": "gradient_tol", "lbfgs_history": "history",
                   "max_trials": "line_search_trials", "armijo_energy_atol": "energy_atol"}
        for old, new in aliases.items():
            if old in values and new not in values:
                values[new] = values[old]
        supported = {f.name for f in fields(OptimizationControls)}
        out = OptimizationControls(**{k: v for k, v in values.items() if k in supported})
    if (out.max_iterations < 0 or out.history < 1 or out.line_search_trials < 1 or
            not np.isfinite(out.gradient_tol) or out.gradient_tol <= 0 or
            not np.isfinite(out.metric_reg) or out.metric_reg < 0 or
            not np.isfinite(out.stationary_tolerance) or out.stationary_tolerance <= 0 or
            not np.isfinite(out.initial_precond_reg) or out.initial_precond_reg <= 0 or
            not np.isfinite(out.precond_prefactor) or out.precond_prefactor <= 0 or
            not np.isfinite(out.initial_step) or out.initial_step <= 0 or
            not np.isfinite(out.initial_direction_norm) or out.initial_direction_norm <= 0 or
            not np.isfinite(out.max_relative_step) or out.max_relative_step <= 0 or
            not 0 < out.armijo_c1 < 1 or not 0 < out.shrink < 1 or
            not np.isfinite(out.energy_atol) or out.energy_atol < 0 or
            (out.max_seconds is not None and (not np.isfinite(out.max_seconds) or out.max_seconds <= 0))):
        raise ValueError("invalid optimization controls")
    return out


def _norm(rho, vector, metric_reg):
    squared = metric(rho, vector, vector, metric_reg)
    if not np.isfinite(squared) or squared < 0:
        raise SolverFailure(f"invalid tangent metric norm squared: {squared}")
    return float(np.sqrt(squared))


def _initial_direction(rho, gradient, metric_reg, precond_reg, desired_norm):
    pg = precondition(rho, gradient, metric_reg, precond_reg)
    norm_pg = _norm(rho, pg, metric_reg)
    if norm_pg == 0:
        raise SolverFailure("preconditioned direction vanished before convergence")
    return -(desired_norm/norm_pg)*pg


def _lbfgs_direction(rho, gradient, pairs, metric_reg, precond_reg):
    direction = gradient.copy()
    coefficients = []
    for step, change, inverse_curvature in reversed(pairs):
        coefficient = inverse_curvature*metric(rho, step, direction, metric_reg)
        coefficients.append(coefficient)
        direction -= coefficient*change
    # OptimKit rescales the current preconditioner to satisfy the latest secant
    # relation. This scaling is material when precond_reg changes with ||g||².
    step, change, _ = pairs[-1]
    pchange = precondition(rho, change, metric_reg, precond_reg)
    numerator = metric(rho, step, change, metric_reg)
    denominator = metric(rho, change, pchange, metric_reg)
    gamma = numerator/denominator if denominator > 0 else np.nan
    if not np.isfinite(gamma) or gamma <= 0:
        return None, gamma
    direction = gamma*precondition(rho, direction, metric_reg, precond_reg)
    for (step, change, inverse_curvature), coefficient in zip(pairs, reversed(coefficients)):
        direction += step*(coefficient-inverse_curvature*metric(rho, change, direction, metric_reg))
    return -direction, float(gamma)


def optimize(r, k, rho, energy, gradient, objective_callable, controls,
             resume_state=None):
    """Yield events while optimizing an already-evaluated accepted initial state.

    ``objective_callable(r,k,rho) -> (energy,gradient,stats)`` uses NumPy matrices
    for inputs/returned gradient; its observable work may run on a GPU. Accepted
    events have ``state`` containing R/K/rho/gradient/energy, normalized LBFGS
    pairs, precond_reg, next_step, iteration, evaluations, and elapsed_seconds.
    Resuming takes these fields from ``resume_state`` without a redundant initial
    objective call. ``max_iterations`` is the total accepted-iteration cap across
    resumes; ``max_seconds`` applies only to this invocation.
    """
    p = _controls(controls)
    resumed = {} if resume_state is None else resume_state
    r = np.array(resumed.get("R", resumed.get("r", r)), dtype=np.complex128, copy=True)
    k = np.array(resumed.get("K", resumed.get("k", k)), dtype=np.complex128, copy=True)
    rho = np.array(resumed.get("rho", rho), dtype=np.complex128, copy=True)
    gradient = np.array(resumed.get("gradient", resumed.get("g", gradient)), dtype=np.complex128, copy=True)
    energy = float(resumed.get("energy", energy))
    if (r.ndim != 2 or r.shape[0] != r.shape[1] or
            any(a.shape != r.shape for a in (k, rho, gradient)) or
            not np.isfinite(energy) or any(not np.isfinite(a).all() for a in (r, k, rho, gradient))):
        raise ValueError("invalid accepted optimization state")
    pairs = [(np.array(s, dtype=np.complex128, copy=True),
              np.array(y, dtype=np.complex128, copy=True), float(inverse))
             for s, y, inverse in resumed.get("pairs", [])][-p.history:]
    if any(s.shape != r.shape or y.shape != r.shape or not np.isfinite(inverse) or inverse <= 0 or
           not np.isfinite(s).all() or not np.isfinite(y).all() for s, y, inverse in pairs):
        raise ValueError("invalid resumed LBFGS history")
    iteration = int(resumed.get("iteration", 0))
    evaluations = int(resumed.get("evaluations", 0))
    precond_reg = float(resumed.get("precond_reg", p.initial_precond_reg))
    next_step = float(resumed.get("next_step", p.initial_step))
    elapsed_before = float(resumed.get("elapsed_seconds", 0.0))
    if (iteration < 0 or evaluations < 0 or not np.isfinite(precond_reg) or precond_reg <= 0 or
            not np.isfinite(next_step) or next_step <= 0 or
            not np.isfinite(elapsed_before) or elapsed_before < 0):
        raise ValueError("invalid resumed optimizer controls or counters")
    last_stats = resumed.get("stats")
    begin = perf_counter()

    def snapshot():
        # Subsequent iterations replace arrays and pair lists; yielded accepted
        # snapshots therefore remain unchanged while the generator resumes.
        return dict(R=r, K=k, rho=rho, gradient=gradient, g=gradient, energy=energy,
                    pairs=list(pairs), precond_reg=precond_reg, next_step=next_step,
                    iteration=iteration, evaluations=evaluations,
                    elapsed_seconds=elapsed_before+perf_counter()-begin,
                    gradient_norm=_norm(rho, gradient, p.metric_reg), stats=last_stats)

    def stopped(reason, converged=False, **details):
        return dict(type="stopped", event="stopped", reason=reason, converged=converged,
                    state=snapshot(), **details)

    def over_budget():
        return p.max_seconds is not None and perf_counter()-begin >= p.max_seconds

    while True:
        gn = _norm(rho, gradient, p.metric_reg)
        if gn <= p.gradient_tol:
            yield stopped("gradient tolerance reached", converged=True)
            return
        if iteration >= p.max_iterations:
            yield stopped("requested iteration limit reached")
            return
        if over_budget():
            yield stopped("optimization time budget reached")
            return
        iteration_begin = perf_counter()
        if p.refresh_curvatures:
            refreshed = []
            for s, y, _ in pairs:
                sy = metric(rho, s, y, p.metric_reg)
                ss = metric(rho, s, s, p.metric_reg)
                if ss > 0 and sy/ss > gn/10000:
                    refreshed.append((s, y, 1/sy))
            pairs = refreshed
        gamma = None
        if pairs:
            direction, gamma = _lbfgs_direction(rho, gradient, pairs, p.metric_reg, precond_reg)
        else:
            direction = _initial_direction(rho, gradient, p.metric_reg, precond_reg,
                                           p.initial_direction_norm)
        slope = metric(rho, gradient, direction, p.metric_reg) if direction is not None else np.nan
        if not np.isfinite(slope) or slope >= 0:
            pairs = []
            direction = _initial_direction(rho, gradient, p.metric_reg, precond_reg,
                                           p.initial_direction_norm)
            slope = metric(rho, gradient, direction, p.metric_reg)
        if not np.isfinite(slope) or slope >= 0:
            raise SolverFailure("preconditioned direction is not a descent direction")
        alpha_max = p.max_relative_step*max(1., np.linalg.norm(r))/max(np.linalg.norm(direction), np.finfo(float).eps)
        alpha = min(next_step, alpha_max)
        accepted = False
        failures = []
        for trial in range(1, p.line_search_trials+1):
            if over_budget():
                yield stopped("optimization time budget reached during line search",
                              numerical_failures=failures)
                return
            rr, kk = retract(r, k, direction, alpha)
            qq = -1j*kk-rr.conj().T@rr/2
            try:
                rhonew, stationary_stats = stationary_state(qq, rr, tolerance=p.stationary_tolerance)
                evaluations += 1
                enew, gnew, newstats = objective_callable(rr, kk, rhonew)
                enew = float(enew)
                gnew = np.asarray(gnew, dtype=np.complex128)
                if not np.isfinite(enew) or not np.isfinite(gnew).all() or gnew.shape != r.shape:
                    raise SolverFailure("trial returned nonfinite or invalid objective/gradient")
                bound = energy+p.armijo_c1*alpha*slope+p.energy_atol
                # alpha can become a NumPy scalar after the relative-step cap;
                # comparison then returns np.bool_, which JSON cannot encode.
                accepted = bool(enew <= bound)
                yield dict(type="trial", event="trial", iteration=iteration+1, trial=trial,
                           alpha=float(alpha), energy=enew, armijo_bound=float(bound),
                           accepted=accepted, stats=newstats, stationary=stationary_stats,
                           evaluations=evaluations)
                if accepted:
                    break
            except (SolverFailure, RuntimeError) as error:
                detail = str(error)
                failures.append(detail)
                yield dict(type="trial", event="trial", iteration=iteration+1, trial=trial,
                           alpha=float(alpha), accepted=False, error=detail, evaluations=evaluations)
            alpha *= p.shrink
        if not accepted:
            yield stopped("Armijo line search exhausted; last accepted state retained",
                          numerical_failures=failures)
            return
        step, change = alpha*direction, gnew-gradient
        new_gn = _norm(rhonew, gnew, p.metric_reg)
        sy = metric(rhonew, step, change, p.metric_reg)
        ss = metric(rhonew, step, step, p.metric_reg)
        cautious_pass = np.isfinite(sy) and np.isfinite(ss) and ss > 0 and sy/ss > new_gn/10000
        if cautious_pass:
            norm_step = np.sqrt(ss)
            pairs = (pairs+[(step/norm_step, change/norm_step, ss/sy)])[-p.history:]
        r, k, rho, energy, gradient = rr, kk, rhonew, enew, gnew
        last_stats = newstats
        iteration += 1
        next_step = min(1., 2*alpha) if p.warm_start_line_search else p.initial_step
        precond_reg = max(10*np.finfo(float).eps, p.precond_prefactor*min(new_gn*new_gn, 1.))
        state = snapshot()
        yield dict(type="accepted", event="accepted", state=state,
                   iteration=iteration, energy=energy, gradient_norm=new_gn,
                   alpha=float(alpha), trials=trial, seconds=perf_counter()-iteration_begin,
                   stats=newstats, stationary=stationary_stats, gamma=gamma,
                   cautious_update=bool(cautious_pass), history_size=len(pairs),
                   evaluations=evaluations)
