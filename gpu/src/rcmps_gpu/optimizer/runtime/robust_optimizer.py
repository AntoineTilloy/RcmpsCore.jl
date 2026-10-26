"""Isolated resolution-aware variant of the existing FP64 optimizer controller.

The controller loop is derived from optimizer.py. Its physical objective,
metric, retraction, preconditioning, L-BFGS direction and cautious condition
reuse the original numerical helpers. Added guards skip unresolved trials,
exclude unresolved/noisy secants, and bound recovery to a history/step reset
then one metric steepest direction before reporting numerical stagnation.

line_search="wolfe" (opt-in; "armijo", the bounded backtracking, is the
default) first runs the Hager-Zhang search of RcmpsCore's SafeguardedLineSearch,
which accepts exact or approximate Wolfe steps from the slope along the
retraction, then the same Armijo backtracking if that search fails.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from . import optimizer
from .optimizer import _controls, _norm, _initial_direction, _lbfgs_direction
from .numerics import SolverFailure, metric, retract
from .sweep_stopping import ProgressMonitor


LINE_SEARCHES = ("armijo", "wolfe")
# energy_atol of RcmpsCore's SafeguardedLineSearch: the energy increase an
# approximate Wolfe step may show (the epsilon of the Hager-Zhang search).
WOLFE_ENERGY_ATOL = 1e-10


@dataclass(frozen=True)
class RobustnessControls:
    energy_resolution_factor: float = 64.
    step_resolution_factor: float = 64.
    predicted_decrease_factor: float = 10.
    secant_gradient_floor: float = 1e-10
    max_recoveries: int = 2
    line_search: str = "armijo"
    wolfe_energy_atol: float | None = None

    def __post_init__(self):
        if (any(not np.isfinite(value) or value <= 0 for value in
                (self.energy_resolution_factor,self.step_resolution_factor,
                 self.predicted_decrease_factor)) or
                not np.isfinite(self.secant_gradient_floor) or self.secant_gradient_floor < 0 or
                self.max_recoveries not in (1,2)):
            raise ValueError("invalid robust optimization controls")
        # The Wolfe search takes its energy tolerance explicitly; the Armijo one has none.
        if (self.line_search not in LINE_SEARCHES or
                (self.line_search == "armijo") != (self.wolfe_energy_atol is None) or
                self.wolfe_energy_atol is not None and
                (not np.isfinite(self.wolfe_energy_atol) or self.wolfe_energy_atol < 0)):
            raise ValueError("invalid line search controls")


@dataclass(frozen=True)
class HagerZhangControls:
    """OptimKit 0.5.1's HagerZhangLineSearch as RcmpsCore's SafeguardedLineSearch
    constructs it: the defaults c1, c2, theta, gamma and rho, and the iteration
    and evaluation limits LS_MAXITER = 10 and LS_MAXFG = 20 it reads then."""
    c1: float = 0.1
    c2: float = 0.9
    theta: float = 0.5
    gamma: float = 2/3
    rho: float = 5.
    max_iterations: int = 10
    max_evaluations: int = 20


HAGER_ZHANG = HagerZhangControls()


@dataclass(frozen=True)
class LinePoint:
    """A line search trial: step, energy, slope along the retraction curve
    (metric(rho, g, direction) at the trial, whose tangent is the direction)
    and what the controller keeps of an accepted trial."""
    alpha: float
    energy: float
    slope: float
    data: tuple | None = None


class LineSearchFailure(Exception):
    """Ends a Hager-Zhang search; the Armijo backtracking runs instead, as in
    SafeguardedLineSearch on a SolverFailure."""


class _OutOfTime(Exception):
    pass


def wolfe_conditions(point, origin, epsilon, hz=HAGER_ZHANG):
    """OptimKit's checkexactwolfe and checkapproxwolfe of a trial from origin."""
    exact = (point.energy <= origin.energy+hz.c1*point.alpha*origin.slope and
             point.slope > hz.c2*origin.slope)
    approximate = (point.energy <= origin.energy+epsilon and
                   (2*hz.c1-1)*origin.slope >= point.slope >= hz.c2*origin.slope)
    return bool(exact), bool(approximate)


def wolfe_acceptance(point, origin, epsilon, hz=HAGER_ZHANG):
    """SafeguardedLineSearch's check of the trial the search returns (OptimKit
    also returns its last trial after exhausting its limits): "exact",
    "approximate", or None, for which the Armijo backtracking runs."""
    if not (np.isfinite(point.energy) and np.isfinite(point.slope) and point.alpha > 0):
        return None
    if (point.energy-origin.energy <= hz.c1*point.alpha*origin.slope and
            point.slope >= hz.c2*origin.slope):
        return "exact"
    if (point.energy <= origin.energy+epsilon and
            hz.c2*origin.slope <= point.slope <= (2*hz.c1-1)*origin.slope):
        return "approximate"
    return None


def hager_zhang(take, origin, alpha, epsilon, hz=HAGER_ZHANG):
    """Generator of OptimKit 0.5.1's Hager-Zhang search (Algorithm 851, CG_DESCENT)
    from origin (step 0) with first trial alpha, which it accepts if it satisfies
    either Wolfe condition (acceptfirst, as OptimKit's L-BFGS sets it).

    ``take(alpha, stage)`` is a generator returning the LinePoint of that step;
    its events pass through, and its exceptions end the search. Returns the
    point the search ends at, as OptimKit, also when its limits stopped it.
    """
    eps = np.finfo(float).eps
    fmax = origin.energy+epsilon

    def wolfe(c):
        return any(wolfe_conditions(c, origin, epsilon, hz))

    def secant(a, b):
        # Julia's division by zero gives a step outside every bracket, as NaN does.
        if b.slope == a.slope:
            return np.nan
        return (a.alpha*b.slope-b.alpha*a.slope)/(b.slope-a.slope)

    def bisect(a, b):
        # Applied when a has slope < 0 and energy <= fmax, b slope < 0 and energy > fmax.
        count = 0
        while True:
            if b.alpha-a.alpha <= eps**0.75 or count >= hz.max_evaluations:
                return a, b, count
            d = yield from take((1-hz.theta)*a.alpha+hz.theta*b.alpha, "bisect")
            count += 1
            if wolfe(d):
                return d, d, count
            if d.slope >= 0:
                return a, d, count
            if d.energy <= fmax:
                a = d
            else:
                b = d

    def update(a, b, alpha):
        if not a.alpha < alpha < b.alpha:
            return a, b, 0
        c = yield from take(alpha, "update")
        if not np.isfinite(c.energy):
            raise LineSearchFailure(f"nonfinite energy at step {alpha}")
        if wolfe(c):
            return c, c, 1
        if c.slope >= 0:
            return a, c, 1
        if c.energy <= fmax:
            return c, b, 1
        a, b, count = yield from bisect(a, c)
        return a, b, count+1

    def bracket(c):
        count, a, alpha = 0, origin, c.alpha
        while True:
            while not (np.isfinite(c.energy) and np.isfinite(c.slope)) and count < hz.max_evaluations:
                alpha = (a.alpha+alpha)/2
                c = yield from take(alpha, "bracket")
                count += 1
            if not (np.isfinite(c.energy) and np.isfinite(c.slope)):
                return a, a, count
            if c.slope >= 0:
                return a, c, count
            if c.energy > fmax:
                # OptimKit bisects from the origin here, not from the last a.
                a, b, more = yield from bisect(origin, c)
                return a, b, count+more
            a = c
            if count >= hz.max_evaluations:
                return a, a, count
            alpha *= hz.rho
            c = yield from take(alpha, "bracket")
            count += 1

    c = yield from take(alpha, "initial")
    if wolfe(c):
        return c
    a, b, evaluations = yield from bracket(c)
    evaluations += 1
    iteration = 1
    while a.alpha != b.alpha and b.alpha-a.alpha >= eps:
        if iteration >= hz.max_iterations or evaluations >= hz.max_evaluations:
            break
        # secant2, then a bisection if the bracket shrank by less than gamma.
        width = b.alpha-a.alpha
        alpha = secant(a, b)
        A, B, count = yield from update(a, b, alpha)
        evaluations += count
        if A.alpha == B.alpha:
            return A
        if alpha == B.alpha:
            a, b, count = yield from update(A, B, secant(b, B))
            evaluations += count
        elif alpha == A.alpha:
            a, b, count = yield from update(A, B, secant(a, A))
            evaluations += count
        else:
            a, b = A, B
        if a.alpha != b.alpha and b.alpha-a.alpha > hz.gamma*width:
            a, b, count = yield from update(a, b, (a.alpha+b.alpha)/2)
            evaluations += count
        iteration += 1
    return a


def optimize(r, k, rho, energy, gradient, objective_callable, controls,
             resume_state=None, *, robustness=None):
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
    robustness = RobustnessControls() if robustness is None else robustness
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
    recovery_mode = 0
    unresolved_acceptances = 0

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
        energy_resolution = max(p.energy_atol,
            robustness.energy_resolution_factor*np.finfo(float).eps*max(1., abs(energy)))
        step_resolution = robustness.step_resolution_factor*np.finfo(float).eps*max(1., np.linalg.norm(r))
        if p.refresh_curvatures:
            refreshed = []
            for s, y, _ in pairs:
                sy = metric(rho, s, y, p.metric_reg)
                ss = metric(rho, s, s, p.metric_reg)
                if ss > 0 and sy/ss > gn/10000:
                    refreshed.append((s, y, 1/sy))
            pairs = refreshed
        gamma = None
        if recovery_mode == 2:
            direction = -(p.initial_direction_norm/gn)*gradient
        elif pairs:
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
        unresolved_trial = False
        wolfe = robustness.line_search == "wolfe"
        searched = {}
        if wolfe:
            origin = LinePoint(0., energy, float(slope))
            wolfe_trials = 0

            def take(step, stage):
                nonlocal evaluations, wolfe_trials
                if over_budget():
                    raise _OutOfTime
                # RcmpsCore bounds the trials of the search, not the retraction curve.
                if step > alpha_max:
                    raise LineSearchFailure(f"trial step {step:.3e} exceeds the step bound {alpha_max:.3e}")
                if step*np.linalg.norm(direction) <= step_resolution:
                    raise LineSearchFailure(f"trial step {step:.3e} is below the step resolution")
                wolfe_trials += 1
                rr, kk = retract(r, k, direction, step)
                qq = -1j*kk-rr.conj().T@rr/2
                try:
                    rhonew, stationary_stats = optimizer.stationary_state(qq, rr, tolerance=p.stationary_tolerance)
                    evaluations += 1
                    enew, gnew, newstats = objective_callable(rr, kk, rhonew)
                    enew = float(enew)
                    gnew = np.asarray(gnew, dtype=np.complex128)
                    if not np.isfinite(enew) or not np.isfinite(gnew).all() or gnew.shape != r.shape:
                        raise SolverFailure("trial returned nonfinite or invalid objective/gradient")
                    point = LinePoint(float(step), enew, metric(rhonew, gnew, direction, p.metric_reg),
                                      (rr, kk, rhonew, gnew, newstats, stationary_stats))
                except (SolverFailure, RuntimeError) as error:
                    detail = str(error)
                    failures.append(detail)
                    yield dict(type="trial", event="trial", iteration=iteration+1, trial=wolfe_trials,
                               line_search="wolfe", stage=stage, alpha=float(step), accepted=False,
                               error=detail, evaluations=evaluations)
                    raise LineSearchFailure(detail) from None
                exact, approximate = wolfe_conditions(point, origin, robustness.wolfe_energy_atol)
                yield dict(type="trial", event="trial", iteration=iteration+1, trial=wolfe_trials,
                           line_search="wolfe", stage=stage, alpha=point.alpha, energy=enew,
                           slope=point.slope, initial_slope=origin.slope, exact_wolfe=exact,
                           approximate_wolfe=approximate, accepted=False, stats=newstats,
                           stationary=stationary_stats, evaluations=evaluations)
                return point

            condition = failure = None
            try:
                point = yield from hager_zhang(take, origin, alpha, robustness.wolfe_energy_atol)
                condition = wolfe_acceptance(point, origin, robustness.wolfe_energy_atol)
                if condition is None:
                    failure = f"the search ended at step {point.alpha:.3e} without the Wolfe conditions"
            except LineSearchFailure as error:
                failure = str(error)
            except _OutOfTime:
                yield stopped("optimization time budget reached during line search",
                              numerical_failures=failures)
                return
            yield dict(type="trial", event="trial", iteration=iteration+1, trial=wolfe_trials,
                       line_search="wolfe", stage="result", alpha=point.alpha if condition else None,
                       accepted=condition is not None, wolfe_condition=condition,
                       armijo_fallback=condition is None, reason=failure, evaluations=evaluations)
            searched = dict(line_search="wolfe", wolfe_condition=condition, wolfe_trials=wolfe_trials,
                            wolfe_failure=failure, armijo_fallback=condition is None)
            if condition is not None:
                rr, kk, rhonew, gnew, newstats, stationary_stats = point.data
                alpha, enew, accepted, trial = point.alpha, point.energy, True, wolfe_trials
                expected_decrease = -alpha*slope
        fallback = dict(line_search="armijo") if wolfe else {}
        if not accepted:
            for trial in range(1, p.line_search_trials+1):
                if over_budget():
                    yield stopped("optimization time budget reached during line search",
                                  numerical_failures=failures)
                    return
                expected_decrease = -alpha*slope
                nominal_step_norm = float(alpha*np.linalg.norm(direction))
                if (expected_decrease <= robustness.predicted_decrease_factor*energy_resolution or
                        nominal_step_norm <= step_resolution):
                    unresolved_trial = True
                    yield dict(type="trial", event="trial", iteration=iteration+1, trial=trial,
                        alpha=float(alpha), accepted=False, robust_skipped_unresolved=True,
                        expected_decrease=float(expected_decrease), energy_resolution=float(energy_resolution),
                        nominal_step_norm=nominal_step_norm, step_resolution=float(step_resolution),
                        recovery_mode=recovery_mode, evaluations=evaluations, **fallback)
                    break
                rr, kk = retract(r, k, direction, alpha)
                qq = -1j*kk-rr.conj().T@rr/2
                try:
                    rhonew, stationary_stats = optimizer.stationary_state(qq, rr, tolerance=p.stationary_tolerance)
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
                               evaluations=evaluations, **fallback)
                    if accepted:
                        break
                except (SolverFailure, RuntimeError) as error:
                    detail = str(error)
                    failures.append(detail)
                    yield dict(type="trial", event="trial", iteration=iteration+1, trial=trial,
                               alpha=float(alpha), accepted=False, error=detail, evaluations=evaluations,
                               **fallback)
                alpha *= p.shrink
            if wolfe:
                trial += wolfe_trials
        if not accepted:
            if recovery_mode < robustness.max_recoveries:
                pairs = []
                next_step = p.initial_step
                recovery_mode += 1
                yield dict(type="trial", event="trial", iteration=iteration+1, trial=trial,
                    alpha=float(next_step), accepted=False, robust_restart=True,
                    reason="unresolved trial" if unresolved_trial else "line search exhausted",
                    recovery_mode=recovery_mode, history_size=0, evaluations=evaluations)
                continue
            if unresolved_trial:
                yield stopped("Numerical stagnation: unresolved trial after bounded clean recovery",
                              numerical_failures=failures, recovery_mode=recovery_mode)
                return
            yield stopped(("Wolfe search and Armijo backtracking" if wolfe else "Armijo line search")
                          + " exhausted; last accepted state retained",
                          numerical_failures=failures, recovery_mode=recovery_mode)
            return
        step, change = alpha*direction, gnew-gradient
        new_gn = _norm(rhonew, gnew, p.metric_reg)
        sy = metric(rhonew, step, change, p.metric_reg)
        ss = metric(rhonew, step, step, p.metric_reg)
        gradient_change_norm = _norm(rhonew, change, p.metric_reg)
        actual_step_norm = float(np.linalg.norm(rr-r))
        actual_energy_decrease = energy-enew
        secant_resolved = (actual_energy_decrease > energy_resolution and
            expected_decrease > robustness.predicted_decrease_factor*energy_resolution and
            actual_step_norm > step_resolution and
            gradient_change_norm > robustness.secant_gradient_floor)
        if wolfe:
            # The gradient resolves the secant where the energy cannot; the
            # cautious condition below decides, as for OptimKit's L-BFGS.
            secant_resolved = (actual_step_norm > step_resolution and
                               gradient_change_norm > robustness.secant_gradient_floor)
        cautious_pass = (secant_resolved and np.isfinite(sy) and np.isfinite(ss) and
                         ss > 0 and sy/ss > new_gn/10000)
        if cautious_pass:
            norm_step = np.sqrt(ss)
            pairs = (pairs+[(step/norm_step, change/norm_step, ss/sy)])[-p.history:]
        r, k, rho, energy, gradient = rr, kk, rhonew, enew, gnew
        last_stats = newstats
        iteration += 1
        next_step = min(1., 2*alpha) if p.warm_start_line_search else p.initial_step
        precond_reg = max(10*np.finfo(float).eps, p.precond_prefactor*min(new_gn*new_gn, 1.))
        unresolved_accepted = (actual_step_norm <= step_resolution or
            (actual_energy_decrease <= energy_resolution and
             gradient_change_norm <= robustness.secant_gradient_floor))
        stop_unresolved = False
        if unresolved_accepted:
            pairs = []
            next_step = p.initial_step
            unresolved_acceptances += 1
            stop_unresolved = unresolved_acceptances > robustness.max_recoveries
            recovery_mode = min(robustness.max_recoveries, recovery_mode+1)
        else:
            unresolved_acceptances = 0
            recovery_mode = 0
        state = snapshot()
        yield dict(type="accepted", event="accepted", state=state,
                   iteration=iteration, energy=energy, gradient_norm=new_gn,
                   alpha=float(alpha), trials=trial, seconds=perf_counter()-iteration_begin,
                   stats=newstats, stationary=stationary_stats, gamma=gamma,
                   cautious_update=bool(cautious_pass), history_size=len(pairs),
                   evaluations=evaluations, robust_secant_resolved=bool(secant_resolved),
                   robust_unresolved_accepted=bool(unresolved_accepted),
                   actual_step_norm=actual_step_norm, step_resolution=float(step_resolution),
                   gradient_change_norm=gradient_change_norm,
                   actual_energy_decrease=float(actual_energy_decrease),
                   expected_decrease=float(expected_decrease),
                   energy_resolution=float(energy_resolution), recovery_mode=recovery_mode,
                   **searched)
        if stop_unresolved and new_gn > p.gradient_tol:
            yield stopped("Numerical stagnation: repeated unresolved accepted steps after bounded clean recovery",
                          unresolved_acceptances=unresolved_acceptances)
            return


def optimize_with_progress(r, k, rho, energy, gradient, objective_callable,
                           controls, resume_state=None, *, progress_controls,
                           initial_gradient_norm, prior_history=(), robustness=None):
    """Yield ordinary optimizer events and a truthful, nonconverged plateau stop.

Resumed history reconstructs the progress anchors without another objective.
The last accepted state and L-BFGS pairs remain available for continuation.
"""
    monitor = ProgressMonitor(energy, initial_gradient_norm,
                              (resume_state or {}).get("iteration", 0), progress_controls)
    if prior_history:
        first = prior_history[0]
        monitor = ProgressMonitor(first["energy"], first["gradient_norm"],
                                  first["iteration"], progress_controls)
        for row in prior_history[1:]:
            monitor.observe(row["energy"], row["gradient_norm"], row["iteration"])
    events = optimize(r, k, rho, energy, gradient, objective_callable, controls, resume_state, robustness=robustness)
    try:
        for event in events:
            yield event
            if event["type"] == "accepted":
                state = event["state"]
                stalled = monitor.observe(state["energy"], state["gradient_norm"], state["iteration"])
                # Let the optimizer report a reached target on its next yield.
                if stalled and state["gradient_norm"] > controls.gradient_tol:
                    yield dict(type="stopped", event="stopped", converged=False,
                               reason="energy and gradient progress stalled",
                               state=state, last_progress_iteration=monitor.last_progress)
                    return
    finally:
        events.close()
