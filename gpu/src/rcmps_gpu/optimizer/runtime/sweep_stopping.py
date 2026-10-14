"""Optional progress stop for the dimension sweep; no observable evaluation.

Energy or gradient progress keeps a run alive. A progress anchor only moves
after a material cumulative change, so many small improvements still count.
The underlying optimizer retains its independent target and time limits.
"""
from dataclasses import dataclass
import math

from optimizer import optimize


@dataclass(frozen=True)
class ProgressControls:
    patience: int = 100
    minimum_iterations: int = 200
    energy_improvement: float = 1e-10
    gradient_relative_improvement: float = 0.05

    def __post_init__(self):
        if (self.patience < 1 or self.minimum_iterations < 0 or
                not math.isfinite(self.energy_improvement) or self.energy_improvement <= 0 or
                not 0 < self.gradient_relative_improvement < 1):
            raise ValueError("invalid progress stopping controls")


class ProgressMonitor:
    def __init__(self, energy, gradient_norm, iteration, controls):
        self.controls = controls
        self.energy_anchor = float(energy)
        self.gradient_anchor = float(gradient_norm)
        self.last_progress = int(iteration)

    def observe(self, energy, gradient_norm, iteration):
        p = self.controls
        progress = False
        if self.energy_anchor-float(energy) >= p.energy_improvement:
            self.energy_anchor = float(energy)
            progress = True
        if float(gradient_norm) <= (1-p.gradient_relative_improvement)*self.gradient_anchor:
            self.gradient_anchor = float(gradient_norm)
            progress = True
        if progress:
            self.last_progress = int(iteration)
        return (iteration >= p.minimum_iterations and
                iteration-self.last_progress >= p.patience)


def optimize_with_progress(r, k, rho, energy, gradient, objective_callable,
                           controls, resume_state=None, *, progress_controls,
                           initial_gradient_norm, prior_history=()):
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
    events = optimize(r, k, rho, energy, gradient, objective_callable, controls, resume_state)
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
