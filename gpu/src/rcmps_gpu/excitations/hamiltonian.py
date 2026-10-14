"""Matrix-free Hamiltonian action F(W) in scaled variables.

Port of excitation_hamiltonian_action (src/excitation_action.jl) with
kinetic_backend=:resolvent and a fixed theta grid: kinetic specs and quadratic
monomials go to the resolvent quadrature (kinetic.py), every other term to the ODE
route (potential.py or a GPU integrator). The boundary terms are linear, so the
tangent-independent boundary data of all terms are summed once and each action
does a single shifted dual Poisson solve.

Input and output are hat variables, W = Wh S^-1 and F = Fh S, so that
tr(U' F) = tr(Uh' Fh) for any pair of tangents.
"""
from __future__ import annotations

from .kinetic import KineticAction, TangentFrame, finish_boundary, kinetic_terms
from .potential import PotentialAction, integrate_scipy
from .scaled import stack
from .states import SolverFailure


class HamiltonianAction:
    """`terms`: [(coefficient, Observable)]; `integrator`: IntegratorParameters for the
    ODE terms; `kinetic_nodes`: fixed theta nodes; `tol`: relative residual of every
    linear solve; `integrate`, `system`: ODE backend and equations (potential.PotentialAction)."""
    def __init__(self, ss, momentum, terms, *, integrator, kinetic_nodes=256, tol=1e-12,
                 integrate=integrate_scipy, system=None, **controls):
        self.ss, self.p, self.tol, self.controls = ss, float(momentum), tol, controls
        self.integrator, self.kinetic_nodes = integrator, kinetic_nodes
        resolved = kinetic_terms(terms)
        self.kinetic = (KineticAction(ss, momentum, resolved, kinetic_nodes, tol=tol, **controls)
                        if resolved else None)
        self.potentials = [(float(c), PotentialAction(ss, momentum, spec, integrator,
                                                      integrate=integrate, system=system, tol=tol,
                                                      **controls))
                           for c, spec in terms if (c, spec) not in resolved]
        parts = ([self.kinetic.boundary] if self.kinetic else []) + [
            p.boundary.scaled(c) for c, p in self.potentials]
        if not parts:
            raise ValueError("empty Hamiltonian")
        self.boundary = parts[0]
        for part in parts[1:]:
            self.boundary = self.boundary + part
        self.calls = 0

    def frame(self, wh):
        """TangentFrame of a batch (D, n, D) or of one (D, D) tangent."""
        xp = self.ss.xp
        if wh.ndim == 2:
            wh = stack([wh], xp)
        if not bool(xp.isfinite(wh).all()):
            raise ValueError("nonfinite tangent")
        return TangentFrame(self.ss, self.p, wh, tol=self.tol, **self.controls)

    def __call__(self, wh):
        single = wh.ndim == 2
        frame = self.frame(wh)
        xp = self.ss.xp
        out, a0 = xp.zeros_like(frame.wh), xp.zeros_like(frame.wh)
        if self.kinetic:
            f, a = self.kinetic.bulk(frame)
            out += f
            a0 += a
        for coefficient, potential in self.potentials:
            f, a = potential.bulk(frame)
            out += coefficient * f
            a0 += coefficient * a
        boundary, _ = finish_boundary(frame, self.boundary, a0, tol=self.tol, **self.controls)
        out += boundary
        if not bool(xp.isfinite(out).all()):
            raise SolverFailure("nonfinite Hamiltonian action")
        self.calls += frame.count
        return out[:, 0, :] if single else out
