"""Bilinear energies with the potential terms on the fused kernel.

bilinear.BilinearElement integrates the four-block source ODE with a host-controlled
integrator, which is the reference but too slow on a GPU beyond small bond dimension.
For the diagonal elements <Phi(W)|H|Phi(W)> that the validation step needs, the same
ODE is the Z2-packed system of packed_pair.py, which the fused trajectory kernel
integrates in one launch. Everything else (environments, kinetic resolvent sum, and
any element or term the packed system does not cover) is inherited unchanged.
"""
from __future__ import annotations

from bilinear import BilinearElement
from packed_pair import diagonal_observable


class FusedBilinearElement(BilinearElement):
    """BilinearElement whose monomial terms of diagonal elements use
    `packed_integrate(system, points, endpoint, integrator)`, e.g. fused.fused_integrator(cp).
    `fused_trajectories` counts them; other cases fall back to the inherited route."""
    def __init__(self, ss, momentum, terms, *, packed_integrate, **options):
        super().__init__(ss, momentum, terms, **options)
        self.packed_integrate = packed_integrate
        self.fused_trajectories = 0

    def observable(self, pair, spec):
        try:
            value = diagonal_observable(pair, spec, self.integrator, self.packed_integrate)
        except ValueError:      # not a diagonal Z2 monomial element: reference route
            return super().observable(pair, spec)
        self.trajectories += 1
        self.fused_trajectories += 1
        return value


def bound(packed_integrate):
    """A BilinearElement factory with `packed_integrate` fixed (same call signature)."""
    def make(ss, momentum, terms, **options):
        return FusedBilinearElement(ss, momentum, terms, packed_integrate=packed_integrate, **options)
    return make
