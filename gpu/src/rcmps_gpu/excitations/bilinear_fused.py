"""Bilinear energies with the potential terms on the fused kernel.

bilinear.BilinearElement integrates the four-block source ODE with a host-controlled
integrator, which is the reference but too slow on a GPU beyond small bond dimension.
For the diagonal elements <Phi(W)|H|Phi(W)> that the validation step needs, the same
ODE is the Z2-packed system of packed_pair.py, which the fused trajectory kernel
integrates in one launch (or, for a state without symmetry or the sector 'all', its
unpacked form packed_pair.UnpackedDiagonalPair). Everything else (environments, kinetic
resolvent sum, and any element or term the packed system does not cover) is inherited
unchanged.
"""
from __future__ import annotations

from .bilinear import BilinearElement
from .packed_pair import PackedDiagonalPair, diagonal_observable


class FusedBilinearElement(BilinearElement):
    """BilinearElement whose monomial terms of diagonal elements use
    `packed_integrate(system, points, endpoint, integrator)`, e.g. fused.fused_integrator(cp),
    on the system `pair_system` (packed_pair.PackedDiagonalPair or UnpackedDiagonalPair).
    `fused_trajectories` counts them; other cases fall back to the inherited route."""
    def __init__(self, ss, momentum, terms, *, packed_integrate, pair_system=PackedDiagonalPair, **options):
        super().__init__(ss, momentum, terms, **options)
        self.packed_integrate, self.pair_system = packed_integrate, pair_system
        self.fused_trajectories = 0

    def observable(self, pair, spec):
        try:
            value = diagonal_observable(pair, spec, self.integrator, self.packed_integrate, self.pair_system)
        except ValueError:      # not a diagonal Z2 monomial element: reference route
            return super().observable(pair, spec)
        self.trajectories += 1
        self.fused_trajectories += 1
        return value


def bound(packed_integrate, pair_system=PackedDiagonalPair):
    """A BilinearElement factory with `packed_integrate` and `pair_system` fixed (same call
    signature)."""
    def make(ss, momentum, terms, **options):
        return FusedBilinearElement(ss, momentum, terms, packed_integrate=packed_integrate,
                                    pair_system=pair_system, **options)
    return make
