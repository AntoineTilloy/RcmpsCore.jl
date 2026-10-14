"""Hamiltonian terms, source kernel and quadrature grids (src/observables.jl, src/kinetic.jl).

NumPy/SciPy only. The term lists are the ones shared by the CPU ground-state
objectives and excitation routines; only the phi^4 family is in scope for the
GPU path, but edges carry complex coefficients as on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gamma, kv, roots_legendre


@dataclass(frozen=True)
class Edge:
    """Source insertion from channel `src` to `dst` (zero based); `left` inserts C x,
    otherwise x C'. The transformed source multiplies `coefficient`."""
    src: int
    dst: int
    left: bool
    coefficient: complex


@dataclass(frozen=True)
class Observable:
    channels: int
    edges: tuple
    commutator: bool = False  # C = [Q, R] instead of R
    kinetic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "edges", tuple(self.edges))  # so that specs compare by value


def monomial_spec(n):
    """:phi^n: as a chain of n+1 channels; the edge from channel k to k+1 (zero based)
    carries the factor k+1, once as a left and once as a right insertion."""
    if n < 0:
        raise ValueError("monomial degree must be nonnegative")
    edges = tuple(Edge(k, k + 1, left, k + 1) for k in range(n) for left in (True, False))
    return Observable(n + 1, edges)


def kinetic_spec(commutator):
    s = math.sqrt(2)
    return Observable(4, (Edge(0, 1, True, s), Edge(0, 2, False, s),
                          Edge(2, 3, True, s), Edge(1, 3, False, s)), commutator, True)


def phi4_terms(g):
    """[(coefficient, Observable)] of H = H_free(1) + g :phi^4:, zero terms dropped."""
    g = float(g)
    if not math.isfinite(g):
        raise ValueError("nonfinite model parameter")
    terms = [(1.0, kinetic_spec(False)), (1.0, kinetic_spec(True)), (g, monomial_spec(4))]
    return [term for term in terms if term[0] != 0]


def default_terms(state):
    """ExcitationModel(s) (src/excitations.jl) for the one model ported here: the phi^4
    terms at the stored coupling. Julia dispatches on the stored model name; any other
    name is refused instead of silently receiving phi^4 terms."""
    if state.model != "phi4":
        raise ValueError(f"only the phi4 model has default terms, not {state.model!r}; pass terms")
    return phi4_terms(state.coupling)


def is_quadratic(spec):
    return (not spec.kinetic and spec.channels == 3 and spec.edges == monomial_spec(2).edges)


def jsource(x):
    """Mass-one smearing kernel J(x); integrable inverse-square-root singularity at 0."""
    x = abs(float(x))
    if x == 0:
        return math.inf
    return kv(0.25, x) / (2 ** 2.25 * math.sqrt(math.pi) * gamma(1.25) * x ** 0.25)


def transformed_source(t):
    """2|t| J(t^2): the substitution x = t|t| removes the singularity; 1/sqrt(pi) at 0."""
    t = abs(float(t))
    return 1 / math.sqrt(math.pi) if t == 0 else 2 * t * jsource(t * t)


def quadrature(integrator):
    """2 n_quad signed Gauss-Legendre nodes in t on (-a, a), a = sqrt(max_point), with
    weights; each half is an n_quad-point rule on (0, a), mirrored (src/observables.jl)."""
    p = integrator
    if not (p.n_quad >= 8 and p.max_point > 0 and p.atol > 0 and p.rtol > 0 and p.maxiters > 0):
        raise ValueError("invalid integration controls")
    x, w = roots_legendre(p.n_quad)
    a = math.sqrt(p.max_point)
    points, weights = a / 2 * (x + 1), a / 2 * w
    return np.r_[-points[::-1], points], np.r_[weights[::-1], weights]


def kinetic_grid(n):
    """n-point Gauss-Legendre rule in theta on (0, pi/2); the weights sum to one, i.e.
    they include the factor 2/pi of the kinetic resolvent integral (src/kinetic.jl)."""
    if not (isinstance(n, (int, np.integer)) and n >= 2):
        raise ValueError("kinetic_nodes must be an integer of at least 2")
    x, w = roots_legendre(int(n))
    return math.pi / 4 * (x + 1), w / 2
