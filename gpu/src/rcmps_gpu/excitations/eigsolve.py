"""Matrix-free iterative excitation solver in scaled variables: the production driver.

Port of ExcitationOperator, kinetic_preconditioner, field_seed, initial_vectors,
refined_integration, fresh_level, bilinear_level, check_levels_request, solve_levels,
excitation_eigsolve and iterative_dispersion of src/excitation_eigsolve.jl. The dense
transfer context of the CPU code is replaced by a ScaledState (scaled.py): every
linear solve is a batched preconditioned GMRES and no D^2 x D^2 object exists. The
only dense objects are n x n in the sector dimension n (the coarse kinetic matrix of
the preconditioner and the Davidson projections), kept on the host in NumPy.

Variable types (scaled.py). A white coefficient vector x (host, length n) maps to the
hat tangent Wh = M Bh' with M[rows, cols] = x, W = Wh S^-1, and Bh' rh Bh = 1 on the
kept columns, so that tr(W rho W') = tr(Wh rh Wh') = |x|^2. The Hamiltonian action is
F = Fh S and its covector is (Fh Bh)[rows, cols] = (F B)[rows, cols] with B = S^-1 Bh
the CPU whitening: coefficient vectors, operator matrices and preconditioners are
those of the CPU code in every scaling, dropped metric columns included, up to the
refinement of the fixed point. Returned `vectors` are physical tangents W in the gauge
of the input state.

The density. The CPU code evaluates every formula on the stored rho. Here rho = S rh S
with rh re-solved in hat variables (scaled.py); with scaling 'density' that corrects the
relative error of the small eigenvalues of the stored rho, about eps cond(rho). Energies
then differ from the CPU ones, and from scaling 'none' (which keeps the stored rho), by
up to about eps cond(rho) |H|, and equal the CPU formulas evaluated on S rh S
(test_eigsolve.IllConditioned: 9e-8 against 1.5e-11 at cond(rho) = 1.6e9).

Defaults. kinetic_nodes=None (every entry point of this module) is default_kinetic_nodes(D):
256, the Julia default, up to D = 64 and 512 beyond, where 256 nodes are measurably not
converged; the building blocks hamiltonian.HamiltonianAction and bilinear.BilinearElement
keep Julia's 256. davidson_options=None is the Julia library schedule
(library_davidson_options: search space max(8, 2 nev + 2), retain nev, fresh restarts).
research_davidson_options(n, nev) is the Davidson part of the Julia D <= 48 research
configuration (search space min(160, n), retain min(max(12, nev), maxspace - 1), thick
restarts: Julia's retain 12 wherever Julia accepts it), which cli.py uses by default;
sector_dimension gives the n of a request without building the solver.

Differences from the Julia source: fixed kinetic grids only (Julia's kinetic_nodes=nothing,
an adaptive theta grid, is not ported; None is the D-dependent default above, which
differs from Julia's 256 for D > 64); `environment_tol` and `max_bond` are replaced by
`solver_tol`, the relative residual of every GMRES solve; the spectral test of
ExcitationContext (unique, mixing fixed point) is replaced by `check_mixing`, one pinned
Poisson solve; terms=None exists for the phi4 model only and the result has no `model`
entry (Julia: model.name), a model being a plain list of terms here; ArgumentError and
DimensionMismatch become ValueError; the random start draws from
np.random.default_rng(1234).
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from .coordinates import make_coordinates
from .davidson import DavidsonOptions, check_davidson, davidson
from .hamiltonian import HamiltonianAction
from .kinetic import KineticAction, TangentFrame
from .models import default_terms, quadrature
from .ode import integrate_host
from .scaled import ScaledState, rmul, stack
from .states import IntegratorParameters, SolverFailure, adj

norm = np.linalg.norm
STARTS = ("auto", "field", "kinetic", "random")
START_ERROR = "start must be 'auto', 'field', 'kinetic', 'random', coordinates or tangents"


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def check_kinetic_nodes(n):
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 2):
        raise ValueError("kinetic_nodes must be an integer of at least 2")
    return int(n)


def default_kinetic_nodes(d):
    """Fixed theta nodes of the kinetic resolvent quadrature at bond dimension d: 256 (the
    CPU default) for d <= 64, 512 for d > 64. Measured on the g = 2.7 phi^4 states against
    1024 nodes (README, "Kinetic nodes at large D"): with 256 nodes the resolvent part of
    the action of the converged vector is off by 9.6e-11 at D = 64, 2.1e-7 at D = 96 and
    1.2e-5 at D = 128; with 512 nodes by at most 3.1e-10 up to D = 128. Nothing was
    measured beyond D = 128 and the count is not extrapolated: it stays 512 there, and the
    refined check of the validation (twice the nodes by default) is what would expose an
    unconverged quadrature."""
    if not (isinstance(d, (int, np.integer)) and not isinstance(d, bool) and d >= 1):
        raise ValueError("D must be a positive integer")
    return 256 if d <= 64 else 512


def resolve_kinetic_nodes(kinetic_nodes, d):
    """`kinetic_nodes` unchanged, or default_kinetic_nodes(d) for None."""
    return default_kinetic_nodes(d) if kinetic_nodes is None else kinetic_nodes


RESEARCH_MAXSPACE, RESEARCH_RETAIN = 160, 12


def library_davidson_options(nev=1, *, maxspace=None, retain=None, **controls):
    """The library schedule, davidson_options=None of excitation_eigsolve (as in Julia):
    search space maxspace = max(8, 2 nev + 2), retain = nev (None), and every restart
    recomputes the actions of the retained vectors (refresh=True). `maxspace` and
    `retain` replace the respective rule; `controls` are the other DavidsonOptions fields."""
    return DavidsonOptions(maxspace=max(8, 2 * nev + 2) if maxspace is None else maxspace, retain=retain,
                           refresh=True, **controls)


def research_davidson_options(n, nev=1, *, maxspace=None, retain=None, **controls):
    """The Davidson part of the Julia D <= 48 research configuration (docs/excitations.md)
    for a sector of dimension n, and the default of cli.py: maxspace = min(160, n), thick
    restarts (refresh=False) keeping retain = min(max(12, nev), maxspace - 1) Ritz vectors.
    That is Julia's DavidsonOptions(maxspace=min(160,n), retain=12, refresh=false) wherever
    Julia's check_davidson accepts it (nev <= 12 and n >= 13); where it refuses retain 12,
    retain is raised to nev (nev > 12) or lowered to maxspace - 1 (n <= 12).
    The A100 runs of the README's mass and sector tables used it (with maxspace 160 given
    also at D = 16, where n = 128; no search there came near n) and never restarted; those
    of its comparison of scaled and unscaled variables used the library schedule and
    restarted. `maxspace` or
    `retain` replace the respective rule (retain then follows the given maxspace).
    nev >= maxspace leaves no admissible retain (check_davidson needs
    nev <= retain < maxspace) and is a ValueError, as is any combination check_davidson
    refuses. `controls` are the other DavidsonOptions fields. The stall controls of the
    Julia research configuration (stall_residual_limit=1e-6, stall_energy_atol=1e-10) are
    not implied: unless given, the DavidsonOptions defaults apply, as in the A100 runs."""
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 1):
        raise ValueError("the sector dimension n must be a positive integer")
    maxspace = min(RESEARCH_MAXSPACE, int(n)) if maxspace is None else maxspace
    if retain is None:
        if not nev < maxspace:
            raise ValueError(f"nev={nev} leaves no room in a search space of maxspace={maxspace} vectors "
                             f"(sector dimension {n}): nev must be below maxspace (give a larger maxspace)")
        retain = min(max(RESEARCH_RETAIN, nev), maxspace - 1)
    options = DavidsonOptions(maxspace=maxspace, retain=retain, refresh=False, **controls)
    check_davidson(options, nev)
    return options


class SolverState(ScaledState):
    """ScaledState that accumulates the statistics of its batched GMRES solves in
    `stats`: systems solved, their iterations, batched Arnoldi steps, largest residual.
    `attained` holds the largest residual per requested tolerance (the preconditioner
    and the actions ask for different ones)."""
    def __init__(self, *args, **kw):
        self.stats = dict(solves=0, iterations=0, steps=0, max_residual=0.0)
        self.attained = {}
        super().__init__(*args, **kw)

    def _record(self, info, tol):
        worst = float(np.max(info["residuals"]))
        self.stats["solves"] += len(info["iterations"])
        self.stats["iterations"] += int(np.sum(info["iterations"]))
        self.stats["steps"] += int(info["steps"])
        self.stats["max_residual"] = max(self.stats["max_residual"], worst)
        key = f"{tol:g}"
        self.attained[key] = max(self.attained.get(key, 0.0), worst)

    def resolvent(self, *args, **kw):
        x, info = super().resolvent(*args, **kw)
        self._record(info, kw.get("tol", 1e-12))
        return x, info

    def poisson(self, *args, **kw):
        x, info = super().poisson(*args, **kw)
        self._record(info, kw.get("tol", 1e-12))
        return x, info


def solver_statistics(ss):
    """GMRES statistics of a ScaledState (cumulative when it is shared between solves)."""
    attained = {"residual_by_tolerance": dict(ss.attained)} if hasattr(ss, "attained") else {}
    return dict(getattr(ss, "stats", {}), **attained, fixed_point_residual=float(ss.fixed_point["residual"]),
                eigenbasis_condition=dict(ss.eigenbasis_condition),
                scale_range=(float(ss.s_host.min()), float(ss.s_host.max())))


def inverse_cholesky(b):
    """Whether b has the form inv(U) of coordinates.py's Cholesky whitening, also after a
    positive row scaling: square, upper triangular, positive diagonal. An eigen-whitened
    block has it only if its eigenvectors are unit vectors of phase +1, where both agree."""
    diagonal = np.diag(b)
    return (b.shape[0] == b.shape[1] and not np.tril(b, -1).any()
            and bool((diagonal.imag == 0).all() and (diagonal.real > 0).all()))


class WhiteBasis:
    """White coordinates in hat variables for TangentCoordinates `chart` and a ScaledState
    built on chart.state (tangent_matrix, tangent_coordinates, covector_coordinates and
    original_tangent of src/excitation_eigsolve.jl).

    With S = 1 the whitening is chart.whitening (rh = rho). Otherwise (density gauge) it
    is S B made orthonormal for rh, block by parity block. Where the chart whitened by
    Cholesky (B = inv(U), rho = U'U, every column kept) this is the inverse upper Cholesky
    factor of rh: rh = (U S^-1)'(U S^-1). Where it whitened by eigenvectors (a dropped
    column, or a failed Cholesky) it is B0 G^-1/2 with B0 = S B on the kept columns and
    G = B0' rh B0, which keeps the roundoff-determined phases of those eigenvectors. In
    both cases Bh = S B, and the coefficients are the CPU ones, up to the refinement of
    the fixed point (scaled.py: rh is re-solved, S^-1 rho S^-1 only its first guess).
    `defect` is |Bh' rh Bh - 1_kept|. Coefficient vectors live on the host, tangents and
    covectors on the device; batches of vectors are arrays (m, n) and batches of matrices
    (D, m, D).
    """
    def __init__(self, chart, ss, scaling=None):
        if ss.state is not chart.state:
            raise ValueError("the scaled state must be built on the state of the coordinates")
        xp, d = ss.xp, chart.state.D
        self.chart, self.ss, self.xp, self.D = chart, ss, xp, d
        rh = to_host(ss.rh)
        kept = np.zeros(d, dtype=bool)
        kept[chart.cols] = True
        plain = bool((ss.s_host == 1).all())
        self.scaling = scaling if isinstance(scaling, str) else "none" if plain else "vector"
        if plain:
            bh = np.array(chart.whitening, dtype=np.complex128)
        else:
            if chart.gauge != "density":
                raise ValueError("scaled variables require gauge='density'")
            z = chart.parity
            blocks = [np.arange(d)] if z is None else [np.flatnonzero(z == 1), np.flatnonzero(z == -1)]
            bh = np.zeros((d, d), dtype=np.complex128)
            for blk in blocks:
                ix = np.ix_(blk, blk[kept[blk]])
                b0 = ss.s_host[blk, None] * chart.whitening[ix]    # S B on the kept columns
                if b0.shape[1] == 0:
                    continue
                try:
                    if inverse_cholesky(b0):    # the chart's b = inv(U); b0 = inv(U S^-1)
                        u = cholesky(rh[ix], lower=False)
                        bh[ix] = solve_triangular(u, np.eye(len(blk)), lower=False)
                    else:                       # eigen-whitened: nearest basis orthonormal for rh
                        g = adj(b0) @ rh[np.ix_(blk, blk)] @ b0
                        values, vectors = np.linalg.eigh((g + adj(g)) / 2)
                        if not values.min() > 0:
                            raise LinAlgError("nonpositive Gram matrix")
                        bh[ix] = b0 @ (vectors / np.sqrt(values)) @ adj(vectors)
                except LinAlgError as error:
                    raise SolverFailure("the scaled tangent metric is not positive definite") from error
        self.defect = float(norm(adj(bh) @ rh @ bh - np.diag(kept.astype(float))))
        if not self.defect <= 1e-6:
            raise SolverFailure(f"tangent whitening is unresolved: metric defect {self.defect}")
        self.kept = kept
        self.rows, self.cols = xp.asarray(chart.rows), xp.asarray(chart.cols)
        self.bh = xp.ascontiguousarray(xp.asarray(bh))
        self.bhd = xp.ascontiguousarray(xp.asarray(adj(bh)))
        self.metric = xp.ascontiguousarray(ss.rh @ self.bh)

    def __len__(self):
        return len(self.chart)

    def _sector(self, g):
        """Sector entries of every matrix of a batch (D, m, D), as a host array (m, n)."""
        return to_host(self.xp.moveaxis(g, 1, 0)[:, self.rows, self.cols])

    def tangents(self, xs):
        """Batch Wh (D, m, D) of the coefficient vectors xs (m, n)."""
        xp = self.xp
        xs = np.asarray(xs, dtype=np.complex128)
        if xs.ndim != 2 or xs.shape[1] != len(self):
            raise ValueError("coordinate vector has the wrong length")
        m = xp.zeros((len(xs), self.D, self.D), dtype=xp.complex128)
        m[:, self.rows, self.cols] = xp.asarray(xs)
        return rmul(xp.ascontiguousarray(xp.moveaxis(m, 0, 1)), self.bhd)

    def tangent(self, x):
        """Wh = M Bh' of one coefficient vector (tangent_matrix)."""
        return self.xp.ascontiguousarray(self.tangents(np.asarray(x)[None])[:, 0, :])

    def covectors(self, fh):
        """(Fh Bh)[rows, cols] for a batch of actions, as rows of a host array (m, n)."""
        return self._sector(rmul(fh, self.bh))

    def covector(self, fh):
        """Coordinates of one action: vdot(x, covector(Fh)) = tr(Wh(x)' Fh) (covector_coordinates)."""
        return self.covectors(fh[:, None, :])[0]

    def coordinates_batch(self, wh):
        return self._sector(rmul(wh, self.metric))

    def coordinates(self, wh):
        """Metric-orthogonal projection (Wh rh Bh)[rows, cols] of a hat tangent (tangent_coordinates)."""
        return self.coordinates_batch(wh[:, None, :])[0]

    def hat(self, w):
        """Wh of a physical tangent W given in the gauge of the input state (coordinate_tangent)."""
        u, w = self.chart.rotation, np.asarray(w, dtype=np.complex128)
        if w.shape != (self.D, self.D):
            raise ValueError("tangent dimensions do not match the state")
        return self.xp.asarray((adj(u) @ w @ u) * self.ss.s_host[None, :])

    def physical_tangent(self, wh):
        """W = Wh S^-1 rotated back to the gauge of the input state (original_tangent), on the host."""
        u = self.chart.rotation
        return u @ (to_host(wh) / self.ss.s_host[None, :]) @ adj(u)

    def physical(self, x):
        """Physical tangent W, in the gauge of the input state, of a coefficient vector."""
        return self.physical_tangent(self.tangent(x))


class ExcitationOperator:
    """Hermitian excitation operator in white coordinates, x -> [Fh(Wh(x)) Bh]_sector, on a
    fixed kinetic grid of `kinetic_nodes` theta nodes (None: default_kinetic_nodes(D))
    (exactly linear kinetic part; ODE terms linear to their tolerance). `calls` counts
    applications. `integrate=None` is ode.integrate_host."""
    def __init__(self, basis, momentum, terms, *, integrator, kinetic_nodes=None, tol=1e-12,
                 integrate=None, system=None, **controls):
        self.basis, self.ss, self.momentum, self.terms = basis, basis.ss, float(momentum), terms
        self.integrator = replace(integrator)
        self.kinetic_nodes = check_kinetic_nodes(resolve_kinetic_nodes(kinetic_nodes, basis.D))
        self.tol = tol
        self.settings = dict(tol=tol, integrate=integrate_host if integrate is None else integrate,
                             system=system, **controls)
        self.action = HamiltonianAction(self.ss, self.momentum, terms, integrator=self.integrator,
                                        kinetic_nodes=self.kinetic_nodes, **self.settings)
        self.calls = 0

    def __call__(self, x):
        fh = self.action(self.basis.tangent(x))
        self.calls += 1
        return self.basis.covector(fh)

    def with_action(self, *, integrator=None, kinetic_nodes=None):
        """with_context: the same operator on other integration settings, with calls = 0."""
        return ExcitationOperator(self.basis, self.momentum, self.terms,
                                  integrator=self.integrator if integrator is None else integrator,
                                  kinetic_nodes=self.kinetic_nodes if kinetic_nodes is None else kinetic_nodes,
                                  **self.settings)


def kinetic_preconditioner(basis, ss, momentum, terms, *, nodes=8, floor=None, tol=1e-10, chunk=64,
                           **controls):
    """Positive preconditioner from the coarse kinetic matrix K (kinetic specs of `terms`
    only) on a `nodes`-point theta grid: r -> sum_k v_k <v_k, r> / max(floor, kappa_k).
    The floor defaults to the largest kinetic coefficient. The n basis tangents go through
    one KineticAction in chunks of `chunk`; K and its eigenvectors stay on the host."""
    kinetics = [(coefficient, spec) for coefficient, spec in terms if spec.kinetic]
    if not kinetics:
        raise ValueError("the kinetic preconditioner requires kinetic terms")
    nodes = check_kinetic_nodes(nodes)
    lower = max(coefficient for coefficient, _ in kinetics) if floor is None else float(floor)
    if not (math.isfinite(lower) and lower > 0):
        raise ValueError("preconditioner floor must be positive")
    n = len(basis)
    action = KineticAction(ss, momentum, kinetics, nodes, tol=tol, **controls)
    k = np.empty((n, n), dtype=np.complex128)
    for first in range(0, n, chunk):
        index = np.arange(first, min(n, first + chunk))
        units = np.zeros((len(index), n))
        units[np.arange(len(index)), index] = 1
        frame = TangentFrame(ss, momentum, basis.tangents(units), tol=tol, **controls)
        k[:, index] = basis.covectors(action.action(frame)).T
    if not np.isfinite(k).all():
        raise SolverFailure("nonfinite kinetic preconditioner")
    values, vectors = np.linalg.eigh((k + adj(k)) / 2)

    def apply(r):
        return vectors @ ((adj(vectors) @ r) / np.maximum(lower, values))

    return dict(apply=apply, matrix=k, values=values, vectors=vectors,
                skew=float(norm(k - adj(k)) / max(1, norm(k))), floor=lower, nodes=nodes)


def field_seed(basis, ss, momentum, *, tol=1e-12, **controls):
    """White coordinates of the gauge-fixed field insertion phi(0)|Psi> at momentum p:
    W = 1 + [R, X] with (L* + i p) X = R + R' (trace-projected; X is a dual LL variable),
    Wh = W S. Only a start vector."""
    xp, r = ss.xp, ss.R["L"]
    x, _ = ss.poisson(stack([r + ss.Rd["L"]], xp), "L", "L", -1j * float(momentum), dual=True, tol=tol,
                      **controls)
    x = -x[:, 0, :]
    return basis.coordinates((ss.eye + r @ x - x @ r) * ss.s[None, :])


def start_list(start):
    """One coordinate vector (1-D) or one tangent ((D, D) array), else a sequence of them."""
    if isinstance(start, np.ndarray) and start.ndim <= 2:
        return [start]
    starts = list(start)
    return [np.asarray(starts)] if starts and all(np.ndim(x) == 0 for x in starts) else starts


def initial_vectors(op, start, nev, pre):
    basis = op.basis
    n = len(basis)

    def as_coordinates(x):
        x = np.asarray(x)
        if x.ndim == 2:  # a tangent in the gauge of the input state
            if x.shape != (basis.D, basis.D):
                raise ValueError("start tangent has the wrong size")
            return basis.coordinates(basis.hat(x))
        if x.shape != (n,):
            raise ValueError("start coordinates have the wrong length")
        return x.astype(np.complex128)

    def kinetic(k):
        return [] if pre is None else [pre["vectors"][:, j].copy() for j in range(min(k, n))]

    if isinstance(start, str):
        if start not in STARTS:
            raise ValueError(START_ERROR)
        if start in ("auto", "field"):
            x = field_seed(basis, op.ss, op.momentum, tol=op.tol)
            # The field insertion has unit norm up to its gauge part; ~0 means another sector.
            if norm(x) > 1e-6:
                return [x] + kinetic(nev - 1), "field"
            if start == "field":
                raise ValueError(f"the field insertion vanishes in the {basis.chart.sector} sector")
        if start in ("auto", "kinetic") and pre is not None:
            return kinetic(nev), "kinetic"
        if start == "kinetic":
            raise ValueError("start='kinetic' requires preconditioner='kinetic'")
        rng = np.random.default_rng(1234)  # Julia: MersenneTwister(1234)
        return [(rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
                for _ in range(nev)], "random"
    return [as_coordinates(x) for x in start_list(start)], "supplied"


def refined_integration(p):
    return IntegratorParameters(n_quad=2 * p.n_quad, max_point=p.max_point + 10, atol=p.atol / 10,
                                rtol=p.rtol / 10, maxiters=p.maxiters)


def fresh_level(op, x):
    """One fresh action: a continuum Rayleigh quotient with an exact (for that action) residual."""
    h = op(x)
    value = np.vdot(x, h) / np.vdot(x, x)
    return dict(energy=float(value.real), imaginary=float(value.imag),
                residual=float(norm(h - value.real * x) / norm(x)), action=h)


def bilinear_level(element, basis, x):
    """Independent bilinear energy H(W, W) / tr(W rho W') of a coefficient vector."""
    value = complex(element.energy(basis.tangent(x)))
    return dict(energy=value.real, imaginary=value.imag)


LEVEL_KEYWORDS = ("integrate", "system", "bilinear_integrate", "bilinear_element", "callback", "log")


def check_levels_request(chart, *, nev=1, davidson_options=None, start="auto", preconditioner="kinetic",
                         refine_steps=0, kinetic_nodes=None, preconditioner_nodes=8,
                         acceptance_residual=1e-5, validation_tol=1e-7, validate=True, refined=None,
                         refined_kinetic_nodes=None, preconditioner_tol=1e-10, solver_tol=1e-14, chunk=64,
                         **other):
    """Validate the cheap request before any solve; returns the Davidson options
    (library_davidson_options(nev) for None). kinetic_nodes=None is default_kinetic_nodes(D)."""
    n, d = len(chart), chart.state.D
    integer = (int, np.integer)
    unknown = sorted(set(other) - set(LEVEL_KEYWORDS))
    if unknown:
        raise ValueError(f"unknown excitation_eigsolve keywords: {', '.join(unknown)}")
    if not (isinstance(nev, integer) and 1 <= nev <= n):
        raise ValueError(f"nev={nev} must lie between 1 and the sector dimension {n}")
    options = library_davidson_options(nev) if davidson_options is None else davidson_options
    check_davidson(options, nev)
    if not (isinstance(refine_steps, integer) and refine_steps >= 0 and (refine_steps == 0 or nev == 1)):
        raise ValueError("refine_steps must be a nonnegative integer and requires nev=1")
    named = preconditioner is None or isinstance(preconditioner, str)
    if not (preconditioner in ("kinetic", "none", None) if named else callable(preconditioner)):
        raise ValueError("preconditioner must be 'kinetic', 'none', or a function")
    if isinstance(start, str):
        if start not in STARTS:
            raise ValueError(START_ERROR)
        if start == "kinetic" and preconditioner != "kinetic":
            raise ValueError("start='kinetic' requires preconditioner='kinetic'")
    else:
        starts = start_list(start)
        for x in starts:
            if np.shape(x) not in ((n,), (d, d)):
                raise ValueError(f"start entries must be length-{n} coordinates or {d}x{d} tangents")
        if not 1 <= len(starts) < options.maxspace:
            raise ValueError(f"between 1 and maxspace-1={options.maxspace - 1} start vectors are accepted")
    if not isinstance(validate, bool):
        raise ValueError("validate must be true or false")
    if refined is not None:
        if not isinstance(refined, IntegratorParameters):
            raise ValueError("refined must be None or IntegratorParameters")
        quadrature(refined)
    if refined_kinetic_nodes is not None:
        check_kinetic_nodes(refined_kinetic_nodes)
    check_kinetic_nodes(resolve_kinetic_nodes(kinetic_nodes, d))
    if named and preconditioner == "kinetic":
        check_kinetic_nodes(preconditioner_nodes)
    if not (math.isfinite(acceptance_residual) and acceptance_residual > 0
            and math.isfinite(validation_tol) and validation_tol > 0):
        raise ValueError("acceptance_residual and validation_tol must be positive")
    if not (math.isfinite(solver_tol) and solver_tol > 0 and math.isfinite(preconditioner_tol)
            and preconditioner_tol > 0):
        raise ValueError("solver_tol and preconditioner_tol must be positive")
    if not (isinstance(chunk, integer) and chunk >= 1):
        raise ValueError("chunk must be a positive integer")
    return options


def solve_levels(basis, momentum, terms, *, nev=1, kinetic_nodes=None, preconditioner="kinetic",
                 preconditioner_nodes=8, preconditioner_tol=1e-10, start="auto", davidson_options=None,
                 validate=True, refined=None, refined_kinetic_nodes=None, refine_steps=0,
                 validation_tol=1e-7, acceptance_residual=1e-5, solver_tol=1e-14, integrate=None,
                 system=None, bilinear_integrate=None, bilinear_element=None, chunk=64, callback=None,
                 log=None, context_seconds=0.0):
    """solve_levels: preconditioned Davidson search on Hamiltonian actions, optional
    refinement of the first level under the refined action, and validation by fresh
    actions and bilinear energies (`bilinear_element`, default bilinear.BilinearElement,
    on `bilinear_integrate`). The integration settings are those of basis.chart.state;
    kinetic_nodes=None is default_kinetic_nodes(D)."""
    chart, ss = basis.chart, basis.ss
    if not math.isfinite(momentum):
        raise ValueError("invalid momentum")
    momentum = float(momentum)
    kinetic_nodes = resolve_kinetic_nodes(kinetic_nodes, basis.D)
    options = check_levels_request(
        chart, nev=nev, davidson_options=davidson_options, start=start, preconditioner=preconditioner,
        refine_steps=refine_steps, kinetic_nodes=kinetic_nodes, preconditioner_nodes=preconditioner_nodes,
        acceptance_residual=acceptance_residual, validation_tol=validation_tol, validate=validate,
        refined=refined, refined_kinetic_nodes=refined_kinetic_nodes, preconditioner_tol=preconditioner_tol,
        solver_tol=solver_tol, chunk=chunk)
    if validate and bilinear_element is None:
        from .bilinear import BilinearElement as bilinear_element  # independent route: validation only
    say = log if log is not None else (lambda line: None)
    kinetic = isinstance(preconditioner, str) and preconditioner == "kinetic"
    tick = time.time()
    op = ExcitationOperator(basis, momentum, terms, integrator=chart.state.integrator,
                            kinetic_nodes=kinetic_nodes, tol=solver_tol, integrate=integrate, system=system)
    operator_seconds = time.time() - tick
    say(f"operator p={momentum:g} dimension={len(basis)} kinetic_nodes={op.kinetic_nodes} "
        f"({operator_seconds:.1f} s)")
    tick = time.time()
    pre = (kinetic_preconditioner(basis, ss, momentum, terms, nodes=preconditioner_nodes,
                                  tol=preconditioner_tol, chunk=chunk) if kinetic else None)
    apply = (pre["apply"] if pre is not None else (lambda r: r) if preconditioner in ("none", None)
             else preconditioner)
    preconditioner_seconds = time.time() - tick
    if pre is not None:
        say(f"preconditioner nodes={pre['nodes']} skew={pre['skew']:.3e} minimum={pre['values'].min():.6g} "
            f"({preconditioner_seconds:.1f} s)")
    starts, startkind = initial_vectors(op, start, nev, pre)
    say(f"start {startkind} ({len(starts)} vectors)")

    def report(row, u, au):
        say(f"davidson {row['count']:4d} cycle {row['cycle']} dimension {row['dimension']} "
            f"energy {row['energy']:.14f} residual {row['residual']:.3e}")
        if callback is not None:
            callback(row, u, au)

    tick = time.time()
    result = davidson(op, starts, nev=nev, precondition=apply, options=options,
                      callback=None if log is None and callback is None else report)
    solve_seconds = time.time() - tick
    say(f"davidson {result['stop_reason']} after {result['numops']} operations ({solve_seconds:.1f} s)")
    coefficients = [np.array(x, dtype=np.complex128) for x in result["vectors"]]
    energies, residuals, refined_converged = list(result["energies"]), list(result["residuals"]), None
    tick = time.time()
    reference = refined_params = None
    corrections = []
    if validate or refine_steps > 0:
        refined_params = refined_integration(op.integrator) if refined is None else refined
        refined_nodes = 2 * op.kinetic_nodes if refined_kinetic_nodes is None else refined_kinetic_nodes
        reference = op.with_action(integrator=refined_params, kinetic_nodes=refined_nodes)
        say(f"refined operator n_quad={refined_params.n_quad} max_point={refined_params.max_point:g} "
            f"kinetic_nodes={reference.kinetic_nodes} ({time.time() - tick:.1f} s)")
    if refine_steps > 0:
        # Preconditioned corrections under the refined action, accepted only when the fresh
        # residual decreases. The refined vector's own refined Rayleigh quotient and residual
        # replace the Davidson values.
        x = coefficients[0]
        level = fresh_level(reference, x)
        corrections.append(dict(step=0, energy=level["energy"], residual=level["residual"], accepted=True))
        for step in range(1, refine_steps + 1):
            if level["residual"] <= options.tol:
                break
            r = level["action"] - level["energy"] * x
            d = apply(r)
            d = d - x * np.vdot(x, d)
            d = d / norm(d)
            shifted = op(d) - level["energy"] * d
            alpha = np.vdot(shifted, r) / np.vdot(shifted, shifted).real
            trial = x - alpha * d
            trial = trial / norm(trial)
            candidate = fresh_level(reference, trial)
            accepted = candidate["residual"] < level["residual"]
            corrections.append(dict(step=step, energy=candidate["energy"], residual=candidate["residual"],
                                    accepted=accepted))
            say(f"refine {step} energy {candidate['energy']:.14f} residual {candidate['residual']:.3e} "
                f"accepted {accepted}")
            if not accepted:
                break
            x, level = trial, candidate
        coefficients[0] = x
        energies[0], residuals[0] = level["energy"], level["residual"]
        refined_converged = level["residual"] <= options.tol
    validation = None
    if validate:
        controls = dict(tol=solver_tol,
                        integrate=integrate_host if bilinear_integrate is None else bilinear_integrate)
        element = bilinear_element(ss, momentum, terms, integrator=op.integrator,
                                   kinetic_nodes=op.kinetic_nodes, **controls)
        refined_element = bilinear_element(ss, momentum, terms, integrator=reference.integrator,
                                           kinetic_nodes=reference.kinetic_nodes, **controls)
        validation = []
        for k, x in enumerate(coefficients):
            standard, fine = fresh_level(op, x), fresh_level(reference, x)
            bilinear = bilinear_level(element, basis, x)
            refined_bilinear = bilinear_level(refined_element, basis, x)
            values = (standard["energy"], fine["energy"], bilinear["energy"], refined_bilinear["energy"])
            # np.max and np.min propagate nan, as Julia's maximum and minimum: a nan energy is rejected.
            imaginary = float(np.max([abs(level["imaginary"])
                                      for level in (standard, fine, bilinear, refined_bilinear)]))
            spread = float(np.max(values) - np.min(values))
            residual = float(np.max((standard["residual"], fine["residual"])))
            consistent = spread <= validation_tol and imaginary <= validation_tol
            validation.append(dict(
                standard_energy=standard["energy"], standard_residual=standard["residual"],
                refined_energy=fine["energy"], refined_residual=fine["residual"],
                bilinear_energy=bilinear["energy"], refined_bilinear_energy=refined_bilinear["energy"],
                energy_spread=spread, max_imaginary=imaginary,
                residual_target_met=residual <= validation_tol, consistent=consistent,
                accepted=consistent and residual <= acceptance_residual))
            say(f"validation {k} energies {values[0]:.14f} {values[1]:.14f} {values[2]:.14f} {values[3]:.14f} "
                f"spread {spread:.3e} residual {residual:.3e} accepted {validation[-1]['accepted']}")
    validation_seconds = time.time() - tick
    # Validated levels are accepted on the research criteria (energy agreement, imaginary
    # parts, fresh residuals); without validation only a converged search (and refinement)
    # is accepted. Every requested level must be present.
    accepted = len(coefficients) == nev and (
        (result["converged"] and (True if refined_converged is None else refined_converged))
        if validation is None else all(v["accepted"] for v in validation))
    return dict(
        energies=energies, vectors=[basis.physical(x) for x in coefficients], coefficients=coefficients,
        residuals=residuals, converged=result["converged"], stop_reason=result["stop_reason"],
        refined_converged=refined_converged, accepted=bool(accepted), nev=nev,
        validation_tol=validation_tol, acceptance_residual=acceptance_residual,
        numops=result["numops"], action_calls=op.calls, history=result["history"],
        momentum=momentum, sector=chart.sector, parity=chart.parity, parity_error=chart.parity_error,
        gauge=chart.gauge, dimension=len(chart), metric_rank=chart.metric_rank,
        discarded_metric_weight=chart.discarded_metric_weight, metric_defect=chart.metric_defect,
        density_condition=chart.density_condition, start=startkind, davidson=options,
        preconditioner=None if pre is None else dict(nodes=pre["nodes"], skew=pre["skew"], floor=pre["floor"],
                                                     minimum=float(pre["values"].min())),
        corrections=corrections, validation=validation,
        validated_energies=None if validation is None else [v["refined_bilinear_energy"] for v in validation],
        integration=op.integrator, refined_integration=None if reference is None else reference.integrator,
        kinetic_nodes=op.kinetic_nodes,
        refined_kinetic_nodes=None if reference is None else reference.kinetic_nodes,
        timings=dict(context=context_seconds, operator=operator_seconds, preconditioner=preconditioner_seconds,
                     solve=solve_seconds, validation=validation_seconds),
        scaling=basis.scaling, solver=dict(solver_statistics(ss), white_defect=basis.defect),
        coordinates=chart, basis=basis)


def reported_energies(r):
    return r["energies"] if r["validated_energies"] is None else r["validated_energies"]


def accepted_energies(r):
    """Energies of an accepted result, else a SolverFailure describing why it is not accepted."""
    if r["accepted"]:
        return reported_energies(r)
    if len(r["energies"]) < r["nev"]:
        detail = f"only {len(r['energies'])} of {r['nev']} levels returned"
    elif r["validation"] is None:
        detail = f"residuals {r['residuals']}, refined_converged={r['refined_converged']}"
    else:
        v = r["validation"]
        rejected = [k for k, level in enumerate(v) if not level["accepted"]]
        detail = (f"levels {rejected} rejected: max fresh residual "
                  f"{max(max(l['standard_residual'], l['refined_residual']) for l in v)} "
                  f"(limit {r['acceptance_residual']}), energy spread {max(l['energy_spread'] for l in v)} "
                  f"and imaginary part {max(l['max_imaginary'] for l in v)} (limit {r['validation_tol']} each)")
    raise SolverFailure(f"iterative excitation levels at p={r['momentum']} not accepted (stop: "
                        f"{r['stop_reason']}, {detail}); use excitation_eigsolve or strict=False to inspect them")


MIXING_ERROR = "excitation environments require a unique, mixing transfer fixed point"


def check_mixing(ss, *, tol=1e-6):
    """Matrix-free counterpart of the spectral test of ExcitationContext
    (src/excitation_environments.jl:24-31), which no later solve replaces: a reducible
    state can pass every solve of an action (e.g. a Z2 cat state at p != 0). L is a
    Lindblad generator in finite dimension, so a simple eigenvalue 0 leaves no other
    eigenvalue on the imaginary axis (a unique stationary state is approached from every
    state), and the test amounts to -L + |rh><tr| being invertible. That operator then
    solves a generic right-hand side, here one fixed Gaussian matrix, to the loose
    relative residual `tol`; with a second zero mode the residual stalls at the overlap
    with its left partner, about 1/D. A nearly degenerate second eigenvalue lambda is
    refused where roundoff keeps the residual above 20 tol: measured on a D = 4 family,
    accepted at |lambda| = 3e-12 |L|_2 and refused at 3e-15 |L|_2. Julia refuses below
    1e-10 max(1, |L|_F): this is the looser test. The stored optimizer states (D = 16 to
    256, both scalings) pass in 16 to 37 iterations. The solve is not counted in the
    statistics."""
    xp, d = ss.xp, ss.D
    rng = np.random.default_rng(20261014)
    rhs = (rng.standard_normal((d, 1, d)) + 1j * rng.standard_normal((d, 1, d))) / math.sqrt(2)
    try:
        ScaledState.poisson(ss, xp.asarray(rhs), "H", "H", tol=tol)
    except SolverFailure as error:
        raise SolverFailure(f"{MIXING_ERROR} ({error})") from error


def sector_dimension(state, *, sector="auto", parity="auto", parity_tol=1e-12, gauge="density", metric_tol=0.0):
    """Dimension n of the white coordinates that excitation_eigsolve and iterative_dispersion
    use for the same arguments: make_basis builds them by the same make_coordinates call,
    so sector resolution ('auto'), parity and metric cutoff are the same. Costs one
    Hermitian eigensolve per parity block of rho and no transfer solve (e.g. for
    research_davidson_options before the solve)."""
    return len(make_coordinates(state, sector=sector, parity=parity, parity_tol=parity_tol, gauge=gauge,
                                metric_tol=metric_tol))


def make_basis(state, request, *, sector="auto", parity="auto", parity_tol=1e-12, gauge="density",
               scaling="density", metric_tol=0.0, xp=np, max_basis_bytes=None, log=None):
    """Coordinates, the request check, the ScaledState, the fixed-point test and the white
    basis: everything that does not depend on the momentum. Returns the basis and the
    seconds spent."""
    tick = time.time()
    plain = isinstance(scaling, str) and scaling == "none"
    if gauge == "original" and not plain:
        raise ValueError("gauge='original' requires scaling='none'")
    chart = make_coordinates(state, sector=sector, parity=parity, parity_tol=parity_tol, gauge=gauge,
                             metric_tol=metric_tol)
    check_levels_request(chart, **request)
    ss = SolverState(chart.state, scaling, xp=xp, max_basis_bytes=max_basis_bytes)
    check_mixing(ss)
    basis = WhiteBasis(chart, ss, scaling if isinstance(scaling, str) else "vector")
    seconds = time.time() - tick
    if log is not None:
        log(f"context D={chart.state.D} sector={chart.sector} gauge={chart.gauge} scaling={basis.scaling} "
            f"dimension={len(chart)} density_condition={chart.density_condition:.3e} "
            f"fixed_point_residual={ss.fixed_point['residual']:.3e} ({seconds:.1f} s)")
    return basis, seconds


def excitation_eigsolve(state, p=0.0, terms=None, *, nev=1, sector="auto", parity="auto", parity_tol=1e-12,
                        gauge="density", scaling="density", metric_tol=0.0, kinetic_nodes=None,
                        preconditioner="kinetic", preconditioner_nodes=8, preconditioner_tol=1e-10,
                        start="auto", davidson_options=None, validate=True, refined=None,
                        refined_kinetic_nodes=None, refine_steps=0, validation_tol=1e-7,
                        acceptance_residual=1e-5, solver_tol=1e-14, xp=np, integrate=None, system=None,
                        chunk=64, callback=None, log=None, bilinear_integrate=None, bilinear_element=None,
                        max_basis_bytes=None):
    """excitation_eigsolve: lowest `nev` tangent-space excitation energies at momentum p by a
    preconditioned Davidson iteration on matrix-free Hamiltonian actions.

    terms=None is models.default_terms(state): phi4_terms(state.coupling) for a phi4 state,
    a ValueError for any other stored model. Sector and coordinates as on the CPU
    (`sector`, `parity`, `parity_tol`, `gauge`, `metric_tol`); `scaling` ('density', 'none'
    or a positive vector) selects the scaled variables, and gauge='original' requires
    scaling='none'. Potential terms use state.integrator and `integrate` (None:
    ode.integrate_host) on the equations `system` (None: potential.ActionSystem); the
    kinetic part uses `kinetic_nodes` fixed theta nodes, None meaning
    default_kinetic_nodes(state.D) (256 up to D = 64, 512 beyond; resolved before any check,
    so the result reports the integer used). `davidson_options` None is the library
    schedule of the CPU code (library_davidson_options: search space max(8, 2 nev + 2),
    retain nev, fresh restarts); research_davidson_options(sector_dimension(state, ...),
    nev) gives the schedule of the A100 mass and sector runs and of cli.py. `solver_tol` is
    the relative residual of every GMRES solve of the actions, `preconditioner_tol` that of
    the coarse kinetic matrix, built `chunk` tangents at a time; `max_basis_bytes` bounds one
    Krylov basis. Energies inherit an error that grows with D: at solver_tol = 1e-12 the validated
    mass is 2e-11 from Julia's at D = 16 (-3e-12 at 1e-14) and the Davidson energy is off
    by 1e-8 at D = 64, hence the default 1e-14. Where roundoff stops GMRES above the
    request (pinned Poisson problems from D = 96 on) the attained residual is accepted
    (scaled.gmres) and reported in `solver['residual_by_tolerance']`.
    `start`: 'auto' (field insertion, else the lowest coarse kinetic modes), 'field',
    'kinetic', 'random', white coordinate vectors, or (D, D) tangents in the gauge of `state`.

    validate=True needs bilinear.py: each vector gets fresh actions at the standard and
    refined settings (`refined`, default refined_integration; `refined_kinetic_nodes`,
    default twice the nodes) and two bilinear energies, whose four-block ODEs use
    `bilinear_integrate` (None: ode.integrate_host, whatever `integrate` is);
    `bilinear_element` replaces the class bilinear.BilinearElement (same constructor), e.g.
    bilinear_fused.bound(fused_integrator) for the fused kernel on a GPU.
    `validated_energies` are the refined bilinear ones. `log(str)` receives one line per
    stage and per Davidson step, `callback(row, U, AU)` the Davidson rows. Returns the dict
    of dense_reference.eigsolve plus `scaling`, `solver` (GMRES statistics) and `basis`;
    `vectors` are metric-normalized tangents W in the gauge of `state`."""
    if not math.isfinite(p):
        raise ValueError("invalid momentum")
    terms = default_terms(state) if terms is None else terms
    kinetic_nodes = resolve_kinetic_nodes(kinetic_nodes, state.D)
    request = dict(nev=nev, kinetic_nodes=kinetic_nodes, preconditioner=preconditioner,
                   preconditioner_nodes=preconditioner_nodes, preconditioner_tol=preconditioner_tol,
                   start=start, davidson_options=davidson_options, validate=validate, refined=refined,
                   refined_kinetic_nodes=refined_kinetic_nodes, refine_steps=refine_steps,
                   validation_tol=validation_tol, acceptance_residual=acceptance_residual,
                   solver_tol=solver_tol, chunk=chunk)
    basis, seconds = make_basis(state, request, sector=sector, parity=parity, parity_tol=parity_tol,
                                gauge=gauge, scaling=scaling, metric_tol=metric_tol, xp=xp,
                                max_basis_bytes=max_basis_bytes, log=log)
    return solve_levels(basis, p, terms, integrate=integrate, system=system,
                        bilinear_integrate=bilinear_integrate, bilinear_element=bilinear_element,
                        callback=callback, log=log, context_seconds=seconds, **request)


def iterative_dispersion(state, momenta, terms=None, *, sector="auto", parity="auto", parity_tol=1e-12,
                         gauge="density", scaling="density", metric_tol=0.0, start="auto", strict=True,
                         xp=np, max_basis_bytes=None, kinetic_nodes=None, **request):
    """iterative_dispersion: one solve per momentum on shared coordinates and a shared
    ScaledState (both are momentum independent). Each momentum warm-starts from the
    previous coefficients unless `start` is supplied. With `strict`, stop with a
    SolverFailure at the first momentum whose levels are not accepted. `request` holds the
    other solver keywords of excitation_eigsolve, with its defaults (kinetic_nodes=None:
    default_kinetic_nodes(state.D)). Returns the list of results."""
    terms = default_terms(state) if terms is None else terms
    request = dict(request, kinetic_nodes=resolve_kinetic_nodes(kinetic_nodes, state.D))
    basis, seconds = make_basis(state, dict(request, start=start), sector=sector, parity=parity,
                                parity_tol=parity_tol, gauge=gauge, scaling=scaling, metric_tol=metric_tol,
                                xp=xp, max_basis_bytes=max_basis_bytes, log=request.get("log"))
    auto = isinstance(start, str) and start == "auto"
    results = []
    for k, p in enumerate(momenta):
        seed = results[-1]["coefficients"] if k > 0 and auto else start
        results.append(solve_levels(basis, p, terms, start=seed, context_seconds=seconds if k == 0 else 0.0,
                                    **request))
        if strict:
            accepted_energies(results[-1])
    return results
