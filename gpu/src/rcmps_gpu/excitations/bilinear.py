"""Matrix-free bilinear matrix element <Phi_p(U)|(H - E0)|Phi_p(W)> in scaled variables.

Port of ExcitationPair, excitation_transfer!, excitation_contract, excitation_operators,
excitation_source_add! and excitation_resolvent (src/excitation_environments.jl), of
excitation_rhs!, excitation_observable, excitation_kinetic_sum and excitation_element
(src/excitations.jl; kinetic_backend=:resolvent, fixed theta grid) and of bilinear_level
(src/excitation_eigsolve.jl). It is the independent check of the action (hamiltonian.py):
forward, value-only contractions with no covector and no pullback. Only kinetic specs
use the resolvent quadrature; every other term, quadratic monomials included, is a
four-block source ODE on the window.

Structural changes that leave the result unchanged: every Poisson and resolvent solve
is a batched preconditioned GMRES (scaled.py); the resolvent chains of all theta nodes,
source operators and source orders form one batch per block; the tangent-independent
ground family x1(z) = (1 - z L)^-1 (c rho or rho c') is solved once per BilinearElement.

Variable types (scaled.py). Tangents W = Wh S^-1, U = Uh S^-1, with gauge parts
Vh = -R' Wh and Vbrah = -R' Uh. The four tangent blocks (00, ket, bra, ket+bra) of the
CPU code are x1 = S x1h S (HH), x2 = y2h S (LH), x3 = S b3h (HL) and x4 unscaled (LL);
the left tail is kept as hh = S h S (dual HH), so that tr(x4) + tr(h x1) =
tr(x4) + tr(hh x1h). With scaling "none" every hat quantity is the CPU one. Four blocks
are one array (4, D, n, D) of batches, n running over ODE channels or resolvent systems.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .models import kinetic_grid, transformed_source
from .scaled import lmul, rmul
from .states import SolverFailure, adj

TYPES = (("H", "H"), ("L", "H"), ("H", "L"), ("L", "L"))   # of the blocks x1, x2, x3, x4


def integrate_scipy(system, points, endpoint, integrator):
    """CPU reference with the calling convention of ode.integrate_host: the state at the
    nonzero `points` and at +endpoint by SciPy's DOP853, restarted at t = 0 where the
    source has a cusp (tstops=[0.] on the CPU). `maxiters` bounds the step attempts
    (12 right-hand sides each), as on the CPU."""
    if system.xp is not np:
        raise ValueError("the SciPy reference integrator is CPU only")
    points = np.asarray(points, dtype=float)
    shape = system.initial.shape
    rtol = max(integrator.rtol, 2.3e-14)   # SciPy clips at 100 eps; do it explicitly

    def fun(t, y):
        return system.rhs(y.reshape(shape), 2 * abs(t), transformed_source(t)).reshape(-1)

    y, saved, evaluations = system.initial.reshape(-1), [], 0
    for lower, upper in ((-endpoint, 0.0), (0.0, endpoint)):
        inside = points[(points > lower) & (points < upper)]
        solution = solve_ivp(fun, (lower, upper), y, method="DOP853", t_eval=np.r_[inside, upper],
                             rtol=rtol, atol=integrator.atol)
        if not solution.success:
            raise SolverFailure(f"excitation ODE failed: {solution.message}")
        evaluations += solution.nfev
        saved.append(solution.y.T[:-1])
        y = solution.y[:, -1].copy()
    if evaluations > 12 * integrator.maxiters:
        raise SolverFailure("excitation ODE failed: MaxIters")
    samples = np.concatenate(saved + [y[None, :]])
    if len(samples) != len(points) + 1:
        raise ValueError("sample points must be nonzero and inside the window")
    return samples.reshape((-1,) + shape)


class Pair:
    """ExcitationPair(ctx, u, w) for hat tangents: gauge parts, overlap n = tr(Wh rh Uh'),
    the right environments `right` (4, D, 1, D) of types (HH, LH, HL, LL) and the left
    tail `left` = S left00 S (dual HH). `infos` holds the GMRES reports of the four solves."""
    def __init__(self, ss, momentum, uh, wh, *, tol=1e-12, **controls):
        xp = ss.xp
        self.ss, self.p = ss, float(momentum)
        self.uh, self.wh = self._tangent(uh), self._tangent(wh)
        self.vh, self.vbrah = -ss.Rd["L"] @ self.wh, -ss.Rd["L"] @ self.uh
        self.uhd = xp.ascontiguousarray(adj(self.uh))
        self.vbrahd = xp.ascontiguousarray(adj(self.vbrah))
        self.overlap = xp.sum((self.wh @ ss.rh) * xp.conj(self.uh))
        self.infos = []

        def poisson(rhs, k, shift):
            x, info = ss.poisson(xp.ascontiguousarray(rhs), *TYPES[k], shift, tol=tol, **controls)
            self.infos.append(info)
            return x

        right = xp.zeros((4, ss.D, 1, ss.D), dtype=xp.complex128)
        right[0, :, 0, :] = ss.rh
        right[1] = poisson(self.ket(right[0], "H"), 1, 1j * self.p)
        right[2] = poisson(self.bra(right[0], "H"), 2, -1j * self.p)
        right[3] = poisson(self.ket(right[2], "L") + self.bra(right[1], "L")
                           + rmul(lmul(self.wh, right[0]), self.uhd), 3, 0.0)
        self.right = right
        b = self.uhd @ self.wh - self.overlap * ss.zero_mode("H", "H", dual=True)
        h, info = ss.poisson(xp.ascontiguousarray(b[:, None, :]), "H", "H", dual=True, tol=tol,
                             **controls)
        self.infos.append(info)
        self.left = h[:, 0, :]

    def _tangent(self, x):
        xp, d = self.ss.xp, self.ss.D
        x = xp.ascontiguousarray(xp.asarray(x, dtype=xp.complex128))
        if x.shape != (d, d):
            raise ValueError("tangent dimensions do not match the state")
        if not bool(xp.isfinite(x).all()):
            raise ValueError("nonfinite tangent")
        return x

    def ket(self, x, column):
        """ket_insertion(v, w, x) = v x + w x R' on a batch whose column index has gauge
        `column`: HH -> LH, HL -> LL."""
        return lmul(self.vh, x) + rmul(lmul(self.wh, x), self.ss.Rd[column])

    def bra(self, x, row):
        """bra_insertion(vbra, u, x) = x vbra' + R x u' on a batch whose row index has gauge
        `row`: HH -> HL, LH -> LL."""
        return rmul(x, self.vbrahd) + rmul(lmul(self.ss.R[row], x), self.uhd)

    def double(self, x):
        """w x u' - overlap x on an HH batch -> LL; the second term is -n S xh S."""
        s = self.ss.s
        return (rmul(lmul(self.wh, x), self.uhd)
                - self.overlap * (s[:, None, None] * x * s[None, None, :]))

    def transfer(self, x):
        """excitation_transfer!: the connected generator on four blocks (4, D, n, D)."""
        ss, p = self.ss, self.p
        out = ss.xp.empty_like(x)
        out[0] = ss.lmap(x[0], "H", "H")
        out[1] = ss.lmap(x[1], "L", "H") - 1j * p * x[1] + self.ket(x[0], "H")
        out[2] = ss.lmap(x[2], "H", "L") + 1j * p * x[2] + self.bra(x[0], "H")
        out[3] = (ss.lmap(x[3], "L", "L") + self.ket(x[2], "L") + self.bra(x[1], "L")
                  + self.double(x[0]))
        return out

    def contract(self, x):
        """excitation_contract per system: tr(x4) + tr(left00 x1) = tr(x4) + tr(hh x1h)."""
        xp = self.ss.xp
        return (self.ss.pairing(x[3], "L", "L")
                + xp.sum(self.left.T[:, None, :] * x[0], axis=(0, 2)))

    def operators(self, commutator):
        """excitation_operators as two triples keyed by `left`: (c, ch, cw S) for the
        annihilation source and (c', ch', (cu S)') for the creation source, where c is R or
        [Q, R] in the L gauge, ch the same in the hat gauge, cw and cu its tangent lifts."""
        ss, p, xp = self.ss, self.p, self.ss.xp
        if commutator:
            ql, rl, qh, rhh = ss.Q["L"], ss.R["L"], ss.Q["H"], ss.R["H"]

            def lift(t, v):
                return ql @ t - t @ qh + v @ rhh - rl @ v - 1j * p * t

            c, ch = ql @ rl - rl @ ql, qh @ rhh - rhh @ qh
            cwh, cuh = lift(self.wh, self.vh), lift(self.uh, self.vbrah)
        else:
            c, ch, cwh, cuh = ss.R["L"], ss.R["H"], self.wh, self.uh
        return {True: (c, ch, cwh),
                False: tuple(xp.ascontiguousarray(adj(m)) for m in (c, ch, cuh))}

    def source(self, x, operators, left):
        """excitation_source_add! with unit weight on four blocks: the normal-ordered
        annihilation source c x (left) or creation source x c', with its tangent lift."""
        c, ch, lifted = operators[left]
        out = self.ss.xp.empty_like(x)
        if left:
            out[0] = lmul(ch, x[0])
            out[1] = lmul(c, x[1]) + lmul(lifted, x[0])
            out[2] = lmul(ch, x[2])
            out[3] = lmul(c, x[3]) + lmul(lifted, x[2])
        else:
            out[0] = rmul(x[0], ch)
            out[1] = rmul(x[1], ch)
            out[2] = rmul(x[2], c) + rmul(x[0], lifted)
            out[3] = rmul(x[3], c) + rmul(x[1], lifted)
        return out


class PairSystem:
    """excitation_rhs!: the value-only four-block ODE of one term, for ode.integrate_host or
    integrate_scipy. State (4, D, channels, D): each block is a batch over the channels."""
    def __init__(self, pair, spec):
        xp = pair.ss.xp
        self.pair, self.spec, self.xp = pair, spec, xp
        self.operators = pair.operators(spec.commutator)
        self.shape = (4, pair.ss.D, spec.channels, pair.ss.D)
        initial = xp.zeros(self.shape, dtype=xp.complex128)
        initial[:, :, :1, :] = pair.right
        self.initial = initial

    def rhs(self, y, jac, source):
        """d/dt of the state for scalars jac = 2|t| and source = 2|t| J(t^2)."""
        out = jac * self.pair.transfer(y)
        for edge in self.spec.edges:
            out[:, :, edge.dst:edge.dst + 1, :] += (edge.coefficient * source) * self.pair.source(
                y[:, :, edge.src:edge.src + 1, :], self.operators, edge.left)
        return out


class BilinearElement:
    """`terms`: [(coefficient, Observable)]; `integrator`: IntegratorParameters of the ODE
    terms; `kinetic_nodes`: fixed theta nodes of the kinetic specs; `tol`: relative residual
    of every linear solve; `integrate(system, points, endpoint, integrator)`: ODE backend
    (default integrate_scipy; ode.integrate_host for NumPy or CuPy). `stats` accumulates the
    linear solves, `trajectories` the ODE integrations and `elements` the evaluations."""
    def __init__(self, ss, momentum, terms, *, integrator, kinetic_nodes=256, tol=1e-12,
                 integrate=None, **controls):
        if not math.isfinite(momentum):
            raise ValueError("invalid momentum")
        xp = ss.xp
        self.ss, self.p, self.tol, self.controls = ss, float(momentum), tol, controls
        self.integrator = integrator
        self.integrate = integrate_scipy if integrate is None else integrate
        self.kinetics = [(float(c), spec) for c, spec in terms if spec.kinetic]
        self.potentials = [(float(c), spec) for c, spec in terms if not spec.kinetic]
        self.kinetic_nodes = self.ground = None
        if self.kinetics:
            theta, weights = kinetic_grid(kinetic_nodes)
            self.kinetic_nodes = int(kinetic_nodes)
            self.z_host, self.weights = np.cos(theta), xp.asarray(weights)
            self.sources = sorted({spec.commutator for _, spec in self.kinetics})
            self.coefficients = {c: sum(a for a, spec in self.kinetics if spec.commutator == c)
                                 for c in self.sources}
        self.stats = dict(solves=0, iterations=0, max_residual=0.0)
        self.trajectories = self.elements = 0

    def _record(self, info):
        self.stats["solves"] += len(info["iterations"])
        self.stats["iterations"] += int(info["iterations"].sum())
        self.stats["max_residual"] = max(self.stats["max_residual"], float(info["residuals"].max()))

    def pair(self, uh, wh):
        """The environments of <Phi_p(U)| . |Phi_p(W)> (ExcitationPair)."""
        pair = Pair(self.ss, self.p, uh, wh, tol=self.tol, **self.controls)
        for info in pair.infos:
            self._record(info)
        return pair

    def observable(self, pair, spec):
        """excitation_observable: source ODE on (-a, a) from the right environments,
        contracted with the left ones at t = a in the last channel."""
        p = self.integrator
        if not (p.max_point > 0 and p.atol > 0 and p.rtol > 0 and p.maxiters > 0):
            raise ValueError("invalid integration controls")
        final = self.integrate(PairSystem(pair, spec), (), math.sqrt(p.max_point), p)[-1]
        self.trajectories += 1
        if not bool(self.ss.xp.isfinite(final).all()):
            raise SolverFailure("nonfinite excitation trajectory")
        return complex(pair.contract(final[:, :, -1:, :])[0])

    def kinetic_sum(self, pair):
        """excitation_kinetic_sum: the two mixed source orders of every kinetic spec. Per
        theta node and first source, excitation_resolvent is the chain x1 -> (x2, x3) -> x4
        with shifts 1, 1 + i z p, 1 - i z p, 1; here one batched solve per block over
        (source operator, source order, node)."""
        ss, xp, n, p = self.ss, self.ss.xp, self.kinetic_nodes, self.p
        keys = [(c, left) for c in self.sources for left in (False, True)]
        operators = {c: pair.operators(c) for c in self.sources}
        first = [pair.source(pair.right, operators[c], left) for c, left in keys]
        z_host = np.concatenate([self.z_host] * len(keys))
        z = xp.asarray(z_host)[None, :, None]

        def spread(k):   # block k of the first sources, constant over the nodes
            return xp.concatenate([xp.broadcast_to(b[k], (ss.D, n, ss.D)) for b in first], axis=1)

        def solve(rhs, k, sigma):
            x, info = ss.resolvent(xp.ascontiguousarray(rhs), *TYPES[k], z_host, sigma, tol=self.tol,
                                   **self.controls)
            self._record(info)
            return x

        if self.ground is None:   # block 1 of the sources is c rh or rh c': no tangent enters
            self.ground = solve(spread(0), 0, 1.0)
        x = xp.empty((4,) + self.ground.shape, dtype=xp.complex128)
        x[0] = self.ground
        x[1] = solve(spread(1) + z * pair.ket(x[0], "H"), 1, 1 + 1j * z_host * p)
        x[2] = solve(spread(2) + z * pair.bra(x[0], "H"), 2, 1 - 1j * z_host * p)
        x[3] = solve(spread(3) + z * (pair.ket(x[2], "L") + pair.bra(x[1], "L") + pair.double(x[0])),
                     3, 1.0)
        result = 0j
        for i, (c, left) in enumerate(keys):
            y = pair.source(x[:, :, i * n:(i + 1) * n, :], operators[c], not left)
            result += self.coefficients[c] / 2 * complex(xp.sum(self.weights * pair.contract(y)))
        return result

    def _element(self, pair):
        result = 0j
        if self.kinetics:
            result += self.kinetic_sum(pair)
        for coefficient, spec in self.potentials:
            result += coefficient * self.observable(pair, spec)
        if not (math.isfinite(result.real) and math.isfinite(result.imag)):
            raise SolverFailure("nonfinite excitation matrix element")
        self.elements += 1
        return result

    def element(self, uh, wh):
        """excitation_element(ExcitationPair(ctx, u, w), model; kinetic_backend=:resolvent,
        kinetic_nodes) for hat tangents: the sesquilinear form tr(Uh' Fh(Wh))."""
        return self._element(self.pair(uh, wh))

    def energy(self, wh):
        """bilinear_level: element(W, W) / tr(W rho W'), complex (energy + i imaginary).
        A vanishing norm gives nan + nan i, as Julia's 0/0 does (the level is then rejected),
        instead of Python's ZeroDivisionError."""
        pair = self.pair(wh, wh)
        value, overlap = self._element(pair), complex(pair.overlap)
        return complex(math.nan, math.nan) if overlap == 0 else value / overlap
