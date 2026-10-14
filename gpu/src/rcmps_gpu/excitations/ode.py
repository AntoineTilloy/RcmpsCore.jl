"""Host-controlled adaptive DOP853 for the excitation ODE systems (NumPy or CuPy).

The integrator of the simple GPU path and of CPU tests: the state lives on the
device (`system.xp`), each right-hand side is a handful of array operations, and
the step controller runs on the host. Method, error norm, step controller, cusp
handling at t = 0 and seventh-order dense output are those of the GPU optimizer's
device controller (research/gpu_d64/generic_gpu/runtime), which follow SciPy's
DOP853; the source 2|t| J(t^2) is evaluated exactly on the host.

A system provides `xp`, `initial` (any shape, complex128) and
`rhs(y, jac, source)` for scalars jac = 2|t| and source = 2|t| J(t^2).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as dop

from models import transformed_source
from states import SolverFailure


def integrate_host(system, points, endpoint, integrator, *, initial_step=0.01, statistics=None):
    """State at the strictly increasing `points` in (-endpoint, endpoint) and at +endpoint:
    array (len(points) + 1,) + system.initial.shape. No step crosses t = 0."""
    xp = system.xp
    points = np.asarray(points, dtype=float)
    a, atol, rtol = float(endpoint), integrator.atol, max(integrator.rtol, 100 * np.finfo(float).eps)
    y = system.initial.copy()
    size = y.size
    k = xp.empty((16,) + y.shape, dtype=xp.complex128)
    samples = xp.empty((len(points) + 1,) + y.shape, dtype=xp.complex128)

    def rhs(t, z):
        return system.rhs(z, 2 * abs(t), transformed_source(t))

    def combination(weights, count, h):
        z = y.copy()
        for j in range(count):
            if weights[j] != 0:
                z += (h * weights[j]) * k[j]
        return z

    t, h = -a, min(initial_step, 2 * a)
    f = rhs(t, y)
    attempts = accepted = rejected = dense_steps = next_point = 0
    was_rejected = False
    while t < a:
        if attempts >= integrator.maxiters:
            raise SolverFailure("excitation ODE failed: MaxIters")
        boundary = 0.0 if t < 0 else a
        h = min(h, boundary - t)
        if not math.isfinite(h) or h <= 10 * abs(np.nextafter(t, math.inf) - t):
            raise SolverFailure("excitation ODE failed: step underflow")
        attempts += 1
        k[0] = f
        for s in range(1, 12):
            k[s] = rhs(t + dop.C[s] * h, combination(dop.A[s], s, h))
        proposed = combination(dop.B, 12, h)
        k[12] = rhs(t + h, proposed)
        err5 = sum(dop.E5[s] * k[s] for s in range(13) if dop.E5[s] != 0)
        err3 = sum(dop.E3[s] * k[s] for s in range(13) if dop.E3[s] != 0)
        scale = atol + rtol * xp.maximum(xp.abs(y), xp.abs(proposed))
        e5 = float(xp.sum(xp.abs(err5 / scale) ** 2))
        e3 = float(xp.sum(xp.abs(err3 / scale) ** 2))
        denominator = e5 + 0.01 * e3
        error = h * e5 / math.sqrt(denominator * size) if denominator > 0 else 0.0
        if not math.isfinite(error):
            raise SolverFailure("nonfinite excitation ODE error estimate")
        factor = min(10.0, 0.9 * error ** (-1 / 8)) if error > 0 else 10.0
        if error > 1:
            rejected += 1
            was_rejected = True
            h *= max(0.2, factor)
            continue
        accepted += 1
        t_new = boundary if h == boundary - t else t + h
        if was_rejected:
            factor = min(1.0, factor)
        was_rejected = False
        first = next_point
        while next_point < len(points) and points[next_point] <= t_new:
            next_point += 1
        if next_point > first:
            dense_steps += 1
            for s in range(13, 16):
                k[s] = rhs(t + dop.C[s] * h, combination(dop.A[s], s, h))
            delta = proposed - y
            coefficients = [delta, h * f - delta, 2 * delta - h * (k[12] + f)]
            for row in range(4):
                coefficients.append(sum((h * dop.D[row, s]) * k[s] for s in range(16) if dop.D[row, s] != 0))
            for node in range(first, next_point):
                if points[node] == t_new:
                    samples[node] = proposed
                    continue
                x = (points[node] - t) / h
                value = xp.zeros_like(y)
                for s in range(7):
                    value = (value + coefficients[6 - s]) * (x if s % 2 == 0 else 1 - x)
                samples[node] = value + y
        y, f, t = proposed, k[12].copy(), t_new
        h *= factor
    if next_point != len(points):
        raise SolverFailure("excitation ODE failed: missing samples")
    samples[len(points)] = y
    if not bool(xp.isfinite(y).all()):
        raise SolverFailure("nonfinite excitation trajectory")
    if statistics is not None:
        statistics.update(attempts=attempts, accepted=accepted, rejected=rejected,
                          dense_steps=dense_steps, rhs_calls=1 + 12 * attempts + 3 * dense_steps)
    return samples
