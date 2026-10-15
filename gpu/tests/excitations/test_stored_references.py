"""Stored Julia numbers: the acceptance gate of the dense oracle at D = 16.

Needs RCMPS_EXCITATION_SLOW=1 (about three minutes). Two full iterative solves need a
second variable each and are meant to be run one at a time: RCMPS_EXCITATION_D16_SOLVE=1
(library defaults, about eight minutes on one thread) and RCMPS_EXCITATION_D16_HISTORY=1
(the research search against its stored history, about six minutes).

Reference: research/mass_scan/results/D16/g2.0000/{result,checks}.jld2 of the rcmps-phi4-z2
repository (read only), written on 2026-09-06 by its research/mass_scan/run_mass.jl with the
Julia implementation, for the Z2 phi^4 state at g = 2, D = 16, which is also
test/fixtures/phi4_g2_D16_z2_state.jld2 here. Momentum 0, odd sector, ORIGINAL gauge:
  b             block eigen-whitening of rho, b' rho b = 1 (run_mass.jl:40-45)
  inds          [CartesianIndex(i,j) for j in 1:d for i in 1:d if (i<=half)==(j<=half)] (:46)
  coefficients  c, the lowest odd level after refinement, |c| = 1
  vector        W = X b' with X[inds] = c (:47, 118)
  action        ((F_kinetic(W) + g F_phi4(W)) b)[inds] at the TIGHT settings (:49-51, 63-64,
                104): window 40, n_quad 768, atol 1e-15, rtol 1e-14, 512 kinetic nodes
  energy        real(dot(c, action)) = 0.3523798021196878; residual |action - energy c| = 3.80e-8
  checks        bilinear energies <W|H|W> (:124-129): 0.35237980211934294 at the STANDARD
                settings (window 30, n_quad 512, atol 1e-13, rtol 1e-12, 256 kinetic nodes) and
                the recorded mass at the tight ones (level mass_scan_D16_g2 of
                test/fixtures/excitation_references.toml; the test checks that they are equal).
Through h5py a JLD2 complex array is compound (re, im), a matrix comes back transposed and
a CartesianIndex is the compound I = ('1', '2'), one based.

Measured with this oracle on 2026-10-13 (OPENBLAS_NUM_THREADS=1, SciPy 1.16.2), W stored:
  tight action       |a - a_Julia|: max-abs 2.64e-9, 2-norm 6.02e-9;
                     Rayleigh quotient 0.35237980209059616 = stored energy - 2.91e-11
  standard action    Rayleigh quotient 0.3523798023406403 = recorded mass + 2.21e-10;
                     against the stored TIGHT action: max-abs 9.39e-8, 2-norm 2.10e-7
  bilinear energies  0.3523798021132538 (standard, Julia - 6.09e-12) and
                     0.35237980211619924 (tight, Julia - 3.09e-12)
The tolerances below are these numbers with a factor of about five. The standard action is
compared with a tight Julia action, and the 2e-7 is the oracle's, not Julia's: the
trajectories saved at the quadrature points carry the error of SciPy's DOP853 dense output
(t_eval; about 4e-11 here), which the whitening multiplies by up to 1/sqrt(min rho) = 5.8e3.
test_standard_action_without_dense_output shows it: with the ODE stepped onto the save
points the standard action is within 5.1e-9 (2-norm; max-abs 2.3e-9) of the stored one.
For this eigenvector the Rayleigh quotient moves by 2e-10 only; for the field seed (first
row of the stored solver history, 0.40017337311918866) the standard action gives -4.3e-9
(original gauge) and +8.9e-9 (density gauge), and -4.9e-11 and -5.8e-12 without dense
output. The value-only bilinear ODE saves no intermediate point and is not affected.

Iterative solves at the standard settings (one scratch run each, 2026-10-13):
  library defaults (odd sector, density gauge, fresh restarts at maxspace 8): converged
      after 23 operator calls, Davidson residual 9.06e-8, accepted; validated (refined
      bilinear) energy 0.3523798021158191 = recorded mass - 3.47e-12 (test_eigsolve)
  research search (original gauge, no restart; Julia: 16 calls, residual 3.76e-8, Ritz
      energy 0.35237980212352027), with `stepping_integrate`: 16 calls, residual 3.78e-8,
      every Ritz energy at Julia - 5e-11 (test_research_history)
  the same search with the oracle's own integrator: every Ritz energy at Julia - 2.4e-9
      to - 2.9e-9 and the residual stays between 1.07e-7 and 1.53e-7 from the 16th call on:
      not converged at tol = 1e-7 after 24 calls (no test)
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.integrate import solve_ivp

from rcmps_gpu.excitations import coordinates as tc
import dense_reference as ref
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import states
from rcmps_gpu.excitations.davidson import DavidsonOptions
from rcmps_gpu.excitations.states import IntegratorParameters, SolverFailure, adj
from _paths import FIXTURES, PHI4_Z2
import _references

FIXTURE = str(FIXTURES / "phi4_g2_D16_z2_state.jld2")
RESULTS = str(PHI4_Z2 / "research" / "mass_scan" / "results" / "D16" / "g2.0000")
ENERGY = 0.3523798021196878          # stored real(dot(c, action)), tight action
MASS_SCAN = _references.level("mass_scan_D16_g2")
MASS = None if MASS_SCAN is None else MASS_SCAN["validated_energy"]   # recorded mass: tight bilinear energy
STANDARD_BILINEAR = 0.35237980211934294
STANDARD = IntegratorParameters(n_quad=512, max_point=30.0, atol=1e-13, rtol=1e-12)
TIGHT = IntegratorParameters(n_quad=768, max_point=40.0, atol=1e-15, rtol=1e-14)
norm = np.linalg.norm

available = os.path.exists(FIXTURE) and os.path.exists(os.path.join(RESULTS, "result.jld2")) and MASS is not None
slow = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_SLOW") == "1" and available,
                           "set RCMPS_EXCITATION_SLOW=1 (D = 16; needs the rcmps-phi4-z2 results and the "
                           "reference file)")
solve = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_D16_SOLVE") == "1",
                            "set RCMPS_EXCITATION_D16_SOLVE=1 as well (about eight minutes)")
history = unittest.skipUnless(os.environ.get("RCMPS_EXCITATION_D16_HISTORY") == "1",
                              "set RCMPS_EXCITATION_D16_HISTORY=1 as well (about six minutes)")


def julia_complex(dataset):
    value = dataset[()]
    return value["re"] + 1j * value["im"]


def stepping_integrate(rhs, u0, integrator, points, label):
    """dense_reference.integrate with the ODE stepped onto every save point: the saved
    states are then solver states, not values of DOP853's dense output. Diagnostic only."""
    p, a = integrator, math.sqrt(integrator.max_point)
    points = np.asarray(points, dtype=float)
    shape, y, saved = u0.shape, u0.reshape(-1), []
    for t0, t1, inner in ((-a, 0.0, points[points < 0]), (0.0, a, points[points > 0])):
        t, h = t0, None
        for stop in np.r_[inner, t1]:
            solution = solve_ivp(lambda t, y: rhs(t, y.reshape(shape)).reshape(-1), (t, stop), y, method="DOP853",
                                 rtol=max(p.rtol, 2.3e-14), atol=p.atol, first_step=None if h is None else min(h, stop - t))
            if not solution.success:
                raise SolverFailure(f"{label} ODE failed: {solution.message}")
            steps = np.diff(solution.t)
            y, t, h = solution.y[:, -1], stop, steps.max() if len(steps) else h
            saved.append(y)
        if t1 == 0:
            saved.pop()                                        # t = 0 is a stop, not a save point
    return np.array(saved).reshape(-1, *shape)


@slow
class StoredReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import h5py

        with h5py.File(os.path.join(RESULTS, "result.jld2"), "r") as f:
            cls.R, cls.K, cls.W, cls.b = (julia_complex(f[name]).T for name in ("R", "K", "vector", "b"))
            cls.c, cls.a = julia_complex(f["coefficients"]), julia_complex(f["action"])
            index = f["inds"][()]["I"]
            cls.rows, cls.cols = index["1"] - 1, index["2"] - 1          # zero based
            cls.energy, cls.residual = float(f["energy"][()]), float(f["residual"][()])
        with h5py.File(os.path.join(RESULTS, "checks.jld2"), "r") as f:
            cls.checks = [tuple(f[reference][()]) for reference in f["checks"][()]]
        cls.s = states.load_jld2(FIXTURE)
        cls.terms = models.phi4_terms(cls.s.coupling)

    def white(self, f):
        """The stored white coordinates of an action F (run_mass.jl:67)."""
        return (f @ self.b)[self.rows, self.cols]

    def context(self, integrator):
        return ref.make_context(self.s.with_integrator(integrator), 0.0)

    def test_file(self):
        """(i) How vector, b, inds, coefficients, action and energy relate (run_mass.jl:40-47,
        104, 118-121), and the state: R and K are those of the fixture, bit for bit."""
        s, d, c, a = self.s, 16, self.c, self.a
        self.assertEqual((s.D, s.model, s.coupling), (d, "phi4", 2.0))
        self.assertEqual((norm(s.R - self.R), norm(s.K - self.K)), (0.0, 0.0))
        pairs = list(zip(self.rows.tolist(), self.cols.tolist()))
        self.assertEqual(pairs, [(i, j) for j in range(d) for i in range(d) if (i < 8) == (j < 8)])
        m = np.zeros((d, d), dtype=np.complex128)
        m[self.rows, self.cols] = c
        self.assertLessEqual(norm(m @ adj(self.b) - self.W), 1e-13)          # measured 3.7e-15
        self.assertLessEqual(abs(norm(c) - 1), 1e-14)
        self.assertLessEqual(abs(np.vdot(c, a).real - self.energy), 1e-15)
        self.assertLessEqual(abs(norm(a - self.energy * c) - self.residual), 1e-15)
        self.assertEqual((self.energy, self.residual), (ENERGY, 3.8008250285582583e-08))
        self.assertEqual([(q[0], q[1], q[2]) for q in self.checks], [(30.0, 256, STANDARD_BILINEAR), (40.0, 512, MASS)])
        # b whitens the density recomputed by states.load_jld2 (measured defect 2.2e-9, set by
        # the eigenvalue 3.0e-8 of rho), W has unit metric norm, and the stored index list is
        # the odd sector of TangentCoordinates in the original gauge, in the same order.
        self.assertLessEqual(norm(adj(self.b) @ s.rho @ self.b - np.eye(d)), 1e-8)
        self.assertLessEqual(abs(ref.tangent_metric(s, self.W, self.W) - 1), 1e-12)
        coordinates = tc.make_coordinates(s, sector="odd", gauge="original")
        self.assertEqual(list(zip(coordinates.rows.tolist(), coordinates.cols.tolist())), pairs)
        x = tc.tangent_coordinates(coordinates, self.W)
        self.assertLessEqual(abs(norm(x) - 1), 1e-12)
        self.assertLessEqual(norm(tc.tangent_matrix(coordinates, x) - self.W), 3e-8)   # 5.6e-9 = eps cond(rho)

    def test_tight_action(self):
        """(ii) hamiltonian_action at the tight settings against the stored action. Measured:
        max-abs 2.64e-9, 2-norm 6.02e-9, Rayleigh quotient - stored energy = -2.91e-11."""
        x = self.white(ref.hamiltonian_action(self.context(TIGHT), self.W, self.terms, kinetic_nodes=512))
        quotient = np.vdot(self.c, x)
        self.assertLessEqual(np.abs(x - self.a).max(), 1.5e-8)
        self.assertLessEqual(norm(x - self.a), 3e-8)
        self.assertLessEqual(abs(quotient.real - ENERGY), 1.5e-10)
        self.assertLessEqual(abs(quotient.real - MASS), 1.5e-10)
        self.assertLessEqual(abs(quotient.imag), 1e-10)                      # measured -8.8e-12 (Julia: -4.2e-12)
        self.assertLessEqual(abs(norm(x - quotient.real * self.c) - self.residual), 1e-8)   # 3.97e-8 against 3.80e-8

    def test_standard_action(self):
        """(iii) The same at the standard settings (256 kinetic nodes). Measured: Rayleigh
        quotient - recorded mass = +2.21e-10; against the stored tight action max-abs 9.39e-8
        and 2-norm 2.10e-7 (dense-output error times the whitening, see the module docstring)."""
        x = self.white(ref.hamiltonian_action(self.context(STANDARD), self.W, self.terms, kinetic_nodes=256))
        quotient = np.vdot(self.c, x)
        self.assertLessEqual(abs(quotient.real - MASS), 1e-9)
        self.assertLessEqual(abs(quotient.imag), 5e-9)                       # measured -7.6e-10
        self.assertLessEqual(np.abs(x - self.a).max(), 5e-7)
        self.assertLessEqual(norm(x - self.a), 1e-6)

    def test_standard_action_without_dense_output(self):
        """Where the 2e-7 comes from: the same standard action with the ODE stepped onto the
        save points (`stepping_integrate` in place of dense_reference.integrate) is within
        5.10e-9 (2-norm; max-abs 2.30e-9) of the stored tight action, with a Rayleigh
        quotient at the recorded mass - 2.70e-11."""
        with mock.patch.object(ref, "integrate", stepping_integrate):
            x = self.white(ref.hamiltonian_action(self.context(STANDARD), self.W, self.terms, kinetic_nodes=256))
        self.assertLessEqual(np.abs(x - self.a).max(), 1.5e-8)
        self.assertLessEqual(norm(x - self.a), 3e-8)
        self.assertLessEqual(abs(np.vdot(self.c, x).real - MASS), 1.5e-10)

    def test_bilinear_energies(self):
        """The two stored bilinear checks (run_mass.jl:124-129): excitation_kinetic_sum plus
        g times excitation_observable(:phi^4:) on the pair (W, W), at the standard and at the
        tight settings. Measured: oracle - Julia = -6.09e-12 and -3.09e-12 (imaginary parts
        2.3e-13 and 2.6e-13; Julia 1.1e-12)."""
        for integrator, nodes, stored in ((STANDARD, 256, STANDARD_BILINEAR), (TIGHT, 512, MASS)):
            value = ref.matrix_element(ref.make_pair(self.context(integrator), self.W, self.W), self.terms,
                                       kinetic_nodes=nodes)
            self.assertLessEqual(abs(value.real - stored), 3e-11, nodes)
            self.assertLessEqual(abs(value.imag), 1e-11, nodes)

    @history
    def test_research_history(self):
        """run_mass.jl:52-56, 81-99 in full against the stored solver_history, row by row: field
        start, coarse kinetic preconditioner with floor 1, original gauge, no restart
        (maxspace 128), tol 1e-7, standard settings. The oracle whitens by Cholesky and the
        script by eigenvectors: a unitary change of white coordinates, which leaves Ritz
        energies and residual norms unchanged. Run with `stepping_integrate`. Measured: 16
        operator calls as in Julia, every Ritz energy at Julia - 4.9e-11 to - 5.7e-11,
        residuals within 4e-7 relative for the twelve rows above 1e-5 and within 5e-3 for the
        last four (final 3.781e-8 against 3.764e-8). With the oracle's own integrator
        (dense output) the same search is recorded in the module docstring."""
        import h5py

        with h5py.File(os.path.join(RESULTS, "result.jld2"), "r") as f:
            stored = [tuple(f[reference][()]) for reference in f["solver_history"][()]]
        self.assertEqual([(row[0], row[1]) for row in stored], [(k, k) for k in range(1, 17)])
        options = DavidsonOptions(tol=1e-7, maxops=24, maxspace=128, retain=12, stall_window=20, stall_minops=40,
                                  stall_rtol=0.1, stall_residual_limit=1e-6, stall_energy_atol=1e-10)
        with mock.patch.object(ref, "integrate", stepping_integrate):
            r = ref.eigsolve(self.s.with_integrator(STANDARD), 0.0, gauge="original", validate=False,
                             davidson_options=options)
        self.assertEqual((r["start"], r["stop_reason"], r["numops"], r["dimension"]), ("field", "converged", 16, 128))
        for row, (count, dimension, energy, residual, hermiticity) in zip(r["history"], stored):
            self.assertEqual((row["count"], row["dimension"]), (count, dimension))
            self.assertLessEqual(abs(row["energy"] - energy), 3e-10, count)
            self.assertLessEqual(abs(row["residual"] / residual - 1), 1e-5 if residual > 1e-5 else 2e-2, count)

    @solve
    def test_eigsolve(self):
        """The library call excitation_eigsolve(s, 0.) at the standard settings (odd sector,
        density gauge, field start, 256 kinetic nodes, fresh restarts at maxspace 8), as in
        benchmark/excitation_mass_d16.jl. Measured: converged after 23 operator calls
        (run_mass.jl, original gauge and no restart: 16), Davidson residual 9.06e-8, accepted;
        validated (refined bilinear) energy 0.3523798021158191 = recorded mass - 3.47e-12,
        spread of the four energies 9.70e-11, fresh residuals 1.12e-7 (standard) and 1.33e-7
        (refined), i.e. residual_target_met is False at validation_tol = 1e-7."""
        r = ref.eigsolve(self.s.with_integrator(STANDARD), 0.0)
        v = r["validation"][0]
        self.assertEqual((r["sector"], r["gauge"], r["dimension"], r["metric_rank"], r["start"]),
                         ("odd", "density", 128, 16, "field"))
        self.assertEqual((r["converged"], r["stop_reason"], r["accepted"]), (True, "converged", True))
        self.assertLessEqual(r["numops"], 30)
        self.assertLessEqual(abs(r["validated_energies"][0] - MASS), 5e-11)
        self.assertLessEqual(abs(r["energies"][0] - MASS), 2e-9)             # measured +2.68e-10
        self.assertLessEqual(v["energy_spread"], 5e-10)
        self.assertLessEqual(max(v["standard_residual"], v["refined_residual"]), 5e-7)
        print(f"\nD=16 eigsolve: numops {r['numops']}, residual {r['residuals'][0]:.3e}, validated energy "
              f"{r['validated_energies'][0]!r} (recorded mass {MASS!r}), validation {v}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
