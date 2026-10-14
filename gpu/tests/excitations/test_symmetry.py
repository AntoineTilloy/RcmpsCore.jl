"""Sign of the momentum: time reversal.

The Hamiltonian is real, so complex conjugation maps the state (R, K) to (conj R, -conj K)
and momentum p to -p. The levels of one state at p therefore equal those of the other at
-p. A generic state is not invariant, and its own levels at p and -p differ; that is the
case of an optimized state up to its convergence (1.6e-8 at D = 16, g = 2, p = 0.5 on the
test fixture).

    OPENBLAS_NUM_THREADS=1 python -m unittest test_symmetry -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eigsolve  # noqa: E402
import models  # noqa: E402
import ode  # noqa: E402
import packed  # noqa: E402
import states  # noqa: E402
from davidson import DavidsonOptions  # noqa: E402


class TimeReversal(unittest.TestCase):
    def levels(self, state, p):
        result = eigsolve.excitation_eigsolve(
            state, p, models.phi4_terms(1.3), nev=2, kinetic_nodes=48, validate=False,
            davidson_options=DavidsonOptions(tol=1e-9, maxops=200, maxspace=40),
            integrate=ode.integrate_host, system=packed.PackedActionSystem)
        self.assertTrue(result["converged"])
        return np.array(result["energies"])

    def test_conjugate_state_at_opposite_momentum(self):
        integrator = states.IntegratorParameters(n_quad=48, max_point=16.0, atol=1e-13, rtol=1e-12)
        state = states.random_state(4, seed=11, scaling=0.5, z2=True)
        state.integrator = integrator
        conjugate = states.make_state(np.conj(state.R), -np.conj(state.K), integrator=integrator,
                                      model=state.model, coupling=state.coupling)
        forward, backward = self.levels(state, 0.4), self.levels(state, -0.4)
        np.testing.assert_allclose(self.levels(conjugate, -0.4), forward, rtol=0, atol=1e-11)
        np.testing.assert_allclose(self.levels(conjugate, 0.4), backward, rtol=0, atol=1e-11)
        # The random state is not time-reversal invariant: the test above is not vacuous.
        self.assertGreater(abs(forward[0] - backward[0]), 0.1)


if __name__ == "__main__":
    unittest.main()
