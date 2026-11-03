# Tangent-space excitations

This guide describes the Hamiltonian excitation routines. The
[mathematical derivation](excitations.tex) covers the environments, the
`-im*p*W` kinetic insertion, the finite-box subtraction, and literature
references. The [README](../README.md#tangent-space-excitations) gives a short
overview.

## Dense reference at small bond dimension

The one-argument `mass_gap(s)` retains its historical meaning: the **transfer
gap**, now also named `transfer_gap(s)`. For a Hamiltonian rest gap and dispersion,
use the complex tangent-space calculation:

```julia
model = ExcitationModel(:phi4, 2.0) # same coefficients as fg_phi4(s, 2.0)
# s should be an optimized ground state for this model.
levels = excitation_spectrum(s, 0.5; model, nev=2, assembly=:action)
gap = excitation_gap(s; model, assembly=:action) # lowest represented energy at p=0
gap = mass_gap(s, model)  # explicit-model overload of excitation_gap (default assembly=:bilinear)
dispersion = dispersion_relation(s, [-1.0, -0.5, 0.0, 0.5, 1.0]; model, assembly=:action)
check = relativistic_diagnostics(dispersion)
# check.invariant_mass_squared: E(p)^2-p^2
# check.c_squared: slope of the least-squares fit E(p)^2 = a+c_squared*p^2
```

`ExcitationModel(s)` reads the state's model metadata by default. Explicit models
support `:phi4`, `:shg`, `:sg`, and `:phi4_m` (with `massmod`), sharing the ground
objectives' term definitions. The modified-mass objective uses rescaled energy
units; convert those before applying a speed-one dispersion check (or set
`target_speed` in `relativistic_diagnostics`). Since state metadata does not
store `massmod`, that model requires `ExcitationModel(s; massmod=...)` explicitly.
The explicit constructor `ExcitationModel(:phi4_m, g)`, like `fg_phi4_massmod`,
defaults to `massmod=0.1`, whereas `optimize_state` defaults to `massmod=1.0`;
pass the value used for the ground state.

The calculation uses `V=-R'W`, the physical complex norm `tr(W*rho*U')`
(`tangent_metric(s, U, W)`), momentum-shifted single-insertion environments, a
connected double-insertion environment, and an adjoint environment for the right
tail. It subtracts the extensive ground contribution analytically. Kinetic terms
default to an infinite-line resolvent quadrature; `kinetic_backend=:ode` provides
a separate finite-window evaluation using the same source graphs as the ground
observables. Potential terms use `s.integrator_param.max_point` and the ODE
tolerances; `n_quad` is not used for these values-only excitation ODEs. Increase
the source window and tighten tolerances independently to check convergence.

This is a **small-bond dense reference implementation**. It builds all complex
matrix elements, checks Hermiticity before symmetrization, and returns normalized
`vectors`, finite-matrix `residuals`, `metric_rank`, and
`discarded_metric_weight`. Assembly needs O(D⁴) observable evaluations, with
O(D⁶) dense transfer factorization. The default D≤16 guard can be changed with
`max_bond`; it is not a runtime guarantee. `excitation_hamiltonian_action(s,p,W)`
avoids the O(D⁴) bilinear evaluations by contracting the bra tangent analytically;
it returns `F` such that `dot(U,F)` is the excitation matrix element. It still uses
dense D² transfer factorizations, and potential actions also require convergence
of `n_quad`. Use `assembly=:action` in `excitation_spectrum`, `excitation_gap`, or
`dispersion_relation` to assemble the matrix from these actions. This uses the
same adaptive ODE and Gauss quadrature strategy as the ground-state gradient;
it stores trajectories, not a dense LU factor at every integration point.
The default `assembly=:bilinear` retains the independent reference calculation.
Actions accept `parallel=:trajectories` for concurrent forward/backward ODEs,
`:terms` for concurrent Hamiltonian terms, and `:both` for both levels. This
also provides parallel work inside an iterative eigensolver's sequential
Hamiltonian calls. No optimizer metric regularizer enters the calculation;
`metric_tol` controls the resolved density-matrix support.

## Large bond dimension: iterative solver

`excitation_eigsolve` computes the lowest levels without assembling a matrix. It
is the library form of the pipeline used for the D=16–64 mass scans:

```julia
s.integrator_param.max_point = 30.0; s.integrator_param.n_quad = 512
s.integrator_param.atol = 1e-13;     s.integrator_param.rtol = 1e-12
model = ExcitationModel(:phi4, 2.0)
r = excitation_eigsolve(s, 0.0; model)  # sector=:auto: :odd for a Z2 state, :all otherwise
r.validated_energies[1]  # refined bilinear energy; r.validation[1] has all checks
gap = excitation_gap(s; model, solver=:iterative)
dispersion = dispersion_relation(s, 0:0.25:1; model, solver=:iterative)
```

- **Sectors.** `excitation_parity(s)` detects the balanced Z₂ layout of
  `RandomStateZ2` (or pass `parity`). Physical parity acts as W ↦ -ZWZ, so
  block-diagonal W form the `:odd` sector and off-diagonal W the `:even` sector.
  `sector=:auto` selects `:odd` for Z₂ states and `:all` otherwise. The dense
  routines accept the same `sector` keyword and provide a small-D reference.
- **Coordinates.** `TangentCoordinates` uses white coordinates with
  tr(WρW†)=‖c‖². `gauge=:density` (the default here) rotates each parity block
  into the eigenbasis of ρ and whitens by a Cholesky factor. This is the
  Float64-stable choice used at D≥56. Returned `vectors` are in the gauge of `s`.
- **Operator.** It uses the reverse-contracted Hamiltonian action with
  `kinetic_nodes=256` fixed θ nodes, which makes the kinetic part exactly linear,
  and potential terms on `s.integrator_param`, which are linear to the ODE
  tolerance. With the default `parallel=:both`, kinetic/potential terms
  and forward/backward trajectories run concurrently.
- **Solver.** Davidson (`DavidsonOptions`) with a positive preconditioner built
  from the coarse 8-node kinetic matrix. The start vector is the gauge-fixed field
  insertion φ(0)|Ψ⟩, gauge-fixed at the requested momentum; in the even sector it
  is the lowest coarse kinetic mode. The default search space holds at most
  `max(8, 2nev+2)` vectors; a restart keeps the `nev` lowest Ritz vectors and
  recomputes their actions (fresh restart, as at D≥56).
  The D≤48 research configuration is `gauge=:original` with
  `DavidsonOptions(maxspace=min(160,n), retain=12, refresh=false,
  stall_residual_limit=1e-6, stall_energy_atol=1e-10)`, where n is the sector
  dimension. On the D=16 g=2 state it reproduced the research run (16
  iterations) and its mass to 3e-12 when measured during development. The stall
  rule stops a plateaued search, and the result then reports `converged=false`.
- **Validation.** For every returned level (`validate=true`), it computes fresh
  actions at the standard and refined settings, plus two independent bilinear
  energies. The refined settings default to window +10, 2×`n_quad`, tolerances/10
  and 2× kinetic nodes. `refine_steps` adds the D≤48 preconditioned corrections.
  A level is `accepted` when its four energies agree and their imaginary parts
  vanish within `validation_tol=1e-7`, and its fresh residuals are at most
  `acceptance_residual=1e-5`; these are the research acceptance criteria.
  `excitation_gap` and `dispersion_relation` with `solver=:iterative` throw
  `SolverFailure` for levels that are not accepted (the default `strict=true`);
  `dispersion_relation(...; strict=false)` instead computes every momentum and
  returns the unaccepted levels too, flagged in its `accepted` field. These
  checks are consistency measurements, not error bounds, and they do not test
  the background state.
- **Cost.** Dense transfer factors of ≈48D⁴ bytes at p=0 (≈80D⁴ otherwise) are
  still required, peaking near twice that during construction; `max_bond=64`
  guards this. Each concurrent kinetic action also allocates a 16D⁴-byte factor
  per node (32D⁴ at p≠0), and the preconditioner runs `Threads.nthreads()` of
  them at once. At D=64 this is 268 MB per thread. Each concurrent potential
  action stores ≈640·`n_quad`·D² bytes of trajectories. The coarse preconditioner
  needs one 8-node kinetic action per coordinate (D²/2 for a sector). Each
  iteration needs one full action. `ExcitationContext(base; momentum, integrator)`
  reuses the momentum-independent factors, which
  `dispersion_relation(...; solver=:iterative)` does automatically, warm-starting
  each momentum from the previous vectors.

## Physical scope

The lowest p=0 level is the rest gap **within the same-vacuum tangent sector**.
Kinks between different vacua and general multiparticle states are not covered.
Negative levels remain visible, and the routine does not certify that its input
is a converged ground state. Check background optimization and bond dimension
before interpreting interacting energies. Branches are sorted independently at
each momentum; track their identity near crossings before using the relativistic
diagnostic.

## Tests, example and research runs

[The runnable example](../examples/excitations.jl) verifies the exact free
dispersion and can also load an optimized saved state. Tests compare against
analytic coherent states and an independent finite-box contraction without
Poisson environments, as well as both kinetic backends and auxiliary gauge
invariance. `benchmark/excitation_mass_d16.jl` reproduces the recorded mass of
the D=16, g=2 state in `test/fixtures/` with the default solver
(`julia --project=. --threads=2 benchmark/excitation_mass_d16.jl`, about 8 minutes with
two threads); it fails if the mass differs from the recorded 0.3523798021192911 by more
than 1e-9.

The research runs are in the companion repository rcmps-phi4-z2. Its Atlas mass
scan (`research/mass_scan/README.md` there) applies the preconditioned
iterative method to frozen Z₂-symmetric φ⁴ ground states from D=16 through D=64,
with fresh residual checks, independent energy contractions, mass-versus-coupling
plots and critical-coupling extrapolations.

The D=16 quartic research run (`research/excitations_d16/README.md` in the
companion repository) records an explicitly Z₂-symmetric optimization,
parity-resolved spectra, adaptive refinement, integration checks, and the
Hamiltonian-truncation comparison. Its iterative trial
(`research/excitations_d16/ITERATIVE.md` there) compares Lanczos using only full
Hamiltonian actions with a Davidson variant that retains the quadratic matrix as
a preconditioner and applies the quartic part through ODEs.

## State handling

The state-handling rules of the [README](../README.md#quick-start) and
[numerics guide](numerics.md#numerical-controls-and-checks) apply unchanged.
