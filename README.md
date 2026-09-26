# RcmpsCore 0.2 — independent refactor

A CPU Julia implementation of single-field relativistic continuous matrix product
states, in the mass-one convention. This folder is a standalone package with the
same name and UUID as the original. The original source and data are preserved.

The first change is mathematical: **every observable gradient differentiates the
stationary initial state**, using one adjoint Poisson solve. The old finite-window
bulk gradient omits this contribution, which can remain large at a small transfer
gap. See [the derivation](docs/boundary-gradient.md) and [measured results](docs/benchmarks.md).

## Run

From the repository root:

```sh
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor refactor/examples/phi4.jl
```

Your current global `JULIA_LOAD_PATH` puts `/home/antoine/Projects/` before the
active environment. The explicit setting above is essential on this machine:
otherwise `using RcmpsCore` loads the original package despite `--project=refactor`.
After extracting this directory elsewhere, use `--project=.` instead.

Julia 1.12.7 and the versions in `Manifest.toml` were used for validation. The
project allows Julia 1.10+, subject to dependency compatibility; other Julia
versions have not been validated here. The manifest records the resolved current
releases; it does not reuse the original project's pins or modified OptimKit.

```julia
using RcmpsCore
s = RandomState(8; seed=42, scaling=0.2)
s.model_param.name = "phi4"
s.model_param.coupling = 2.0
energy, gradient = fg_phi4(s, 2.0)
energy_only = fg_phi4(s, 2.0; gradient=false)
result = optimize_state(s; method="LBFGS", n_iter=100, diagnostics=true)
rcmps_store(result.state, "result.jld2")
```

`metric(s, gradient, W)` gives the directional derivative along the canonical
retraction. `vertex(s,beta)` returns a complex value and the gradient of its
**real part**. This convention handles real and imaginary beta consistently.

## Structure

| File | Responsibility |
|---|---|
| `src/state.jl` | Validated canonical states, typed settings, deterministic local RNG |
| `src/transfer.jl` | Lindblad actions, CPTP steady states, boundary solve, entropy and transfer gap |
| `src/observables.jl` | Shared augmented ODE and adjoint engine, covectors, energy assembly |
| `src/kinetic.jl`, `src/kinetic_krylov.jl` | Adaptive kinetic resolvents, dense reference and matrix-free Arnoldi |
| `src/excitation_environments.jl`, `src/excitations.jl` | Momentum-resolved tangent environments, Hamiltonian spectrum and dispersion |
| `src/excitation_action.jl` | Reverse contraction of the bra tangent for a Hamiltonian action |
| `src/davidson.jl` | Restarted Hermitian Davidson with fresh or thick restarts and a stall rule |
| `src/excitation_eigsolve.jl` | Parity sectors, white coordinates, preconditioned iterative excitation solver |
| `src/optimization.jl` | Metric, retraction, released OptimKit integration, bounded Armijo search |
| `src/storage.jl` | Versioned JLD2 schema, atomic writes, loading and selection |
| `research/` | Independent dense Python derivation check and plotting |
| `benchmark/` | Reproducible Julia comparisons, raw CSV results, original-code adapter |

Matrices use `ComplexF64`. Scratch matrices are owned by individual evaluations,
not by the state. The ODE stores channels as `D × D × n`, so each matrix slice is
contiguous; the warmed RHS allocates zero bytes. Values-only calls skip adjoints
and gradient quadrature. Energy sums share a quadrature grid and combine their
boundary contributions before one Poisson solve. No distributed workers or
forced garbage collection are required.

## Numerical controls and checks

```julia
s.integrator_param.max_point = 20.0 # physical half-window, bare mass = 1
s.integrator_param.n_quad = 128     # Gauss nodes PER HALF in x=t|t|
s.integrator_param.atol = 1e-11
s.integrator_param.rtol = 1e-10
s.manifold_param.metric_reg = 1e-3
```

Check three errors separately: increase the physical window, increase `n_quad`,
and tighten the ODE tolerances. The boundary correction removes the omitted
stationary-state derivative; it does not replace these convergence checks.
The source transformation has an explicit finite limit at zero and needs no
arbitrary lower cutoff.

`steady_state(s)` returns the state, physical residual, solver method, and number
of map applications. It uses the CPTP construction of
[Beugnot, Gregory, Robin and Tilloy](https://arxiv.org/abs/2608.30860), with a cached
Schur no-jump inverse. Density matrices are checked for trace, Hermiticity,
positivity and stationarity. Unsuccessful ODE/Krylov solves throw `SolverFailure`.
A stalled boundary solve at D≤16 gets one dense fallback, checked against the
same physical equation. No eigenvalue clipping or unbounded power-method
fallback masks an error.

The stationary state is assumed unique. D≤8 additionally receives a dense
uniqueness check; `steady_state(s; method=:dense)` is a reference for small systems.
For larger D a residual does not prove uniqueness or a small forward error.
`mass_gap` uses full spectra at D≤16 by default and checked Arnoldi beyond that;
it may fail to converge for difficult large systems. CPU timings and Krylov
iteration counts depend on the state and BLAS configuration. GPU execution and
arbitrary precision are outside this implementation's scope.

## Hamiltonian excitations

The one-argument `mass_gap(s)` retains its historical meaning: the **transfer
gap**, now also named `transfer_gap(s)`. For a Hamiltonian rest gap and dispersion,
use the new complex tangent-space calculation:

```julia
model = ExcitationModel(:phi4, 2.0) # same coefficients as fg_phi4(s, 2.0)
# s should be an optimized ground state for this model.
levels = excitation_spectrum(s, 0.5; model, nev=2, assembly=:action)
gap = excitation_gap(s; model, assembly=:action) # lowest represented energy at p=0
gap = mass_gap(s, model)           # explicit-model overload, same calculation
dispersion = dispersion_relation(s, [-1.0, -0.5, 0.0, 0.5, 1.0]; model)
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

The calculation uses `V=-R'W`, the physical complex norm `tr(W*rho*U')`,
momentum-shifted single-insertion environments, a connected double-insertion
environment, and an adjoint environment for the right tail. It subtracts the
extensive ground contribution analytically. Kinetic terms default to an
infinite-line resolvent quadrature; `kinetic_backend=:ode` provides a separate
finite-window evaluation using the same source graphs as the ground observables.
Potential terms use `s.integrator_param.max_point` and the ODE tolerances;
`n_quad` is not used for these values-only excitation ODEs. Increase the source
window and tighten tolerances independently to check convergence.

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
Hamiltonian calls.
No optimizer metric regularizer enters
the calculation; `metric_tol` controls the resolved density-matrix support.

### Large bond dimension: iterative solver

`excitation_eigsolve` computes the lowest levels without assembling a matrix. It
is the library form of the pipeline used for the D=16–64 mass scans:

```julia
s.integrator_param.max_point = 30.0; s.integrator_param.n_quad = 512
s.integrator_param.atol = 1e-13;     s.integrator_param.rtol = 1e-12
r = excitation_eigsolve(s, 0.0; model=ExcitationModel(:phi4, 2.0))  # odd sector of a Z2 state
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
  tr(WρW†)=‖c‖². `gauge=:density` rotates each parity block into the eigenbasis of
  ρ and whitens by a Cholesky factor. This is the Float64-stable choice used at
  D≥56. Returned `vectors` are in the gauge of `s`.
- **Operator.** It uses the reverse-contracted Hamiltonian action with
  `kinetic_nodes=256` fixed θ nodes, which makes the kinetic part exactly linear,
  and potential terms on `s.integrator_param`, which are linear to the ODE
  tolerance. With `parallel=:both`, kinetic/potential terms
  and forward/backward trajectories run concurrently.
- **Solver.** Davidson (`DavidsonOptions`) with a positive preconditioner built
  from the coarse 8-node kinetic matrix. The start vector is the gauge-fixed field
  insertion φ(0)|Ψ⟩, gauge-fixed at the requested momentum; in the even sector it
  is the lowest coarse kinetic mode. The default restarts from `max(8, 2nev+2)`
  vectors and recomputes the retained actions (fresh restart, as at D≥56).
  The D≤48 research configuration is `gauge=:original` with
  `DavidsonOptions(maxspace=min(160,n), retain=12, refresh=false,
  stall_residual_limit=1e-6, stall_energy_atol=1e-10)`, where n is the sector
  dimension. On the D=16 g=2 state it reproduces the research run (16
  iterations) and its mass to 3e-12. The stall rule stops a plateaued search, and
  the result then reports `converged=false`.
- **Validation.** For every returned level (`validate=true`), it computes fresh
  actions at the standard and refined settings, plus two independent bilinear
  energies. The refined settings default to window +10, 2×`n_quad`, tolerances/10
  and 2× kinetic nodes. `refine_steps` adds the D≤48 preconditioned corrections.
  A level is `accepted` when its four energies and imaginary parts agree within
  `validation_tol=1e-7` and its fresh residuals are at most
  `acceptance_residual=1e-5`; these are the research acceptance criteria.
  `excitation_gap` and `dispersion_relation` with `solver=:iterative` throw
  `SolverFailure` for levels that are not accepted. These checks are consistency
  measurements, not error bounds, and they do not test the background state.
- **Cost.** Dense transfer factors of ≈48D⁴ bytes at p=0 (≈80D⁴ otherwise) are
  still required, peaking near twice that during construction; `max_bond=64`
  guards this. Each concurrent kinetic action also allocates a 16D⁴-byte factor
  per node (32D⁴ at p≠0), and the preconditioner runs `Threads.nthreads()` of
  them at once. At D=64 this is 268 MB per thread. The coarse preconditioner needs one
  8-node kinetic action per coordinate (D²/2 for a sector). Each iteration needs
  one full action. `ExcitationContext(base; momentum, integrator)` reuses the
  momentum-independent factors, which `dispersion_relation(...; solver=:iterative)`
  does automatically, warm-starting each momentum from the previous vectors.

The lowest p=0 level is the rest gap **within the same-vacuum tangent sector**.
Kinks between different vacua and general multiparticle states are not covered.
Negative levels remain visible, and the routine does not certify that its input
is a converged ground state. Check background optimization and bond dimension
before interpreting interacting energies. Branches are sorted independently at
each momentum; track their identity near crossings before using the relativistic
diagnostic.

The [mathematical derivation](docs/excitations.tex) covers the new environments,
the `-im*p*W` kinetic insertion, finite-box subtraction, and literature references.
[The runnable example](examples/excitations.jl) verifies the exact free dispersion
and can also load an optimized saved state. Tests compare against analytic
coherent states and an independent finite-box contraction without Poisson
environments, as well as both kinetic backends and auxiliary gauge invariance.
The [Atlas mass scan](research/mass_scan/README.md) applies the preconditioned
iterative method to saved Z₂-symmetric D=16 and D=32 φ⁴ states from g=2 to 2.77,
with frozen inputs, fresh residual checks, independent energy contractions,
and a mass-versus-coupling plot with the difference between bond dimensions.

The [D=16 quartic research run](research/excitations_d16/README.md) records an
explicitly Z₂-symmetric optimization, parity-resolved spectra, adaptive refinement,
integration checks, and the Hamiltonian-truncation comparison.
Its [iterative trial](research/excitations_d16/ITERATIVE.md) compares Lanczos
using only full Hamiltonian actions with a Davidson variant that retains the
quadratic matrix as a preconditioner and applies the quartic part through ODEs.

Treat `s.R`, `s.K`, `s.Q`, and `s.ρ0` as read-only. Rebuild with `State(R,K)` or use
`retraction` to update them together. Configure a state before concurrent
observables; do not mutate settings while evaluations are running.

## Migration

The main observable and optimizer names are retained. Deliberate changes:

- `n_quad` counts Gauss nodes per half instead of exp-sinh steps; `min_point` is removed.
- `max_point` directly specifies the physical window.
- Random constructors use a local reproducible RNG. Odd Z2 dimensions are rejected.
- `generate_T` follows Julia's column-major vectorization exactly.
- Invalid model/optimizer names and solver failures raise exceptions.
- `retraction_regulated` aliases the canonical retraction. Step selection belongs
  to the line search; a nonlinear step cap must not silently change its derivative.
  The default now bounds parameter excursions in the Wolfe search and falls back
  to Armijo backtracking if a numerical trial fails. Failed trials are rejected,
  preserving the last accepted state and the exact canonical retraction.
  Approximate Wolfe acceptance allows energy noise up to `1e-10` while explicitly
  checking the slope conditions; this avoids false failures near convergence.
- `rcmps_store(s,path)` and `rcmps_load(path)` take explicit paths. New files are
  `.jld2`, record numerical settings, and recompute rho on load.
- `get_exact_state` returns `nothing` when no state matches. Selection uses actual
  couplings rather than flooring them into potentially colliding bins.

Historical JLD reading is a separate, optional dependency. Prepare the benchmark
and migration environment once:

```sh
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor/benchmark -e 'using Pkg; Pkg.develop(path="refactor"); Pkg.instantiate()'
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor/benchmark refactor/examples/migrate_legacy.jl data/phi4_g2.0_D2_.jld migrated.jld2
```

The migration script preserves R, K and model metadata and recomputes energy and
gradient. Historical `phi4_m` files did not store the modified mass; the script requires
an explicit third argument with the mass originally used for such a file.
The script refuses to overwrite an existing destination.

## Reproduce the research and benchmarks

```sh
python refactor/research/boundary_reference.py # numpy, scipy
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor refactor/benchmark/boundary.jl
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor refactor/benchmark/optimization.jl
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor/benchmark refactor/benchmark/performance.jl
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor/benchmark refactor/benchmark/saved_states.jl
JULIA_LOAD_PATH='@:@stdlib' julia --project=refactor/benchmark refactor/benchmark/converged_optimization.jl
python refactor/research/plot_boundary.py # numpy, matplotlib
```

The legacy adapter reads the original source directly. Its only compatibility
changes are the dedicated Vern7 import and SciML 3's saved-state indexing; it
retains the original numerical algorithms and Octavian kernels. Raw timing
results are descriptive measurements, not portable speed guarantees.

The [converged g=2 comparison](benchmark/converged/README.md) reports matched
old/new optimizations at D=2,4,8,16, including independent gradient checks,
precision refinement, timings, and saved parameter matrices.

## Parallel evaluation and structured kinetic terms

Start Julia with `--threads=6 --gcthreads=1`, then select parallel execution:

```julia
using RcmpsCore, LinearAlgebra
BLAS.set_num_threads(1)
energy, gradient = fg_phi4(state, 2.0; parallel=:both)
result = optimize_state(state; parallel=:both, diagnostics=true)
```

The modes are `:serial` (default), `:terms` (independent energy contributions),
`:trajectories` (forward/backward ODEs), and `:both`. Julia's worker pool limits
concurrency; with one compute thread all modes run serially. Each evaluation
owns its scratch space, all workers are joined on failure, and results are
summed in a fixed order. Treat the state's matrices and parameters as read-only
while evaluating it. The library does not change your global BLAS settings.

An optional kinetic backend integrates the source kernel analytically and
computes the resulting shifted transfer resolvents:

```julia
energy, gradient = fg_phi4(state, 2.0;
    parallel=:both, kinetic_backend=:resolvent)
result = optimize_state(state; parallel=:both, kinetic_backend=:resolvent,
    kinetic_options=KineticOptions(atol=1e-11, rtol=1e-10), diagnostics=true)
```

For larger bond dimensions, a matrix-free shifted Arnoldi backend avoids the
dense transfer matrix:

```julia
energy, gradient = fg_phi4(state, 2.0; parallel=:both, kinetic_backend=:krylov,
    krylov_options=KineticKrylovOptions(tolerance=1e-12, initialdim=16, maxdim=512))
```

The same options work for individual kinetic terms, optimization, and the other
energy models. The resolvent backend shares a dense Schur factorization; the
Krylov backend reuses forward and adjoint bases across all shifts and angular
refinements, checks every true physical residual, and grows storage only as
needed up to the basis cap. Contexts and scratch arrays belong to one evaluation.
Angular quadrature doubles
from 32 through at most 512 nodes until both energy and the complete gradient
pass refinement checks. Insufficient refinement raises `SolverFailure`.
Interaction terms still use the original ODE settings. Resolvent kinetic terms
are infinite-line quantities; the ODE backend (`kinetic_backend=:ode`, default)
retains the finite-window reference. Schur setup costs O(D⁶) and storage O(D⁴),
so the dense backend is intended for small/moderate bond dimensions. For Krylov
basis size k, persistent basis storage is O(kD²+k²) per right-hand side;
orthogonalization costs O(k²D²), in addition to O(kD³) Lindblad actions.
Wide transfer spectra can require larger bases: insufficient convergence is
reported explicitly rather than silently falling back to a large dense solve.
Angular refinement and true shifted residuals are complementary checks, not
certified bounds on the final gradient. Stationary-response and metric solves
can amplify their errors, especially near a small transfer gap.

See the [kinetic derivation and independent checks](research/structured_observables/README.md)
and [bond-enlargement research](research/bond_enlargement/README.md).

Measured [complete optimizations](benchmark/threading/README.md) and
[large-D evaluations](benchmark/large_scaling/README.md) distinguish threading
gains from changes in the kinetic algorithm. The latter include random states
through D64 and the saved g=2 D28 state. The saved shg D32 stress state fails the
default Krylov residual requirement; this backend remains an explicit option.
