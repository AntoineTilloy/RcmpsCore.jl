# Numerical settings, checks and parallel evaluation

This page collects the numerical controls of the ground-state observables and
optimizer. Excitation-specific settings are in [excitations.md](excitations.md);
measured results are in [benchmarks.md](benchmarks.md).

## Array layout and allocations

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

These are the defaults. Check three errors separately: increase the physical
window, increase `n_quad`, and tighten the ODE tolerances. The boundary
correction removes the omitted stationary-state derivative; it does not replace
these convergence checks. The source transformation has an explicit finite
limit at zero and needs no arbitrary lower cutoff.

`steady_state(s)` returns the density matrix `rho`, the physical residual, the
solver method, and the number of map applications. It uses the CPTP construction of
[Beugnot, Gregory, Robin and Tilloy](https://arxiv.org/abs/2608.30860), with a cached
Schur no-jump inverse. Density matrices are checked for trace, Hermiticity,
positivity and stationarity. Unsuccessful ODE/Krylov solves throw `SolverFailure`.
A stalled boundary solve at D≤16 gets one dense fallback, checked against the
same physical equation. No eigenvalue clipping or unbounded power-method
fallback masks an error.

The stationary state is assumed unique. D≤8 additionally receives a dense
uniqueness check; `steady_state(s; method=:dense)` is a reference for small systems.
For larger D a residual does not prove uniqueness or a small forward error.
`transfer_gap(s)` (= `mass_gap(s)`) uses full spectra at D≤16 by default and checked Arnoldi beyond that;
it may fail to converge for difficult large systems. CPU timings and Krylov
iteration counts depend on the state and BLAS configuration. GPU execution and
arbitrary precision are outside this implementation's scope.

## Parallel evaluation and structured kinetic terms

Start Julia with `--threads=6 --gcthreads=1`, then select parallel execution:

```julia
using RcmpsCore, LinearAlgebra
BLAS.set_num_threads(1)
# state: a State whose model_param is set as in the README quick start ("phi4", 2.0);
# optimize_state uses that model, not the fg_phi4 call, and defaults to n_iter=1000.
energy, gradient = fg_phi4(state, 2.0; parallel=:both)
result = optimize_state(state; parallel=:both, diagnostics=true)
```

The modes are `:serial` (default), `:terms` (independent energy contributions),
`:trajectories` (forward/backward ODEs), and `:both`. Julia's worker pool limits
concurrency; with one compute thread all modes run serially. Each evaluation
owns its scratch space, all workers are joined on failure, and results are
summed in a fixed order. Treat `s.R`, `s.K`, `s.Q`, `s.ρ0` and the settings as
read-only while evaluating a state; rebuild with `State(R,K)` or use
`retraction` to update the matrices together. The library does not change your global BLAS settings.

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
Angular quadrature doubles from 32 through at most 512 nodes until both energy
and the complete gradient pass refinement checks. Insufficient refinement raises
`SolverFailure`, unless the opt-in `gradient_fallback_atol` of `KineticOptions`
(default 0, disabled) accepts a gradient plateau below that absolute error; its
docstring states the conditions. Interaction terms still use the original ODE
settings. Resolvent kinetic terms are infinite-line quantities; the ODE backend
(`kinetic_backend=:ode`, default) retains the finite-window reference. Schur
setup costs O(D⁶) and storage O(D⁴), so the dense backend is intended for
small/moderate bond dimensions. For Krylov basis size k, persistent basis storage
is O(kD²+k²) per right-hand side; orthogonalization costs O(k²D²), in addition
to O(kD³) Lindblad actions. Wide transfer spectra can require larger bases:
insufficient convergence is reported explicitly rather than silently falling
back to a large dense solve. Angular refinement and true shifted residuals are
complementary checks, not certified bounds on the final gradient.
Stationary-response and metric solves can amplify their errors, especially near
a small transfer gap.

The kinetic derivation and independent checks
(`research/structured_observables/README.md`) and the bond-enlargement research
(`research/bond_enlargement/README.md`) are in the companion repository
rcmps-phi4-z2.

Measured complete optimizations (`benchmark/threading/README.md` in the
companion repository) and large-D evaluations (`benchmark/large_scaling/README.md`
there) distinguish threading gains from changes in the kinetic algorithm. The
latter include random states through D64 and the saved g=2 D28 state. The saved
shg D32 stress state fails the default Krylov residual requirement; this backend
remains an explicit option.
