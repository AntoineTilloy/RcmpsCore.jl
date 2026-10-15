# RcmpsCore.jl

RcmpsCore.jl is a CPU Julia implementation of relativistic continuous matrix
product states (RCMPS) for a single scalar field in 1+1 dimensions, with
Hamiltonians normal ordered at mass one. It optimizes translation-invariant
ground states and computes tangent-space excitations at fixed momentum.
The models are φ⁴ (`fg_phi4`), sinh-Gordon and sine-Gordon (`fg_sgordon` with
real or imaginary β) and the modified-mass φ⁴ objective (`fg_phi4_massmod`),
named `:phi4`, `:shg`, `:sg` and `:phi4_m` by `ExcitationModel`. `fg_phi4(s, g)`
is the energy density of H_free(m=1) + g ∫ dx :φ⁴: (no 1/4!); the sinh-Gordon
and sine-Gordon Hamiltonians are given in [docs/excitations.tex](docs/excitations.tex).
This is version 0.2, an independent rewrite with the same name and UUID; the
original implementation is frozen as tag `v0.1.0`.

## Installation

The package is not registered. Add it by URL, or clone the repository and run
`julia --project=. -e 'using Pkg; Pkg.instantiate()'` in the checkout:

```julia
using Pkg
Pkg.add(url="https://gitlab.inria.fr/atilloy/RcmpsCore.jl.git")
# legacy implementation: Pkg.add(url="https://gitlab.inria.fr/atilloy/RcmpsCore.jl.git", rev="v0.1.0")
```

`Project.toml` requires Julia 1.10 or later and bounds JLD2 (0.6), KrylovKit
(0.10), OptimKit (0.5), OrdinaryDiffEqVerner (2), SciMLBase (3) and
SpecialFunctions (2). Validation used Julia 1.12.7 with the dependency versions
recorded in `benchmark/environment.txt`; other Julia versions have not been
tested. No `Manifest.toml` is committed, so dependencies resolve to current
releases within these bounds.

The v0.1.0 `Project.toml` bounds only OptimKit (0.4) and Julia, and 0.1 was
developed with a modified OptimKit. A fresh install of the tag resolves
OrdinaryDiffEq to current releases, which require SciMLBase 3, and the 0.1 sources
use saved-state indexing that SciMLBase 3 no longer supports. `benchmark/legacy.jl`
runs the tagged sources in the `benchmark` environment with the required import
and indexing changes.

Troubleshooting: a directory listed in `JULIA_LOAD_PATH` ahead of the active project
(`@`) that contains another RcmpsCore.jl checkout takes precedence over the active
project, and `using RcmpsCore` loads that copy; `JULIA_LOAD_PATH='@:@stdlib'` avoids this.

## Implementations

| Package | Language, hardware | Scope | Role |
|---|---|---|---|
| `RcmpsCore` (`src/`) | Julia, CPU | all models | reference |
| `rcmps_gpu` ([`gpu/`](gpu/README.md)) | Python with CuPy, CUDA, FP64; NVIDIA GPUs of compute capability 6.0 or later (FP64 tensor cores from 8.0), run on A100 and H100 | φ⁴ on Z₂-symmetric states, even D: optimizer and excitation solver | accelerator; excitations qualified to D=128 |

They share one contract: `rcmps_store` checkpoints (JLD2 schema 1), the portable state
directories of format 1 (Julia side in `gpu/julia/`), and the excitation levels of
`test/fixtures/excitation_references.toml`, which `benchmark/excitation_references.jl`
and the `gpu/` tests recompute. The D=4 and D=32 Z₂ states of those levels and a portable
export of the D=16 state are in `test/fixtures/` too. GitLab CI also runs the CPU tests of
`gpu/` on Python 3.11 and 3.12 (see [gpu/README.md](gpu/README.md#tests)).

## Quick start

```julia
using RcmpsCore
s = RandomState(8; seed=42, scaling=0.2)
s.model_param.name = "phi4"          # "phi4", "phi4_m", "shg" or "sg"
s.model_param.coupling = 2.0
energy, gradient = fg_phi4(s, 2.0)
energy_only = fg_phi4(s, 2.0; gradient=false)
result = optimize_state(s; method="LBFGS", n_iter=100, diagnostics=true)
result.converged                     # whether gradtol was reached within n_iter
rcmps_store(result.state, "result.jld2")
t = rcmps_load("result.jld2")        # recomputes and checks rho
```

States are left canonical, Q = -iK - R†R/2; a tangent W changes R by W and Q by
-R†W. `metric(s, gradient, W)` is the directional derivative along the canonical
retraction, for the metric 2 Re tr(W₁(ρ+ε)W₂†), ε = `s.manifold_param.metric_reg`
(default 1e-3). `vertex(s, beta)` returns a complex value and the gradient of its
**real part**; this convention handles real and imaginary beta consistently.
Treat `s.R`, `s.K`, `s.Q` and `s.ρ0` as read-only: rebuild with `State(R,K)` or
use `retraction` to update them together, and do not mutate settings while
evaluations are running.

## Tangent-space excitations

The momentum-p tangent ansatz |Φ_p(W)⟩ replaces one (Q, R) pair by (V, W), with
V = -R†W (left gauge), and integrates against e^{ipx}. Its physical metric is
⟨Φ_p(U)|Φ_q(W)⟩ = 2πδ(p-q) tr(WρU†) (`tangent_metric(s, U, W)`, no optimizer
regularizer). The routines diagonalize ⟨Φ_p(U)|(H-E₀)|Φ_p(W)⟩ per 2πδ(0), with the
extensive ground contribution subtracted analytically. `mass_gap(s)` keeps its 0.1
meaning, the transfer gap (also `transfer_gap(s)`); `mass_gap(s, model)` is the
Hamiltonian gap. Dense reference for an optimized ground state `s`, D≤16 by default:

```julia
model = ExcitationModel(:phi4, 2.0) # same coefficients as fg_phi4(s, 2.0)
levels = excitation_spectrum(s, 0.5; model, nev=2, assembly=:action)
gap = excitation_gap(s; model, assembly=:action) # lowest represented energy at p=0
dispersion = dispersion_relation(s, [-1.0, -0.5, 0.0, 0.5, 1.0]; model, assembly=:action)
check = relativistic_diagnostics(dispersion) # E(p)^2-p^2 and a fit of E(p)^2
```

Iterative solver without an assembled matrix (the D=16–64 mass-scan pipeline), D≤64 by default:

```julia
s.integrator_param.max_point = 30.0; s.integrator_param.n_quad = 512
s.integrator_param.atol = 1e-13;     s.integrator_param.rtol = 1e-12
r = excitation_eigsolve(s, 0.0; model)  # sector=:auto: :odd for a Z2 state, :all otherwise
r.validated_energies[1]  # refined bilinear energy; r.validation[1] has all checks
gap = excitation_gap(s; model, solver=:iterative)
dispersion = dispersion_relation(s, 0:0.25:1; model, solver=:iterative)
```

The dense routines check Hermiticity before symmetrization. `excitation_eigsolve`
validates each level with fresh actions at standard and refined settings and two
independent bilinear energies; it is `accepted` when the four energies agree and
their imaginary parts vanish within `validation_tol=1e-7`, and the fresh residuals
are at most `acceptance_residual=1e-5`. With `solver=:iterative`, `excitation_gap`
and `dispersion_relation` (by default, `strict=true`) throw `SolverFailure` for
levels that are not accepted.
These checks are consistency measurements, not error bounds.
The lowest p=0 level is the rest gap **within the same-vacuum tangent sector**:
kinks between different vacua and multiparticle states are not covered. The
input is not certified to be a converged ground state; negative levels remain
visible. Branches are sorted independently at each momentum. See
[docs/excitations.md](docs/excitations.md) for details and
[docs/excitations.tex](docs/excitations.tex) for the derivation.

## Numerical settings

Check three errors separately: increase the physical half-window `max_point` (default
20.0) and the Gauss nodes per half `n_quad` (default 128) of `s.integrator_param`, and
tighten its ODE tolerances `atol`, `rtol` (defaults 1e-11, 1e-10). The boundary correction
does not replace these checks. See [docs/numerics.md](docs/numerics.md) for solvers,
failure handling, threads and kinetic backends.

## Changes from 0.1 and migration

Every observable gradient now differentiates the stationary initial state,
using one adjoint Poisson solve; the 0.1 finite-window bulk gradient omits this
contribution, which can remain large at a small transfer gap. See
[the derivation](docs/boundary-gradient.md) and [measured results](docs/benchmarks.md).
All 34 names exported by 0.1 are still exported:

| Status | Names | Change |
|---|---|---|
| kept | `lindblad`, `lindblad_adjoint`, `lindblad!`, `lindblad_adjoint!`, `jsource`, `robust_inv`, `entanglement_entropy`, `metric`, `precondition`, `retraction`, `naive_transport!`, `finalize!`, `ls_lenient` | same formulas and parameters; `lindblad!` and `lindblad_adjoint!` take an optional scratch matrix instead of state buffers |
| kept | `get_ρ0`, `mass_gap(s)` | same quantities; CPTP Arnoldi for ρ; full spectrum at D≤16, checked Arnoldi above |
| changed | `State`, `RandomState`, `RandomStateZ2` | `ComplexF64` only; no cached adjoints, commutators or buffers; a supplied `ρ0` is checked; `e0` starts as `NaN`; see the list below |
| changed | `copy_param!` | deep-copies the settings; copies `info` but no longer `e0` |
| changed | `vertex`, `phi_monomial`, `kinetic1`, `kinetic2`, `fg_phi4`, `fg_sgordon` | gradients include the stationary-state derivative; keywords `gradient`, `parallel`, `kinetic_backend`; threads replace `Distributed` workers |
| changed | `optimize_state` | defaults `method="LBFGS"` (was `"GD"`), `mlbfgs=20` (100), `verbosity=2` (3), `ls=ls_default` (`ls_lenient`), `retract=retraction` (`retraction_regulated`, now an alias), `massmod=1.0` (0.1); `diagnostics=true` returns a named tuple |
| changed | `generate_T`, `retraction_regulated`, `ls_default`, `ls_backward` | see the list below; `ls_backward` is bounded Armijo backtracking |
| changed | `rcmps_store`, `rcmps_load`, `get_states`, `get_exact_state` | explicit paths, versioned `.jld2` files; `get_states("data")` (positional directory, default `"data"`) reads only `.jld2` files |
| newly exported | `IntegratorParameters`, `ModelParameters`, `ManifoldParameters` | defaults `n_quad=128`, `atol=1e-11`, `rtol=1e-10` (were 1000, 1e-13, 1e-11), `metric_reg=1e-3` (1e-2); `maxiters` added |
| newly exported | `fg_phi4_massmod`, `select_best` | same objective; `select_best` keys on exact couplings |
| removed | none exported | exp-sinh quadrature, `update_stored_states`, `ls_big_steps`; dependencies `Distributed`, `JLD`, `LinearMaps`, `Octavian`, `Parameters`, `Printf`, `ProgressMeter`, `Revise`; `OrdinaryDiffEq` replaced by `OrdinaryDiffEqVerner`; OptimKit 0.4 (a modified copy in the 0.1 environment) replaced by released OptimKit 0.5; Julia 1.10 or later (was 1.6) |
| new | ground state: `steady_state`, `boundary_environment`, `transfer_gap`, `SolverFailure`, `SafeguardedLineSearch`, `KineticOptions`, `KineticKrylovOptions`; excitations: `ExcitationModel`, `tangent_metric`, `excitation_environments`, `excitation_matrix_element`, `excitation_hamiltonian`, `excitation_hamiltonian_action`, `excitation_spectrum`, `excitation_gap`, `dispersion_relation`, `relativistic_diagnostics`, `excitation_eigsolve`, `DavidsonOptions`, `TangentCoordinates`, `ExcitationContext`, `excitation_parity`, and the method `mass_gap(s, model)` | see [numerics](docs/numerics.md) and [excitations](docs/excitations.md) |

Deliberate changes:

- `n_quad` counts Gauss nodes per half instead of exp-sinh steps; `min_point` is
  removed; `max_point` directly specifies the physical window.
- Random constructors use a local reproducible RNG; odd Z2 dimensions are
  rejected. `generate_T` follows Julia's column-major vectorization exactly.
- Invalid model/optimizer names and solver failures raise exceptions.
- `retraction_regulated` aliases the canonical retraction: step selection belongs
  to the line search; a nonlinear step cap must not silently change its derivative.
  The default search bounds parameter excursions in the Wolfe search and falls
  back to Armijo backtracking when a Wolfe trial fails numerically, exceeds the
  step bound, or does not satisfy the rechecked Wolfe conditions; failed trials
  are rejected, preserving the last accepted state and the exact canonical
  retraction. Approximate Wolfe acceptance allows energy noise up to `1e-10` while
  explicitly checking the slope conditions; this avoids false failures near
  convergence.
- `rcmps_store(s,path)` and `rcmps_load(path)` take explicit paths. New files are
  `.jld2`, record numerical settings, and recompute rho on load.
- `get_exact_state` returns `nothing` when no state matches; selection uses actual
  couplings rather than flooring them into potentially colliding bins.

### Migrating saved states

Reading historical JLD files needs the optional `benchmark/` environment. A fresh
clone has no `benchmark/Manifest.toml` and RcmpsCore is unregistered, so prepare
it once from the repository root (network access is needed the first time):

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark examples/migrate_legacy.jl benchmark/legacy_states/phi4_g2.0_D2_.jld examples/output/migrated.jld2
```

The script preserves R, K and model metadata, recomputes energy and gradient, and
refuses to overwrite an existing destination (`examples/output/` is gitignored).
Historical `phi4_m` files did not store the modified mass; the script refuses to
run on them without a third argument giving the mass originally used (the 0.1
default of `optimize_state` was 0.1). Other models are recomputed with
`massmod=1.0`, which they do not use.

## Repository layout

| Path | Contents |
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
| `test/` | `runtests.jl` and the kinetic and excitation test files it includes |
| `test/fixtures/` | Captured D=48 Davidson residual history (stall-rule test); D=16, g=2 Z₂ φ⁴ ground state (mass regression) |
| `docs/` | Boundary-gradient derivation, benchmark report, excitation guide and derivation, numerics guide |
| `examples/` | `phi4.jl`, `excitations.jl` (free dispersion or a saved state), `migrate_legacy.jl` |
| `benchmark/` | Validation, comparison and scaling scripts, the dense Python derivation check `boundary_reference.py`, recorded tables and reports, the 0.1 adapter `legacy.jl`, its own `Project.toml` |
| `benchmark/legacy_states/` | Four 0.1 JLD states read by the saved-state, scaling and migration scripts |

## Tests and benchmarks

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                 # several minutes
julia --project=. --threads=2 benchmark/excitation_mass_d16.jl
julia --project=. examples/phi4.jl                            # writes examples/output/
julia --project=. examples/excitations.jl
python benchmark/boundary_reference.py                        # numpy, scipy
julia --project=. benchmark/boundary.jl
julia --project=. benchmark/optimization.jl
julia --project=benchmark benchmark/performance.jl
julia --project=benchmark benchmark/saved_states.jl
julia --project=benchmark benchmark/converged_optimization.jl
python benchmark/plot_boundary.py                             # numpy, matplotlib
```

GitLab CI (`.gitlab-ci.yml`) runs `Pkg.test()` on every push with Julia 1.12, and
with Julia 1.10, the lowest supported version, as a non-blocking job.

The D=16 regression (8 minutes with two threads on a loaded 12-core workstation)
recomputes the odd-sector p=0 mass of the g=2 Z₂ state in `test/fixtures/` with the
default solver at the research integration settings, and fails if it differs from
the recorded 0.3523798021192911 by more than 1e-9; a run at commit becd9d5, with
library defaults, gave a difference of -1.03e-12.

The other commands produce the results in
[docs/benchmarks.md](docs/benchmarks.md); prepare the `benchmark` environment as in
[Migrating saved states](#migrating-saved-states).
The legacy adapter `benchmark/legacy.jl` reads the v0.1.0 sources with `git show`
(or from `RCMPS_LEGACY_SRC`); its only compatibility changes are the dedicated
Vern7 import and SciML 3's saved-state indexing, and it retains the original
numerical algorithms and Octavian kernels. Raw timings are descriptive, not
portable speed guarantees.

Several scripts rewrite their CSV tables in
`benchmark/` (`plot_boundary.py` the figures in `docs/`); others write result
folders under `benchmark/` (`converged/`, `threading/`, `large_scaling/`, ...),
which are gitignored. The recorded September 2026 results are in the companion
repository rcmps-phi4-z2 at the same paths; scripts reading
`benchmark/converged/*.jld2` or `benchmark/d64_50/final.jld2` need them from there.

## Research record

The companion repository rcmps-phi4-z2 (not yet published) holds the record of
the Z₂-symmetric φ⁴ RCMPS campaigns of September 2026, produced with RcmpsCore
0.2: ground-state optimizations at D=16..64, zero-momentum excitation masses,
critical-coupling analysis and numerical diagnostics, with datasets, drivers,
studies and benchmark result folders, for H = H_free(m=1) + g ∫ dx :φ⁴: (no 1/4!).

## References and license

- A. Tilloy, *Relativistic continuous matrix product states for quantum fields without cutoff*, Phys. Rev. D 104, 096007 (2021), [arXiv:2102.07741](https://arxiv.org/abs/2102.07741).
- G. Beugnot, P. Gregory, R. Robin, A. Tilloy, *Well-conditioned iterative methods for large open quantum systems*, [arXiv:2608.30860](https://arxiv.org/abs/2608.30860).
- J. Haegeman, J. I. Cirac, T. J. Osborne, F. Verstraete, *Calculus of continuous matrix product states*, Phys. Rev. B 88, 085118 (2013), [arXiv:1211.3935](https://arxiv.org/abs/1211.3935).
- D. Draxler, J. Haegeman, T. J. Osborne, V. Stojevic, L. Vanderstraeten, F. Verstraete, *Particles, holes and solitons: a matrix product state approach*, Phys. Rev. Lett. 111, 020402 (2013), [arXiv:1212.1114](https://arxiv.org/abs/1212.1114).
- L. Vanderstraeten, J. Haegeman, F. Verstraete, *Tangent-space methods for uniform matrix product states*, SciPost Phys. Lect. Notes 7 (2019), [doi:10.21468/SciPostPhysLectNotes.7](https://scipost.org/10.21468/SciPostPhysLectNotes.7).

The package is released under the MIT license; see [LICENSE](LICENSE).
