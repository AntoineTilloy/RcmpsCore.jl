# RcmpsCore.jl

Relativistic continuous matrix product states (RCMPS) for quantum field theories in
1+1 dimensions, without UV cutoff and directly in the thermodynamic limit.

## The ansatz

A translation-invariant RCMPS of bond dimension D is

```math
|Q,R\rangle = \mathrm{tr}\Big[\mathcal P\exp\Big(\int_{-\infty}^{+\infty}\! dx\;
Q\otimes\mathbb 1 + R\otimes a^\dagger(x)\Big)\Big]\,|0\rangle_a ,
```

with $`Q, R`$ two $`D\times D`$ matrices, and $`a(x)`$ the annihilation operators of the free
field of mass $`m=1`$ in position space, $`[a(x),a^\dagger(y)]=\delta(x-y)`$, $`a(x)|0\rangle_a=0`$:
the field is $`\phi(x)=\int dy\,J(x-y)\,[a(y)+a^\dagger(y)]`$ with $`\hat J(k)=(2\omega_k)^{-1/2}`$,
$`\omega_k=\sqrt{1+k^2}`$. States are kept left canonical, $`Q=-iK-R^\dagger R/2`$ with
$`K=K^\dagger`$, and Hamiltonians are normal ordered at mass one.

For now the package handles only this simplest case: **a single scalar bosonic field**.
Extensions will be added soon. The models are φ⁴
(`fg_phi4`, $`H_\text{free}(m{=}1)+g\int dx\,{:}\phi^4{:}`$, no 1/4!), sinh-Gordon and
sine-Gordon (`fg_sgordon` with real or imaginary β), and a modified-mass φ⁴ objective
(`fg_phi4_massmod`); see [docs/excitations.tex](docs/excitations.tex) for the conventions.

The package optimizes ground states, and computes excitations at fixed momentum p with
the tangent-space (quasiparticle) ansatz, which replaces one $`(Q,R)`$ by $`(V,W)`$:

```math
|\Phi_p(W)\rangle=\int dx\,e^{ipx}\,\mathrm{tr}\Big[\mathcal P e^{\int_{-\infty}^{x}(\dots)}
\big(V\otimes\mathbb 1+W\otimes a^\dagger(x)\big)\,\mathcal P e^{\int_{x}^{+\infty}(\dots)}\Big]|0\rangle_a,
\qquad V=-R^\dagger W .
```

This is the quasiparticle ansatz of Pufu, Søgaard and Terlemes
([arXiv:2610.13824](https://arxiv.org/abs/2610.13824)). The implementation here is
independent and differs from theirs; it is derived in
[docs/excitations.tex](docs/excitations.tex).

## Implementations

| | Hardware | Scope |
|---|---|---|
| `RcmpsCore` (Julia, `src/`) | CPU | every model; the reference implementation |
| `rcmps_gpu` (Python, [`gpu/`](gpu/README.md)) | NVIDIA GPU, FP64 (compute capability ≥ 6.0) | φ⁴ only: optimizer and excitation solver, for Z₂-symmetric states and states without symmetry |

`gpu/` is a GPU implementation in Python (CuPy and generated CUDA kernels) that was
**entirely vibe coded**. It is tested against the Julia package through shared fixtures and
reference levels. It is much faster than the CPU code at large D, so **use it if you
have access to a GPU**. It reads `rcmps_store`
checkpoints, and exchanges states with Julia in a portable format (`gpu/julia/`); see
[gpu/README.md](gpu/README.md) for installation, commands and the GPUs it has been
qualified on.

## Installation

The Julia package is not registered (Julia 1.10 or later):

```julia
using Pkg
Pkg.add(url="https://github.com/AntoineTilloy/RcmpsCore.jl.git")
```

The GPU package, from a clone of this repository:

```sh
git clone https://github.com/AntoineTilloy/RcmpsCore.jl.git
pip install "./RcmpsCore.jl/gpu[cuda12]"
```

## Quick start

```julia
using RcmpsCore
s = RandomState(8; seed=42, scaling=0.2)
s.model_param.name = "phi4"          # "phi4", "phi4_m", "shg" or "sg"
s.model_param.coupling = 2.0
energy, gradient = fg_phi4(s, 2.0)
result = optimize_state(s; method="LBFGS", n_iter=100, diagnostics=true)
result.converged                     # whether gradtol was reached within n_iter
rcmps_store(result.state, "result.jld2")
t = rcmps_load("result.jld2")        # recomputes and checks rho
```

Treat `s.R`, `s.K`, `s.Q` and `s.ρ0` as read-only: rebuild with `State(R, K)` or use
`retraction`, which update them together. `RandomStateZ2(D)` draws a Z₂-symmetric state (even D).

## Tangent-space excitations

```julia
model = ExcitationModel(:phi4, 2.0)      # same coefficients as fg_phi4(s, 2.0)
levels = excitation_spectrum(s, 0.5; model, nev=2, assembly=:action)  # dense, small D
r = excitation_eigsolve(s, 0.0; model)   # matrix-free Davidson, larger D
r.validated_energies[1]
gap = excitation_gap(s; model, solver=:iterative)
dispersion = dispersion_relation(s, 0:0.25:1; model, solver=:iterative)
```

`excitation_eigsolve` validates each level by independent routes; with
`solver=:iterative`, `excitation_gap` and `dispersion_relation` throw `SolverFailure` for
levels that fail. Only excitations above the same vacuum are covered
(no kinks between distinct vacua). See [docs/excitations.md](docs/excitations.md).

## Numerical settings

Check three errors separately: the half-window `max_point` and the Gauss nodes per half
`n_quad` of `s.integrator_param`, and its ODE tolerances `atol`, `rtol`. See
[docs/numerics.md](docs/numerics.md).

## Tests and benchmarks

```sh
julia --project=. -e 'using Pkg; Pkg.test()'     # several minutes
julia --project=. examples/phi4.jl
julia --project=. examples/excitations.jl
```

CI runs these tests and the CPU tests of `gpu/` on the GitLab mirror,
[gitlab.inria.fr/atilloy/RcmpsCore.jl](https://gitlab.inria.fr/atilloy/RcmpsCore.jl). Benchmark scripts are in
`benchmark/` (own environment), with results in [docs/benchmarks.md](docs/benchmarks.md).

## References

- A. Tilloy, *Relativistic continuous matrix product states for quantum fields without cutoff*, Phys. Rev. D 104, 096007 (2021), [arXiv:2102.07741](https://arxiv.org/abs/2102.07741).
- S. S. Pufu, B. T. Søgaard, S. Terlemes, *The Quasiparticle Ansatz for Relativistic Continuous Matrix Product States*, [arXiv:2610.13824](https://arxiv.org/abs/2610.13824).
- G. Beugnot, P. Gregory, R. Robin, A. Tilloy, *Well-conditioned iterative methods for large open quantum systems*, [arXiv:2608.30860](https://arxiv.org/abs/2608.30860).
- J. Haegeman, J. I. Cirac, T. J. Osborne, F. Verstraete, *Calculus of continuous matrix product states*, Phys. Rev. B 88, 085118 (2013), [arXiv:1211.3935](https://arxiv.org/abs/1211.3935).
- D. Draxler, J. Haegeman, T. J. Osborne, V. Stojevic, L. Vanderstraeten, F. Verstraete, *Particles, holes and solitons: a matrix product state approach*, Phys. Rev. Lett. 111, 020402 (2013), [arXiv:1212.1114](https://arxiv.org/abs/1212.1114).
- L. Vanderstraeten, J. Haegeman, F. Verstraete, *Tangent-space methods for uniform matrix product states*, SciPost Phys. Lect. Notes 7 (2019), [doi:10.21468/SciPostPhysLectNotes.7](https://scipost.org/10.21468/SciPostPhysLectNotes.7).

MIT license; see [LICENSE](LICENSE).
