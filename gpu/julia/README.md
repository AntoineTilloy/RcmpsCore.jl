# Julia scripts for portable states

Scripts that move states between the Julia package (`rcmps_store` checkpoints, JLD2 schema 1)
and the portable state directories of `rcmps_gpu` ("rcmps-portable-state" format 1, specified in
the docstring of `gpu/src/rcmps_gpu/io.py` and summarized in `gpu/README.md`). They use only the
public API of RcmpsCore 0.2 and the LinearAlgebra, TOML and Random standard libraries, so they run
in the package's own environment, from the repository root:

```bash
julia --project=. gpu/julia/export_state.jl CHECKPOINT.jld2 NEW_DIRECTORY
julia --project=. gpu/julia/import_state.jl PORTABLE_DIRECTORY NEW_CHECKPOINT.jld2
julia --project=. gpu/julia/prepare_initial.jl D SEED COUPLING NEW_DIRECTORY [SCALING]
julia --project=. gpu/julia/check_portable.jl [FIXTURES_DIRECTORY]
```

- `export_state.jl`: a checkpoint as a format-1 directory: R, K, the stationary density of the
  loaded state, and, if the state was evaluated (finite `e0`), its energy, gradient and gradient
  norm; model, integration and manifold parameters and `info` from the checkpoint. The source
  recorded is the checkpoint's file name. The result is a valid `--initial` of
  `rcmps-gpu-optimize` and STATE of `rcmps-gpu-excitations`.
- `import_state.jl`: a portable directory (format 1, such as `optimized_state/` of a GPU run, or a
  directory of the research code) as a new checkpoint. A given `rho.bin` is checked as the
  stationary density of R and K. A state with `validation_passed = false` is refused; with a
  gradient and its `gradient_norm`, the metric norm is recomputed and must agree to 1e-9, and
  must not exceed `gradient_target` when `gradient_target_reached` is set. A checkpoint holds
  an energy `e0` only together with its gradient `grad` and the integration settings of both
  (readers take a finite `e0` to mean all three, and the default `grad` is zero), so the
  energy and gradient are imported only when the directory has an energy, a gradient and its
  integration settings. Otherwise `e0` stays NaN, and the energy, gradient norm and settings
  that the directory has are appended to `info`, with a warning: an export of the checkpoint
  then carries no energy.
- `prepare_initial.jl`: a random Z2 initial state reproducible in Julia (the matrices of
  `RandomStateZ2(D; seed, scaling)`, scaling 1/sqrt(D) by default, with the stationary density
  solved at tolerance 1e-13), for phi4 at the given coupling, as a format-1 directory for
  `rcmps-gpu-optimize --initial`. Nothing is evaluated.
- `check_portable.jl`: export and import of the shared fixture `test/fixtures/phi4_g2_D16_z2_state.jld2`
  in a temporary directory: the export reproduces the committed `test/fixtures/phi4_g2_D16_z2_portable`
  (made by `export_state.jl` of RcmpsCore 0.2.0; only the version in `producer` may differ), and
  importing either gives back the checkpoint exactly.

`portable.jl`, included by the scripts, reads and writes format-1 directories and reads the
research dialects as `rcmps_gpu.io.read_portable` does. None of the scripts overwrites an
existing file or directory.
