# Full-energy optimization comparison at g=2

Quick fixed-budget comparison of the original and refactored phi4 optimization
pipelines. Each run starts from identical seeded random R and K matrices, with
R and K scaled by 1/sqrt(D), for D=2,4,8,16. These are independent random starts
at each dimension, not previously optimized checkpoints or bond-dimension growth.

Both use released OptimKit 0.5 LBFGS (history 20), the same Hager–Zhang line
search, canonical retraction, maximum 30 iterations, gradient tolerance 1e-5,
metric regulator 1e-3, and identical preconditioner settings. ODE tolerances are
atol=1e-11, rtol=1e-10, with physical half-window 20. Quadrature remains at each
implementation's default: 1000 exp-sinh steps in the original, 128 Gauss nodes
per half in the refactor.

The original pipeline includes its actual kinetic terms, fourth moment,
stationary-state solves, distributed dispatch, and forced garbage collection.
The refactor includes its corrected full gradient and checked stationary solver.
Both run on one Julia thread and one OpenBLAS thread with no extra workers.
This comparison standardizes optimizer controls rather than comparing different
optimizer defaults or the old nonlinear step cap.

Timings exclude a one-step compilation warmup for each pipeline, initial-state
construction, and independent final validation. They include all optimization
function/gradient evaluations, line searches, retractions, and stationary-state
recomputations during optimization. These are single-run wall times, not medians
across random seeds. All final energies and gradient norms are recomputed with
the same corrected implementation outside the timed region.

The CSV records the stopping status explicitly. Reaching the iteration limit
is not convergence; a speed ratio here compares the same iteration budget,
not necessarily time to a given physical accuracy.

Run from the repository root:

```sh
JULIA_LOAD_PATH='@:@stdlib' julia --project=benchmark benchmark/full_optimization.jl
```

`RCMPS_BENCH_STEPS` and `RCMPS_BENCH_DIMS` override the default budget and dimensions.
`full_optimization.csv` contains the summary; `full_optimization_D*_*.csv` contain
per-iteration histories. Environment details are in `environment.txt`.

## Results

| D | Old time | New time | Speedup at equal iteration count | Old final energy | New final energy | Old/new iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 30.0 s | 2.8 s | 10.82× | -0.028659814 | -0.028659812 | 30/30 |
| 4 | 150.9 s | 54.4 s | incomplete | -0.052587890 | -0.078305423 | 30/20 |
| 8 | 39.2 s | 14.3 s | 2.75× | -0.121277730 | -0.140013496 | 30/30 |
| 16 | 92.9 s | 41.7 s | 2.23× | -0.150157498 | -0.149491925 | 30/30 |

No run met the gradient tolerance of 1e-5. At D=4 the new implementation
stopped after 20 accepted iterations when a subsequent line-search trial hit
the ODE's 1,000,000-step limit; its reported 54.4 s includes 43.3 s spent on
that failed trial. The last accepted iterate was reached in 11.1 s. The old
implementation has a 10,000,000-step limit and does not check its ODE return
codes. The failure is retained as a reliability limitation, not counted as
an equal-budget speedup.

The new code is faster on all three completed pairs, but per-iteration energy
convergence is mixed: it reaches a lower energy at D=8, while the old code
reaches a slightly lower energy at D=16. These wall-time ratios do not establish
time-to-convergence speedups.

| D | Old corrected gradient norm | New corrected gradient norm | Old transfer gap | New transfer gap |
|---:|---:|---:|---:|---:|
| 2 | 1.689e-05 | 1.284e-04 | 2.670e+00 | 2.670e+00 |
| 4 | 1.357e+00 | 4.175e-01 | 9.869e-04 | 1.026e+00 |
| 8 | 1.769e-01 | 2.173e-01 | 4.415e-04 | 3.795e-01 |
| 16 | 7.083e-02 | 1.195e-01 | 2.079e-01 | 1.743e-01 |

The old D=4 trajectory stagnates near a transfer gap of 9.9e-4. Its own gradient
norm is 0.233 at the last step, whereas the corrected norm is 1.357. Its final
five iterations consume 112 additional energy/gradient evaluations with almost
no energy improvement. This is consistent with the omitted boundary response
becoming important at a small transfer gap.
