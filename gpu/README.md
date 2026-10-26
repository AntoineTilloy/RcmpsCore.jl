# rcmps_gpu

CUDA implementations of two algorithms of RcmpsCore, in Python with CuPy, for the φ⁴
model on relativistic continuous matrix product states: the ground-state optimizer for
Z2-symmetric states (even bond dimension D) and for states without symmetry (any D), the
excitation solver for Z2-symmetric states. Everything is float64/complex128.

The Julia package RcmpsCore in the repository root is the reference implementation (CPU,
every model) and is not modified by this one. This package accelerates its ground-state
optimizer and its tangent-space excitation solver for φ⁴ on Z2-symmetric states, and is
tested against it through shared fixtures and reference levels (`test/fixtures`). When the
two disagree beyond the tolerances documented here, the Julia result is the right one, with
one documented exception: the excitation solver evaluates the Julia formulas on a
stationary density re-solved in scaled variables, whose small eigenvalues are accurate to
full relative precision where those of the stored ρ that the Julia code uses are not (see
"Excitation solver: differences from the Julia code").

- `rcmps_gpu.optimizer`: ground-state optimizer. One persistent cooperative CUDA kernel
  integrates the whole adaptive DOP853 trajectory of the quartic term in FP64, its matrix
  products on the tensor cores from compute capability 8.0 on and in scalar FMAs below;
  L-BFGS on the manifold, as in the Julia code. The kernel is generated for each D: on the
  packed Z2 parity blocks (the default, every even D from 2 to 512) or, with
  `--symmetry none`, on full complex R and K (every D from 1 to 256), subject to the GPU
  holding the whole grid resident (see "Optimizer: states without Z2 symmetry").
- `rcmps_gpu.excitations`: tangent-space excitation solver (matrix-free Hamiltonian
  action, Davidson, validation by the bilinear route), both parity sectors, any momentum.
  The action trajectory of a Z2 sector runs in one fused cooperative kernel.
- `rcmps_gpu.io`: the state files of both: checkpoints of the Julia package and portable
  state directories.

The optimizer records the one-point function ⟨φ⟩ of its states (its history, final
validation and `optimized_state/`), with the normalization of RcmpsCore's
`phi_monomial(s, 1)`.

Status (details under "GPU qualification"): the package passed its qualification batch on
an A100, with the FP64 tensor cores and with the scalar tiles, and on a V100, with the
scalar tiles (2026-10-15). The excitation solver was qualified on an A100 up to D = 128
(with the sources this package imported, and earlier ones; with this package, D = 64 on
the A100 and D = 32 on the V100); at D = 16 and 32 its energies agree with the Julia code
to about 1e-11 on both GPUs, and above that its two independent routes (action and
bilinear) agree to 4e-10. The optimizer's runtime is the generic version of the H100
runtime of the research campaign: for the sizes that runtime supported (packed d = D/2 in
{2, 3, 4, 8, 16, 32, 48, 64, 128}) it generates the same CUDA source and compiler options,
bit for bit. The generic runtime itself ran short optimizations (3 and 30 iterations) at
D = 8 to 128 on both GPUs, with both tile paths on the A100, but no optimization to
convergence. The optimizer on states without symmetry (`--symmetry none`) passed its checks
on an H100 NVL with both tile paths (2026-10-26: trajectories, residency, runs at D = 8 to
64, the energy, gradient and ⟨φ⟩ of RcmpsCore at D = 4 to 32 to 2e-13, 3e-11 and 1e-14, and
optimizations at D = 16, 32 and 64), and its kernel ran on an RTX 2070. Both kernels choose
their tile product by device, for NVIDIA GPUs with or without FP64 tensor cores (see
"GPUs"); no other GPU has run this package.

## Installation

Python 3.11 or later. On a machine with an NVIDIA GPU of compute capability 6.0 or later
(Pascal and later; cooperative launch) and a driver for CUDA 12, from the repository root:

```bash
pip install "./gpu[cuda12]"     # with CuPy for CUDA 12 (cupy-cuda12x[ctk] ~= 14.2)
```

Both kernels compute their matrix products as 8 x 8 x 4 tiles, with the native FP64
tensor-core instruction `mma.sync` m8n8k4 (compute capability 8.0 or later) or with scalar
tiles of the same layout (warp shuffles and FP64 FMAs, any GPU). Both command lines choose
by device with one rule, `--tiles auto` (the default): tensor cores from compute capability
8.0 on, scalar tiles below; `--tiles mma` is refused below 8.0, `--tiles scalar` runs
anywhere (see "GPUs"). Keep CUDA 12 for Pascal and Volta GPUs (P100, V100): CUDA 13 no
longer supports them (the V100 was qualified with the CUDA 12.9 libraries of the CuPy
wheels, under a driver for CUDA 13.0). The qualification runs used Python 3.11, NumPy
2.4.6, SciPy 1.17.1 and CuPy 14.2.0.

For development:

```bash
pip install -e "./gpu[cuda12]"  # editable, on a GPU machine
pip install -e "./gpu[test]"    # CPU only: enough for the tests and for the CUDA source
```

Importing `rcmps_gpu` or its subpackages never imports CuPy; only a GPU run does. With
`-e` (editable), the installed package is the checkout itself: the tests test it, and a
run records in `plan.json` the git commit and whether tracked files of the package were
modified. A regular install copies the package into site-packages: the tests and runs then
use that copy, edits to the checkout take effect only after a reinstall, and `plan.json`
has no git commit (`package.git_commit` null), only the version.

## GPUs

Both kernels, the optimizer's trajectory kernel and the excitation solver's fused kernel,
compute every d x d product as 8 x 8 x 4 tiles in the fragment layout of the PTX
instruction `mma.sync.aligned.m8n8k4.row.col.f64`, in one of two ways:

- `mma`: that instruction, on the FP64 tensor cores (compute capability 8.0 or later);
- `scalar`: the same fragments moved to each lane by warp shuffles, and every element
  accumulated as a chain of FP64 FMAs in k order, on any GPU with cooperative launch
  (compute capability 6.0 or later).

The rule is the same in both command lines (`--tiles`, default `auto`): `auto` gives `mma`
from compute capability 8.0 on, any minor version, and `scalar` below; `--tiles mma` below
8.0 is refused before anything is compiled, with a message; `--tiles scalar` runs anywhere.
Everything else (layout, controller, FP64 arithmetic, tolerances) is the same. The PTX of
either path is audited: FP64 `mma.sync` for a target 8.0 or later on the tensor-core path,
no `mma` at all on the scalar path, no FP32 arithmetic on either. The optimizer audits its
kernel's PTX at every compilation; the excitation kernel's PTX is audited by the CPU tests
(NVRTC for compute_60 to 90) and on the device by the qualification (`gpu_bringup.py
--stages compile`), not by a run. On one GPU the two paths agree bit for bit wherever the
tensor cores compute the same chain of FMAs, which `gpu_optimizer_check.py --stages unit`
measures. The A100's do: its tensor-core tiles equal the chain of FMAs and the scalar tiles
in every case of that test (random tiles, a tie, a cancellation, a signed zero, a subnormal
product), and with either path both kernels gave the same bytes where the bytes were
compared (the optimizer's trajectories, first gradients and final states; the excitation
trajectories and bilinear terms) and the same values elsewhere (the optimizer's histories,
the excitation actions, levels and D = 64 solve; see "GPU qualification"). Across GPUs,
results agree to rounding: the V100 and the A100 differ by at most 6.7e-12 in the energies
along 30 optimizer iterations and 1.4e-12 in the validated excitation energies. The
resolved tiles are recorded (`compile.json`, the checkpoint and `final_validation.json` of
the optimizer; `settings.tiles` and the kernel occupancies in the excitation
`result.json`). The scalar path issues 24 shuffles and 8 FMAs per real tile where the
tensor cores issue one instruction: on the A100 the scalar tiles took 1.5 to 2.1 times as
long per optimizer trajectory at D = 8 to 64, 1.2 to 1.4 times as long per right-hand side
of the excitation kernel at D = 4 to 16 (1.5 to 2.5 times at D = 32 to 128, with earlier
sources), and 1.4 times as long for a D = 64 excitation solve. The V100 took 1.1 to 1.9
times the A100's tensor-core time per optimizer trajectory (0.8 to 1.0 times its scalar
time), and 1.3 to 1.5 times for the reference levels at D = 16 and 32.

| GPU | Compute capability | Tiles (`auto`) | Status |
|---|---|---|---|
| H100 | 9.0 | mma | the research runtime this one derives from ran the optimizer (D = 32 to 256); this package ran the optimizer on states without symmetry on an H100 NVL (2026-10-26, both tile paths; grid resident up to D = 144, refused at 152), not the Z2 optimizer or the excitation solver |
| A100 | 8.0 | mma | qualified with both tile paths (2026-10-15), which gave identical results: optimizer runs at D = 8 to 128 (3 and 30 iterations), grids compiled and resident up to D = 256; excitation kernels resident up to D = 128, reference levels at D = 16 and 32, a solve at D = 64; earlier, the excitation solver up to D = 128 (tensor cores, imported and earlier sources) and the research optimizer runtimes (D = 32 to 128) |
| A30, A10, RTX 30, L4, L40, RTX 40 | 8.0, 8.6, 8.9 | mma | compiled only (the tests compile for compute_80 and 86; compute_89 was checked once); FP64 at 1/64 of FP32 except the A30 |
| V100 | 7.0 | scalar | qualified with the scalar tiles (2026-10-15): optimizer runs at D = 8 to 128 (3 and 30 iterations), grid resident up to D = 160 and refused at 192 and 256; excitation action kernel resident up to D = 96 and refused at 128, pair kernel resident at D = 64 and refused at 96 (its limit estimated at 94 in the table below), reference levels at D = 16 and 32; `--tiles mma` refused |
| RTX 2070 | 7.5 | scalar | the optimizer on states without symmetry: its trajectories, the comparison with RcmpsCore and optimizations at D = 8 and 16 (2026-10-26) |
| T4, other RTX 20 | 7.5 | scalar | compiled only; FP64 at 1/32 of FP32 |
| P100 | 6.0 | scalar | compiled only |

"Compiled only" means: NVRTC 12.8 compiles both kernels to PTX for that target, both tile
paths where they apply, and the PTX passes the audit (the optimizer's kernel at d = 4, 64
and 128 in `tests/optimizer/test_tiles.py`, the excitation kernels at d = 3 and 32 in
`tests/excitations/test_fused.py`, compute_60, 70, 75, 80, 86 and 90); and ptxas, through
NVRTC, assembles them to a cubin (the optimizer's kernel and the excitation action kernel
at d = 8 for sm_60, 70 and 75 with the scalar tiles and for sm_80, 86 and 90 with the
tensor cores, the excitation pair kernel for sm_70 and 86, the optimizer's kernel also at
d = 128 for sm_70; sm_89 not), which PTX alone does not show: NVRTC writes FP64 `mma.sync`
for any target, and only ptxas refuses it below sm_80. Nothing has run there. The
qualification of a new GPU is `qualification/run_all.sh` (see "GPU qualification").

On slow FP64 GPUs one trajectory of the optimizer may exceed its device watchdog
(`--max-device-seconds`, 300 s), which stops the run cleanly at the first evaluation. A GPU
that drives a display may have a kernel run-time limit that kills the persistent kernels;
`compile.json` and the excitation `result.json` record it (`kernel_run_time_limit`).

Sizes are limited by what the GPU holds resident: both kernels are cooperative and need
their whole grid resident, which `--compile-only` (optimizer) and the start of an
excitation run check, refusing a size that does not fit before any numerical launch. The
grids (w = ceil(d/8), d = D/2; without symmetry d = D):

| Kernel | Threads per block | Thread blocks | Shared memory per block (bytes) |
|---|---|---|---|
| optimizer trajectory | 32 w | 10 w + 1 | 512 d + 16 w (dynamic from D = 192) |
| optimizer trajectory, `--symmetry none` | 32 w | 7 w + 1 | 512 d + 16 w (dynamic from D = 96) |
| excitation, action system | 32 w | 26 w | 768 d + 16 w |
| excitation, pair system (validation) | 32 w | 20 w | 1024 d + 16 w |

Largest even D whose grid is resident (any D without symmetry), estimated with the CUDA
occupancy calculator from the register counts of NVRTC 12.8 for each target (the occupancy
query of a run decides; CuPy's compilation can differ by a few registers, which moves a
limit by one even D). The optimizer columns include its register caps (the kernel without
symmetry has the registers of the packed kernel of the same matrix dimension: its column
follows from the same model, which reproduces the Z2 column exactly); the excitation columns
use the caps of `fused.REGISTER_CAPS`, and a run with validation needs both of its kernels:

| GPU | Multiprocessors | Optimizer: Z2 | no symmetry | Excitations: action kernel | pair kernel | run with validation |
|---|---|---|---|---|---|---|
| P100 | 56 | 126 | 63 | 64 | 62 | 62 |
| V100 | 80 | 190 | 95 | 96 | 94 | 94 (96 with `--host-bilinear`) |
| T4 | 40 | 112 | 63 | 56 | 62 | 56 |
| A30 | 56 | 176 | 120 | 106 | 106 | 106 |
| A100 | 108 | 256 | 128 | about 142 (208 of 216 blocks measured at D = 128) | 160 | 128 qualified, about 142 estimated |
| RTX 3090 | 82 | 194 | 97 | 96 | 96 | 96 |
| L4 | 58 | 176 | 97 | 84 | 80 | 80 |
| RTX 4090 | 128 | 194 | 144 | 130 | 96 | 96 |
| H100 SXM, H100 NVL | 132 | 256 | 144 | 160 | 208 | 160 |

The occupancy queries of the qualification batch agree with the V100 and A100 rows at the
sizes it checked: on the V100 the optimizer's grid is resident at D = 160 and refused at
192 (121 blocks, 80 resident), the action kernel resident at D = 96 (156 of 160 blocks) and
refused at 128, the pair kernel resident at D = 64 and refused at 96 (120 blocks, 80
resident at every cap); on the A100 every size is resident, up to D = 256 for the optimizer
and 128 for both excitation kernels.

Device memory does not limit these sizes on 16 GB cards (the largest buffers are the
optimizer's validation samples, 3 GB at D = 190); the excitation solver bounds its own with
`--basis-bytes`.

## Command line

Optimizer (`rcmps-gpu-optimize`, or `python -m rcmps_gpu.optimizer`):

```bash
rcmps-gpu-optimize --D 160                         # plan only, on the CPU
rcmps-gpu-optimize --D 160 --emit-cuda D160.cu     # generated CUDA source, on the CPU
rcmps-gpu-optimize --D 160 --compile-only --run-gpu --output RUN   # compile and resource check
rcmps-gpu-optimize --D 160 --seed 42 --coupling 2.7 --run-gpu --output RUN
rcmps-gpu-optimize --D 160 --resume-from RUN/optimization.json --run-gpu --output RUN2
rcmps-gpu-optimize --D 64 --symmetry none --seed 1 --coupling 3 --run-gpu --output RUN   # no symmetry
rcmps-gpu-optimize --D 64 --symmetry none --initial STATE --run-gpu --output RUN
```

Only `--D` changes between sizes. Output directories must be new. `--symmetry z2` (the
default) optimizes Z2-symmetric states, R odd and K even under the parity with the first
D/2 basis states even, on the packed parity blocks; `--symmetry none` optimizes states with
full complex R and K (see "Optimizer: states without Z2 symmetry"); a resume keeps the
symmetry of its checkpoint and must name it. A run starts from a random state (`--seed`: Z2,
or without symmetry) or from `--initial DIR`, a portable state directory (see State files
below): the `initial/` or `optimized_state/` of an earlier run, a checkpoint of the Julia
package exported by `gpu/julia/export_state.jl`, or a random state of
`gpu/julia/prepare_initial.jl`; a Z2 run refuses, when planning, an initial state whose R
and K are not exactly odd and even (`--symmetry none` takes any state, a Z2 one included).
The random start of `--seed` uses NumPy PCG64 complex Gaussian draws (scaling 1/sqrt(D))
and solves for the stationary density; it does not reproduce Julia's `RandomStateZ2` or
`RandomState`, whose draws `prepare_initial.jl` reproduces (`--symmetry none` for the
latter). Either way the run writes the state it starts from to `initial/`, with its
symmetry. If that state carries an energy (and
a gradient), the first evaluation must reproduce them, the energy to 1e-8 and the gradient
to 3e-6 in the metric norm, or the optimizer does not start.

The default target is a gradient norm of 1e-7 (metric norm), with at most 20,000 accepted
iterations per invocation. There is no default wall-time limit (`--max-seconds` sets one);
`--max-device-seconds` (300) is the watchdog of one quartic trajectory, not a limit on the
run. Reaching the target, the iteration limit, numerical stagnation, a numerical failure
and the time limit are distinct recorded outcomes. `--resume-from` continues a stopped
checkpoint of this runtime without resetting its L-BFGS pairs, next step, counters or
history; the objective controls are inherited and cannot change, and the original files
stay as they are.

Excitations (`rcmps-gpu-excitations`, or `python -m rcmps_gpu.excitations`):

```bash
rcmps-gpu-excitations STATE --output DIR [--momentum 0.0] [--nev 1] [--sector odd|even|all]
```

STATE is an `rcmps_store` checkpoint (.jld2) or a portable state directory. `--help`
describes every option. The defaults:

- **Problem:** odd sector for a Z2 state, p = 0, one level, φ⁴ at the stored coupling.
- **Discretization:** window 30, 512 quadrature nodes, ODE tolerances 1e-13/1e-12; kinetic
  nodes `eigsolve.default_kinetic_nodes(D)`, 256 up to D = 64 and 512 beyond (see
  "Kinetic nodes at large D"); 8 preconditioner nodes; solver tolerance 1e-14; scaled
  variables.
- **Validation:** on; refined settings window 40, 1024 nodes, tolerances divided by 10,
  twice the kinetic nodes.
- **Davidson:** residual target 1e-7, at most 200 actions, and the schedule of the A100
  mass and sector runs (the Davidson part of the Julia D ≤ 48 research configuration,
  `eigsolve.research_davidson_options`): for a sector of dimension n, a search space of
  min(160, n) vectors with thick restarts keeping min(max(12, nev), maxspace - 1) Ritz
  vectors. `--fresh-restarts` selects the library schedule of the Julia code instead
  (search space max(8, 2 nev + 2), retain nev, every restart recomputes the actions), and
  `--maxspace` and `--retain` override either schedule. The stall rule is the
  `DavidsonOptions` default (`--stall-residual-limit` max(1e-5, 10 tol),
  `--stall-energy-atol` 1e-9).

`--backend fused` (the default) integrates one parity sector of a Z2 state per launch with
the tile products of `--tiles` (auto by default: see the fused kernel below), so
`--sector all` needs `--backend host` (host-controlled integrator on the device, slow) or
`--backend numpy` (the same code on the CPU). `--basis-bytes` and `--chunk` bound the GPU
memory of the batched solves (1.0e10 and 32 were used at D = 128 on 40 GB). The output
directory receives `result.json` (energies, the four validation energies, history,
timings, solver and kernel statistics, source hashes, and `settings`: the options with the
values used), `vectors.npz` (white coefficients and tangents W in the gauge of the state),
`log.jsonl`, and during the search `checkpoint.npz` (current Ritz vectors after every
action). `--start FILE` restarts from either file for the same state and sector, so a
failure after the search costs one action; the white coefficients refer to the
eigenvectors of ρ computed in that run, so reuse a checkpoint in the same software
environment.

From Python the entry points are `rcmps_gpu.excitations.eigsolve.excitation_eigsolve(state,
p, ...)` and `iterative_dispersion` (a loop over momenta with warm starts). They resolve
`kinetic_nodes=None` to `default_kinetic_nodes(state.D)`, but their Davidson default stays
the library schedule of the Julia code; the command line's schedule is
`davidson_options=eigsolve.research_davidson_options(eigsolve.sector_dimension(state), nev)`
(with the same `sector`, `parity`, `gauge` and `metric_tol` in both calls).

## Optimizer: sizes, devices and checks

The packed Z2 representation requires even D. The row layout generates every even D from
2 to 512; this is a code-generation range, not a promise that every size fits every GPU.
The whole cooperative grid must be resident: shared memory, registers and the device's
thread-block limits can impose a smaller maximum, and compilation refuses an unsafe grid
before the trajectory sample buffers are allocated (`--compile-only` checks a size on the
target GPU; without `--run-gpu` the plan gives the size of those buffers). Panels larger
than 48 KiB use dynamic shared memory, and resource use and spills are reported. Up to
D = 128 the kernel compiles without a register cap (about 100 to 250 registers per thread
with NVRTC 12.8, depending on the target, the tiles and D), so a GPU with fewer or smaller
multiprocessors than the A100 may hold too few blocks: the worker then recompiles with the
caps 128, 96 and 64 in turn and takes the first whose grid is resident (spills grow from
about 50 bytes at cap 128 to 1.5 KB at cap 64), recording the cap and every attempt in
`compile.json`; the run, its qualification and its validation use that cap (on the V100 at
D = 128: 168 registers hold 80 of the 81 blocks, cap 128 holds 160 with 48 bytes of
spills). Above D = 128 the historical cap 64 is the only one. On the A100 the first attempt
was resident at every size the qualification compiled (D = 8 to 256), and on the H100 it
is expected to be, so their compiler options stay the historical ones. There is
no automatic slower backend, reduced precision or relaxed tolerance. Sizes above 512 need
another row layout.

The tile products of the kernel follow the device (`--tiles`, recorded in `plan.json`):
`auto`, the default, uses the FP64 tensor cores (`mma`) on compute capability 8.0 or later
and the scalar tiles (`scalar`) below; `--tiles mma` is refused below 8.0 before anything
is compiled, and `--tiles scalar` runs anywhere, on an A100 or H100 too. The scalar tiles
compute each element as a chain of FMAs in k order. Where the tensor cores compute the same
chain, both paths agree bit for bit on one GPU (on the A100 the two paths gave the same
trajectories and final states, byte for byte, and the same histories), and across GPUs to
rounding. The worker records the resolved tiles in `compile.json`, the checkpoint
(`trajectory_controls`, for every run) and `final_validation.json`; a resume of a
checkpoint may use other tiles, since it forks. The tensor-core path generates the source
of `golden_cuda.json`, unchanged; the scalar path differs only by the tile function
(`tests/optimizer/golden_cuda_scalar.json`).

Before optimizing, a GPU run compares the fast energy and gradient, and their raw terms,
with an independent CuPy propagation and the baseline kinetic solver at the initial state;
a failure keeps the raw data and stops before the optimizer starts. Compilation audits the
PTX: FP64 `mma.sync` instructions for a target of compute capability 8.0 or later on the
tensor-core path, none on the scalar path, and no FP32 arithmetic on either. After stopping,
the five-row consistency validator re-evaluates the final state (three standard
evaluations, two refined ones) and, if it passes, writes `optimized_state/` with the energy
and gradient of the finest row (window 40, 1024 nodes, 1e-14/1e-13);
`gpu/julia/import_state.jl` turns it into a checkpoint. Reports keep validation separate
from convergence, and failed or partial results are kept. A run directory holds the plan,
the compiler, resource and PTX report, the initial qualification, the complete optimizer
history (JSON and NPZ), the final consistency report and, after a pass,
`optimized_state/`. These are consistency checks, not an independent error certificate
for the gradient, nor a guarantee that the target can be reached.

## Optimizer: states without Z2 symmetry

`--symmetry none` optimizes φ⁴ states with full complex R and K, any D from 1 to 256: the
symmetry-broken states of the ordered phase, whose one-point function ⟨φ⟩ is not zero. The
optimizer is the same (stationary density, kinetic resolvent with the cooperative Arnoldi,
boundary Poisson problem, metric, preconditioner, retraction, L-BFGS and its controller,
checkpoints, initial qualification, five-row validation), none of which assumes a symmetry;
only the quartic trajectory changes. Its seven Hermitian channels (F1 to F3, from ρ, and
the four adjoint channels, the fifth being I), which the Z2 kernel packs in ten blocks of
dimension D/2, are integrated unpacked: seven D x D channels and the scalar accumulator, the
equations of the host-driven `--reduced` objective of the research code
(`ode_backend._ReducedQuarticRHS`). The kernel is the packed one with three changes, made by
anchored rewrites of its source (`persistent_trajectory_backend.UNPACKED_REWRITES`): seven
channels of row blocks, the gather offsets of the insertions (each channel takes the product
with its own neighbour in the hierarchy), and the scalar, 4 j(t) Re tr(R F3 + F3 R†), which
does not assume F3 Hermitian. The controller, dense output, tile products and resource checks
are those of the Z2 kernel; the runtime drivers select the layout by their objective control
`--reduced` in place of `--packed-z2`, so a checkpoint records it and a resume cannot change
it. Per right-hand side the unpacked kernel computes 7 x 6 products of dimension D against
10 x 6 of dimension D/2: 5.6 times the arithmetic of a Z2 state of the same D, on a grid of
7 ceil(D/8) + 1 blocks, which sets the sizes of the table under "GPUs" (D = 128 on the A100,
144 on the H100, 95 on the V100). The pullback is chunked from D = 128 (the packed one from
D = 192), and the validation samples at the refined settings take 16 bytes for each of
2049 nodes of 7 D² + 1 complex values), 3.8 GB at D = 128.

The one-point function is that of RcmpsCore's `phi_monomial(s, 1)`: tr F1(a) for the first
monomial, F0 = ρ and dF1 = 2|t| L(F1) dt + j(t) (R F0 + F0 R†) dt over the window [-W, W],
which is exactly ⟨φ⟩ = c(W) tr(Rρ + ρR†) = 2 c(W) Re tr(ρR), since L preserves the trace;
c(W), the integral of the source J over the window, is 2^-1/2 minus a tail of 2.3e-15 at
W = 30 (`ode_backend.source_integral`). `ode_backend.phi_expectation` evaluates it; every
accepted row of the history records it (`phi`, at the run's window) and the checkpoint the
last one (`final_phi`); the validation records it for every row and, at the refined
settings, also by the ODE of the first monomial (`phi_ode`, the route of `phi_monomial`,
host-driven: 24 s on the H100 at D = 32; it agreed to 3e-13 or better), and
`optimized_state/` holds it as `phi` with `symmetry = "none"`.
Its sign is that of the branch the optimizer reached: R and -R are the two broken states.

Validation against RcmpsCore (`qualification/optimizer/check_nonz2.py`, Julia references of
`qualification/optimizer/nonz2_references.jl`: `fg_phi4` at the optimizer's settings, window
30, 512 quadrature nodes, 1e-13/1e-12, Krylov kinetic with 512 angular nodes, and
`phi_monomial(s, 1)`), φ⁴ at g = 3, the GPU objective of a run (`worker.qualify`) with R, K
and ρ as stored:

| State (φ⁴, g = 3) | D | Julia gradient norm | GPU - Julia: energy | gradient (metric norm) | ⟨φ⟩ |
|---|---|---|---|---|---|
| `RandomState(D; seed=1, scaling=1/sqrt(D))` | 4 | 75 | +4.3e-14 | 9.7e-12 | +8.7e-15 |
| | 5 | 12 | +2.1e-14 | 1.9e-12 | +8.3e-16 |
| | 8 | 14 | -5.3e-14 | 2.4e-11 | -1.3e-15 |
| | 16 | 11 | -1.9e-14 | 2.4e-11 | -9.1e-15 |
| | 32 | 8.8 | -1.6e-13 | 2.6e-11 | +4.2e-16 |
| iterates 5, 15, 25 of a Julia optimization | 8 | 0.81, 1.0, 0.51 | +1.8e-15, +9.6e-15, -2.3e-15 | 2.7e-11, 2.0e-10, 9.5e-11 | |
| Julia minimum (216 iterations) | 8 | 7.4e-9 | -2.7e-14 | 1.5e-9 | -1.3e-14 |
| Julia minimum (207 iterations) | 16 | 9.8e-8 | +7.9e-14 | 2.6e-9 | -6.5e-14 |

These ran on an RTX 2070 (compute capability 7.5, scalar tiles) on 2026-10-26; on an H100
NVL (tensor cores) the differences are the same, at most 1.6e-13, 2.6e-11 and 1.3e-14 on the
random states and iterates, 2.9e-13 and 2.6e-9 at the minima. Up to the
minima the GPU is as close to Julia as the energy can be resolved, and its gradient
agrees to 1e-11 relative. At the two minima the gradient difference, 1.5e-9 and 2.6e-9, is
the floor of the two quartic integrations (Julia's Vern7 and the GPU's DOP853, both at
1e-13/1e-12; 1024 kinetic nodes instead of 512 change Julia's gradient by 1.3e-12 and
5.8e-12); the fast objective and the independent CuPy oracle of the qualification agree to
3e-13 in energy and 6e-12 in gradient on every state. Julia's ⟨φ⟩ by the ODE of
`phi_monomial` and its closed form agree to 7e-14.

Optimizations, on the same GPU. From the start of a Julia optimization (D = 8,
`RandomState(8; seed=4)`), the first iteration has Julia's energy to 12 digits; then the
line searches differ (the GPU's bounded Armijo backtracking, Julia's Hager-Zhang search)
and after 30 iterations the energies are -0.33751 (GPU) and -0.32314 (Julia). Julia
re-evaluates the GPU's iterates 10, 20 and 30 within 5e-15, 4e-14 and 1.4e-14 of the
energies the GPU recorded (a run is deterministic: the runs of 10 and 20 iterations repeat
the first iterations of that of 30 bit for bit). From the start of the D = 16 Julia
optimization (`RandomState(16; seed=3)`) the GPU stopped after 175 iterations (313 s) at
numerical stagnation with gradient norm 3.8e-6 (target 1e-7, which Julia reached in 207):
energy -0.388678942012248 (refined validation), 5.4e-12 above Julia's minimum
-0.388678942017663, and ⟨φ⟩ = 0.6172540552 against Julia's 0.6172521147 (1.9e-6, the
first-order effect of the residual gradient, while the energy is second order); Julia's
evaluation of the GPU's state agrees with it to 3e-13 in energy and 9e-14 in ⟨φ⟩. The
stagnation is that of the optimizer's controller, the same for Z2 states (see the D = 64
runs under "GPU qualification"): near 1e-6 the L-BFGS steps need up to 13 backtracking trials.

The CPU tests (`tests/optimizer/test_unpacked.py`) check the source rewrite (golden hashes
in `tests/optimizer/golden_cuda_unpacked.json`), that the kernel's tables evaluate the host
equations, that these equal the full ten-channel system built from the observable's
specification (the route of RcmpsCore's `evaluate`) to 1e-12 and, on a Z2 state, the packed
channels, the chunked pullback, ⟨φ⟩ against the ODE of the first monomial, the planner and
the random start; with NVRTC, compilation, PTX audit and assembly for compute capability 6.0
to 9.0; `tests/optimizer/test_worker_cpu.py` runs a D = 5 optimization end to end on the CPU.

## Excitation solver: differences from the Julia code

The Hamiltonian action F(W), the bilinear cross-check, the preconditioner, the field seed
and the Davidson iteration follow the Julia code (`src/excitation_*.jl`, `src/davidson.jl`;
mathematics in `docs/excitations.tex`) formula by formula. Nothing uses a D² x D² transfer
matrix, Schur form or LU factor, so memory and time are set by D x D products. There are
three structural changes. The first two leave the result unchanged; the third keeps the
formulas and changes their conditioning, and with it the density they are evaluated on:

1. **Matrix-free solves** (`scaled.py`). Every resolvent `(sigma - z L) X = B` and every
   pinned Poisson problem is solved by right-preconditioned GMRES on a whole batch of
   independent systems at once (all theta nodes, source operators and tangents). The
   preconditioner is the exact inverse of the no-jump part `X -> sigma X - z (Q X + X Q')`,
   diagonal in the eigenbasis of Q. A solve takes 30 iterations on average at D = 64-128
   (up to 55) for a relative residual of 1e-14. The spectral test of the Julia context (a
   unique, mixing transfer fixed point) is one such solve, of a generic right-hand side on
   the pinned operator (`eigsolve.check_mixing`): a reducible state is refused before the
   first action.
2. **Tangent-independent halves are computed once** (`kinetic.py`, `potential.py`). In the
   resolvent quadrature and in the ODE, the ground-state halves do not depend on W.
   Boundary terms are linear, so one shifted dual Poisson solve serves all terms.
3. **Scaled variables** (`scaled.py`). With S = sqrt(ρ) in the density gauge, each matrix
   index is kept either in the stored left-canonical gauge or rescaled by S, so that every
   intermediate is O(1): a density x is stored as xh with x = S xh S, a tangent as
   Wh = W S, and so on. The right fixed point becomes close to 1 and no result is divided
   by a small Schmidt value. `--scaling none` (S = 1) evaluates the Julia formulas
   literally; both modes are the same code.

The third point matters at large D. With the Julia formulas the white-coordinate action
carries Float64 roundoff amplified by 1/λ_min(ρ); this is the residual floor near 1e-6 of
the D ≥ 56 CPU runs. On the D = 64, g = 2.7 CPU checkpoint, with the library Davidson
schedule (`--fresh-restarts`) and solver tolerance 1e-12, `--scaling none` stalls at
residual 2.1e-6 after 146 actions, as the CPU run did, and the scaled variables converge
to 8.8e-8 in 124.

It also changes which density the formulas see. The Julia code uses the stored ρ, whose
small eigenvalues are known to a relative accuracy of about eps x cond(ρ). The scaled route
re-solves the fixed point in the scaled variables (`fixed_point_residual` in the log),
which determines those eigenvalues to full relative accuracy, and both the action and the
bilinear route use it. The two therefore differ by up to about eps x cond(ρ) times the
weight of the vector on the small-ρ columns, and a comparison with Julia at large D cannot
be better than that. On crafted D = 4 states with cond(ρ) = 1.3e7, 1e10 and 6e12, one
bilinear energy of the scaled route agrees to 1e-15 (relative) with the Julia formulas
evaluated on the refined density and to 2e-14, 2e-13 and 5e-12 with the same formulas on
a 50-digit stationary density, while the stored density gives 9e-12, 2e-8 and 3e-6. On
the optimized states used here the effect is small: at D = 64 (cond 5.6e10) the
field-seed energy is 0.06920762511 scaled, 0.06920762518 with the Julia formulas on the
GPU, and 0.06920762547 on the CPU.

Modules of `rcmps_gpu.excitations`:

| Module | Role |
|---|---|
| `states`, `models` | states (loaded through `rcmps_gpu.io`), φ⁴ terms, grids, source kernel |
| `coordinates`, `davidson` | white tangent coordinates and Davidson: line-by-line ports |
| `scaled` | scaled variables, typed Lindblad maps, batched preconditioned GMRES |
| `kinetic` | kinetic and quadratic terms by the theta resolvent quadrature |
| `potential`, `ode` | ODE terms: equations, pullback, host-controlled DOP853 (reference path) |
| `packed` | Z2-packed, reduced action ODE as term tables (26 blocks for φ⁴) |
| `fused` | fused cooperative FP64 kernel that integrates a packed system in one launch |
| `hamiltonian` | the complete action F(W) |
| `bilinear`, `packed_pair`, `bilinear_fused` | bilinear energies: reference route, packed diagonal system, fused route |
| `eigsolve` | operator in white coordinates, preconditioner, seed, Davidson driver, validation, defaults (kinetic nodes, Davidson schedules) |
| `cli` | command line |

The fused kernel (`fused.py`) runs the whole adaptive DOP853 trajectory of the action
(forward and adjoint together, as in the optimizer) inside one cooperative kernel: step
controller, twelve stages, error norm and dense output at the quadrature nodes, with no
host synchronization per step. It reuses the controller arithmetic, source approximation
(the Chebyshev table of `optimizer/runtime/jax_source.py`), thread-to-tile mapping and
`mma.sync` m8n8k4 FP64 products of the optimizer's trajectory kernel. The right-hand side
is table driven: each packed block has a list of d x d products in a jac group, a source
group and two-sided groups (`packed.py`), so one kernel serves both sectors, any momentum,
both scalings, and the bilinear system (`packed_pair.py`, three two-sided groups). Stages
alternate between two buffers, which removes one grid barrier per stage. The scalar tiles
(`--tiles scalar`, or `--scalar-tiles`; `use_mma=False`) replace the tensor-core tile
product by warp shuffles and FMAs with the same layout; on the A100 the two gave the same
trajectories, byte for byte, with the same steps. `--tiles auto` (`use_mma=None`, the
default) chooses by device as the optimizer does, and `result.json` records the tiles that
ran (`settings.tiles`, and per kernel in `kernels`). The φ⁴ action system has 26 packed
blocks and 190 products per right-hand side (the optimizer's ground-state system: 10
blocks, 60 products); the bilinear system of the validation has 20 blocks and needs more
shared memory per block (1024 d + 16 w bytes against 768 d + 16 w, w = ceil(d/8)), so it
can fail to be resident where the action kernel is. The command line therefore compiles both kernels, and checks their cooperative
grids, before the output directory is created (`fused.preflight`): a GPU that cannot hold
one is refused at once, naming it, and `--host-bilinear` validates without the second one.

Three settings matter for accuracy:

- **Solver tolerance.** An energy inherits about tolerance x ||H||. At 1e-12 the D = 64
  mass is off by 1e-8; the default 1e-14 costs about 15% more iterations. GMRES stops by
  itself when it reaches the roundoff level of a system (largest attained residual 1e-14
  at D = 16, 9e-14 at D = 64, 2e-13 at D = 96, 4e-13 at D = 128) and accepts residuals up
  to max(20 tol, 1e-11); the attained residuals are in `result.json`
  (`solver.residual_by_tolerance`).
- **Kinetic nodes.** The Julia default of 256 nodes is converged up to D = 64 only, hence
  the default of 512 beyond (`eigsolve.default_kinetic_nodes`, used by the command line and
  by the entry points of `eigsolve.py`: `excitation_eigsolve`, `iterative_dispersion`,
  `solve_levels`, `ExcitationOperator`; the building blocks `hamiltonian.HamiltonianAction`
  and `bilinear.BilinearElement` keep Julia's 256). See "Kinetic nodes at large D".
- **Davidson schedule.** The library schedule (search space 8, every restart recomputes
  the actions) is safe against drift but needs 124 actions at D = 64 where one unrestarted
  space needs 44. With scaled variables the stored actions stay consistent with fresh ones
  (fresh and recombined residuals agree to better than 1% up to D = 96), so the command
  line's default keeps up to min(160, n) vectors and recombines the stored actions at a
  restart; none of the runs under "GPU qualification" needed one.

## Kinetic nodes at large D

Measured on an A100 with `qualification/excitations/gpu_settings_check.py`, with sources
earlier than those this package imported (see "GPU qualification"): the two parts of the
action of a converged vector, at each setting, against the finest one (1024 kinetic nodes;
2048 window nodes, window 40, ODE tolerances 1e-14/1e-13), on the D = 16 fixture and the
g = 2.7 GPU-optimizer states at D = 64, 96 and 128 (at D = 64 not the CPU checkpoint of the
mass runs). Each part has norm 5 (D = 16) to 21 (D = 128), and
at D = 128 the two energies cancel from +1.29 and -1.26 to the mass 0.035.

| D | Resolvent part, 256 nodes: action, energy | 512 nodes: action, energy | ODE part, standard settings: action, energy |
|---|---|---|---|
| 16 | 4.0e-12, 1.2e-12 | 5.4e-12, 6.8e-13 | 1.0e-10, 2.4e-11 |
| 64 | 9.6e-11, 1.9e-12 | 5.8e-11, 3.4e-12 | 5.2e-10, 3.7e-12 |
| 96 | 2.1e-7, 4.8e-12 | 1.0e-10, 1.8e-11 | 1.7e-9, 2.0e-11 |
| 128 | 1.2e-5, 7.0e-10 | 3.1e-10, 2.3e-11 | 5.3e-9, 9.0e-11 |

The Julia default of 256 kinetic nodes is converged up to D = 64 and fails abruptly
beyond: at D = 128 a vector converged with 256 nodes has a residual of 1.2e-5 under 512
nodes, above the acceptance limit of 1e-5. 512 nodes are converged to 3e-10 up to
D = 128, at 4 s more per action there (8.6 s instead of 4.4 s for the resolvent part), so
`default_kinetic_nodes(D)` is 256 up to D = 64 and 512 above (with twice as many for the
refined check of the validation). The window settings of the ODE part are adequate at
every D. Nothing was measured beyond D = 128; the command line notes it there, and warns
when `--kinetic-nodes` is below the default.

## State files

Two formats connect this package with the Julia one. `rcmps_gpu.io` reads the states of
both subpackages and writes those of the optimizer:

- `read_jld2`: an `rcmps_store` checkpoint of the Julia package (JLD2 schema 1 of
  `src/storage.jl`), read with h5py; this direction only. `mass` is read when present (1.0
  otherwise); a φ⁴ state with another mass is refused. `rho` is not stored, and is
  recomputed.
- `read_portable`, `write_portable`: a portable state directory, "rcmps-portable-state"
  format 1, the format exchanged with Julia in both directions. `metadata.toml` holds
  `format = "rcmps-portable-state"`, `format_version = 1`, `D`, `model`, `coupling` (real),
  and optionally `mass` (default 1.0, and 1.0 for φ⁴), `symmetry` (`"Z2"` or `"none"`: the
  symmetry the producer imposed), `energy`, `gradient_norm`, `phi` (⟨φ⟩ as `phi_monomial(s,
  1)` evaluates it), `validation_passed`, `producer`, `source`, `info`, an `[integration]` table (`n_quad`,
  `max_point`, `atol`, `rtol`, optional `maxiters`: the settings of the energy and
  gradient) and a `[manifold]` table (`metric_reg`, `precond_reg`, optional
  `precond_reg_prefactor`). `R.bin` and `K.bin` are required, `rho.bin` (recomputed when
  absent) and `gradient.bin` optional: each a headerless D x D complex128 matrix,
  little-endian, column-major, exactly 16 D² bytes. Readers ignore other keys and files, so
  `symmetry` and `phi`, added after rcmps_gpu 0.1.0, leave the format version at 1. The
  module docstring is the specification.

`read_portable` also reads the unversioned directories of the research code (the
optimizer's former `initial/` and `optimized_state/`, the Julia exports of the research
tree; phi4): `quadrature`/`n_quad`, `window`/`half_window`, `energy`/`saved_energy` and
`gradient.bin`/`saved_gradient.bin` are mapped as the docstring describes. Values read from
the `saved_*` keys get the settings of their `[saved_integration]` table, and none (unknown)
in the earlier exports that have no such table, whose top-level settings are those of the
GPU run, not of the saved values. The optimizer writes `initial/` and `optimized_state/` in format 1; `initial/` has besides the files and
keys that the runtime reads (`saved_gradient.bin`, the `saved_*` keys, `precond_reg`).

On the Julia side, the scripts of `gpu/julia/` (see its README) export a checkpoint as a
format-1 directory, import one as a checkpoint, and prepare a random initial state with
Julia's random draws; they use only the public API of RcmpsCore. A checkpoint holds an
energy only with its gradient and integration settings, so a directory that lacks one of
the three is imported without energy (`e0` NaN), what it has being recorded in `info`. The shared fixture
`test/fixtures/phi4_g2_D16_z2_portable` is the export of `phi4_g2_D16_z2_state.jld2`:
`tests/test_io.py` checks that Python reads the same state from both, and
`gpu/julia/check_portable.jl` that the export and import reproduce the checkpoint.

## Reference levels

`test/fixtures/excitation_references.toml` holds excitation levels shared with the Julia
package. Each `[[levels]]` entry is one level of one solve: a state of `test/fixtures`
(checked by sha256), a `[settings]` table, sector, momentum and number of levels, the Julia
validated energy, four validation energies and Ritz energy, and the tolerances a
recomputation must meet; the D = 16 and D = 32 entries also hold the values of the A100 runs
(`gpu_*`). The file's `about` and `[fields]` describe every key.
`benchmark/excitation_references.jl` recomputes the regenerable Julia levels (D = 4 and
D = 16) with RcmpsCore and checks them, or with `--update` rewrites them. Here:

- `tests/excitations/test_references.py` runs the four D = 4 solves (a random Z2 state,
  `nev = 2`, both sectors, p = 0 and 0.4) with the NumPy backend in about a minute; they
  agree with Julia within 1e-12, and the tolerance is 1e-11. With `RCMPS_EXCITATION_SLOW=1`
  it also runs the three D = 16 solves at the settings of the A100 runs (about 9 minutes
  each on one thread; validated energies within 2.4e-12 of Julia's, tolerance 1e-10).
- `qualification/excitations/check_references.py` runs the D = 16 and D = 32 levels through
  the command line on a GPU and prints, as JSON lines, their differences from the Julia
  values (with the tolerances) and from the A100 values.

The recorded mass of the research mass scan at D = 16 (level `mass_scan_D16_g2`), which
`tests/excitations/test_eigsolve.py` and `test_stored_references.py` compare with, is read
from the same file.

## GPU qualification

What has run on which GPU, summarized; the detailed run records (logs, result files,
environments) are kept outside the repository. All runs used Python 3.11, NumPy 2.4.6,
SciPy 1.17.1 and CuPy 14.2.0, φ⁴ at g = 2 (the D = 16 fixture) or g = 2.7.

To qualify a GPU, `bash gpu/qualification/run_all.sh NEW_DIR` (from the repository root,
with the package and CuPy installed) runs, with the tiles the device selects and, from
compute capability 8.0 on, with the scalar tiles forced: `gpu_optimizer_check.py` (the
optimizer's tile product, trajectories, compilation and residency per D, short runs at
D = 8 to 64, and 96 and 128 when resident, and 30 iterations at D = 16 and 32);
`gpu_bringup.py --stages device compile` (both excitation kernels per D: compilation,
residency, PTX audit); `gpu_fused_check.py`, `gpu_pair_check.py --reference` and
`gpu_action_check.py --reference` (the fused kernel, the bilinear term and the complete
action against the host integrator and the CPU equations, on random states and the D = 16
fixture); `check_references.py` (the reference levels of `excitation_references.toml`
through the command line, against Julia); with `RCMPS_STATE_D64`, a D = 64 solve; and
`gpu_optimizer_check.py --symmetry none` (the kernel of states without symmetry: trajectories
against the host oracle, compilation and residency per D, short runs at D = 5 to 64). It
does not run `gpu_settings_check.py` (the kinetic-node study) or the solver, kinetic and
host-integrator stages of `gpu_bringup.py`. Every script exits with status 1 when a check
fails: an error, a failed compilation or PTX audit, a repeated launch whose result is not
the same bytes (or, for the complete action, differs by more than 1e-9), a difference from
the host integrator or the CPU equations above 1e-9 (relative), a level outside its
tolerance or not accepted (`check_references.py`; the D = 64 solve fails unless its level
is accepted); a size or tile path the GPU cannot hold or run is reported, not failed. Each
step writes one log of JSON lines, and `NEW_DIR/status.tsv` lists the exit status and time
of every step; the script's header lists the steps.

**Qualification batch, A100 and V100, 2026-10-15.** `run_all.sh` at the commit "README:
exact cubin coverage, failed solves, test times" (the device log of each machine records
it, with no tracked file modified); every step exited with status 0. The machines: an A100
40 GB (compute capability 8.0, 108 multiprocessors; the CUDA runtime names it
A100-SXM4-40GB, nvidia-smi A100-PCIE-40GB), every check with the tensor cores (`auto`) and
again with the scalar tiles forced, and the D = 64 solves; a Tesla V100-SXM2-32GB (7.0, 80
multiprocessors), every check with the scalar tiles (`auto`), and `--tiles mma` checked to
be refused. Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, CuPy 14.2.0 with the CUDA 12.9
runtime libraries of the `cupy-cuda12x[ctk]` wheels, drivers for CUDA 13.2 (A100) and 13.0
(V100). A preliminary batch the same day, with the same package sources and the earlier
`gpu_optimizer_check.py` (the commit "Correct the register range of register_caps for every
packed d <= 64"), passed every step it ran: all of them on the V100; on the A100 steps 01
to 11, after which it was stopped during the scalar reference levels (three of the six done,
all accepted), before the D = 64 solves.

The optimizer, with `gpu_optimizer_check.py` (the random Z2 state of seed 0, g = 2.7, the
integration settings of the command line). The tile product alone: on the A100 the
tensor-core tiles equal the host chain of FMAs and the scalar tiles, bit for bit, in every
case (64 random tiles with spread exponents, a tie, a cancellation, a signed zero, a
subnormal product); on the V100 the scalar tiles equal the chain in every case. The
trajectories at D = 8 to 64: the same accepted and rejected steps on both GPUs and with
every path, a repeated launch gives the same bytes, and at D = 8 the host oracle takes the
same 271 attempts, its samples within 1.1e-15 (A100) and 1.2e-15 (V100) relative; on the
A100 the two paths gave the same bytes. Device time per trajectory (ms):

| D | Steps (accepted, rejected) | A100, tensor cores | A100, scalar tiles | Scalar / tensor cores | V100 |
|---|---|---|---|---|---|
| 8 | 267, 4 | 28.6 | 47.2 | 1.65 | 38.1 |
| 16 | 344, 4 | 50.1 | 74.8 | 1.49 | 56.8 |
| 32 | 305, 4 | 44.5 | 83.0 | 1.87 | 84.5 |
| 64 | 313, 5 | 68.8 | 144.2 | 2.09 | 130.9 |

Compilation and residency (`--compile-only`): registers per thread (with the register cap
and the bytes spilled, where there are any) and the thread blocks resident, against the
blocks of the grid. Every resident kernel passed the PTX audit (FP64 `mma.sync` on the
tensor-core path, none on the scalar path, no FP32 arithmetic on either).

| D | Blocks | A100, tensor cores | A100, scalar tiles | V100 |
|---|---|---|---|---|
| 8, 16 | 11 | 166 registers, 1296 resident | 168, 1296 | 168, 960 |
| 32 | 21 | 166, 648 | 168, 648 | 168, 480 |
| 64 | 41 | 166, 324 | 168, 324 | 168, 240 |
| 96 | 61 | 168, 216 | 168, 216 | 168, 160 |
| 112 | 71 | 168, 108 | 168, 108 | 168, 80 |
| 128 | 81 | 132, 108 | 168, 108 | cap 128: 128, 48 bytes spilled, 160 (uncapped: 168, 80) |
| 160 | 101 | cap 64: 64, 136 bytes, 324 | cap 64: 64, 304 bytes, 324 | cap 64: 64, 312 bytes, 160 |
| 192 | 121 | cap 64: 64, 304 bytes, 216 | cap 64: 64, 320 bytes, 216 | refused |
| 256 | 161 | cap 64: 64, 256 bytes, 216 | cap 64: 64, 280 bytes, 216 | refused |

The V100 refused D = 192 and 256 at compilation, before any launch ("D=192 requires 121
resident blocks; this GPU can hold 80. The current one-kernel row layout cannot run safely
at this size.", and 161 blocks at D = 256), and `--tiles mma` at every D before anything
was compiled ("--tiles mma needs FP64 tensor-core instructions (mma.sync m8n8k4 f64,
compute capability 8.0 or later); this GPU is 7.0: use --tiles auto or scalar").

The runs (`rcmps-gpu-optimize --run-gpu`): 3 accepted iterations at D = 8 to 128 (96 and
128 being resident on both GPUs) and 30 at D = 16 and 32, each passing the initial
qualification against the CuPy oracle and the final five-row validation. At the initial
state the fast energy differs from the oracle's by at most 7.8e-15 and the gradient by
1.5e-14 to 6.2e-14 (metric norm), on both GPUs. On the A100 the two paths gave the same
first gradient and the same final R, K and gradient, byte for byte, and the same energy at
every iteration, at every D and for both lengths. The seconds include compilation, initial
qualification and validation:

| D | Iterations | Initial energy | Fast - oracle: A100, V100 | Final energy (A100) | V100 - A100 | Seconds: A100 tensor cores, scalar; V100 |
|---|---|---|---|---|---|---|
| 8 | 3 | 0.635263 | 1.7e-15, 1.4e-15 | 0.0011012344036118774 | -1.4e-15 | 4.4, 4.5; 5.4 |
| 16 | 3 | 0.754995 | 1.9e-15, 2.3e-15 | 0.06923331417005117 | +2.0e-15 | 5.4, 5.6; 7.0 |
| 32 | 3 | 1.210888 | 7.8e-15, 7.1e-15 | 0.2639712900504151 | +3.3e-16 | 6.1, 6.4; 7.7 |
| 64 | 3 | 0.970180 | 4.8e-15, 4.0e-15 | 0.1889563667046353 | -1.6e-15 | 7.0, 8.1; 9.8 |
| 96 | 3 | 0.929534 | 3.3e-15, 3.4e-15 | 0.16661806711404634 | +5.6e-16 | 10.2, 11.8; 13.5 |
| 128 | 3 | 0.963646 | 5.9e-15, 4.0e-15 | 0.1813095301955501 | +1.3e-15 | 15.5, 17.8; 19.5 |
| 16 | 30 | 0.754995 | | -0.27963031615292266 | +1.1e-12 | 9.8, 10.1; 13.2 |
| 32 | 30 | 1.210888 | | -0.27857855201722104 | -3.9e-14 | 11.4, 12.2; 14.8 |

Along the 30 iterations the energies of the two GPUs differ by at most 6.7e-12 (D = 16) and
8.0e-13 (D = 32).

The excitation solver. Compilation and residency (`gpu_bringup.py --stages device
compile`, both kernels at D = 4, 8, 16, 32, 64, 96 and 128 with every path of the device):
on the A100 both kernels are resident with both paths at every D, at D = 128 with register
cap 120 (uncapped and at cap 160 the device holds 108 blocks; at cap 120, 216, for the 208
blocks of the action kernel and the 160 of the pair kernel; up to 88 bytes spilled); on the
V100 the action kernel is resident up to D = 96 (156 of 160 blocks) and the pair kernel up
to D = 64 (80 of 160), and the pair kernel at D = 96 and both at D = 128 are refused
("unsafe cooperative grid: 120 blocks are not resident for any register cap (cap,
registers, capacity): [(None, 168, 80), (160, 160, 80), (120, 120, 80), (96, 96, 80), (64,
64, 80)]" at D = 96). Every resident kernel passed the PTX audit. The kernel checks:

| Script | Check | A100: tensor cores and scalar tiles | V100: scalar tiles |
|---|---|---|---|
| `gpu_fused_check.py` | fused kernel against the host-controlled NumPy integrator: random Z2 states D = 4, 6, 8 and 12 (odd sector at p = 0 with both scalings, even sector at p = 0.4) | same steps; samples to 6.1e-15, action to 1.3e-14 | same steps; samples to 6.3e-15, action to 1.4e-14 |
| | the same on the D = 16 fixture | samples to 7.3e-11, action to 3.5e-10 (even sector: 1545 steps, the host 1541) | samples to 1.2e-11, action to 8.3e-11 (even sector: 1542 steps, the host 1530) |
| | repeated launches; the two paths | same bytes; the two paths the same bytes and steps, the scalar tiles 1.18 to 1.29 times the tensor-core time per right-hand side at D = 4 to 12 and 1.42 at D = 16 (9.8 to 13.8 μs with the tensor cores) | same bytes; 10.5 to 16.4 μs per right-hand side |
| `gpu_pair_check.py --reference` | bilinear φ⁴ term, D = 16, against the host packed and unpacked systems; repeated launches | 1.1e-15, 1.1e-14, the same bytes with either path; repeat the same bytes; 0.34 s (tensor cores) and 0.52 s (scalar) per launch | 7.4e-15, 1.6e-14; repeat the same bytes; 0.48 s |
| `gpu_action_check.py --reference` | complete action, D = 16, against the unpacked equations on the CPU; repeated action | 8.7e-11, the same value with either path; repeat the same bytes; 0.52 to 0.56 s (tensor cores) and 0.67 to 0.71 s (scalar) per action | 2.3e-11; repeat the same bytes; 0.68 to 0.72 s |

`check_references.py`: the six levels of `excitation_references.toml` at D = 16 and 32,
through the command line at the settings of the A100 runs. Every level is within its
tolerance (1e-10) and accepted, with the stored number of actions, on both GPUs; on the A100
the two paths gave the same energies (the four validation energies and the Ritz energy), and
the V100's validated energies differ from the A100's by at most 1.4e-12:

| Level | Actions | Validated energy - Julia: A100, tensor cores | A100, scalar tiles | V100 | Wall (s): A100 tensor cores, scalar; V100 |
|---|---|---|---|---|---|
| D = 16, odd, p = 0 | 16 | -2.0e-12 | -2.0e-12 | -2.9e-12 | 15.3, 18.4; 20.1 |
| D = 16, odd, p = 0.5 | 17 | -1.4e-12 | -1.4e-12 | -2.8e-12 | 15.5, 19.1; 21.0 |
| D = 16, even, p = 0.5 | 20 | +9.5e-13 | +9.5e-13 | +8.3e-13 | 17.0, 21.1; 23.3 |
| D = 32, odd, p = 0.05 | 27 | +4.4e-12 | +4.4e-12 | +2.9e-12 | 57.6, 76.1; 87.1 |
| D = 32, odd, p = 0.1 | 28 | -4.0e-12 | -4.0e-12 | -4.9e-12 | 59.4, 78.2; 88.8 |
| D = 32, even, p = 0 | 30 | -1.1e-11 | -1.1e-11 | -1.1e-11 | 61.7, 81.2; 93.1 |

The D = 64 solve (`RCMPS_STATE_D64`: the CPU checkpoint of the D = 64 row of the next table;
odd sector, p = 0, the defaults), on the A100 with the tensor cores and with the scalar
tiles forced: 44 actions each, the same Davidson history, and the validated energy
0.03469099907685025 in both (four-energy spread 4.3e-11, residual 8.5e-8), 1.2e-11 below
the run of 2026-10-14 and 2.5e-10 above the stalled Julia run; wall 411 s and 577 s (the
search 274 s and 415 s, the preconditioner 81 s and 80 s, the validation 49 s and 72 s).

On the V100 machine, before the batches, the CI subset of the tests (the command under
"Tests") ran under Python 3.11.9 at the commit "Excitation qualification: fail on errors and
wrong numbers, compare launches bitwise" (the log does not record it; identified by the
number of tests and the lines of the skipped ones): 275 passed and 10 skipped (the four
NVRTC compilation tests, the CUDA headers being absent there, and six slow tests), in 513 s.
The three tests added since are two NVRTC tests and a CPU test of
`gpu_optimizer_check.py`.

Not qualified: any other GPU (the P100, the T4 and RTX 20 cards, the A30, the 8.6 and 8.9
cards of the table under "GPUs", and the H100, where only the research runtime ran, and this
package's optimizer on states without symmetry, below); the Z2 optimizer of this package
beyond D = 128 (D = 160 to 256 were compiled and checked for residency, not run) and to
convergence (the batch runs 3 and 30 iterations); the optimizer on states without symmetry
on the A100 and V100 (whose limits, 128 and 95, are estimates) and beyond D = 64; the excitation
solver of this package beyond D = 64 on the A100 and D = 32 on the V100 (the D = 96 and 128
solves below used the imported and earlier sources, with the tensor cores).

**Optimizer on states without symmetry, H100 NVL, 2026-10-26.** One H100 NVL (compute
capability 9.0, 132 multiprocessors, 94 GB; Grid'5000 Sophia `musa`, driver for CUDA 13.0),
Python 3.13.5, NumPy 2.5.4, SciPy 1.17.1, CuPy 14.2.0 with the CUDA 12.9 libraries of the
wheels, the package at the commit "Qualification batch: the optimizer on states without
symmetry" (copied, not a checkout: the plans record no commit). `gpu_optimizer_check.py
--symmetry none` with the tensor cores and the scalar tiles forced (seed 0, g = 3): the tile
product equals the chain of FMAs; at D = 8 to 64 the two paths take the same steps and give
the same samples, byte for byte, a repeated launch the same bytes, and at D = 8 the host
oracle takes the same 465 attempts, its samples within 2.4e-15; the runs of 3 iterations at
D = 8, 16, 32 and 64 pass their initial qualification (fast - oracle: energy at most 1.4e-14,
gradient 7.7e-14) and validation, and the two paths give the same first gradient, history
and final state. Device time per trajectory of the random state (ms; registers per thread,
resident blocks against the grid's):

| D | Steps (accepted, rejected) | Tensor cores | Scalar tiles | Scalar / tensor cores | Registers, resident / blocks (tensor cores) |
|---|---|---|---|---|---|
| 8 | 460, 5 | 46 | 88 | 1.87 | 154, 1584 / 8 |
| 16 | 352, 4 | 48 | 90 | 1.86 | 128, 1056 / 15 |
| 32 | 365, 4 | 72 | 160 | 2.22 | 128, 528 / 29 |
| 64 | 341, 3 | 120 | 348 | 2.90 | 130, 132 / 57 |

Compilation and residency (`--compile-only`, tensor cores): resident at D = 96, 112 and 128
(64 registers, the cap of D > 64; 264 blocks resident) and at 136 and 144 (132 resident, for
120 and 127 blocks); refused at D = 152 ("D=152 requires 134 resident blocks; this GPU can
hold 132"), 160, 176, 192, 224 and 256, as the table under "GPUs" estimates. The
comparison with RcmpsCore (`check_nonz2.py`, the states of the table under "Optimizer:
states without Z2 symmetry"): the differences of the RTX 2070 within rounding. The
optimizations (tensor cores, φ⁴ at g = 3, target 1e-7, the defaults otherwise):

| D | Start | Iterations (evaluations) | Seconds per iteration (median) | Stop: gradient norm | Energy (refined validation) | ⟨φ⟩ |
|---|---|---|---|---|---|---|
| 16 | the start of the Julia minimum | 177 (214) | 0.33 | stagnation: 3.0e-6 | -0.3886789420167186 (Julia's minimum -0.388678942017663) | 0.6172532155 (Julia 0.6172521147) |
| 32 | random (`--seed 1`) | 418 (466) | 1.07 | stagnation: 1.7e-5 | -0.3897114771313478 | -0.6075901266 |
| 64 | random (`--seed 1`), one job | 675 (676) | 3.8 (0.26 at the start, 8.5 at the end) | time budget (2700 s): 1.6e-4 | -0.3898340978399881 | 0.6047528105 |

The time of an iteration is that of the trajectory: optimized states make the quartic
integration stability limited, 6077 accepted steps (1.0 s) at iteration 200 of the D = 32
run against 365 at its random start, while the kinetic resolvent took 0.19 s on the other
stream and the CPU boundary and metric solves 0.01 s. The explicit integrator's steps along
the imaginary axis are bounded by the spread of the spectrum of K, which the minima develop
on almost unpopulated directions: 21 at D = 8 and 62 at D = 16 (the Julia minima; the
GPU's D = 16 state has the same spread, smallest Schmidt value 3.6e-8 and condition number
2.8e7 of ρ), 305 at the end of the D = 32 run (ρ condition number 7e9), and at D = 64 from
5.6 at the random start to 1290 after 675 iterations, still growing (smallest eigenvalue of
ρ 4.7e-13), with 29,517 steps (8.5 s) per trajectory. At D = 64 the optimizer therefore
needs hours to converge; ⟨φ⟩ moved by 1e-5 over the last 100 of those iterations and the
energy by 2e-8.

**Excitation solver, A100 PCIe 40 GB, 2026-10-13 and 14, up to D = 128.** The import commit
of this package holds byte for byte the sources of the last batch of runs (see Provenance):
the masses at D = 16, 32 and 64, the last restart of the D = 128 mass (13 actions, from a
vector computed with earlier sources), the D = 16 even level, and the kernel checks below
not marked *. The other runs used earlier sources of the same algorithms;
across the three source versions the D = 16, 32 and 64 masses agree to 2e-13, 8e-13 and
2e-11. The mass and sector runs (the defaults of the command line, with 256 kinetic nodes;
the D = 96 and 128 rows end with restarts at 512):

| State | D | Sector, p | Actions | Wall | Validated energy | Four-energy spread | Julia | GPU - Julia |
|---|---|---|---|---|---|---|---|---|
| g = 2 fixture | 16 | odd, 0 | 16 | 22 s | 0.3523798021159728 | 8.2e-12 | 0.3523798021182097 | -2.2e-12 |
| | 16 | odd, 0.5 * | 17 | 35 s | 0.6115040261903056 | 5.0e-12 | 0.6115040261920367 | -1.7e-12 |
| | 16 | even, 0.5 | 20 | 22 s | 1.2896320149270475 | 4.2e-11 | 1.2896320149261895 | +8.6e-13 |
| g = 2.7 | 32 | odd, 0 | 25 | 61 s | 0.03377574625890766 | 5.0e-11 | 0.03377574626049129 † | -1.6e-12 |
| | 32 | odd, 0.05 * | 27 | 64 s | 0.05575314792848829 | 4.0e-11 | 0.05575314792508679 | +3.4e-12 |
| | 32 | odd, 0.1 * | 28 | 65 s | 0.10235418166420304 | 3.5e-11 | 0.10235418166753874 | -3.3e-12 |
| | 32 | even, 0 * | 30 | 61 s | 0.31281412705278433 | 3.5e-11 | 0.3128141270643408 | -1.2e-11 |
| g = 2.7, CPU checkpoint | 64 | odd, 0 | 44 | 376 s | 0.03469099908836591 | 5.1e-11 | 0.03469099882896343 ‡ | +2.6e-10 |
| g = 2.7, GPU-optimizer state | 96 | odd, 0 * | 71 + 2 | 1783 s + 508 s | 0.03484655970011241 | 3.5e-10 | | |
| g = 2.7, GPU-optimizer state | 128 | odd, 0 | 108 * + 12 * + 13 | 6454 s + 1983 s + 2033 s | 0.03496269270838775 | 3.2e-10 | | |

\* earlier sources. † recorded mass of the research mass scan (Julia, at its own settings).
‡ Julia run that stalled at residual 2.3e-7, its four energies spread over 6.8e-10. The
D = 16 and 32 Julia values are levels of `test/fixtures/excitation_references.toml`; the
D = 64 value is the mass of that stalled run of the mass scan of the companion repository
rcmps-phi4-z2 (`research/mass_scan/masses.csv`), whose state is not a fixture. In every case
computed by the Julia library at the settings of the GPU run (D = 16 and the D = 32 rows
other than p = 0) the two codes take the same number of actions and both accept the level;
the D = 16 mass is also 3.3e-12 below the recorded one. The D = 96 and 128 rows end with
restarts (`--start`) from the run before with 512 kinetic nodes, the first 256-node
D = 128 run being refused by its refined check (residual 1.2e-5). At D = 128 the state has
four Schmidt values below the metric cutoff 10 D eps, so four columns of the tangent are
dropped, as the Julia code would do (sector dimension 7936 instead of 8192). The
four-energy spread at D = 96 and 128 is between the action and the bilinear route, not
within either; it is the Float64 floor of the linear solves (GMRES residuals 2e-13 and
4e-13, ρ with condition number 2.6e12), an attribution by elimination. The D = 96 and 128
masses of these states are therefore known to about 4e-10.

Time per stage on the A100 (seconds; the Julia code needs 548 s per action and 23,900 s
for the preconditioner at D = 64, and 40 minutes per level at D = 32 on two threads):

| D | Sector dimension | Operator setup | Preconditioner | One action | Validation |
|---|---|---|---|---|---|
| 16 | 128 | 0.7 | 1.1 | 0.54 | 4.0 |
| 32 | 512 | 1.6 | 6.2 | 1.4 | 10.6 |
| 64 | 2048 | 5.7 | 57 | 5.7 | 44 |
| 96 | 4608 | 18 | 287 | 18.8 | 143 |
| 128 | 7936 | 46 | 897 | 47.8 (52.4 with 512 kinetic nodes) | 349 |

Kernel checks, with the scripts of `qualification/excitations/` (each prints JSON lines):

| Script | Check | Result |
|---|---|---|
| `gpu_fused_check.py` | fused kernel against the host-controlled NumPy integrator: random Z2 states D = 4 and 8, both sectors, p = 0 and 0.4, both scalings | same steps; samples to 2e-14, action to 1.2e-14 |
| | the same on the D = 16 fixture (stability-limited steps) | samples to 9e-11, action to 4e-11 |
| | tensor-core against scalar tiles: D = 4 to 16 (D = 32 to 128 with earlier sources) | equal (largest difference 0), same steps; scalar tiles 1.2 to 1.4 times the tensor-core time per right-hand side (1.5 to 2.5 at D = 32 to 128) |
| `gpu_action_check.py --reference` | complete action, fused GPU route against the unpacked equations on the CPU, D = 16; repeated action, D = 16 (D = 64 to 128 with earlier sources) | 3.1e-11; repeat difference 0 |
| `gpu_pair_check.py --reference` | bilinear φ⁴ term, fused kernel against the host packed and unpacked systems, D = 16; repeated launches, D = 16 (D = 64 with earlier sources) | 4.4e-15, 6.2e-15; repeat difference 0 |
| `gpu_bringup.py` * | batched GMRES and kinetic action, GPU against NumPy, D = 64 | 6e-14, 7e-14 |
| `gpu_settings_check.py` * | kinetic nodes and window settings: D = 16 to 128 (the last batch only repeated D = 128 at 512 nodes against itself) | "Kinetic nodes at large D" |
| `rcmps-gpu-excitations --start` * | restart from the checkpoint of a finished D = 16 run | converged in 1 action, same validated energy |
| `check_references.py` | reference levels against Julia (and the A100 values above) | not run in these runs; first run in the batch of 2026-10-15 |

The repeated launches of `gpu_fused_check.py` in these runs compared the second launch with
itself: the kernel returns its sample buffer, which the second launch overwrote before the
comparison. The script now copies the samples of the first launch and compares the bytes,
as in the batch of 2026-10-15; the repeats above are those of `gpu_action_check.py` and
`gpu_pair_check.py`, whose values are copied to the host.

The fused kernel occupies 104 of 324 resident thread blocks of the A100 at D = 64, 156 of
216 at D = 96 and 208 of 216 at D = 128; it has not run beyond D = 128 there (with register
cap 64 and heavy spills, the occupancy calculator estimates that its grid stays resident up
to about D = 142; see "GPUs"). The coarse kinetic preconditioner is a dense matrix of
dimension D²/2 built from D²/2 kinetic actions; it scales as D⁴ and is, with the kernel
layout, the first thing to replace beyond D = 128.

**Optimizer.** The kernels of this family ran full optimizations at g = 2.7 on an A100
SXM4 40 GB (D = 32, 64, 96, 128) and on an H100 SXM 80 GB (D = 32, 64, 128, 256, with the
H100 runtime from which this one derives; for its packed sizes this runtime generates the
same CUDA source and compiler options bit for bit, `tests/optimizer/golden_cuda.json`). On
the A100, D = 64 from the CPU initial state took 704 s (565 iterations) to a gradient norm
of 2.7e-5, with a refined energy 1.9e-10 from the saved CPU checkpoint. On the H100, fresh
runs took 125 s at D = 32 and 629 s at D = 64 (stagnation at gradient norm 6.6e-6, and the
target 3e-5 reached), 5408 s at D = 128 (1106 iterations, norm 2.1e-4) and about 8 h at
D = 256 (723 iterations, norm 1.1e-3, not converged); a warm energy and gradient evaluation
took 0.43 s, 1.39 s and 9.2 s at D = 32, 64 and 128. The selected states pass the five-row
consistency check (at D = 128 on the A100 only with 512 kinetic nodes); the independent
directional-derivative audit passes at D = 32 but not at D = 96 and 128, so there is no
independent error certificate for these gradients. The generic runtime, with the command
line, worker and validator of this package, first ran in the batch of 2026-10-15 (short
runs at D = 8 to 128, sizes of the historical set; the sizes outside it, D = 112, 160 and
192, were only compiled). The plan of every run still records
`layout.generic_runtime_actual_GPU_qualified` false, a constant of the imported runtime;
this section, not that flag, says what has run.

## Tests

CPU only (NumPy, SciPy, h5py); no test imports CuPy. From the repository root:

```bash
pip install -e "./gpu[test]"
cd gpu && OPENBLAS_NUM_THREADS=1 python -m pytest        # whole default suite, about 25 minutes
```

GitLab CI runs, on Python 3.11 and 3.12, the job `python` when `gpu/`, `test/fixtures/` or
`.gitlab-ci.yml` change and on the default branch: the optimizer, state-file and
reference-level tests and the excitation files that exercise every production module (the
driver on the strict CPU stand-in for CuPy, the command line), in about 9 minutes on one
thread of a workstation (10 with the NVRTC compilation tests, which skip on the CI image).
The optimizer tests include a run of `rcmps-gpu-optimize --run-gpu` at D = 8 and its
resume, end to end on the CPU (`tests/optimizer/test_worker_cpu.py`): the command line,
worker, qualification, optimization and validation run as written, with stand-ins for the
GPU leaves only (`tests/optimizer/worker_on_cpu.py`: a NumPy-backed `cupy`, the CPU oracle
of the trajectory kernel, the CPU kinetic solvers), on a stand-in device of compute
capability 8.0 and, for the resume, 7.0 (scalar tiles, and a register cap from the ladder),
and a run at D = 5 without symmetry. `tests/optimizer/test_unpacked.py` checks the kernel of
states without symmetry (see "Optimizer: states without Z2 symmetry").
`tests/optimizer/test_tiles.py` checks the scalar tile product bitwise against the chain of
FMAs it implements, the selection rule, the PTX audit, and with NVRTC (no GPU) both tile
paths of the kernel for compute capability 6.0 to 9.0, to PTX and, with ptxas, to a cubin,
as `tests/excitations/test_fused.py` does for the excitation kernels.
`tests/test_qualification.py` checks that every option `qualification/run_all.sh` gives a
script exists, that `gpu_optimizer_check.py` runs the optimizer with its own settings, and
runs the excitation scripts of the batch on a stand-in device
(`tests/qualification_on_cpu.py`: NumPy arrays, a fused kernel whose launch is the host
integrator), where a compilation error, an FP32 instruction in the PTX, a launch that is
not repeatable bitwise and a result off by 1e-6 each give exit status 1, and a size the
device cannot hold does not. The same command, from `gpu/`:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest tests/optimizer tests/test_io.py tests/test_qualification.py \
  tests/excitations/test_references.py tests/excitations/test_{coordinates,davidson,scaled}.py \
  tests/excitations/test_{kinetic,fused,symmetry,run,potential,packed,packed_pair}.py \
  tests/excitations/test_{device_strict,hamiltonian}.py
```

The manual job `python-full` runs the whole default suite: it adds the eigsolve,
validation, bilinear and dense_reference files (the driver, the validation and the
bilinear route against the dense oracle, and the oracle itself), the provenance and
packaging tests, and the stored-reference tests, which skip without the companion
repository. Without pytest, the files run one by one with unittest (each line from the
repository root):

```bash
cd gpu/tests/excitations && for t in test_*.py; do OPENBLAS_NUM_THREADS=1 python -m unittest ${t%.py}; done
cd gpu/tests/optimizer && OPENBLAS_NUM_THREADS=1 python -m unittest test_generic test_portable test_worker_cpu test_tiles test_unpacked
cd gpu/tests && OPENBLAS_NUM_THREADS=1 python -m unittest test_package test_io test_qualification
```

Without installing, put the absolute path of `gpu/src` on `PYTHONPATH`. The optimizer tests
take about 40 s (85 s with the NVRTC compilations), `test_qualification` 25 s, the 19
excitation files about 25 minutes on one thread. The excitation tests check every module
against a dense NumPy port of the Julia algorithm (`dense_reference.py`), the oracle
against analytic states, the solver against the Julia levels of
`excitation_references.toml` (and, with the slow flag, against the stored Julia results of
the companion repository), invariance under the scaling S, the kernel tables against the
packed systems, the driver on a strict CPU stand-in for CuPy (`fakecupy.py`), the command
line and the defaults it resolves, time reversal (levels of (R, K) at p equal those of
(conj R, -conj K) at -p), and compilation of the kernel source with NVRTC when the CUDA
headers are available. With the slow tests the suite takes roughly an hour on one thread,
run one file at a time. Environment variables:

| Variable | Effect |
|---|---|
| `RCMPS_EXCITATION_SLOW=1` | slow tests: planted-bug mutations, larger grids and restarts, D = 16 references |
| `RCMPS_EXCITATION_D16_SOLVE=1`, `RCMPS_EXCITATION_D16_HISTORY=1` | with the previous one, full D = 16 solves (about 8 and 6 minutes) |
| `RCMPS_EXCITATION_REPORT=1` | print the largest measured errors |
| `RCMPS_TEST_FIXTURES` | fixtures shared with the Julia tests (default: `test/fixtures` of the repository; outside it, as in an unpacked sdist, the tests that read them are skipped unless it is set) |
| `RCMPS_PHI4_Z2` | companion repository with the stored Julia references (default: `rcmps-phi4-z2` next to this repository) |
| `RCMPS_CUDA_INCLUDE` | directory with the CUDA header `nv/target`, for the NVRTC compilation test |

The GPU qualification scripts of the excitation solver are in `qualification/excitations/`
(`gpu_bringup.py`, `gpu_fused_check.py`, `gpu_action_check.py`, `gpu_pair_check.py`,
`gpu_settings_check.py`, `check_references.py`; see "GPU qualification"). They run on a
GPU against the installed package and print JSON lines; their docstrings give the
commands. The optimizer's GPU checks are part of every run (`--compile-only` for the
compiler and resource checks alone); `qualification/optimizer/gpu_optimizer_check.py`
qualifies a GPU: the tile product alone against the chain of FMAs, trajectories with each
tile path, compilation and residency per D, and short runs through the command line.
`qualification/run_all.sh` runs the qualification batch of one GPU (see "GPU
qualification" for what it runs and what it leaves out).

## Provenance

The code was developed in an untracked research tree and imported byte for byte by the
commit "Import the GPU code verbatim" (`IMPORTED_FROM_RESEARCH.sha256` lists every file
with its sha256 and research path); git is the seal from there on, and each later commit
changes only what its message says. For the optimizer runtime,
`src/rcmps_gpu/optimizer/provenance.json` records the sealed research manifest (the
imported sha256 of the 24 runtime modules, the H100 base manifest and the builder) and
every edit made since, commit by commit, in the format described in
`src/rcmps_gpu/provenance.py`: the import rewrite of the packaging commit, then the pruning
commit, which deleted two modules that the generic run never imports and code paths it
never reaches (old evaluators, the CUDA-graph integrator and the driver-library bindings it
needed, the research benchmark command lines) and renamed the quartic backend label of the
run to `persistent_trajectory`, a commit that rewords comments still describing CUDA graph
capture, and the portability commits, which add the scalar tile path and the selection of
the tiles by device, then the register caps, and correct the register range in a docstring.
The optimizer command line reverts those edits in memory when it plans a run, and the
worker again before and after the run: a runtime that is not the imported one with its
recorded edits is refused, as the research launcher refused one that was not derived from
the sealed H100 candidate. The sha256 of the imported manifest is written in the code
(`cli.IMPORTED_MANIFEST_SHA256`) and in the tests, and the package `__init__` modules,
which run with every runtime import, must hold only a docstring and constants. A test pins
the sha256 of the generated CUDA source, compiler options and launch geometry
(`tests/optimizer/golden_cuda.json`). A run records in `plan.json` the package version, the
git commit when run from a checkout, the result of the provenance check, and the sha256 of
every runtime module and of every other file of the package that runs with it
(`cli.LAUNCHER_SOURCES`); the worker requires them unchanged. Checkpoints hash the runtime
modules by file name, and a resume requires exactly the same runtime.

For the excitation solver, the import commit holds byte for byte the sources of the A100
qualification runs (`final2` of the research results, finished 2026-10-14 between 00:45
and 01:40 UTC: odd sector at p = 0 for D = 16, 32, 64 and 128, even sector at p = 0.5 for
D = 16), as their `settings.source_sha256` records them, with `run.py` now `cli.py`.
`src/rcmps_gpu/excitations/provenance.json` holds that table (pinned by
`cli.QUALIFIED_SOURCES_SHA256`) and every edit made since in the same format; the next
commit rewrote `eigsolve.py` and `cli.py` (defaults only, with every option of those runs
given explicitly they compute the same), which the record ties by sha256, the diff being
in git. `cli.source_provenance()` reverts the edits and compares; a test requires it to
pass. A `result.json` records the sha256 of the 16 modules and of the package files on
their import path (`settings.source_sha256`, keyed by file name, the shared ones relative
to `rcmps_gpu`), and the outcome of that check (`settings.source_provenance`); the command
line logs a warning, but runs, when the check fails.

## Layout

```
src/rcmps_gpu/optimizer/         cli.py (command line), worker.py (GPU process), layout.py,
                                 validate_sweep.py, validate_trajectory_sweep.py, provenance.json
src/rcmps_gpu/optimizer/runtime/ numerical runtime, research file names kept
src/rcmps_gpu/excitations/       the modules above, provenance.json
src/rcmps_gpu/io.py              state files: JLD2 checkpoints and portable directories
src/rcmps_gpu/provenance.py      provenance record format and check
julia/                           Julia scripts: export, import and initial portable states
tests/optimizer/, tests/excitations/, tests/test_package.py, tests/test_io.py, tests/test_qualification.py,
                                 tests/qualification_on_cpu.py
qualification/run_all.sh         the qualification batch of one GPU
qualification/optimizer/         GPU qualification script of the optimizer
qualification/excitations/       GPU qualification scripts of the excitation solver
```
