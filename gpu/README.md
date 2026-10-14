# rcmps_gpu

CUDA implementations of two algorithms of RcmpsCore, in Python with CuPy, for the φ⁴
model on Z2-symmetric relativistic continuous matrix product states (even bond dimension
D). Everything is float64/complex128. The Julia package in the repository root is the
reference implementation and is not modified by this one.

- `rcmps_gpu.optimizer`: ground-state optimizer. One persistent cooperative CUDA kernel
  integrates the whole adaptive DOP853 trajectory of the quartic term with native FP64
  tensor-core products; L-BFGS on the manifold, as in the Julia code. The kernel is
  generated for each D (every even D from 2 to 512, subject to the GPU holding the whole
  grid resident).
- `rcmps_gpu.excitations`: tangent-space excitation solver (matrix-free Hamiltonian
  action, Davidson, validation by the bilinear route), both parity sectors, any momentum.
  The action trajectory of a Z2 sector runs in one fused cooperative kernel.

Status: the excitation solver was qualified on an A100 up to D = 128; at D = 16 and 32
its energies agree with the Julia code to about 1e-11, and above that its two
independent routes (action and bilinear) agree to 4e-10. The optimizer's runtime is the
generic version of the H100 runtime of the research campaign: for the sizes that runtime
supported (packed d = D/2 in {2, 3, 4, 8, 16, 32, 48, 64, 128}) it generates the same CUDA
source and compiler options, bit for bit, but the generic runtime itself has not yet run
on a GPU. The optimizer needs FP64 tensor cores (SM80 or SM90); the excitation kernel uses
them too, with a scalar-tile variant (`--scalar-tiles`) meant for debugging.

## Installation

Python 3.11 or later. From the repository root:

```bash
pip install "./gpu[cuda12]"     # with CuPy for CUDA 12 (cupy-cuda12x[ctk])
pip install "./gpu[test]"       # CPU only: enough for the tests and for the CUDA source
```

Importing `rcmps_gpu` or its subpackages never imports CuPy; only a GPU run does.

## Command line

Optimizer (`rcmps-gpu-optimize`, or `python -m rcmps_gpu.optimizer`):

```bash
rcmps-gpu-optimize --D 160                         # plan only, on the CPU
rcmps-gpu-optimize --D 160 --emit-cuda D160.cu     # generated CUDA source, on the CPU
rcmps-gpu-optimize --D 160 --compile-only --run-gpu --output RUN   # compile and resource check
rcmps-gpu-optimize --D 160 --seed 42 --coupling 2.7 --run-gpu --output RUN
rcmps-gpu-optimize --D 160 --resume-from RUN/optimization.json --run-gpu --output RUN2
```

Output directories must be new. `--initial DIR` starts from a portable export (R.bin,
K.bin, rho.bin, saved_gradient.bin as little-endian complex128 in column-major order,
and metadata.toml) instead of a random Z2 state. There is no default wall-time limit
(`--max-seconds` sets one). A GPU run first compares the fast energy and gradient with an
independent CuPy evaluation at the initial state; after stopping it runs a five-row
consistency check at refined settings and, if it passes, writes `optimized_state/`.

Excitations (`rcmps-gpu-excitations`, or `python -m rcmps_gpu.excitations`):

```bash
rcmps-gpu-excitations STATE --output DIR [--momentum 0.0] [--nev 1] [--sector odd|even|all]
```

STATE is an `rcmps_store` checkpoint (.jld2) or a portable export directory. `--backend
numpy` runs the same code on the CPU. `--help` describes the defaults: the Davidson
schedule of the A100 runs (search space min(160, n), thick restarts) and the kinetic
nodes of `eigsolve.default_kinetic_nodes` (below). From Python the entry points are
`rcmps_gpu.excitations.eigsolve.excitation_eigsolve` and `iterative_dispersion`.

## Kinetic nodes at large D

Measured on an A100 with `qualification/excitations/gpu_settings_check.py`: the two parts
of the action of a converged vector, at each setting, against the finest one (1024
kinetic nodes; 2048 window nodes, window 40, ODE tolerances 1e-14/1e-13), on the g = 2.7
states. Each part has norm 5 (D = 16) to 21 (D = 128).

| D | Resolvent part, 256 nodes: action, energy | 512 nodes: action, energy | ODE part, standard settings: action, energy |
|---|---|---|---|
| 16 | 4.0e-12, 1.2e-12 | 5.4e-12, 6.8e-13 | 1.0e-10, 2.4e-11 |
| 64 | 9.6e-11, 1.9e-12 | 5.8e-11, 3.4e-12 | 5.2e-10, 3.7e-12 |
| 96 | 2.1e-7, 4.8e-12 | 1.0e-10, 1.8e-11 | 1.7e-9, 2.0e-11 |
| 128 | 1.2e-5, 7.0e-10 | 3.1e-10, 2.3e-11 | 5.3e-9, 9.0e-11 |

The Julia default of 256 kinetic nodes is converged up to D = 64 and fails abruptly
beyond. 512 nodes are converged to 3e-10 up to D = 128, so `default_kinetic_nodes(D)` is
256 up to D = 64 and 512 above (with twice as many for the refined check of the
validation). Nothing was measured beyond D = 128; the command line notes it there, and
warns when `--kinetic-nodes` is below the default.

## Provenance

The code was developed in an untracked research tree and imported byte for byte by the
commit "Import the GPU code verbatim" (`IMPORTED_FROM_RESEARCH.sha256` lists every file
with its sha256 and research path); git is the seal from there on, and each later commit
changes only what its message says. For the optimizer runtime,
`src/rcmps_gpu/optimizer/provenance.json` records the sealed research manifest (the
imported sha256 of the 24 runtime modules, the H100 base manifest and the builder) and
every edit made since, commit by commit, in the format described in
`src/rcmps_gpu/provenance.py`. The optimizer command line reverts those edits in memory
when it plans a run, and the worker again before and after the run: a runtime that is not
the imported one with its recorded edits is refused, as the research launcher refused one
that was not derived from the sealed H100 candidate. The sha256 of the imported manifest
is written in the code (`cli.IMPORTED_MANIFEST_SHA256`) and in the tests, and the package
`__init__` modules, which run with every runtime import, must hold only a docstring and
constants. A test pins the sha256 of the generated CUDA source, compiler options and
launch geometry (`tests/optimizer/golden_cuda.json`). A run records in `plan.json` the
package version, the git commit when run from a checkout, the result of the provenance
check, and the sha256 of every runtime module and of every other file of the package that
runs with it (`cli.LAUNCHER_SOURCES`); the worker requires them unchanged. Checkpoints
hash the runtime modules by file name, and a resume requires exactly the same runtime.

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

## Tests

CPU only (NumPy, SciPy, h5py); no test imports CuPy.

```bash
pip install "./gpu[test]"
cd gpu && python -m pytest
```

or file by file (each line from the repository root):

```bash
cd gpu/tests/excitations && for t in test_*.py; do OPENBLAS_NUM_THREADS=1 python -m unittest ${t%.py}; done
cd gpu/tests/optimizer && OPENBLAS_NUM_THREADS=1 python -m unittest test_generic
cd gpu/tests && python -m unittest test_package
```

Without installing, put the absolute path of `gpu/src` on `PYTHONPATH`. The optimizer
tests take about 10 s, the 18 excitation files about 24 minutes on one thread. The
excitation tests check every module against a dense NumPy port of the Julia algorithm
(`dense_reference.py`), the driver on a strict CPU stand-in for CuPy (`fakecupy.py`), the
command line, and compilation of the kernel source with NVRTC when the CUDA headers are
available.
Environment variables:

| Variable | Effect |
|---|---|
| `RCMPS_EXCITATION_SLOW=1` | slow tests: planted-bug mutations, larger grids and restarts, D = 16 references |
| `RCMPS_EXCITATION_D16_SOLVE=1`, `RCMPS_EXCITATION_D16_HISTORY=1` | with the previous one, full D = 16 solves (about 8 and 6 minutes) |
| `RCMPS_EXCITATION_REPORT=1` | print the largest measured errors |
| `RCMPS_TEST_FIXTURES` | fixtures shared with the Julia tests (default: `test/fixtures` of the repository) |
| `RCMPS_PHI4_Z2` | companion repository with the stored Julia references (default: `rcmps-phi4-z2` next to this repository) |
| `RCMPS_CUDA_INCLUDE` | directory with the CUDA header `nv/target`, for the NVRTC compilation test |

One optimizer test needs Triton (the historical `persistent_rows_general` source imports
it) and is skipped without it.

The GPU qualification scripts of the excitation solver are in
`qualification/excitations/` (`gpu_bringup.py`, `gpu_fused_check.py`,
`gpu_action_check.py`, `gpu_pair_check.py`, `gpu_settings_check.py`). They run on a GPU
against the installed package and print JSON lines.

## Layout

```
src/rcmps_gpu/optimizer/         cli.py (command line), worker.py (GPU process), layout.py,
                                 validate_sweep.py, validate_trajectory_sweep.py, provenance.json
src/rcmps_gpu/optimizer/runtime/ numerical runtime, research file names kept
src/rcmps_gpu/excitations/       states, models, coordinates, davidson, scaled, kinetic, potential,
                                 ode, packed, packed_pair, fused, hamiltonian, bilinear,
                                 bilinear_fused, eigsolve, cli, provenance.json
src/rcmps_gpu/provenance.py      provenance record format and check
tests/optimizer/, tests/excitations/, tests/test_package.py, qualification/excitations/
```
