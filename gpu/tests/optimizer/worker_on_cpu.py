"""Run rcmps-gpu-optimize --run-gpu on the CPU, with stand-ins for the GPU leaves only.

    python worker_on_cpu.py RCMPS_GPU_INIT OUTPUT [rcmps-gpu-optimize options]

Child process of tests/optimizer/test_worker_cpu.py (it installs a module named cupy and
rebinds runtime functions, so it must not run inside the test interpreter). RCMPS_GPU_INIT is
the rcmps_gpu/__init__.py the parent imported; the child must import the same copy.
RCMPS_CPU_COMPUTE_CAPABILITY (default 8.0) is the compute capability the stand-in device
reports, which selects the tile product (tensor cores from 8.0 on, scalar tiles below);
with RCMPS_CPU_MAX_REGISTERS (default: no limit), a kernel that uses more registers (168
without a cap) is not resident, which makes the worker try its register caps. The
command line runs as written: cli.main plans the run and writes plan.json, and worker.main,
called in this process instead of a child interpreter, compiles, qualifies the first
evaluation against the CuPy oracle, optimizes through trajectory_sweep_driver and validates
through validate_trajectory_sweep. Only these leaves, which need a GPU, are replaced:
  * cupy: a NumPy-backed module (streams, events and devices do nothing; the device reports
    the compute capability above), with the six ElementwiseKernels of ode_backend emulated
    in NumPy (those of the packed Z2 channels, of the unpacked ones and of the batched RHS);
  * cupy.RawKernel: no compilation and no launch, but source with the FP64 tensor-core
    instruction is refused below compute capability 8.0, as ptxas refuses it; 168 registers,
    or the --maxrregcount cap; the occupancy query gives 1 block per multiprocessor, or 0 for
    a kernel above RCMPS_CPU_MAX_REGISTERS;
    persistent_trajectory_backend.trajectory_ptx_report: no NVRTC;
  * persistent_trajectory_overlap.PersistentTrajectoryDOP853, the CUDA trajectory kernel: its
    CPU oracle persistent_trajectory_backend.integrate_trajectory_numpy (D <= 8), on the
    channels of the run (packed Z2, or unpacked with --symmetry none), which refuses the
    tensor-core tiles below 8.0 as the kernel does and records the tiles;
  * the GPU kinetic solvers (kinetic_backend.kinetic_raw with the GPU Arnoldi and projection,
    cooperative_arnoldi.kinetic_raw_cooperative): kinetic_backend.kinetic_raw on the CPU.
The exit status is the worker's (1, with the traceback on stderr, if it raised).
"""
import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import types
from types import SimpleNamespace

import numpy as np

import rcmps_gpu

if Path(rcmps_gpu.__file__).resolve() != Path(sys.argv[1]).resolve():
    sys.exit(f"worker_on_cpu.py imported {rcmps_gpu.__file__}, not {sys.argv[1]}")

# ---------------------------------------------------------------- NumPy-backed cupy
cp = types.ModuleType("cupy")
cp.__version__ = "0+cpu-stand-in"
cp.__getattr__ = lambda name: getattr(np, name)
cp.asnumpy = lambda x: np.array(x, copy=True)
_local = threading.local()
_pointers = itertools.count(1000)


def _streams():
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


class Stream:
    def __init__(self, non_blocking=False, ptr=None):
        self.ptr = next(_pointers) if ptr is None else ptr

    def __enter__(self):
        _streams().append(self)
        return self

    def __exit__(self, *errors):
        _streams().pop()

    def synchronize(self):
        pass

    def wait_event(self, event):
        pass


_DEFAULT_STREAM = Stream(ptr=0)


class Event:
    def __init__(self, *args, **options):
        self.t = time.perf_counter()

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass


CAPABILITY = tuple(int(part) for part in os.environ.get("RCMPS_CPU_COMPUTE_CAPABILITY", "8.0").split("."))
MAX_REGISTERS = int(os.environ.get("RCMPS_CPU_MAX_REGISTERS", "255"))
KERNELS = {}


class Device:
    def __init__(self, id=0):
        self.id, self.compute_capability = id, "%d%d" % CAPABILITY

    def __enter__(self):
        return self

    def __exit__(self, *errors):
        pass


PROPERTIES = dict(name=b"CPU stand-in", major=CAPABILITY[0], minor=CAPABILITY[1], cooperativeLaunch=True,
                  sharedMemPerBlockOptin=166912,
                  sharedMemPerBlock=49152, maxThreadsPerBlock=1024, multiProcessorCount=108,
                  totalGlobalMem=1 << 34, clockRate=1410000)
cp.cuda = SimpleNamespace(
    Stream=Stream, Event=Event, Device=Device,
    get_current_stream=lambda: _streams()[-1] if _streams() else _DEFAULT_STREAM,
    get_elapsed_time=lambda start, end: 1000. * (end.t - start.t),
    runtime=SimpleNamespace(getDeviceProperties=lambda device: dict(PROPERTIES),
                            runtimeGetVersion=lambda: 12090, driverGetVersion=lambda: 12090),
    driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda pointer, threads, shared:
                           int(KERNELS[pointer].num_regs <= MAX_REGISTERS)))


class RawKernel:
    def __init__(self, code, name, options=(), enable_cooperative_groups=False):
        from rcmps_gpu.optimizer.runtime.generic_layout import dynamic_shared_bytes, packed_geometry
        d = int(re.search(r"#define PACKED_D (\d+)\n", code).group(1))
        if "mma.sync" in code and CAPABILITY < (8, 0):
            raise RuntimeError("worker_on_cpu.py: ptxas refuses mma.sync f64 below sm_80")
        self.code, self.name, self.options = code, name, options
        self.shared_size_bytes = 0 if dynamic_shared_bytes(d) else packed_geometry(d)["shared_bytes"]
        self.max_dynamic_shared_size_bytes, self.max_threads_per_block = 0, 1024
        caps = [int(option.split("=")[1]) for option in options if option.startswith("--maxrregcount=")]
        self.kernel, self.num_regs, self.local_size_bytes = SimpleNamespace(ptr=len(KERNELS) + 1), min([168, *caps]), 0
        KERNELS[self.kernel.ptr] = self

    def compile(self):
        pass

    def __call__(self, *arguments, **options):
        raise AssertionError("worker_on_cpu.py: no CUDA kernel launch on the CPU")


cp.RawKernel = RawKernel


def _write(result, out):
    if out is None:
        return result
    out[...] = result
    return out


def rk_linear_combination(y, k, a, n, stages, h, out=None):
    value = np.array(y, dtype=np.complex128, copy=True)
    for stage in range(int(stages)):
        value += h * a[stage] * k[stage]
    return _write(value, out)


def z2_block_gather(y, indices, adjoints, d, jump, left, right):
    d = int(d)
    matrices = y[:10 * d * d].reshape(10, d, d)
    indices, adjoints = indices.reshape(-1), adjoints.reshape(-1)
    for group, target in enumerate((jump, left, right)):
        for channel in range(10):
            source = matrices[indices[10 * group + channel]]
            target[channel] = source.conj().T if adjoints[10 * group + channel] else source
    return jump, left, right


def z2_rhs_combine(l, qr, jump, il, ir, fixed, coefficients, d, jac, source, out=None):
    c = coefficients.reshape(-1)[:, None, None]
    return _write(jac * (l + qr + jump) + source * (c * (il + ir) + fixed), out)


def z2_trace_accumulator(y, a, b, d, source, out=None):
    d = int(d)
    f3 = y[3 * d * d:4 * d * d].reshape(d, d)
    value = a * np.conj(f3) + b * f3.T
    return _write((8 * source * value.real).astype(np.complex128).reshape(-1), out)


def batched_rhs_combine(l, qr, jump, fixed, il, ir, li, ri, lc, rc, matrix_size, width, jac, source, out=None):
    channels, width = len(l), int(width)
    li, ri, lc, rc = (x.reshape(channels, width) for x in (li, ri, lc, rc))
    insertion = np.zeros_like(l)
    for channel in range(channels):
        for e in range(width):
            if li[channel, e] >= 0:
                insertion[channel] += lc[channel, e] * il[li[channel, e]]
            if ri[channel, e] >= 0:
                insertion[channel] += rc[channel, e] * ir[ri[channel, e]]
    return _write(jac * (l + qr + jump) + source * (insertion + fixed), out)


def quartic_trace_accumulator(il, ir, d, source, out):
    return _write(np.full(out.shape, 4 * source * np.trace(il[2] + ir[2]), dtype=np.complex128), out)


ELEMENTWISE = dict(rcmps_rk_linear_combination=rk_linear_combination, rcmps_z2_block_gather=z2_block_gather,
                   rcmps_z2_rhs_combine=z2_rhs_combine, rcmps_z2_trace_accumulator=z2_trace_accumulator,
                   rcmps_batched_rhs_combine=batched_rhs_combine,
                   rcmps_quartic_trace_accumulator=quartic_trace_accumulator)


class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name="", **options):
        if name not in ELEMENTWISE:
            raise AssertionError(f"worker_on_cpu.py: no emulation of the ElementwiseKernel {name!r}")
        self.call = ELEMENTWISE[name]

    def __call__(self, *arguments):
        return self.call(*arguments)


cp.ElementwiseKernel = ElementwiseKernel
sys.modules["cupy"] = cp

# ---------------------------------------------------------------- CPU kinetic and trajectory
from rcmps_gpu.optimizer.runtime import kinetic_backend  # noqa: E402

CPU_KINETIC = kinetic_backend.kinetic_raw


def kinetic_on_cpu(q, r, rho, nodes=128, *, xp=None, tolerance=1e-12, maxdim=1024, fallback_tolerance=None,
                   **ignored):
    return CPU_KINETIC(np.asarray(q), np.asarray(r), np.asarray(rho), nodes, xp=np, tolerance=tolerance,
                       maxdim=maxdim, arnoldi_backend="cpu", projection_backend="schur_cpu",
                       fallback_tolerance=fallback_tolerance)


kinetic_backend.kinetic_raw = kinetic_on_cpu   # before overlap_benchmark binds it
from rcmps_gpu.optimizer.runtime import cooperative_arnoldi  # noqa: E402
cooperative_arnoldi.kinetic_raw_cooperative = kinetic_on_cpu
from rcmps_gpu.optimizer.runtime import generic_layout  # noqa: E402
from rcmps_gpu.optimizer.runtime import ode_backend  # noqa: E402
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend  # noqa: E402
from rcmps_gpu.optimizer.runtime import persistent_trajectory_overlap  # noqa: E402
if persistent_trajectory_overlap.ORIGINAL_KINETIC is not kinetic_on_cpu:
    sys.exit("worker_on_cpu.py: the runtime bound the GPU kinetic before the stand-in")


class TrajectoryOnCPU:
    def __init__(self, rhs, points, *, atol, rtol, max_steps, max_device_seconds, initial_step=None, tiles,
                 register_cap):
        if tiles not in ("mma", "scalar"):
            raise ValueError(f"worker_on_cpu.py: the trajectory kernel takes resolved tiles, not {tiles!r}")
        self.tiles = generic_layout.select_tiles(tiles, *CAPABILITY)
        self.options = generic_layout.compilation_options(rhs.d, register_cap)
        if min([168, *(int(o.split("=")[1]) for o in self.options if o.startswith("--maxrregcount="))]) > MAX_REGISTERS:
            raise RuntimeError("worker_on_cpu.py: unsafe cooperative trajectory grid (register cap)")
        self.host = type(rhs)(rhs.q, rhs.r, rhs.rho, np)   # packed Z2 or unpacked channels
        self.points, self.controls = points, dict(atol=atol, rtol=rtol, max_steps=max_steps)

    def run(self):
        samples, stats = persistent_trajectory_backend.integrate_trajectory_numpy(
            self.host, self.points, **self.controls)
        # The kernel records its tiles and the device in the occupancy of its statistics.
        return samples, dict(stats, cpu_stand_in=True, occupancy=dict(
            tiles=self.tiles, compute_capability=list(CAPABILITY), compiler_options=list(self.options)))

    def close(self):
        pass


persistent_trajectory_overlap.PersistentTrajectoryDOP853 = TrajectoryOnCPU
persistent_trajectory_backend.trajectory_ptx_report = lambda d, cp, directory=None, tiles="mma", register_cap=None, \
    unpacked=False: dict(cpu_stand_in=True, tiles=tiles, register_cap=register_cap, unpacked=unpacked)

# ---------------------------------------------------------------- the command line
from rcmps_gpu.optimizer import cli, worker  # noqa: E402

SUBPROCESS_RUN = subprocess.run


def run(command, *arguments, **options):
    """cli.main starts the worker as `python -P -m rcmps_gpu.optimizer.worker PLAN`; run it here."""
    if list(command[-3:-1]) != ["-m", "rcmps_gpu.optimizer.worker"]:
        return SUBPROCESS_RUN(command, *arguments, **options)
    try:
        status = worker.main([command[-1]])
    except BaseException:
        traceback.print_exc()
        status = 1
    return subprocess.CompletedProcess(command, status)


cli.subprocess.run = run
sys.exit(cli.main(["--output", sys.argv[2], "--run-gpu", *sys.argv[3:]]))
