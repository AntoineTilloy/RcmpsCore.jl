"""Run a GPU qualification script of qualification/excitations on the CPU, with stand-ins for
the GPU leaves only.

    python qualification_on_cpu.py MODE SCRIPT [script options]

Child process of tests/test_qualification.py (it installs a module named cupy, so it must
not run inside the test interpreter). SCRIPT runs as written, as __main__, against:
  * cupy: a module whose arrays are NumPy arrays, with the device of
    tests/excitations/fakedevice.py (compute capability RCMPS_CPU_COMPUTE_CAPABILITY, default
    8.0; cooperative launch; every grid resident; RawKernel compiles nothing) and an NVRTC
    whose PTX is written from the source: the .target of the device, the FP64 mma.sync
    instruction with USE_MMA 1, warp shuffles and FP64 FMAs without;
  * fused.FusedTrajectory: constructed as written (tile selection, compilation through the
    stand-in, register caps, residency); its launch, integrate, is the host-controlled
    integrator ode.integrate_host, written into one sample buffer that the next launch
    overwrites, as the kernel's is.
MODE replaces one leaf by a faulty one:
  exact      none
  perturbed  every launch is off by a relative 1e-6
  unstable   every second launch of a kernel differs from the first in the last bit of its final
             samples
  broken     compilation fails with an error that is not a device limit
  refused    no thread block is resident (a device too small for every kernel)
  fp32       the PTX of the audit holds an FP32 instruction
The exit status is the script's.
"""
import os
from pathlib import Path
import runpy
import sys
import time
import types

import numpy as np

MODES = ("exact", "perturbed", "unstable", "broken", "refused", "fp32")
MODE, SCRIPT = sys.argv[1], Path(sys.argv[2]).resolve()
if MODE not in MODES:
    sys.exit(f"qualification_on_cpu.py: MODE must be one of {', '.join(MODES)}")
sys.path.insert(0, str(Path(__file__).resolve().parent / "excitations"))
from fakedevice import fake_cupy  # noqa: E402

capability = tuple(int(x) for x in os.environ.get("RCMPS_CPU_COMPUTE_CAPABILITY", "8.0").split("."))
device = fake_cupy(capability, resident=(lambda threads, shared, registers: 0) if MODE == "refused"
                   else (lambda threads, shared, registers: 2))


class RawKernel(device.RawKernel):
    def compile(self):
        if MODE == "broken":
            raise RuntimeError(f"stand-in compilation failure of {self.name}")


class NVRTCProgram:
    """cupy.cuda.compiler._NVRTCProgram: PTX written from the source and the -arch option."""
    def __init__(self, source, name, method="ptx"):
        self.source = source

    def compile(self, options):
        arch = next(option for option in options if option.startswith("-arch=compute_"))[len("-arch=compute_"):]
        lines = [".version 8.7", f".target sm_{arch}", ".visible .entry excitation_trajectory("]
        if "#define USE_MMA 1\n" in self.source:
            lines.append("\tmma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1,%fd2}, {%fd3}, {%fd4}, {%fd1,%fd2};")
        else:
            lines += ["\tshfl.sync.idx.b32 %r4|%p1, %r2, %r51, %r55, %r56;", "\tfma.rn.f64 %fd27, %fd2, %fd4, %fd26;"]
        if MODE == "fp32":
            lines.append("\tadd.f32 %f1, %f2, %f3;")
        return ("\n".join(lines) + "\n").encode(), {}


cuda = types.ModuleType("cupy.cuda")
cuda.runtime = device.cuda.runtime
cuda.runtime.runtimeGetVersion = lambda: 12080
cuda.driver = device.cuda.driver
cuda.get_current_stream = device.cuda.get_current_stream
cuda.get_cuda_path = lambda: None
cuda.Device = lambda *arguments: types.SimpleNamespace(id=0, compute_capability="%d%d" % capability,
                                                       mem_info=(16 << 30, 16 << 30))
cuda.compiler = types.ModuleType("cupy.cuda.compiler")
cuda.compiler._NVRTCProgram = NVRTCProgram
core = types.ModuleType("cupy._core.core")
core._get_cccl_include_options = lambda: ()
cupy = types.ModuleType("cupy")
cupy.__getattr__ = lambda name: getattr(np, name)
cupy.__version__ = "0+cpu-stand-in"
cupy.asnumpy = lambda x: np.array(x, copy=True)
cupy.cuda, cupy.RawKernel, cupy._core = cuda, RawKernel, types.ModuleType("cupy._core")
cupy._core.core = core
sys.modules.update({"cupy": cupy, "cupy.cuda": cuda, "cupy.cuda.compiler": cuda.compiler,
                    "cupy._core": cupy._core, "cupy._core.core": core})

from rcmps_gpu.excitations import fused, ode  # noqa: E402


def integrate(self, system, points, endpoint, integrator):
    """The launch: ode.integrate_host into the one sample buffer of the kernel."""
    started, statistics = time.perf_counter(), {}
    samples = ode.integrate_host(system, points, endpoint, integrator, statistics=statistics)
    self.launches = getattr(self, "launches", 0) + 1
    if MODE == "perturbed":
        samples = samples * (1 + 1e-6)
    if MODE == "unstable" and self.launches % 2 == 0:
        samples[-1] = samples[-1] * (1 + 2.0 ** -52)
    if self.samples is None or self.samples.shape != samples.shape:
        self.samples = np.empty_like(samples)
    self.samples[...] = samples
    self.last = dict(statistics, status=0, status_name="success", t=float(endpoint), device_milliseconds=1.0,
                     seconds=time.perf_counter() - started)
    return self.samples


fused.FusedTrajectory.integrate = integrate
sys.path.insert(0, str(SCRIPT.parent))
sys.argv = [str(SCRIPT), *sys.argv[3:]]
runpy.run_path(str(SCRIPT), run_name="__main__")
