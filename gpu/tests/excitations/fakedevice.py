"""CPU stand-in for the CuPy calls that set up fused.FusedTrajectory (no kernel runs).

`fake_cupy(...)` returns an object to pass as `cp` (or to install as sys.modules["cupy"])
with the device properties of a GPU of the given compute capability, a RawKernel that
compiles nothing but records its source and options (`cp.compiled`), the occupancy query
given as `resident(threads, shared_bytes, registers)` -> thread blocks per multiprocessor,
and NumPy for the buffers. Launching a kernel raises AssertionError. Unlike fakecupy, it
emulates no array semantics: it serves the tests of the tile selection and of the
residency checks, which happen before any array reaches the device.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def fake_cupy(capability=(8, 0), *, multiprocessors=108, resident=lambda threads, shared, registers: 2,
              optin=166912, cooperative=True, run_time_limit=False, registers=168):
    compiled = []

    class RawKernel:
        def __init__(self, code, name, options=(), enable_cooperative_groups=False):
            cap = [int(option.split("=")[1]) for option in options if option.startswith("--maxrregcount=")]
            self.code, self.name, self.options = code, name, tuple(options)
            self.cooperative = enable_cooperative_groups
            self.max_dynamic_shared_size_bytes = 0
            self.kernel = SimpleNamespace(ptr=len(compiled) + 1)
            self.num_regs = min([registers, *cap])
            self.local_size_bytes = 0
            compiled.append(self)

        def compile(self):
            pass

        def __call__(self, *arguments, **options):
            raise AssertionError("fakedevice: no kernel launch on the CPU")

    def occupancy(pointer, threads, shared):
        return resident(threads, shared, compiled[pointer - 1].num_regs)

    props = dict(name=b"CPU stand-in", major=capability[0], minor=capability[1], cooperativeLaunch=cooperative,
                 sharedMemPerBlockOptin=optin, sharedMemPerBlock=49152, maxThreadsPerBlock=1024,
                 multiProcessorCount=multiprocessors, clockRate=1410000, kernelExecTimeoutEnabled=run_time_limit)
    cuda = SimpleNamespace(
        Device=lambda *arguments: SimpleNamespace(id=0, mem_info=(0, 16 << 30)),
        runtime=SimpleNamespace(getDeviceProperties=lambda device: dict(props), driverGetVersion=lambda: 12090),
        driver=SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=occupancy),
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None))
    return SimpleNamespace(__name__="cupy", __version__="0+cpu-stand-in", cuda=cuda, RawKernel=RawKernel,
                           compiled=compiled, asarray=np.asarray, empty=np.empty, float64=np.float64,
                           complex128=np.complex128, int64=np.int64)
