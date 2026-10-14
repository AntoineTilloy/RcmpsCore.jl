"""One shape/resource policy for the persistent FP64 Z2 kernel.

D is the full bond dimension; d is the packed parity-block dimension.
The CUDA row mapping assigns one warp to each eight-column tile. Its current
one-row-per-CTA design therefore has a finite 1024-thread block limit.
No CUDA imports, devices, allocations, numerical solves or network calls.
"""
from __future__ import annotations

from numbers import Integral

MAX_PACKED_D = 256
STATIC_SHARED_LIMIT = 48 * 1024
HISTORICAL_PACKED = (2, 3, 4, 8, 16, 32, 48, 64, 128)


def packed_geometry(d):
    if isinstance(d, bool) or not isinstance(d, Integral) or not 1 <= d <= MAX_PACKED_D:
        raise ValueError(f"packed d must be an integer in [1, {MAX_PACKED_D}]")
    d = int(d)
    warps = (d + 7) // 8
    return dict(d=d, warps=warps, threads=32 * warps, blocks=10 * warps + 1,
                shared_bytes=4 * 8 * d * 16 + 2 * warps * 8)


def bond_geometry(D):
    if isinstance(D, bool) or not isinstance(D, Integral) or D < 2 or D % 2:
        raise ValueError("the packed Z2 implementation requires an even integer D >= 2")
    return dict(D=int(D), **packed_geometry(D // 2))


def dynamic_shared_bytes(d):
    size = packed_geometry(d)["shared_bytes"]
    return size if size > STATIC_SHARED_LIMIT else 0


def compilation_options(d):
    packed_geometry(d)
    # Preserve every historical compiler option, including packed128's cap.
    # Larger new row blocks use the same cap; actual occupancy remains decisive.
    return ("--std=c++17", "--maxrregcount=64") if d > 64 else ("--std=c++17",)


def configure_shared_memory(kernel, props, info):
    if info != packed_geometry(info["d"]):
        raise ValueError("kernel geometry differs from the dimension policy")
    if (int(props["major"]), int(props["minor"])) not in ((8, 0), (9, 0)):
        raise ValueError("this native FP64 MMA path supports A100 SM80 and H100 SM90")
    if not props.get("cooperativeLaunch", False):
        raise ValueError("the one-kernel trajectory requires cooperative launch")
    dynamic = dynamic_shared_bytes(info["d"])
    limit = int(props.get("sharedMemPerBlockOptin" if dynamic else "sharedMemPerBlock", 0))
    if info["shared_bytes"] > limit:
        raise RuntimeError(f"kernel shared memory {info['shared_bytes']} exceeds device limit {limit}")
    if dynamic:
        kernel.max_dynamic_shared_size_bytes = dynamic
        kernel.preferred_shared_memory_carveout = 100
        if int(kernel.max_dynamic_shared_size_bytes) < dynamic:
            raise RuntimeError("dynamic shared-memory opt-in did not take effect")
    if info["threads"] > min(int(kernel.max_threads_per_block), int(props["maxThreadsPerBlock"])):
        raise RuntimeError("dimension exceeds the compiled/device thread-block limit")
    expected = 0 if dynamic else info["shared_bytes"]
    if int(kernel.shared_size_bytes) != expected:
        raise RuntimeError("unexpected compiled static shared-memory layout")
    return dynamic


def describe(D, quadrature=512):
    info = bond_geometry(D)
    if isinstance(quadrature, bool) or not isinstance(quadrature, Integral) or quadrature < 8:
        raise ValueError("quadrature must be an integer >= 8")
    samples = (2 * int(quadrature) + 1) * (10 * info["d"] ** 2 + 1) * 16
    return dict(**info, compiler_options=list(compilation_options(info["d"])),
                dynamic_shared_bytes=dynamic_shared_bytes(info["d"]),
                packed_sample_bytes=samples,
                sample_memory_is_not_total_peak=True,
                arithmetic="float64 / complex128; native FP64 MMA",
                full_grid_residency_requires_actual_device_query=True,
                historical_shape=info["d"] in HISTORICAL_PACKED,
                generic_runtime_actual_GPU_qualified=False)
