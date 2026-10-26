"""One shape/resource policy for the persistent FP64 kernel, Z2 (packed) or unpacked.

D is the full bond dimension; d is the matrix dimension of the kernel: the packed
parity-block dimension D/2 for Z2 states, D for the unpacked channels (symmetry none).
The CUDA row mapping assigns one warp to each eight-column tile. Its current
one-row-per-CTA design therefore has a finite 1024-thread block limit.
No CUDA imports, devices, allocations, numerical solves or network calls.
"""
from __future__ import annotations

from numbers import Integral

MAX_PACKED_D = 256
STATIC_SHARED_LIMIT = 48 * 1024
HISTORICAL_PACKED = (2, 3, 4, 8, 16, 32, 48, 64, 128)
TILES = ("mma", "scalar")
TILE_CHOICES = ("auto",) + TILES
SYMMETRIES = ("z2", "none")


def select_tiles(requested, major, minor):
    """The tile product of a kernel on a device of compute capability major.minor: "mma",
    the native FP64 tensor-core instruction mma.sync m8n8k4 (compute capability 8.0 or
    later), or "scalar", the same fragment layout with warp shuffles and a chain of FP64
    FMAs (any GPU with cooperative launch). "auto" gives mma from 8.0 on and scalar below;
    mma below 8.0 is refused. excitations/fused.py applies the same rule."""
    if requested not in TILE_CHOICES:
        raise ValueError(f"tiles must be one of {', '.join(TILE_CHOICES)}, not {requested!r}")
    native = (int(major), int(minor)) >= (8, 0)
    if requested == "auto":
        return "mma" if native else "scalar"
    if requested == "mma" and not native:
        raise ValueError(f"--tiles mma needs FP64 tensor-core instructions (mma.sync m8n8k4 f64, compute "
                         f"capability 8.0 or later); this GPU is {int(major)}.{int(minor)}: use --tiles auto or scalar")
    return requested


def packed_geometry(d):
    if isinstance(d, bool) or not isinstance(d, Integral) or not 1 <= d <= MAX_PACKED_D:
        raise ValueError(f"packed d must be an integer in [1, {MAX_PACKED_D}]")
    d = int(d)
    warps = (d + 7) // 8
    return dict(d=d, warps=warps, threads=32 * warps, blocks=10 * warps + 1,
                shared_bytes=4 * 8 * d * 16 + 2 * warps * 8)


def unpacked_geometry(D):
    """The grid of the kernel on the seven unpacked D x D channels (states without Z2
    symmetry, any D up to MAX_PACKED_D): the row panels of packed_geometry(D), seven
    channels of row blocks and the scalar block."""
    info = packed_geometry(D)
    return dict(info, blocks=7 * info["warps"] + 1, layout="unpacked")


def bond_geometry(D, symmetry="z2"):
    if symmetry not in SYMMETRIES:
        raise ValueError(f"symmetry must be one of {', '.join(SYMMETRIES)}, not {symmetry!r}")
    if symmetry == "none":
        if isinstance(D, bool) or not isinstance(D, Integral) or not 1 <= D <= MAX_PACKED_D:
            raise ValueError(f"the unpacked implementation requires an integer D in [1, {MAX_PACKED_D}]")
        return dict(D=int(D), **unpacked_geometry(D))
    if isinstance(D, bool) or not isinstance(D, Integral) or D < 2 or D % 2:
        raise ValueError("the packed Z2 implementation requires an even integer D >= 2")
    return dict(D=int(D), **packed_geometry(D // 2))


def dynamic_shared_bytes(d):
    size = packed_geometry(d)["shared_bytes"]
    return size if size > STATIC_SHARED_LIMIT else 0


def compilation_options(d, register_cap=None):
    packed_geometry(d)
    if register_cap is not None:
        if register_cap not in register_caps(d)[1:]:
            raise ValueError(f"register cap {register_cap!r} is not one of {register_caps(d)[1:]} for packed d={d}")
        return ("--std=c++17", f"--maxrregcount={register_cap}")
    # Preserve every historical compiler option, including packed128's cap.
    # Larger new row blocks use the same cap; actual occupancy remains decisive.
    return ("--std=c++17", "--maxrregcount=64") if d > 64 else ("--std=c++17",)


def register_caps(d):
    """Register caps to try, in order, until the whole cooperative grid is resident: None
    (the historical options of compilation_options) first, then, for d <= 64, which compiles
    without a cap (about 100 to 250 registers from sm_60 to sm_90 with NVRTC 12.8: 96 at d = 1
    on sm_86, 248 at d = 3 on sm_60; so few blocks per multiprocessor on GPUs with fewer or
    smaller multiprocessors than the A100 and H100), 128, 96 and 64 (more spills)."""
    packed_geometry(d)
    return (None, 128, 96, 64) if d <= 64 else (None,)


def configure_shared_memory(kernel, props, info, tiles="mma"):
    if info != (unpacked_geometry if info.get("layout") == "unpacked" else packed_geometry)(info["d"]):
        raise ValueError("kernel geometry differs from the dimension policy")
    if tiles not in TILES:
        raise ValueError(f"a compiled kernel has tiles {' or '.join(TILES)}, not {tiles!r}")
    select_tiles(tiles, props["major"], props["minor"])
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


def describe(D, quadrature=512, tiles="auto", symmetry="z2"):
    """The plan's layout record. For symmetry none it has the keys symmetry and layout
    besides; packed_sample_bytes is then the size of the unpacked samples."""
    info = bond_geometry(D, symmetry)
    if isinstance(quadrature, bool) or not isinstance(quadrature, Integral) or quadrature < 8:
        raise ValueError("quadrature must be an integer >= 8")
    select_tiles(tiles, 8, 0)
    channels = 7 if symmetry == "none" else 10
    samples = (2 * int(quadrature) + 1) * (channels * info["d"] ** 2 + 1) * 16
    return dict(**info, compiler_options=list(compilation_options(info["d"])),
                dynamic_shared_bytes=dynamic_shared_bytes(info["d"]),
                packed_sample_bytes=samples,
                sample_memory_is_not_total_peak=True,
                arithmetic="float64 / complex128; tile products: native FP64 MMA (mma.sync m8n8k4) "
                           "or a scalar chain of FP64 FMAs with warp shuffles",
                tiles_requested=tiles,
                tile_rule="auto: mma for compute capability >= 8.0, scalar below; mma below 8.0 is refused",
                full_grid_residency_requires_actual_device_query=True,
                historical_shape=symmetry == "z2" and info["d"] in HISTORICAL_PACKED,
                generic_runtime_actual_GPU_qualified=False,
                **(dict(symmetry="none") if symmetry == "none" else {}))
