"""First checks of the excitation modules on a GPU (or, with --xp numpy, on the CPU).

Stages (choose with --stages, default all that apply):
  device    device, driver and library versions
  solver    batched preconditioned GMRES: GPU versus NumPy on the same systems, and timing
  kinetic   kinetic action: GPU versus NumPy, and timing of the tangent-independent part and of one action
  host_ode  complete action with the host-controlled integrator: GPU versus NumPy (small D only)
  compile   fused trajectory kernels of a phi^4 solve (--kernels: the action system, the pair system of
            the validation) for each packed dimension (--packed) and tile path (--tiles; default every
            path of the device: tensor cores and scalar tiles from compute capability 8.0 on, scalar
            tiles below): compilation, registers, register cap and cooperative residency, and the
            audit of the kernel's PTX for this device (ptx_audit). A kernel this device cannot hold
            or run (a cooperative grid not resident at any register cap, shared memory, threads per
            block, --tiles mma below 8.0) is a result ("resident": false, "refused"), not a failure.
Every result is one JSON line on stdout; nothing is written elsewhere. The exit status is 1
if a check failed ("ok": false: a compilation or audit failure of the compile stage).

    python gpu_bringup.py --state STATE --stages device solver kinetic [--scaling density] [--nodes 256]
STATE is an rcmps_store .jld2 file or a portable export directory of the GPU optimizer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from rcmps_gpu.excitations import coordinates
from rcmps_gpu.excitations import fused
from rcmps_gpu.excitations import hamiltonian
from rcmps_gpu.excitations import kinetic
from rcmps_gpu.excitations import models
from rcmps_gpu.excitations import ode
from rcmps_gpu.excitations import scaled
from rcmps_gpu.excitations import states


FAILED = []     # the rows with "ok": false, of every script that imports emit from here
# What FusedTrajectory (and fused.select_tiles) raise for a kernel that this device cannot
# hold or run: a result of the qualification, not a failure.
REFUSALS = ("--tiles mma needs", "unsafe cooperative grid", "exceeds the device limit", "thread-block limit")


def emit(**row):
    if row.get("ok") is False:
        FAILED.append(row)
    print(json.dumps(row, default=float), flush=True)


def finish():
    """Last line of a script: the number of failed checks, and exit status 1 if any."""
    emit(stage="summary", failed=len(FAILED))
    sys.exit(1 if FAILED else 0)


def describe(error):
    return f"{type(error).__name__}: {error}"[:2000]


def refusal(error):
    """The message of `error` if it is the refusal of a kernel this device cannot hold or run
    (REFUSALS), else None."""
    return describe(error) if any(text in str(error) for text in REFUSALS) else None


def bitwise(x, y):
    """Same shape, type and bytes (signed zeros and NaN payloads included)."""
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype
                and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes())


def device_tiles(cp, requested=None):
    """The tile paths to check: `requested`, else every one this device runs (mma and scalar
    from compute capability 8.0 on, scalar below), with the compute capability."""
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    capability = (int(props["major"]), int(props["minor"]))
    return list(requested or (("mma", "scalar") if capability >= (8, 0) else ("scalar",))), capability


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def sync(xp):
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


def load(path):
    path = Path(path)
    state = states.load_portable(path) if path.is_dir() else states.load_jld2(path)
    # Work in the density gauge, as the eigensolver does.
    return coordinates.make_coordinates(state, sector="odd" if states.excitation_parity(state) is not None else "all")


def white_tangent(c, ss, seed):
    """A random white tangent of the sector of c as a hat matrix (rho is diagonal in this gauge)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(len(c)) + 1j * rng.standard_normal(len(c))
    m = np.zeros((c.state.D, c.state.D), dtype=np.complex128)
    m[c.rows, c.cols] = x / np.linalg.norm(x)
    lam = np.real(np.diag(c.state.rho))
    return m * (ss.s_host / np.sqrt(lam))[None, :], m


def stage_device(xp, args):
    if xp.__name__ != "cupy":
        return emit(stage="device", backend="numpy", numpy=np.__version__)
    import cupy
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    free, total = cupy.cuda.Device().mem_info
    emit(stage="device", name=cupy.cuda.runtime.getDeviceName(cupy.cuda.Device().id) if hasattr(cupy.cuda.runtime, "getDeviceName") else "",
         compute_capability=[int(props["major"]), int(props["minor"])], multiprocessors=int(props["multiProcessorCount"]),
         cooperative_launch=bool(props.get("cooperativeLaunch", False)), memory_total=int(total), memory_free=int(free),
         shared_per_block=int(props["sharedMemPerBlock"]), shared_optin=int(props.get("sharedMemPerBlockOptin", 0)),
         cupy=cupy.__version__, numpy=np.__version__, driver=int(cupy.cuda.runtime.driverGetVersion()),
         runtime=int(cupy.cuda.runtime.runtimeGetVersion()))


def stage_solver(xp, args, c):
    s = c.state
    d = s.D
    reference = scaled.ScaledState(s, args.scaling)
    started = time.perf_counter()
    ss = scaled.ScaledState(s, args.scaling, xp=xp, max_basis_bytes=args.basis_bytes)
    sync(xp)
    emit(stage="solver", D=d, setup_seconds=time.perf_counter() - started,
         fixed_point_residual=ss.fixed_point["residual"], fixed_point_steps=ss.fixed_point["steps"],
         eigenbasis_condition=ss.eigenbasis_condition,
         rh_difference=float(np.linalg.norm(host(ss.rh) - reference.rh)))
    theta, _ = models.kinetic_grid(args.nodes)
    z = np.cos(theta)
    rng = np.random.default_rng(0)
    for a, b, dual, sigma in (("H", "H", False, 1.0), ("L", "H", False, 1 + 1j * z * args.momentum),
                              ("L", "L", True, 1.0), ("L", "H", True, 1 - 1j * z * args.momentum)):
        rhs = rng.standard_normal((d, len(z), d)) + 1j * rng.standard_normal((d, len(z), d))
        started = time.perf_counter()
        x, info = ss.resolvent(xp.asarray(rhs), a, b, z, sigma, dual=dual, tol=args.tol)
        sync(xp)
        seconds = time.perf_counter() - started
        row = dict(stage="solver", D=d, type=a + b + ("*" if dual else ""), systems=len(z), seconds=seconds,
                   steps=info["steps"], max_iterations=int(info["iterations"].max()),
                   max_residual=float(info["residuals"].max()))
        if args.compare and xp is not np:
            ref, _ = reference.resolvent(rhs[:, :args.compare, :].copy(), a, b, z[:args.compare],
                                         sigma if np.isscalar(sigma) else sigma[:args.compare], dual=dual, tol=args.tol)
            row["difference_from_numpy"] = float(np.linalg.norm(host(x)[:, :args.compare, :] - ref) / np.linalg.norm(ref))
        emit(**row)
    return ss, reference


def stage_kinetic(xp, args, c, ss, reference):
    terms = kinetic.kinetic_terms(models.phi4_terms(c.state.coupling))
    started = time.perf_counter()
    action = kinetic.KineticAction(ss, args.momentum, terms, args.nodes, tol=args.tol)
    sync(xp)
    prepare = time.perf_counter() - started
    wh, m = white_tangent(c, ss, 1)
    timings = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        frame = kinetic.TangentFrame(ss, args.momentum, scaled.stack([xp.asarray(wh)], xp), tol=args.tol)
        fh = action.action(frame)
        sync(xp)
        timings.append(time.perf_counter() - started)
    f = host(fh)[:, 0, :]
    row = dict(stage="kinetic", D=c.state.D, nodes=args.nodes, prepare_seconds=prepare, action_seconds=timings,
               stats=action.stats, rayleigh=complex(np.vdot(wh, f)).real, imaginary=complex(np.vdot(wh, f)).imag)
    if args.compare and xp is not np:
        ref = kinetic.KineticAction(reference, args.momentum, terms, args.nodes, tol=args.tol).action(
            kinetic.TangentFrame(reference, args.momentum, scaled.stack([wh]), tol=args.tol))[:, 0, :]
        row["difference_from_numpy"] = float(np.linalg.norm(f - ref) / np.linalg.norm(ref))
    emit(**row)


def stage_host_ode(xp, args, c, ss, reference):
    s = c.state
    terms = models.phi4_terms(s.coupling)
    wh, _ = white_tangent(c, ss, 1)
    out = {}
    for name, state_ss, module in (("device", ss, xp), ("numpy", reference, np)):
        if name == "numpy" and (xp is np or not args.compare):
            continue
        started = time.perf_counter()
        h = hamiltonian.HamiltonianAction(state_ss, args.momentum, terms, integrator=s.integrator,
                                          kinetic_nodes=args.nodes, tol=args.tol, integrate=ode.integrate_host)
        f = h(module.asarray(wh))
        sync(module)
        out[name] = host(f)
        emit(stage="host_ode", D=s.D, backend=name, seconds=time.perf_counter() - started,
             rayleigh=complex(np.vdot(wh, out[name])).real)
    if len(out) == 2:
        emit(stage="host_ode", D=s.D, difference_from_numpy=float(
            np.linalg.norm(out["device"] - out["numpy"]) / np.linalg.norm(out["numpy"])))


def ptx_audit(cp, trajectory, d, n_blocks, panels, temps):
    """Audit of the PTX of a compiled FusedTrajectory on this device: its source, compiled by
    NVRTC for this device's compute capability with the kernel's options, as the optimizer's
    trajectory_ptx_report does, passes fused.audit_ptx and the optimizer's audit of the same
    tile products (audit_trajectory_ptx: a .target directive, FP64 mma.sync for a target of
    8.0 or later on the tensor-core path; no mma, and the shuffles and FP64 FMAs of the tile
    product, on the scalar path; no FP32 arithmetic on either). The run itself does not audit
    the excitation kernel's PTX. Raises RuntimeError if the audit fails."""
    from cupy.cuda import compiler
    from cupy._core import core
    from rcmps_gpu.optimizer.runtime.persistent_trajectory_backend import audit_trajectory_ptx
    cuda_path = cp.cuda.get_cuda_path()
    directories = [] if cuda_path is None else [Path(cuda_path) / "include"]
    try:
        package = importlib.util.find_spec("nvidia.cuda_runtime")
    except ModuleNotFoundError:
        package = None
    if package and package.submodule_search_locations:
        directories.extend(Path(path) / "include" for path in package.submodule_search_locations)
    includes = tuple("-I" + str(path) for path in directories if path.is_dir())
    includes += tuple(core._get_cccl_include_options())
    arch = cp.cuda.Device().compute_capability
    source = fused.kernel_source(d, n_blocks, use_mma=trajectory.use_mma, panels=panels, temps=temps)
    program = compiler._NVRTCProgram(source, "excitation_trajectory_audit.cu", method="ptx")
    binary, _ = program.compile(tuple(trajectory.options) + (
        "--device-c", "--device-as-default-execution-space", "-arch=compute_" + arch) + includes)
    ptx = binary.decode()
    audit = dict(fused.audit_ptx(ptx, use_mma=trajectory.use_mma), **audit_trajectory_ptx(ptx, trajectory.tiles))
    if audit["ptx_target"] != "sm_" + arch:
        raise RuntimeError(f"PTX audit failed: target {audit['ptx_target']}, the device is sm_{arch}")
    return dict(audit, arch=arch)


def stage_compile(xp, args):
    if xp.__name__ != "cupy":
        return emit(stage="compile", skipped="needs a GPU")
    paths, capability = device_tiles(xp, args.tiles)
    for d in args.packed:
        for kernel in args.kernels:
            for tiles in paths:
                started = time.perf_counter()
                row = dict(stage="compile", D=2 * d, d=d, kernel=kernel, tiles=tiles, compute_capability=capability)
                n_blocks, panels, temps = fused.PHI4_KERNELS[kernel]
                try:
                    trajectory = fused.FusedTrajectory(xp, d, n_blocks, panels=panels, temps=temps,
                                                       use_mma=fused.USE_MMA[tiles])
                except Exception as error:  # report and continue with the next size
                    message = refusal(error)
                    if message:
                        emit(**row, resident=False, refused=message, ok=True)
                    else:
                        emit(**row, resident=False, error=describe(error), ok=False)
                    continue
                row.update(resident=True, seconds=time.perf_counter() - started, occupancy=trajectory.occupancy)
                try:
                    emit(**row, ptx_audit=ptx_audit(xp, trajectory, d, n_blocks, panels, temps), ok=True)
                except Exception as error:
                    emit(**row, error=describe(error), ok=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state")
    parser.add_argument("--xp", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--stages", nargs="+", default=["device", "solver", "kinetic", "compile"])
    parser.add_argument("--scaling", default="density")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--compare", type=int, default=8, help="systems recomputed with NumPy (0: none)")
    parser.add_argument("--basis-bytes", type=float, default=6e9)
    parser.add_argument("--packed", type=int, nargs="+", default=[2, 4, 8, 16, 32, 48, 64])
    parser.add_argument("--kernels", nargs="+", choices=tuple(fused.PHI4_KERNELS), default=list(fused.PHI4_KERNELS))
    parser.add_argument("--tiles", nargs="+", choices=fused.TILES, default=None,
                        help="tile paths of the compile stage (default: every path of the device)")
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--n-quad", type=int, default=512)
    args = parser.parse_args()
    if args.xp == "cupy":
        import cupy as xp
    else:
        xp = np
    if "device" in args.stages:
        stage_device(xp, args)
    if "compile" in args.stages:
        stage_compile(xp, args)
    if not args.state:
        return finish()
    c = load(args.state)
    c.state.integrator = states.IntegratorParameters(n_quad=args.n_quad, max_point=args.window, atol=1e-13, rtol=1e-12)
    ss = reference = None
    if any(stage in args.stages for stage in ("solver", "kinetic", "host_ode")):
        ss, reference = stage_solver(xp, args, c)
    if "kinetic" in args.stages:
        stage_kinetic(xp, args, c, ss, reference)
    if "host_ode" in args.stages:
        stage_host_ode(xp, args, c, ss, reference)
    finish()


if __name__ == "__main__":
    main()
