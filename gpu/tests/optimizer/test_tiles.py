"""The tile products of the trajectory kernel, on the CPU: model, selection rule, PTX audit.

The kernel computes its d x d products as 8 x 8 x 4 tiles in the fragment layout of the PTX
instruction mma.sync.aligned.m8n8k4.row.col.f64: lane l of a warp holds A[l/4][l%4],
B[l%4][l/4] and the accumulators C[l/4][2 (l%4)] and C[l/4][2 (l%4) + 1].
trajectory_source(d, "mma") computes a tile with that instruction on the FP64 tensor cores
(compute capability 8.0 or later); trajectory_source(d, "scalar") brings the operands to
each lane with warp shuffles and accumulates with one FMA per k, in k order (SCALAR_TILE).

Model    (1) SCALAR_TILE, executed lane by lane in Python with a correctly rounded FMA
         (Fraction arithmetic), equals bitwise the m8n8k4 product computed as a chain of FMAs
         in k order on every element; the comparison is sensitive: on the same random tiles
         a product with separate roundings of the multiplications and of the additions, and a
         product with a single rounding of the four-term sum, differ from it, and so do they
         on a tie (c = 1 plus two products of 2^-53). Whether a GPU's FP64 tensor core
         computes the chain is what qualification/optimizer/gpu_optimizer_check.py --stages
         unit measures on the device; on the A100 the excitation kernel's two tile paths gave
         equal trajectories (largest difference 0).
Rule     (2) generic_layout.select_tiles on a grid of compute capabilities; the launcher copy
         optimizer/layout.py is byte for byte generic_layout.py; describe records the request.
Audit    (3) audit_trajectory_ptx on hand-written PTX.
Compile  (4) NVRTC alone (no GPU, no CuPy): trajectory_source(d, tiles) for d = 4, 64 and 128
         with the options of trajectory_ptx_report, the scalar tiles for compute_60, 70, 75,
         80, 86 and 90 and the tensor-core tiles for compute_80, 86 and 90, pass the audit;
         ptxas (through NVRTC) assembles the kernel to a cubin at d = 8 for sm_60, 70 and 75
         (scalar tiles) and sm_80, 86 and 90 (tensor-core tiles), and at d = 128 for sm_70;
         the tensor-core source for compute_70 compiles to PTX but fails the audit, and ptxas
         refuses it for sm_70; the scalar tile function alone is a chain of 8 FMAs fed by 24
         warp shuffles on every target. Skipped without the nvidia-cuda-nvrtc and
         nvidia-cuda-runtime wheels and a directory holding the header nv/target
         (RCMPS_CUDA_INCLUDE, then ~/.julia/artifacts/*/include).
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import functools
import glob
import importlib.util
import math
import os
from pathlib import Path
import re
import unittest

import numpy as np

from rcmps_gpu.optimizer import layout
from rcmps_gpu.optimizer.runtime import generic_layout
from rcmps_gpu.optimizer.runtime import persistent_trajectory_backend as backend

LANE = np.arange(32)
ROW, K, COL = LANE >> 2, LANE & 3, 2 * (LANE & 3)
CAPABILITIES = ((5, 2), (6, 0), (6, 1), (7, 0), (7, 2), (7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0),
                (12, 0))
GPU_CHECK = Path(__file__).resolve().parents[2] / "qualification" / "optimizer" / "gpu_optimizer_check.py"


def fma(a, b, c):
    """a b + c rounded once to the nearest double (math.fma exists only from Python 3.13);
    an exact zero is -0.0 only for a zero product and c of the same negative sign."""
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        product_negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
        if (a == 0 or b == 0) and c == 0 and product_negative and math.copysign(1.0, c) < 0:
            return -0.0
        return 0.0
    return float(exact)


def product(A, B, C, mode):
    """The 8 x 8 x 4 product A B + C element by element. mode "chain": FMAs in k order 0..3;
    "separate": a rounded multiplication and a rounded addition per k; "dot": the exact
    four-term sum plus C, rounded once."""
    D = np.empty((8, 8))
    for i in range(8):
        for j in range(8):
            if mode == "chain":
                value = C[i, j]
                for k in range(4):
                    value = fma(A[i, k], B[k, j], value)
            elif mode == "separate":
                value = C[i, j]
                for k in range(4):
                    value = value + A[i, k] * B[k, j]
            else:
                value = float(sum(Fraction(A[i, k]) * Fraction(B[k, j]) for k in range(4)) + Fraction(C[i, j]))
            D[i, j] = value
    return D


def scalar_tile(a, b, c0, c1):
    """SCALAR_TILE for the 32 lanes of a warp: __shfl_sync(mask, v, l) is v[l], and
    c0+=av*b0 is the FMA fma(av, b0, c0) (the compiler contracts it: checked in the PTX)."""
    rl, cl = LANE >> 2, 2 * (LANE & 3)
    c0, c1 = c0.copy(), c1.copy()
    for q in range(4):
        av, b0, b1 = a[(rl << 2) | q], b[(cl << 2) | q], b[((cl + 1) << 2) | q]
        c0 = np.array([fma(x, y, z) for x, y, z in zip(av, b0, c0)])
        c1 = np.array([fma(x, y, z) for x, y, z in zip(av, b1, c1)])
    return c0, c1


def tile(A, B, C):
    """A B + C through the fragments of the kernel and SCALAR_TILE."""
    c0, c1 = scalar_tile(A[ROW, K], B[K, ROW], C[ROW, COL], C[ROW, COL + 1])
    D = np.empty((8, 8))
    D[ROW, COL], D[ROW, COL + 1] = c0, c1
    return D


def same(x, y):
    return np.array_equal(np.asarray(x, dtype=np.float64).view(np.uint64), np.asarray(y, dtype=np.float64).view(np.uint64))


class Model(unittest.TestCase):
    def test_scalar_tile_text(self):
        """(1) The lines of SCALAR_TILE that scalar_tile models."""
        text = re.sub(r"\s+", "", backend.SCALAR_TILE)
        for line in ("void mma(double a,double b,double &c0,double &c1){",
                     "const int lane=threadIdx.x&31,rl=lane>>2,cl=2*(lane&3);", "for(int q=0;q<4;++q){",
                     "double av=__shfl_sync(0xffffffffu,a,(rl<<2)|q);",
                     "double b0=__shfl_sync(0xffffffffu,b,(cl<<2)|q),b1=__shfl_sync(0xffffffffu,b,((cl+1)<<2)|q);",
                     "c0+=av*b0;c1+=av*b1;"):
            self.assertIn(re.sub(r"\s+", "", line), text, "the scalar tile product changed: update scalar_tile")
        self.assertEqual(len(set(zip(ROW, K))), 32)                       # the fragments are bijections
        self.assertEqual(len(set(zip(ROW, COL)) | set(zip(ROW, COL + 1))), 64)

    def test_scalar_tile_is_the_chain_of_FMAs_bitwise(self):
        """(1) 200 random tiles, A with exponents spread over 2^-30..2^30: SCALAR_TILE equals
        the FMA chain bitwise, and differs from the separate-rounding and single-rounding
        products in every one of them."""
        rng = np.random.default_rng(3)
        differs = dict(separate=0, dot=0)
        trials = 200
        for _ in range(trials):
            A, B, C = rng.standard_normal((8, 4)), rng.standard_normal((4, 8)), rng.standard_normal((8, 8))
            A *= 2.0 ** rng.integers(-30, 30, size=A.shape)
            got = tile(A, B, C)
            self.assertTrue(same(got, product(A, B, C, "chain")))
            for mode in differs:
                differs[mode] += not same(got, product(A, B, C, mode))
        self.assertEqual(differs, dict(separate=trials, dot=trials))

    def test_ties_and_signed_zeros(self):
        """(1) c = 1 plus two products of 2^-53: the chain rounds to 1 twice, the single
        rounding gives 1 + 2^-52. A tile of zeros with c = -0 keeps -0 where the products
        are -0 and gives +0 where they are +0."""
        A, B, C = np.zeros((8, 4)), np.zeros((4, 8)), np.ones((8, 8))
        A[:, 0] = A[:, 1] = 2.0 ** -27
        B[0, :] = B[1, :] = 2.0 ** -26
        self.assertTrue(same(tile(A, B, C), np.ones((8, 8))))
        self.assertTrue(same(product(A, B, C, "chain"), np.ones((8, 8))))
        self.assertTrue(same(product(A, B, C, "dot"), np.full((8, 8), 1 + 2.0 ** -52)))
        A, B, C = np.zeros((8, 4)), np.zeros((4, 8)), np.full((8, 8), -0.0)
        A[:4] = -0.0
        B[:] = 1.0
        want = np.zeros((8, 8))
        want[:4] = -0.0
        self.assertTrue(same(tile(A, B, C), want) and same(product(A, B, C, "chain"), want))

    @unittest.skipUnless(GPU_CHECK.is_file(), "qualification/optimizer/gpu_optimizer_check.py is not here")
    def test_model_of_the_qualification_script(self):
        """(1) The host model that gpu_optimizer_check.py --stages unit compares the device
        tiles with is this chain model."""
        spec = importlib.util.spec_from_file_location("gpu_optimizer_check", GPU_CHECK)
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        rng = np.random.default_rng(5)
        for A, B, C in check.unit_cases(rng):
            self.assertTrue(same(check.chain_model(A, B, C), product(A, B, C, "chain")))
            self.assertTrue(same(check.chain_model(A, B, C), tile(A, B, C)))


class Rule(unittest.TestCase):
    def test_select_tiles(self):
        """(2) auto: mma from compute capability 8.0 on, scalar below; mma below 8.0 refused
        with a message; scalar everywhere; other requests refused."""
        for major, minor in CAPABILITIES:
            native = (major, minor) >= (8, 0)
            with self.subTest(capability=(major, minor)):
                self.assertEqual(generic_layout.select_tiles("auto", major, minor), "mma" if native else "scalar")
                self.assertEqual(generic_layout.select_tiles("scalar", major, minor), "scalar")
                if native:
                    self.assertEqual(generic_layout.select_tiles("mma", major, minor), "mma")
                else:
                    with self.assertRaisesRegex(ValueError, rf"--tiles mma needs FP64 tensor-core instructions .*"
                                                            rf"this GPU is {major}\.{minor}: use --tiles auto or scalar"):
                        generic_layout.select_tiles("mma", major, minor)
                for request in ("MMA", "", None, "tensor"):
                    with self.assertRaisesRegex(ValueError, "tiles must be one of auto, mma, scalar"):
                        generic_layout.select_tiles(request, major, minor)
        self.assertEqual(generic_layout.select_tiles("auto", "7", "5"), "scalar")   # CuPy may give strings

    def test_launcher_copy_and_plan_description(self):
        """(2) optimizer/layout.py, which the command line imports, is generic_layout.py byte
        for byte; describe records the requested tiles and refuses an unknown request."""
        self.assertEqual(Path(layout.__file__).read_bytes(), Path(generic_layout.__file__).read_bytes())
        for tiles in ("auto", "mma", "scalar"):
            info = layout.describe(64, 512, tiles)
            self.assertEqual(info["tiles_requested"], tiles)
            self.assertIn("scalar", info["arithmetic"])
        self.assertEqual(layout.describe(64)["tiles_requested"], "auto")
        with self.assertRaisesRegex(ValueError, "tiles must be one of"):
            layout.describe(64, 512, "fast")
        with self.assertRaisesRegex(ValueError, "tiles must be one of mma, scalar"):
            backend.trajectory_source(8, "auto")


SCALAR_PTX = ".version 8.7\n.target sm_70\n\tshfl.sync.idx.b32 %r4|%p1, %r2, %r51, %r55, %r56;\n" \
             "\tfma.rn.f64 %fd27, %fd2, %fd4, %fd26;\n"
MMA_PTX = ".version 8.7\n.target sm_80\n\tmma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1,%fd2}, {%fd3}, " \
          "{%fd4}, {%fd1,%fd2};\n\tfma.rn.f64 %fd27, %fd2, %fd4, %fd26;\n"


class Audit(unittest.TestCase):
    def test_audit_on_written_PTX(self):
        """(3) mma: FP64 mma.sync for a target 8.0 or later; scalar: no mma instruction, warp
        shuffles and FP64 FMAs; both: no FP32 arithmetic (FP32 comparisons are not)."""
        audit = backend.audit_trajectory_ptx(MMA_PTX, "mma")
        self.assertEqual((audit["tiles"], audit["ptx_target"], audit["f64_mma_instructions"]),
                         ("mma", "sm_80", ["mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"]))
        audit = backend.audit_trajectory_ptx(SCALAR_PTX, "scalar")
        self.assertEqual((audit["f64_mma_instructions"], audit["shfl_sync_idx_instructions"],
                          audit["fma_rn_f64_instructions"]), ([], 1, 1))
        self.assertEqual(backend.audit_trajectory_ptx(SCALAR_PTX + "\tsetp.lt.f32 %p1, %f1, %f2;\n",
                                                      "scalar")["fp32_arithmetic_instructions"], [])
        for ptx, tiles, message in (
                (MMA_PTX.replace("sm_80", "sm_75"), "mma", "for target sm_75"),
                (SCALAR_PTX, "mma", "no FP64 mma.sync"),
                (MMA_PTX, "scalar", "mma instructions"),
                (SCALAR_PTX + "\twmma.mma.sync.aligned.row.col.m16n16k16.f32.f16 {%f1};\n", "scalar", "wmma"),
                (SCALAR_PTX.replace("fma.rn.f64", "mul.rn.f64"), "scalar", "no warp shuffles or FP64 FMAs"),
                (SCALAR_PTX.replace(".target sm_70\n", ""), "scalar", "no .target"),
                (MMA_PTX + "\tadd.f32 %f1, %f2, %f3;\n", "mma", "FP32 arithmetic"),
                (SCALAR_PTX + "\tfma.rn.ftz.f32 %f1, %f2, %f3, %f4;\n", "scalar", "FP32 arithmetic")):
            with self.subTest(tiles=tiles, message=message), self.assertRaisesRegex(RuntimeError, message):
                backend.audit_trajectory_ptx(ptx, tiles)
        with self.assertRaises(ValueError):
            backend.audit_trajectory_ptx(MMA_PTX, "auto")


@functools.lru_cache(maxsize=None)
def nvrtc():
    """(libnvrtc, include options), or (None, reason to skip)."""
    try:
        specs = [importlib.util.find_spec(name) for name in ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime")]
    except ModuleNotFoundError:
        specs = [None]
    if any(spec is None or not spec.submodule_search_locations for spec in specs):
        return None, "the wheels nvidia-cuda-nvrtc and nvidia-cuda-runtime are not installed"
    roots = [Path(list(spec.submodule_search_locations)[0]) for spec in specs]
    library = sorted((roots[0] / "lib").glob("libnvrtc.so*"))
    if not library:
        return None, "libnvrtc.so was not found in the nvidia-cuda-nvrtc wheel"
    candidates = [os.environ.get("RCMPS_CUDA_INCLUDE", "")]
    candidates += sorted(glob.glob(os.path.expanduser("~/.julia/artifacts/*/include")))
    include = next((path for path in candidates if path and os.path.isfile(os.path.join(path, "nv", "target"))), None)
    if include is None:
        return None, ("no directory with the header nv/target: set RCMPS_CUDA_INCLUDE "
                      "(searched it and ~/.julia/artifacts/*/include)")
    try:
        lib = ctypes.CDLL(str(library[0]), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        return None, f"libnvrtc cannot be loaded: {error}"
    return lib, (f"-I{roots[1] / 'include'}", f"-I{include}")


def compile_cuda(source, arch, options=(), cubin=False):
    """(status, PTX text or cubin bytes or None, log) of NVRTC for -arch=arch."""
    lib, includes = nvrtc()
    program = ctypes.c_void_p()
    if lib.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"tiles.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    flags = [*options, "--device-as-default-execution-space", f"-arch={arch}", *includes]
    status = lib.nvrtcCompileProgram(program, len(flags), (ctypes.c_char_p * len(flags))(*(f.encode() for f in flags)))
    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetProgramLog(program, log)
    output = None
    if status == 0:
        get_size, get = (lib.nvrtcGetCUBINSize, lib.nvrtcGetCUBIN) if cubin else (lib.nvrtcGetPTXSize, lib.nvrtcGetPTX)
        get_size(program, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        get(program, buffer)
        output = buffer.raw if cubin else buffer.value.decode()
    lib.nvrtcDestroyProgram(ctypes.byref(program))
    return status, output, log.value.decode()


TILE_KERNEL = r'''
extern "C" __global__ void tile(const double* a,const double* b,double* c){
    int l=threadIdx.x; double c0=c[2*l],c1=c[2*l+1];
    mma(a[l],b[l],c0,c1);
    c[2*l]=c0;c[2*l+1]=c1;
}
'''


class Compile(unittest.TestCase):
    SCALAR_TARGETS = ("compute_60", "compute_70", "compute_75", "compute_80", "compute_86", "compute_90")
    MMA_TARGETS = ("compute_80", "compute_86", "compute_90")

    def setUp(self):
        lib, reason = nvrtc()
        if lib is None:
            self.skipTest(reason)

    def test_trajectory_kernel_compiles_and_passes_the_audit(self):
        """(4) Both tile paths at d = 4, 64 and 128 (static and dynamic panels, one to sixteen
        warps), with the options of trajectory_ptx_report, for each target of its path."""
        for d in (4, 64, 128):
            options = ("--std=c++17", "--device-c") + backend.compilation_options(d)[1:]
            for tiles, targets in (("scalar", self.SCALAR_TARGETS), ("mma", self.MMA_TARGETS)):
                for arch in targets:
                    with self.subTest(d=d, tiles=tiles, arch=arch):
                        status, ptx, log = compile_cuda(backend.trajectory_source(d, tiles), arch, options)
                        self.assertEqual(status, 0, log)
                        audit = backend.audit_trajectory_ptx(ptx, tiles)
                        self.assertEqual(audit["ptx_target"], arch.replace("compute", "sm"))
                        self.assertIn(".entry persistent_trajectory(", ptx)
                        if tiles == "scalar":
                            self.assertEqual(audit["f64_mma_instructions"], [])
                            self.assertGreaterEqual(audit["shfl_sync_idx_instructions"], 24 * 72)

    def test_trajectory_kernel_assembles_for_each_target(self):
        """(4) PTX alone does not show an instruction that only later targets assemble (NVRTC
        writes mma.sync f64 for compute_70, below). ptxas, run by NVRTC for sm_XX, assembles
        the kernel with the options of a run (cooperative: --device-c) to a cubin: at d = 8
        with the scalar tiles for sm_60, 70 and 75 and the tensor-core tiles for sm_80, 86 and
        90, and at d = 128 (dynamic shared memory, register cap 64) with the scalar tiles for
        sm_70."""
        cases = [(8, "scalar", arch) for arch in ("sm_60", "sm_70", "sm_75")]
        cases += [(8, "mma", arch) for arch in ("sm_80", "sm_86", "sm_90")] + [(128, "scalar", "sm_70")]
        for d, tiles, arch in cases:
            with self.subTest(d=d, tiles=tiles, arch=arch):
                options = ("--std=c++17", "--device-c") + backend.compilation_options(d)[1:]
                status, cubin, log = compile_cuda(backend.trajectory_source(d, tiles), arch, options, cubin=True)
                self.assertEqual(status, 0, log)
                self.assertEqual(cubin[:4], b"\x7fELF")

    def test_tensor_core_source_below_compute_capability_8(self):
        """(4) NVRTC writes PTX with mma.sync f64 for compute_70 (only ptxas refuses it), which
        is why the audit checks the target, and the tile selection refuses mma below 8.0
        before anything is compiled."""
        status, ptx, log = compile_cuda(backend.trajectory_source(4, "mma"), "compute_70", ("--std=c++17", "--device-c"))
        self.assertEqual(status, 0, log)
        self.assertIn("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64", ptx)
        with self.assertRaisesRegex(RuntimeError, "for target sm_70"):
            backend.audit_trajectory_ptx(ptx, "mma")
        status, _, log = compile_cuda(backend.MMA_TILE + TILE_KERNEL, "sm_70", ("--std=c++17",), cubin=True)
        self.assertNotEqual(status, 0)
        status, cubin, log = compile_cuda(backend.MMA_TILE + TILE_KERNEL, "sm_80", ("--std=c++17",), cubin=True)
        self.assertEqual(status, 0, log)
        status, cubin, log = compile_cuda(backend.SCALAR_TILE + TILE_KERNEL, "sm_70", ("--std=c++17",), cubin=True)
        self.assertEqual(status, 0, log)

    def test_scalar_tile_function_is_a_chain_of_FMAs(self):
        """(4) SCALAR_TILE alone: 24 32-bit warp shuffles (three doubles per k) and 8
        fma.rn.f64, each FMA adding to the result of the previous one of its accumulator, the
        last two stored; no separate FP64 multiplication or addition, on every target."""
        for arch in self.SCALAR_TARGETS:
            with self.subTest(arch=arch):
                status, ptx, log = compile_cuda(backend.SCALAR_TILE + TILE_KERNEL, arch, ("--std=c++17",))
                self.assertEqual(status, 0, log)
                body = ptx[ptx.index(".entry tile("):]
                self.assertEqual(len(re.findall(r"\bshfl\.sync\.idx\.b32\b", body)), 24)
                self.assertEqual(re.findall(r"\b(?:mul|add|sub)\.(?:rn\.)?f64\b", body), [])
                fmas = re.findall(r"fma\.rn\.f64\s+(%fd\d+),\s*(%fd\d+),\s*(%fd\d+),\s*(%fd\d+);", body)
                self.assertEqual(len(fmas), 8)
                results = [f[0] for f in fmas]
                chains = [[i] for i, f in enumerate(fmas) if f[3] not in results]   # adds to a loaded c
                for chain in chains:
                    while True:
                        following = [i for i, f in enumerate(fmas) if f[3] == fmas[chain[-1]][0]]
                        self.assertLessEqual(len(following), 1)
                        if not following:
                            break
                        chain.append(following[0])
                self.assertEqual([len(chain) for chain in chains], [4, 4])
                for i, j in zip(*chains):      # step k of c0 and of c1 share their A operand av
                    self.assertTrue(set(fmas[i][1:3]) & set(fmas[j][1:3]))
                stores = re.findall(r"st\.global\.f64\s+\[[^\]]+\],\s*(%fd\d+);", body)
                self.assertEqual(sorted(stores), sorted(fmas[chain[-1]][0] for chain in chains))


if __name__ == "__main__":
    unittest.main()
