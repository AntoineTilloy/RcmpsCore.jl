"""Tests of the CPU-only parts of fused.py (fused.FusedTrajectory needs a GPU: not tested here).

Tables    (1) kernel_tables of packed.PackedActionSystem and packed_pair.PackedDiagonalPair
          (D = 4, 6, both sectors, p = 0 and 0.5, :phi^2: and :phi^4:); (2) emulate_rhs
          against system.rhs; (3) the ValueErrors of kernel_tables on corrupted tables.
Geometry  (4) geometry; (6) the thread-to-element mapping, the scalar tile product and the
          two product passes of block_rhs, re-implemented here from the kernel text.
Source    (5) kernel_source: #define values, the tensor-core branch, FP64 only; (8) the
          Chebyshev source table against models.transformed_source.
Compile   (7) compile_ptx and audit_ptx with NVRTC alone (no GPU, no CuPy); skipped without
          the nvidia-cuda-nvrtc / nvidia-cuda-runtime wheels and a directory holding the
          header nv/target (RCMPS_CUDA_INCLUDE, then ~/.julia/artifacts/*/include).

Systems: states.random_state(D, z2=True), S = sqrt(diag rho), random class tangents.
RCMPS_EXCITATION_REPORT=1 prints the largest measured errors.
"""
from __future__ import annotations

import functools
import glob
import importlib.util
import itertools
import math
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fused
from bilinear import Pair
from kinetic import TangentFrame
from models import monomial_spec, transformed_source
from packed import PackedActionSystem
from packed_pair import PackedDiagonalPair
from scaled import ScaledState, stack
from states import random_state

norm = np.linalg.norm
DIMS, SECTORS, MOMENTA, DEGREES = (4, 6), ("odd", "even"), (0.0, 0.5), (2, 4)
GRID = list(itertools.product(("action", "pair"), DIMS, SECTORS, MOMENTA, DEGREES))
WIDTHS = {"action": (4, 2), "pair": (5, 3)}       # (cached left panels, two-sided groups)
INTEGER = ("terms", "group_first", "left_constants", "temp_right", "fixed")
COLUMNS = ("term_group", "term_left_kind", "term_left_index", "term_left_adjoint", None,
           "term_right_kind", "term_right_index", "term_right_adjoint")     # of `terms`; 4 is the slot
FP32_LITERAL = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]\b")
FP32_CALL = re.compile(r"\b(?:sqrt|fabs|pow|exp|log|fmin|fmax|hypot|floor|ceil)f\s*\(")
WORST = {}


def record(name, value):
    WORST[name] = max(WORST.get(name, 0.0), float(value))


def tearDownModule():
    if os.environ.get("RCMPS_EXCITATION_REPORT") == "1":
        for name, value in WORST.items():
            print(f"{name}: {value:.3e}", file=sys.stderr)


def gaussian(seed, *shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


@functools.lru_cache(maxsize=None)
def scaled(dim):
    return ScaledState(random_state(dim, z2=True), "density")


def tangent(dim, sector):
    """Random hat tangent: block-diagonal ('odd' sector) or block-off-diagonal ('even')."""
    w, d = gaussian(5, dim, dim), dim // 2
    odd = w.copy()
    odd[:d, d:] = odd[d:, :d] = 0
    return odd if sector == "odd" else w - odd


@functools.lru_cache(maxsize=None)
def system(kind, dim, sector, p, n):
    ss, wh, spec = scaled(dim), tangent(dim, sector), monomial_spec(n)
    if kind == "pair":
        return PackedDiagonalPair(Pair(ss, p, wh, wh, tol=1e-13), spec)
    frame = TangentFrame(ss, p, stack([wh]), tol=1e-13)
    return PackedActionSystem(ss, p, spec, tuple(np.ascontiguousarray(m[:, 0, :])
                                                 for m in (frame.wh, frame.vh, frame.rw)))


def corrupted(tables, key, index, value):
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tables.items()}
    out[key][index] = value
    return out


def squash(text):
    return re.sub(r"\s+", "", text)


def defines(source):
    """The specialization header of kernel_source: {name: text of the value}."""
    head = source.split("#include", 1)[0]
    return dict(re.findall(r"^#define (\w+) (\S+)$", head, re.M))


def active(source):
    """The text the preprocessor keeps, for the only conditional of the kernel:
    #if USE_MMA / #else / #endif."""
    use, keep, out = int(defines(source)["USE_MMA"]), True, []
    for line in source.splitlines():
        word = line.strip()
        if word.startswith("#if"):
            if word != "#if USE_MMA":
                raise AssertionError(f"unexpected conditional {word}")
            keep = bool(use)
        elif word == "#else":
            keep = not use
        elif word == "#endif":
            keep = True
        elif keep:
            out.append(line)
    return "\n".join(out)


def owners(d, n_blocks):
    """How many (thread block, thread, slot) own each element of the packed state, by the
    index arithmetic of excitation_trajectory on the grid of fused.geometry."""
    info = fused.geometry(d, n_blocks)
    block_index, thread = np.meshgrid(np.arange(info["blocks"]), np.arange(info["threads"]), indexing="ij")
    lane, warp = thread & 31, thread >> 5
    block, m, n = block_index // info["warps"], (block_index % info["warps"]) * 8, warp * 8
    row, col = m + (lane >> 2), n + 2 * (lane & 3)
    count = np.zeros(info["total"], dtype=int)
    for slot in (0, 1):
        valid = (row < d) & (col + slot < d)
        np.add.at(count, (block * d * d + row * d + col + slot)[valid], 1)
    return count


def mma(a, b, c0, c1):
    """The '#else' branch of the kernel's mma for the 32 lanes of a warp at once:
    __shfl_sync(mask, v, l) is v[l]. c0 and c1 are updated in place."""
    lane = np.arange(32)
    rl, cl = lane >> 2, 2 * (lane & 3)
    for q in range(4):
        av, b0, b1 = a[(rl << 2) | q], b[(cl << 2) | q], b[((cl + 1) << 2) | q]
        c0 += av * b0
        c1 += av * b1


def zmm(a, b, acc):
    """zmm of the kernel: acc = [r0, r1, i0, i1] += complex tile product, four real mma."""
    mma(a.real, b.real, acc[0], acc[1])
    mma(-a.imag, b.imag, acc[0], acc[1])
    mma(a.real, b.imag, acc[2], acc[3])
    mma(a.imag, b.real, acc[2], acc[3])


def walk(t, constants, y, jac, source):
    """block_rhs lane by lane, as excitation_trajectory calls it: one thread block per
    (packed block, 8 rows), one warp per 8 columns, 4-wide k tiles, cached left panels,
    and the second pass on the shared two-sided intermediates. Memory that the kernel
    never writes is NaN here, so that reading it or leaving an element unowned shows."""
    nb, d = y.shape[:2]
    panels, temps = t["left_constants"].shape[1], t["temp_right"].shape[1]
    info = fused.geometry(d, nb, panels, temps)
    lane = np.arange(32)
    kk, out = lane & 3, np.full_like(y, np.nan)

    def pick(matrix, row, col, inside, adjoint=False):      # guarded read: zero where not `inside`
        r, c = np.where(inside, row, 0), np.where(inside, col, 0)
        return np.where(inside, matrix[c, r].conj() if adjoint else matrix[r, c], 0)

    for index in range(info["blocks"]):
        block, m = index // info["warps"], (index % info["warps"]) * 8
        ar = m + (lane >> 2)                                        # = row of the kernel
        panel = np.zeros((panels, 8, d), dtype=np.complex128)      # rows m..m+7 of the left constants
        for slot, c in enumerate(t["left_constants"][block]):
            for r in range(min(8, d - m) if c >= 0 else 0):
                panel[slot, r] = constants[c][m + r]
        row_temp = np.full((temps, 8, d), np.nan, dtype=np.complex128)
        fixed, partial = t["fixed"][block], []
        for warp in range(info["warps"]):
            bc, col = warp * 8 + (lane >> 2), warp * 8 + 2 * (lane & 3)
            acc = np.zeros((2 + temps, 4, 32))
            for g in range(2 + temps):
                for q in t["terms"][t["group_first"][block, g]:t["group_first"][block, g + 1]]:
                    _, lk, li, la, ls, rk, ri, ra = (int(v) for v in q)
                    for start in range(0, d, 4):
                        kp = start + kk
                        left, right = (ar < d) & (kp < d), (kp < d) & (bc < d)
                        a = pick(panel[ls], ar - m, kp, left) if lk == 0 else pick(y[li], ar, kp, left, la)
                        b = pick(constants[ri], kp, bc, right) if rk == 0 else pick(y[ri], kp, bc, right, ra)
                        zmm(a, b, acc[g])
            values = []
            for slot in (0, 1):
                valid = (ar < d) & (col + slot < d)
                group = acc[:, slot] + 1j * acc[:, 2 + slot]
                inserted = pick(constants[fixed], ar, col + slot, valid) if fixed >= 0 else 0.0
                values.append(jac * (group[0] + complex(*t["momentum"][block]) * pick(y[block], ar, col + slot, valid))
                              + source * (t["coefficient"][block] * group[1] + inserted))
                row_temp[:, (ar - m)[valid], (col + slot)[valid]] = group[2:, valid]
            partial.append(values)
        for warp in range(info["warps"]):                           # after __syncthreads()
            bc, col = warp * 8 + (lane >> 2), warp * 8 + 2 * (lane & 3)
            acc = np.zeros((4, 32))
            for g in range(temps):
                if t["temp_right"][block, g] >= 0:
                    for start in range(0, d, 4):
                        kp = start + kk
                        zmm(pick(row_temp[g], ar - m, kp, (ar < d) & (kp < d)),
                            pick(constants[t["temp_right"][block, g]], kp, bc, (kp < d) & (bc < d)), acc)
            for slot in (0, 1):
                valid = (ar < d) & (col + slot < d)
                value = partial[warp][slot] + jac * (acc[slot] + 1j * acc[2 + slot])
                out[block, ar[valid], (col + slot)[valid]] = value[valid]
    return out


def nvrtc_include():
    """Include directory with the libcudacxx header nv/target, or the reason to skip."""
    try:
        roots = [importlib.util.find_spec(name) for name in ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime")]
    except ModuleNotFoundError:
        roots = [None]
    if any(spec is None or not spec.submodule_search_locations for spec in roots):
        return None, "the wheels nvidia-cuda-nvrtc and nvidia-cuda-runtime are not installed"
    if not glob.glob(os.path.join(list(roots[0].submodule_search_locations)[0], "lib", "libnvrtc.so*")):
        return None, "libnvrtc.so was not found in the nvidia-cuda-nvrtc wheel"
    candidates = [os.environ.get("RCMPS_CUDA_INCLUDE", "")]
    candidates += sorted(glob.glob(os.path.expanduser("~/.julia/artifacts/*/include")))
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "nv", "target")):
            return path, ""
    return None, ("no directory with the header nv/target: set RCMPS_CUDA_INCLUDE "
                  "(searched it and ~/.julia/artifacts/*/include)")


class Case(unittest.TestCase):
    def assertClose(self, x, y, rtol, name, msg=""):
        """|x - y|_2 <= rtol |y|_2, recording the relative error."""
        x, y = np.asarray(x), np.asarray(y)
        self.assertEqual(x.shape, y.shape, msg)
        error, scale = norm(x - y), norm(y)
        self.assertTrue(np.isfinite(error), f"{msg} {name}: nonfinite")
        record(name, error / scale if scale > 0 else error)
        self.assertLessEqual(error, rtol * scale, f"{msg} {name}: |x-y| = {error:.3e}, |y| = {scale:.3e}")


class Tables(Case):
    def test_no_cupy(self):
        """Importing fused.py and the packed systems does not import CuPy."""
        self.assertNotIn("cupy", sys.modules)

    def test_structure(self):
        """(1) Shapes and dtypes (int32, float64), 3 + (two-sided groups) monotone offsets per
        block that partition its terms by group, the panel slot of every constant left operand,
        exactly one state operand per term, and the term columns copied from tables()."""
        for case in GRID:
            s = system(*case)
            t = s.tables()
            k = fused.kernel_tables(t)
            panels, temps = WIDTHS[case[0]]
            nb, count, first = len(s.blocks), len(t["term_group"]), t["term_first"]
            self.assertEqual(set(k), {*INTEGER, "coefficient", "momentum"})
            self.assertEqual({key: k[key].shape for key in k}, dict(
                terms=(count, 8), group_first=(nb, 3 + temps), left_constants=(nb, panels),
                temp_right=(nb, temps), coefficient=(nb,), momentum=(nb, 2), fixed=(nb,)), case)
            for key in k:
                self.assertEqual(k[key].dtype, np.int32 if key in INTEGER else np.float64, f"{case} {key}")
            for key in ("left_constants", "temp_right", "coefficient", "momentum", "fixed"):
                self.assertTrue(np.array_equal(k[key], t[key]), f"{case} {key}")
            for column, key in enumerate(COLUMNS):
                if key is not None:
                    self.assertTrue(np.array_equal(k["terms"][:, column], t[key]), f"{case} {key}")
            offsets, group = k["group_first"], k["terms"][:, 0]
            self.assertTrue((np.diff(offsets, axis=1) >= 0).all(), case)
            self.assertTrue(np.array_equal(offsets[:, 0], first[:-1]) and np.array_equal(offsets[:, -1], first[1:]))
            for b, g in itertools.product(range(nb), range(2 + temps)):
                self.assertTrue((group[offsets[b, g]:offsets[b, g + 1]] == g).all(), f"{case} block {b} group {g}")
                if g >= 2:      # a two-sided group has terms exactly when it has a right factor
                    self.assertEqual(offsets[b, g + 1] > offsets[b, g], k["temp_right"][b, g - 2] >= 0)
            left_kind, left_index, slot = k["terms"][:, 1], k["terms"][:, 2], k["terms"][:, 4]
            block_of = np.repeat(np.arange(nb), np.diff(first))
            constant = left_kind == 0
            self.assertTrue(((slot >= 0) & (slot < panels)).all() and not slot[~constant].any(), case)
            self.assertTrue(np.array_equal(k["left_constants"][block_of[constant], slot[constant]],
                                           left_index[constant]), case)
            self.assertTrue(np.isin(k["terms"][:, [1, 5]], (0, 1)).all())
            self.assertTrue((left_kind + k["terms"][:, 5] == 1).all(), case)
        self.assertEqual((fused.PANELS, fused.TEMPS), WIDTHS["action"])

    def test_emulate_rhs(self):
        """(2) emulate_rhs, which reads the device tables as the kernel does (panel slots,
        group offsets), equals system.rhs block by block to 1e-13 on random packed states,
        for the Lindblad part, the source part and both."""
        for case in GRID:
            s = system(*case)
            k, constants = fused.kernel_tables(s.tables()), np.asarray(s.constants)
            y = gaussian(31, *s.shape)
            for jac, source in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3)):
                got, want = fused.emulate_rhs(k, constants, y, jac, source), s.rhs(y, jac, source)
                for b, block in enumerate(s.blocks):
                    self.assertClose(got[b], want[b], 1e-13, "emulate_rhs vs system.rhs", f"{case} {block.name}")

    def test_rejections(self):
        """(3) kernel_tables raises ValueError for terms not sorted by group, a constant left
        operand missing from (or twice in) left_constants, a term with two or no state
        operands, and a group without a right-factor column; a valid copy passes."""
        for kind in WIDTHS:
            t = system(kind, 4, "even", 0.5, 4).tables()
            first, temps = t["term_first"], WIDTHS[kind][1]
            fused.kernel_tables(corrupted(t, "fixed", 0, t["fixed"][0]))
            part = slice(first[0], first[1])
            self.assertGreater(len(set(t["term_group"][part])), 1)
            bad = {"sorted": [corrupted(t, "term_group", part, t["term_group"][part][::-1].copy())]}
            i = int(np.flatnonzero(t["term_left_kind"] == 0)[0])        # constant left, state right
            j = int(np.flatnonzero(t["term_left_kind"] == 1)[0])        # state left, constant right
            b = int(np.searchsorted(first, i, side="right")) - 1
            slot = int(np.flatnonzero(t["left_constants"][b] == t["term_left_index"][i])[0])
            other = t["left_constants"][b, (slot + 1) % 2]
            self.assertTrue(other >= 0 and other != t["term_left_index"][i])
            bad["left constant"] = [corrupted(t, "left_constants", (b, slot), -1),
                                    corrupted(t, "left_constants", (b, (slot + 1) % 2), t["term_left_index"][i])]
            bad["exactly one state operand"] = [
                corrupted(t, "term_left_kind", i, 1), corrupted(t, "term_right_kind", i, 0),
                corrupted(t, "term_right_kind", j, 1)]
            bad["two-sided group"] = [corrupted(t, "term_group", len(t["term_group"]) - 1, 2 + temps)]
            for message, cases in bad.items():
                for tables in cases:
                    with self.assertRaisesRegex(ValueError, message):
                        fused.kernel_tables(tables)


class Geometry(Case):
    DIMS = (2, 3, 8, 12, 32, 33)

    def test_geometry(self):
        """(4) threads = 32 ceil(d/8) (one warp per 8-column tile), blocks = n_blocks ceil(d/8)
        (one thread block per 8 rows of a packed block), and the shared memory: panels + temps
        row panels of 8 x d complex128 and two doubles per warp."""
        for d, nb, (panels, temps) in itertools.product((1, 2, 3, 7, 8, 9, 16, 31, 32, 33, 64), (1, 20, 26),
                                                        WIDTHS.values()):
            warps = math.ceil(d / 8)
            self.assertEqual(fused.geometry(d, nb, panels, temps), dict(
                d=d, n_blocks=nb, panels=panels, temps=temps, warps=warps, threads=32 * warps,
                blocks=nb * warps, shared_bytes=(panels + temps) * 8 * d * 16 + 2 * warps * 8, total=nb * d * d))
        self.assertEqual(fused.geometry(3, 26), fused.geometry(3, 26, 4, 2))
        info = fused.geometry(32, 26)       # D = 64, action system of :phi^4:
        self.assertEqual((info["threads"], info["blocks"], info["shared_bytes"]), (128, 104, 24640))

    def test_thread_mapping(self):
        """(6) Thread block = (packed block, 8-row group), warp = 8-column tile, lane l owns
        (m + l//4, n + 2 (l%4)) and the next column, when inside d: every element of every
        block has exactly one owner, also when d is not a multiple of 8, 4 or 2."""
        text = squash(fused.CUDA)
        for line in ("const int d=PACKED_D,lane=threadIdx.x&31,warp=threadIdx.x>>5;",
                     "const int block=blockIdx.x/ROW_WARPS,m=(blockIdx.x%ROW_WARPS)*8,n=warp*8;",
                     "const int row=m+(lane>>2),col=n+2*(lane&3);",
                     "const int element0=block*DD+row*d+col,element1=element0+1;",
                     "const bool valid0=row<d&&col<d,valid1=row<d&&col+1<d;"):
            self.assertIn(squash(line), text, "the kernel mapping changed: update `owners`")
        for d, nb in itertools.product(self.DIMS, (1, 3)):
            count = owners(d, nb)
            self.assertEqual(count.shape, (nb * d * d,))
            self.assertTrue((count == 1).all(), f"d={d}: owners per element {np.unique(count)}")

    def test_tile_product(self):
        """(6) The scalar tile product ('#else' branch of mma): with lane l holding A[l//4][l%4]
        and B[l%4][l//4], the accumulators C[l//4][2 (l%4)] and C[l//4][2 (l%4) + 1] receive
        the 8x8x4 product C += A B; zmm is the complex product from four of them."""
        text = squash(fused.CUDA.split("#else")[1].split("#endif")[0])
        for line in ("const int lane=threadIdx.x&31,rl=lane>>2,cl=2*(lane&3);",
                     "double av=__shfl_sync(0xffffffffu,a,(rl<<2)|q);",
                     "double b0=__shfl_sync(0xffffffffu,b,(cl<<2)|q),b1=__shfl_sync(0xffffffffu,b,((cl+1)<<2)|q);",
                     "c0+=av*b0;c1+=av*b1;"):
            self.assertIn(squash(line), text, "the scalar tile product changed: update `mma`")
        lane = np.arange(32)
        row, k, col = lane // 4, lane % 4, 2 * (lane % 4)
        self.assertEqual(len(set(zip(row, k))), 32)                    # the fragments are bijections
        self.assertEqual(len(set(zip(row, col)) | set(zip(row, col + 1))), 64)
        rng = np.random.default_rng(7)
        a, b, c = rng.standard_normal((8, 4)), rng.standard_normal((4, 8)), rng.standard_normal((8, 8))
        c0, c1 = c[row, col].copy(), c[row, col + 1].copy()
        mma(a[row, k], b[k, row], c0, c1)
        got = np.empty((8, 8))
        got[row, col], got[row, col + 1] = c0, c1
        self.assertClose(got, c + a @ b, 1e-15, "scalar tile product")
        za, zb = gaussian(8, 8, 4), gaussian(9, 4, 8)
        acc = np.zeros((4, 32))
        zmm(za[row, k], zb[k, row], acc)
        got = np.empty((8, 8), dtype=np.complex128)
        got[row, col], got[row, col + 1] = acc[0] + 1j * acc[2], acc[1] + 1j * acc[3]
        self.assertClose(got, za @ zb, 1e-15, "complex tile product")

    def test_block_walk(self):
        """(6) block_rhs re-implemented lane by lane (`walk`) equals system.rhs to 1e-13 on the
        real systems (d = 2, 3) and, since the tables do not depend on d, emulate_rhs for their
        tables with random constants at d = 5, 8, 9 and 12 (zero padding of the 4-wide k tiles
        and of the 8 x 8 output tiles, two warps, two thread blocks per packed block)."""
        text = squash(fused.CUDA)
        for line in ("const int d=PACKED_D,ar=m+(lane>>2),bc=n+(lane>>2),kk=lane&3;",
                     "if(lk==0)a=(ar<d&&kp<d)?panels[ls*PANEL+(ar-m)*d+kp]:zeroz();",
                     "if(rk==0)b=(kp<d&&bc<d)?consts[ri*DD+kp*d+bc]:zeroz();",
                     "double2 a=(ar<d&&kp<d)?row_temp[g*PANEL+(ar-m)*d+kp]:zeroz();",
                     "double2 b=(kp<d&&bc<d)?consts[tr*DD+kp*d+bc]:zeroz();",
                     "panels[slot*PANEL+offset]=(constant>=0&&pr<d)?consts[constant*DD+pr*d+pc]:zeroz();"):
            self.assertIn(squash(line), text, "block_rhs changed: update `walk`")
        for kind, dim, sector in itertools.product(WIDTHS, DIMS, SECTORS):
            s = system(kind, dim, sector, 0.5, 4)
            k, constants = fused.kernel_tables(s.tables()), np.asarray(s.constants)
            y = gaussian(41, *s.shape)
            self.assertClose(walk(k, constants, y, 0.7, 0.3), s.rhs(y, 0.7, 0.3), 1e-13,
                             "kernel walk vs system.rhs", f"{kind} D={dim} {sector}")
        for kind, d in itertools.product(WIDTHS, (5, 8, 9, 12)):
            s = system(kind, 4, "even", 0.5, 4)
            k = fused.kernel_tables(s.tables())
            constants, y = gaussian(42, len(s.constants), d, d), gaussian(43, len(s.blocks), d, d)
            self.assertClose(walk(k, constants, y, 0.7, 0.3), fused.emulate_rhs(k, constants, y, 0.7, 0.3),
                             1e-13, "kernel walk vs emulate_rhs", f"{kind} d={d}")


class Source(Case):
    CASES = [(d, nb, use_mma, widths) for d, nb in ((3, 26), (8, 20), (32, 26), (33, 20))
             for use_mma in (True, False) for widths in WIDTHS.values()]

    def test_defines(self):
        """(5) The specialization header: packed size, blocks, warps per thread block, panels,
        two-sided groups, NGROUPS = 2 + TEMPS, USE_MMA, and the constants of the source table."""
        table = fused.source_table()
        for d, nb, use_mma, (panels, temps) in self.CASES:
            values = defines(fused.kernel_source(d, nb, use_mma=use_mma, panels=panels, temps=temps))
            self.assertEqual(float(values.pop("SOURCE_MAXIMUM")), math.sqrt(40.0))
            self.assertEqual({key: int(value) for key, value in values.items()}, dict(
                PACKED_D=d, NBLOCKS=nb, ROW_WARPS=math.ceil(d / 8), PANELS=panels, TEMPS=temps,
                NGROUPS=2 + temps, USE_MMA=int(use_mma), SOURCE_SEGMENTS=16, SOURCE_DEGREE=20))
        self.assertEqual((table.MAXIMUM, table.SEGMENTS, table.DEGREE), (math.sqrt(40.0), 16, 20))
        self.assertEqual(defines(fused.kernel_source(3, 26)), defines(fused.kernel_source(3, 26, use_mma=True,
                                                                                        panels=4, temps=2)))
        self.assertTrue(fused.kernel_source(3, 26).endswith(fused.CUDA))

    def test_mma_branch(self):
        """(5) After the preprocessor (one conditional, on USE_MMA) the FP64 tensor-core
        instruction mma.sync...m8n8k4...f64 is there exactly when use_mma, and the warp
        shuffles of the scalar tile product exactly when not."""
        instruction = "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        for use_mma in (True, False):
            source = fused.kernel_source(8, 26, use_mma=use_mma)
            self.assertEqual([len(re.findall(rf"^\s*#{word}\b", source, re.M)) for word in ("if", "else", "endif")],
                             [1, 1, 1])
            self.assertFalse(re.search(r"#(?:ifdef|ifndef|elif)\b", source))
            kept = active(source)
            self.assertEqual(instruction in kept, use_mma)
            self.assertEqual("mma.sync" in kept, use_mma)
            self.assertEqual("__shfl_sync" in kept, not use_mma)
            self.assertEqual(len(re.findall(r"void mma\(double a,double b,double &c0,double &c1\)", kept)), 1)
            self.assertIn('extern "C" __global__ void excitation_trajectory(', kept)

    def test_fp64_only(self):
        """(5) No `float` type, no FP32 literal (1.0f, 1.f, .5f, 2e-3f) and no FP32 math call
        in the source; the regular expressions are checked on examples first."""
        for text in ("x=1.0f;", "x=1.f;", "y*.5f", "2e-3f", "(1F)", "0.5e+2F,"):
            self.assertTrue(FP32_LITERAL.search(text), text)
        for text in ("x=1.0;", "1.e-14", "0xffffffffu", "0x7ff0000000000000LL", "0x1f", "m8n8k4.row.col.f64",
                     "e5[s]", "b1=0.,b2=0.;", "12LL*TOTAL", "buf[2*f]"):
            self.assertFalse(FP32_LITERAL.search(text), text)
        self.assertTrue(FP32_CALL.search("sqrtf (x)") and not FP32_CALL.search("sqrt(x)+fabs(y)+f(x)"))
        for d, nb, use_mma, (panels, temps) in self.CASES:
            source = fused.kernel_source(d, nb, use_mma=use_mma, panels=panels, temps=temps)
            self.assertEqual(re.findall(r"\bfloat\d?\b", source), [])
            self.assertEqual(FP32_LITERAL.findall(source), [])
            self.assertEqual(FP32_CALL.findall(source), [])
            self.assertIn("double2", source)

    def test_source_table(self):
        """(8) The device source table (piecewise Chebyshev fit of 2|t| J(t^2)) equals
        models.transformed_source to 1e-14 for |t| <= sqrt(40): 4001 points, the segment
        boundaries and t = 0 (1/sqrt(pi)); it is even and has the shape the kernel indexes."""
        table = fused.source_table()
        self.assertEqual(table.COEFFICIENTS.shape, (table.SEGMENTS, table.DEGREE + 1))
        self.assertEqual(table.COEFFICIENTS.dtype, np.float64)
        edges = table.MAXIMUM * np.arange(table.SEGMENTS + 1) / table.SEGMENTS
        t = np.unique(np.r_[np.linspace(-table.MAXIMUM, table.MAXIMUM, 4001), edges, -edges,
                            np.nextafter(edges[1:], 0), 0.0])
        got = table.source_numpy(t)
        want = np.array([transformed_source(x) for x in t])
        record("source table vs transformed_source (absolute)", np.abs(got - want).max())
        self.assertLessEqual(np.abs(got - want).max(), 1e-14)
        self.assertTrue(np.array_equal(got, table.source_numpy(-t)))
        self.assertLessEqual(abs(float(table.source_numpy(0.0)) - 1 / math.sqrt(math.pi)), 1e-15)


class Compile(Case):
    PTX = {True: "\tfma.rn.f64 %fd1, %fd2, %fd3, %fd4;\n\tmma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1};\n",
           False: "\tfma.rn.f64 %fd1, %fd2, %fd3, %fd4;\n\tsetp.lt.f32 %p1, %f1, %f2;\n"}

    def test_audit_synthetic(self):
        """audit_ptx on hand-written PTX: reports the FP64 mma instructions, and raises for
        FP32 arithmetic (add, mul, fma, div, sqrt with any modifiers) or, with use_mma, for
        no tensor-core instruction. FP32 comparisons are not arithmetic."""
        audit = fused.audit_ptx(self.PTX[True], use_mma=True)
        self.assertEqual(audit, dict(f64_mma_instructions=["mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"],
                                     fp32_arithmetic_instructions=[]))
        self.assertEqual(fused.audit_ptx(self.PTX[False], use_mma=False)["f64_mma_instructions"], [])
        with self.assertRaises(RuntimeError):
            fused.audit_ptx(self.PTX[False], use_mma=True)
        for line in ("add.f32 %f1, %f2, %f3;", "mul.rn.f32 %f1, %f2, %f3;", "fma.rn.f32 %f1, %f2, %f3, %f4;",
                     "div.full.f32 %f1, %f2, %f3;", "sqrt.rn.f32 %f1, %f2;"):
            for use_mma in (True, False):
                with self.assertRaises(RuntimeError, msg=line):
                    fused.audit_ptx(self.PTX[use_mma] + "\t" + line + "\n", use_mma=use_mma)

    def test_compile_and_audit(self):
        """(7) NVRTC compiles the kernel (sm_80, no GPU) for d = 3 and 32, the widths of the
        action system (4 panels, 2 two-sided groups) and of the pair system (5, 3), with and
        without tensor cores; the PTX has FP64 mma instructions exactly when use_mma and no
        FP32 arithmetic."""
        include, reason = nvrtc_include()
        if include is None:
            self.skipTest(reason)
        for d, (panels, temps), use_mma in itertools.product((3, 32), WIDTHS.values(), (True, False)):
            msg = f"d={d} panels={panels} temps={temps} use_mma={use_mma}"
            try:
                ptx, _ = fused.compile_ptx(d, 26 if temps == 2 else 20, use_mma=use_mma, include_dirs=(include,),
                                           panels=panels, temps=temps)
            except OSError as error:        # libnvrtc itself cannot be loaded here
                self.skipTest(f"libnvrtc cannot be loaded: {error}")
            audit = fused.audit_ptx(ptx, use_mma=use_mma)
            self.assertEqual(bool(audit["f64_mma_instructions"]), use_mma, msg)
            self.assertEqual(audit["fp32_arithmetic_instructions"], [], msg)
            self.assertEqual("mma.sync" in ptx, use_mma, msg)
            self.assertFalse(re.search(r"\bmma\.sync\S*\.f(?:16|32)\b", ptx), msg)
            self.assertIn(".entry excitation_trajectory", ptx, msg)
            self.assertIn(".target sm_80", ptx, msg)


if __name__ == "__main__":
    unittest.main()
