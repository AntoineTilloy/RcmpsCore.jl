#!/usr/bin/env bash
# The GPU qualification batch of rcmps_gpu on one machine (one GPU), for any NVIDIA GPU with
# cooperative launch: the checks of gpu/qualification listed below in turn, one log per step, a
# status file.
#
#     bash gpu/qualification/run_all.sh OUTPUT_DIR
#
# Needs the package installed with CuPy (pip install -e "./gpu[cuda12]" from the repository
# root) and the shared fixtures of test/fixtures. OUTPUT_DIR must not exist. Environment:
#   PYTHON           interpreter (default: python)
#   RCMPS_DEVICE     CUDA device number (sets CUDA_VISIBLE_DEVICES; default: device 0)
#   RCMPS_STEPS      names of the steps to run, separated by spaces (default: all)
#   RCMPS_STATE_D64  a D = 64 state (rcmps_store .jld2 or portable directory) for the
#                    excitation runs at D = 64; they run only with it, and only on a GPU of
#                    compute capability 8.0 or later (the A100 of the plan)
#
# On a GPU of compute capability 8.0 or later every check runs with the tiles the device
# selects (auto: FP64 tensor cores) and again with the scalar tiles forced; below 8.0 auto is
# the scalar tiles, and the tensor-core path is only checked to be refused.
#
# Steps (OUTPUT_DIR/logs/NN_<step>.log holds the output of step NN, JSON lines and messages):
#   device                nvidia-smi, versions, device properties (also OUTPUT_DIR/device.json)
#   optimizer_tiles       gpu_optimizer_check.py --stages unit trajectory, D = 8 16 32 64: the tile
#                         product against the host FMA chain, trajectories of each path
#   optimizer_compile     gpu_optimizer_check.py --stages compile, D = 8 to 256: compilation,
#                         register caps, residency and PTX audit per D and path (and --tiles mma,
#                         expected to be refused, below 8.0)
#   optimizer_runs        gpu_optimizer_check.py --stages run, D = 8 16 32 64, and 96 and 128 when
#                         optimizer_compile found them resident: compile, initial qualification
#                         against the CuPy oracle, 3 accepted iterations, final validation
#   optimizer_short       gpu_optimizer_check.py --stages run, D = 16 32, 30 accepted iterations
#                         (with both paths, their histories compared)
#   excitation_compile    gpu_bringup.py --stages device compile: both kernels of a solve, every
#                         path, packed d = 2 to 64 (D = 4 to 128): compilation, residency (a size
#                         the GPU cannot hold is reported, not failed) and the PTX audit
#   excitation_fused      gpu_fused_check.py on random Z2 states (D = 4 6 8 12) and on the D = 16
#                         fixture: every path against the host-controlled integrator, repeated
#                         launches compared bitwise
#   excitation_pair       gpu_pair_check.py --reference on the D = 16 fixture: the bilinear term of
#                         every path against the host packed and unpacked systems
#   excitation_action     gpu_action_check.py --reference on the D = 16 fixture, --tiles auto: the
#                         complete action on the GPU against the unpacked equations on the CPU
#   excitation_action_scalar  the same with --tiles scalar (compute capability 8.0 or later)
#   references            check_references.py: the D = 16 and D = 32 levels of
#                         test/fixtures/excitation_references.toml through the command line,
#                         against Julia, with --tiles auto
#   references_scalar     the same with --tiles scalar (compute capability 8.0 or later)
#   excitation_d64        rcmps-gpu-excitations on RCMPS_STATE_D64 (odd sector, p = 0, the
#                         defaults) with --tiles auto (8.0 or later, with the state); fails unless
#                         result.json says the level is accepted
#   excitation_d64_scalar the same with --tiles scalar
#
# OUTPUT_DIR/status.tsv: one line per step (step, exit status, seconds, log); the exit status
# of this script is 1 if any step failed. Steps continue after a failed one.
set -u
if [ $# -ne 1 ]; then
    sed -n '2,54p' "$0" >&2
    exit 2
fi
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python}
FIXTURES=$ROOT/test/fixtures
D16=$FIXTURES/phi4_g2_D16_z2_state.jld2
if [ -e "$OUT" ]; then
    echo "run_all.sh: $OUT exists; give a new directory" >&2
    exit 2
fi
mkdir -p "$OUT/logs" || exit 2
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
if [ -n "${RCMPS_DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$RCMPS_DEVICE   # every script then uses device 0, this one
fi
printf "step\texit\tseconds\tlog\n" > "$OUT/status.tsv"
FAILED=0

run_step() {   # run_step NUMBER NAME COMMAND...: the command's output goes to logs/NUMBER_NAME.log
    local label="$1_$2"
    shift 2
    if [ -n "${RCMPS_STEPS:-}" ] && [[ " $RCMPS_STEPS " != *" ${label#*_} "* ]]; then
        return 0
    fi
    local start=$SECONDS
    echo "$(date -u +%FT%TZ) $label: $*" | tee -a "$OUT/commands.log"
    "$@" > "$OUT/logs/$label.log" 2>&1
    local status=$?
    printf "%s\t%d\t%d\t%s\n" "$label" "$status" $((SECONDS - start)) "logs/$label.log" >> "$OUT/status.tsv"
    echo "$(date -u +%FT%TZ) $label: exit status $status after $((SECONDS - start)) s"
    [ "$status" -eq 0 ] || FAILED=1
}

device_report() {
    nvidia-smi
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    "$PY" -m pip freeze
    "$PY" - "$OUT/device.json" <<'EOF'
import json, sys
import cupy as cp
import numpy, scipy
import rcmps_gpu
props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
row = dict(GPU=name, compute_capability=[int(props["major"]), int(props["minor"])],
           multiprocessors=int(props["multiProcessorCount"]), cooperative_launch=bool(props.get("cooperativeLaunch")),
           shared_memory_per_block_optin=int(props.get("sharedMemPerBlockOptin", 0)),
           shared_memory_per_multiprocessor=int(props.get("sharedMemPerMultiprocessor", 0)),
           registers_per_multiprocessor=int(props.get("regsPerMultiprocessor", 0)),
           memory=int(props["totalGlobalMem"]), kernel_run_time_limit=bool(props.get("kernelExecTimeoutEnabled", False)),
           cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()), driver=int(cp.cuda.runtime.driverGetVersion()),
           cupy=cp.__version__, numpy=numpy.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
           rcmps_gpu=rcmps_gpu.__version__, rcmps_gpu_path=rcmps_gpu.__file__)
open(sys.argv[1], "w").write(json.dumps(row, indent=2) + "\n")
print(json.dumps(row))
EOF
}

solve() {   # solve OUTPUT STATE [options]: rcmps-gpu-excitations, failing unless its level is accepted
    local directory=$1
    shift
    "$PY" -m rcmps_gpu.excitations "$@" --output "$directory" || return $?
    "$PY" - "$directory/result.json" <<'EOF'
import json, sys
result = json.load(open(sys.argv[1]))
print(json.dumps(dict(stage="solve", accepted=result.get("accepted"), stop_reason=result.get("stop_reason"),
                      energies=result.get("energies"), validated_energies=result.get("validated_energies"))))
sys.exit(0 if result.get("accepted") is True else 1)
EOF
}

resident_sizes() {   # D among 96 and 128 that the compile step found resident with auto tiles
    "$PY" - "$OUT/logs/03_optimizer_compile.log" <<'EOF'
import json, sys
found = set()
try:
    lines = open(sys.argv[1]).read().splitlines()
except OSError:
    lines = []
for line in lines:
    if line.startswith("{"):
        row = json.loads(line)
        if row.get("stage") == "compile" and row.get("tiles_requested") == "auto" and row.get("resident"):
            found.add(row["D"])
print(" ".join(str(D) for D in (96, 128) if D in found))
EOF
}

run_step 01 device device_report
CC=$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1]))["compute_capability"];print(10*c[0]+c[1])' "$OUT/device.json" 2>/dev/null)
if [ -z "$CC" ]; then
    CC=$("$PY" -c 'import cupy as cp;p=cp.cuda.runtime.getDeviceProperties(0);print(10*int(p["major"])+int(p["minor"]))') || {
        echo "run_all.sh: no CUDA device" >&2; exit 1; }
fi
if [ "$CC" -ge 80 ]; then
    TILES="auto scalar"      # tensor cores (auto) and the scalar tiles forced
    REFUSED=""
else
    TILES="auto"             # auto is the scalar tiles
    REFUSED="mma"            # checked to be refused before compilation
fi
echo "compute capability $CC: tile paths $TILES ${REFUSED:+(and $REFUSED, to be refused)}"
OPT="$PY $HERE/optimizer/gpu_optimizer_check.py"
EXC=$HERE/excitations

run_step 02 optimizer_tiles $OPT --output "$OUT/optimizer_tiles" --stages unit trajectory --D 8 16 32 64 --tiles $TILES
run_step 03 optimizer_compile $OPT --output "$OUT/optimizer_compile" --stages compile \
    --D 8 16 32 64 96 112 128 160 192 256 --tiles $TILES $REFUSED
run_step 04 optimizer_runs $OPT --output "$OUT/optimizer_runs" --stages run --D 8 16 32 64 $(resident_sizes) \
    --tiles $TILES --iterations 3
run_step 05 optimizer_short $OPT --output "$OUT/optimizer_short" --stages run --D 16 32 --tiles $TILES --iterations 30
run_step 06 excitation_compile "$PY" "$EXC/gpu_bringup.py" --stages device compile --packed 2 4 8 16 32 48 64
run_step 07 excitation_fused "$PY" "$EXC/gpu_fused_check.py" --random 4 6 8 12 --state "$D16"
run_step 08 excitation_pair "$PY" "$EXC/gpu_pair_check.py" --state "$D16" --reference
run_step 09 excitation_action "$PY" "$EXC/gpu_action_check.py" --state "$D16" --reference --tiles auto
if [ "$CC" -ge 80 ]; then
    run_step 10 excitation_action_scalar "$PY" "$EXC/gpu_action_check.py" --state "$D16" --reference --tiles scalar
fi
run_step 11 references "$PY" "$EXC/check_references.py" --output "$OUT/references" --tiles auto
if [ "$CC" -ge 80 ]; then
    run_step 12 references_scalar "$PY" "$EXC/check_references.py" --output "$OUT/references_scalar" --tiles scalar
fi
if [ -n "${RCMPS_STATE_D64:-}" ] && [ "$CC" -ge 80 ]; then
    run_step 13 excitation_d64 solve "$OUT/excitation_d64" "$RCMPS_STATE_D64" --tiles auto
    run_step 14 excitation_d64_scalar solve "$OUT/excitation_d64_scalar" "$RCMPS_STATE_D64" --tiles scalar
fi
echo "status: $OUT/status.tsv"
column -t -s $'\t' "$OUT/status.tsv" 2>/dev/null || cat "$OUT/status.tsv"
exit $FAILED
