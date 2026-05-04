#!/bin/bash
# Run a predict leaf (or per-plate leaf set) locally on the current host's GPU.
#
# Stages each leaf through submit_benchmark_job.py --dry-run (so the
# resolved YAML lands under launcher.run_root/resolved/ with a
# timestamped filename for provenance), then invokes
# `uv run dynacell predict -c <resolved>` directly — no sbatch.
#
# Usage:
#   predict_local.sh <organelle> <model> <train_set> <test_set> [--overwrite] [--parallel N]
#
# Args:
#   <organelle>  e.g. nucleus, membrane, er, mito
#   <model>      e.g. fnet3d_paper, fcmae_vscyto3d_scratch, fcmae_vscyto3d_pretrained,
#                unetvit3d, celldiff_*
#   <train_set>  ipsc | a549 | joint  (or full names: ipsc_confocal,
#                a549_mantis, joint_ipsc_confocal_a549_mantis)
#   <test_set>   ipsc | a549          (or full names: ipsc_confocal, a549_mantis)
#
# Examples:
#   predict_local.sh er    fnet3d_paper             ipsc  ipsc
#   predict_local.sh er    fnet3d_paper             ipsc  a549  --parallel 2
#   predict_local.sh nucleus fcmae_vscyto3d_scratch a549  ipsc  --overwrite
#   predict_local.sh nucleus fcmae_vscyto3d_scratch a549  a549  --parallel 2
#   predict_local.sh nucleus fcmae_vscyto3d_scratch joint a549  --parallel 2
#
# Notes:
# - Discovers leaves under
#     configs/benchmarks/virtual_staining/<organelle>/<model>/<train_set>/
#   matching `predict__<test_set>.yml` (1 leaf for ipsc) or
#   `predict__<test_set>_*.yml` (per-plate leaves for a549, typically 3:
#   mock, denv, zikv).
# - --overwrite passes through to submit_benchmark_job.py, which sets
#   HCSPredictionWriter.init_args.overwrite=True in the resolved YAML.
# - --parallel N runs N predicts concurrently on the same GPU, waiting
#   between batches of N. Tune to fit VRAM (~2 fnet predicts fit on an A40).
#   Single-leaf runs (test_set=ipsc) ignore --parallel above 1.
# - Logs land at $run_root/slurm/local_<TS>_<organelle>_<model>_<plate>.log
# - Fail-fast: a failing plate aborts the script. Re-run remaining plates
#   manually if needed.

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 <organelle> <model> <train_set> <test_set> [--overwrite] [--parallel N]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
TRAIN_RAW=$3
TEST_RAW=$4
shift 4

OVERWRITE=""
PARALLEL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --overwrite)   OVERWRITE="--overwrite"; shift ;;
    --parallel)    PARALLEL="$2"; shift 2 ;;
    --parallel=*)  PARALLEL="${1#--parallel=}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --parallel must be a positive integer (got '$PARALLEL')" >&2
  exit 2
fi

# Map shorthand to the directory / filename names actually on disk.
case "$TRAIN_RAW" in
  ipsc)        TRAIN_SET=ipsc_confocal ;;
  a549)        TRAIN_SET=a549_mantis ;;
  joint)       TRAIN_SET=joint_ipsc_confocal_a549_mantis ;;
  ipsc_confocal|a549_mantis|joint_ipsc_confocal_a549_mantis) TRAIN_SET="$TRAIN_RAW" ;;
  *) echo "error: unknown train_set '$TRAIN_RAW' (want ipsc | a549 | joint)" >&2; exit 2 ;;
esac
case "$TEST_RAW" in
  ipsc)         TEST_SET=ipsc_confocal ;;
  a549)         TEST_SET=a549_mantis ;;
  ipsc_confocal|a549_mantis) TEST_SET="$TEST_RAW" ;;
  *) echo "error: unknown test_set '$TEST_RAW' (want ipsc | a549)" >&2; exit 2 ;;
esac

VISCY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LEAF_DIR=$VISCY_ROOT/applications/dynacell/configs/benchmarks/virtual_staining/$ORGANELLE/$MODEL/$TRAIN_SET
if [ ! -d "$LEAF_DIR" ]; then
  echo "error: leaf directory does not exist: $LEAF_DIR" >&2
  echo "       (no $MODEL trained on $TRAIN_SET for $ORGANELLE?)" >&2
  exit 1
fi

# Single-leaf for ipsc test set, glob for a549 (per-plate: mock/denv/zikv).
if [ "$TEST_SET" = "ipsc_confocal" ]; then
  GLOB="$LEAF_DIR/predict__ipsc_confocal.yml"
else
  GLOB="$LEAF_DIR/predict__${TEST_SET}_*.yml"
fi
mapfile -t LEAVES < <(ls $GLOB 2>/dev/null | sort)
if [ ${#LEAVES[@]} -eq 0 ]; then
  echo "error: no leaves match $GLOB" >&2
  echo "       (expected predict__${TEST_SET}*.yml under $LEAF_DIR/)" >&2
  exit 1
fi

cd "$VISCY_ROOT"

# Extract launcher.{run_root,job_name} + model.init_args.ckpt_path from each
# leaf in one python call so we can (a) locate the resolved YAML staged below
# by job_name suffix, and (b) fail fast before the slow compose / Lightning
# init if any leaf has a placeholder or missing-on-disk ckpt_path.
META=$(uv run python - "${LEAVES[@]}" <<'PY'
import sys, yaml
for path in sys.argv[1:]:
    with open(path) as f:
        d = yaml.safe_load(f)
    ckpt = d.get("model", {}).get("init_args", {}).get("ckpt_path", "") or ""
    print(f"{path}\t{d['launcher']['job_name']}\t{d['launcher']['run_root']}\t{ckpt}")
PY
)

# Pre-flight: validate every leaf's ckpt_path before doing anything else.
BAD=0
while IFS=$'\t' read -r leaf _job _root ckpt; do
  if [ -z "$ckpt" ]; then
    echo "error: leaf has no model.init_args.ckpt_path: $leaf" >&2
    BAD=1
  elif echo "$ckpt" | grep -qE "TODO|FIXME|TBD"; then
    echo "error: ckpt_path is still a placeholder: $ckpt" >&2
    echo "       set a real path in: $leaf" >&2
    BAD=1
  elif [ ! -f "$ckpt" ]; then
    echo "error: ckpt_path file does not exist on disk: $ckpt" >&2
    echo "       referenced by:                          $leaf" >&2
    BAD=1
  fi
done <<< "$META"
[ "$BAD" -eq 0 ] || exit 1

RUN_ROOT=$(echo "$META" | head -1 | cut -f3)
mkdir -p "$RUN_ROOT/slurm"

echo "[stage] composing ${#LEAVES[@]} leaves ($ORGANELLE/$MODEL/$TRAIN_SET → $TEST_SET)${OVERWRITE:+ + overwrite}"
for leaf in "${LEAVES[@]}"; do
  uv run python applications/dynacell/tools/submit_benchmark_job.py \
    "$leaf" $OVERWRITE --dry-run >/dev/null
done

# Pick the runner: unbuffer keeps Lightning's TQDM progress bar visible
# in the log (Python sees a pseudo-tty); fall back to plain uv if missing
# OR if unbuffer is on PATH but its Tcl Expect runtime isn't installed
# (some HPC nodes ship the wrapper without the package, so a PATH check
# alone is insufficient). Probe by running it on `true` and checking the
# exit code.
if command -v unbuffer >/dev/null 2>&1 && unbuffer true 2>/dev/null; then
  RUNNER=(unbuffer uv run dynacell predict)
else
  RUNNER=(uv run dynacell predict)
  echo "[warn] 'unbuffer' unavailable or broken — TQDM progress bar will be hidden in logs"
fi

# Trap to kill any backgrounded predicts if we exit early (Ctrl+C, error).
trap 'kill 0 2>/dev/null || true' EXIT

# Single timestamp for the whole batch so all logs share a clock label.
TS=$(date +%Y%m%d-%H%M%S)

echo "[run] parallel=$PARALLEL leaves=${#LEAVES[@]} run_root=$RUN_ROOT"
declare -a PIDS=()
declare -a PID_PLATES=()
declare -a PID_LOGS=()
declare -a FAILED_PLATES=()

# Wait for all currently backgrounded predicts and record any failures.
# Bash's set -e does NOT propagate background-job failures through `wait`
# (treated as already-handled async), so we check exit codes manually.
flush_batch() {
  local idx pid status
  for idx in "${!PIDS[@]}"; do
    pid=${PIDS[$idx]}
    if wait "$pid"; then
      status=0
    else
      status=$?
    fi
    if [ "$status" -ne 0 ]; then
      FAILED_PLATES+=("${PID_PLATES[$idx]} (exit=$status, log=$(basename "${PID_LOGS[$idx]}"))")
    fi
  done
  PIDS=()
  PID_PLATES=()
  PID_LOGS=()
}

i=0
while IFS=$'\t' read -r leaf job_name _run_root _ckpt; do
  resolved=$(ls -t "$RUN_ROOT/resolved/predict_${job_name}_"*.yml 2>/dev/null | head -1)
  if [ -z "$resolved" ]; then
    echo "error: no resolved yaml found for job_name=$job_name" >&2
    exit 1
  fi
  # Strip the predict__<test_set>_ prefix to get a per-plate label;
  # for the single ipsc leaf this leaves an empty label so use the bare
  # test_set name instead.
  base=$(basename "$leaf" .yml)
  plate=${base#predict__${TEST_SET}_}
  if [ "$plate" = "$base" ]; then  # no prefix stripped → ipsc single leaf
    plate=$TEST_SET
  fi
  log="$RUN_ROOT/slurm/local_${TS}_${ORGANELLE}_${MODEL}_${plate}.log"

  echo "  [start] plate=$plate log=$(basename "$log")"
  "${RUNNER[@]}" -c "$resolved" >"$log" 2>&1 &
  PIDS+=($!)
  PID_PLATES+=("$plate")
  PID_LOGS+=("$log")

  # Pure-assignment increment, NOT ((i++)). With set -e, ((i++)) returns
  # its pre-increment value (0 on first iteration), which bash treats as
  # exit status 1 → the EXIT trap fires kill 0 → script + just-started
  # child both die.
  i=$((i + 1))
  if [ $((i % PARALLEL)) -eq 0 ]; then
    flush_batch
  fi
done <<< "$META"

flush_batch
trap - EXIT

if [ ${#FAILED_PLATES[@]} -gt 0 ]; then
  echo "[fail] ${#FAILED_PLATES[@]}/${#LEAVES[@]} plate(s) failed:" >&2
  for f in "${FAILED_PLATES[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi
echo "[done] all leaves complete; logs: $RUN_ROOT/slurm/local_${TS}_${ORGANELLE}_${MODEL}_*.log"
