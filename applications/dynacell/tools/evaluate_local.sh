#!/bin/bash
# Run `dynacell evaluate` for a (organelle, model, train_set, test_set) tuple
# locally on the current host's GPU. Discovers per-plate predict leaves the
# same way predict_local.sh does, stages each leaf's eval invocation via
# submit_evaluation_job.py, then executes `uv run dynacell evaluate ...` in
# series (or in parallel batches of N with --parallel).
#
# Usage:
#   evaluate_local.sh <organelle> <model> <train_set> <test_set> \
#       [--overwrite | --regen-metrics] [--parallel N] [--cross-condition-probe]
#
# Args:
#   <organelle>  e.g. nucleus, membrane, er, mito
#   <model>      e.g. fnet3d_paper, fcmae_vscyto3d_scratch,
#                fcmae_vscyto3d_pretrained, unetvit3d, celldiff_*
#   <train_set>  ipsc | a549 | joint
#                (or: ipsc_confocal, a549_mantis,
#                joint_ipsc_confocal_a549_mantis)
#   <test_set>   ipsc | a549   (or: ipsc_confocal, a549_mantis)
#
# Flags:
#   --overwrite               adds force_recompute.all=true on every leaf
#                             (regenerates GT masks + features + metrics)
#   --regen-metrics           adds force_recompute.final_metrics=true on every
#                             leaf — recomputes metrics + rewrites embeddings,
#                             reuses cached GT masks / CP / deep features.
#                             Mutually exclusive with --overwrite.
#   --parallel N              run N evals concurrently on the same GPU
#   --cross-condition-probe   after all per-plate evals succeed AND
#                             test_set=a549_mantis, run
#                             `python -m dynacell.evaluation.cross_condition_probe
#                              --eval_dirs <save_dir_mock> <save_dir_denv>
#                              <save_dir_zikv>
#                              --out <save_dir_mock>/../cross_condition_probe.csv`
#
# Examples:
#   evaluate_local.sh er    fnet3d_paper             ipsc  ipsc
#   evaluate_local.sh er    fnet3d_paper             ipsc  a549 --parallel 2
#   evaluate_local.sh er    fnet3d_paper             ipsc  a549 --cross-condition-probe

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 <organelle> <model> <train_set> <test_set> [--overwrite] [--parallel N] [--cross-condition-probe]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
TRAIN_RAW=$3
TEST_RAW=$4
shift 4

OVERWRITE=""
REGEN_METRICS=""
PARALLEL=1
CROSS_PROBE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --overwrite)              OVERWRITE="--overwrite"; shift ;;
    --regen-metrics)          REGEN_METRICS="--regen-metrics"; shift ;;
    --parallel)               PARALLEL="$2"; shift 2 ;;
    --parallel=*)             PARALLEL="${1#--parallel=}"; shift ;;
    --cross-condition-probe)  CROSS_PROBE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [ -n "$OVERWRITE" ] && [ -n "$REGEN_METRICS" ]; then
  echo "error: --overwrite and --regen-metrics are mutually exclusive" >&2
  exit 2
fi

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

if [ "$TEST_SET" = "ipsc_confocal" ]; then
  GLOB="$LEAF_DIR/predict__ipsc_confocal.yml"
else
  GLOB="$LEAF_DIR/predict__${TEST_SET}_*.yml"
fi
mapfile -t LEAVES < <(ls $GLOB 2>/dev/null | sort)
if [ ${#LEAVES[@]} -eq 0 ]; then
  echo "error: no leaves match $GLOB" >&2
  exit 1
fi

cd "$VISCY_ROOT"

# Stage every leaf via --dry-run --print-cmd to (a) validate output_store
# exists and (b) capture the resolved command line + save_dir for execution
# and the optional cross-condition probe.
declare -a CMDS=()
declare -a PLATES=()
declare -a SAVE_DIRS=()
for leaf in "${LEAVES[@]}"; do
  base=$(basename "$leaf" .yml)
  plate=${base#predict__${TEST_SET}_}
  if [ "$plate" = "$base" ]; then
    plate=$TEST_SET
  fi
  PLATES+=("$plate")

  # Capture the literal eval command tokens (one per line).
  cmd_text=$(uv run python applications/dynacell/tools/submit_evaluation_job.py \
    "$leaf" $OVERWRITE $REGEN_METRICS --dry-run --print-cmd 2>/dev/null | grep -v '^\[dry-run\]')
  CMDS+=("$cmd_text")

  # Extract save.save_dir override from the printed command.
  save_dir=$(printf '%s\n' "$cmd_text" | sed -n 's/^save\.save_dir=//p')
  if [ -z "$save_dir" ]; then
    echo "error: failed to extract save.save_dir from staged command for $leaf" >&2
    exit 1
  fi
  SAVE_DIRS+=("$save_dir")
done

# Single timestamp for the whole batch.
TS=$(date +%Y%m%d-%H%M%S)
LOG_ROOT=$(dirname "${SAVE_DIRS[0]}")
mkdir -p "$LOG_ROOT/slurm"
echo "[evaluate_local] leaves=${#LEAVES[@]} parallel=$PARALLEL log_root=$LOG_ROOT"

# Trap to kill backgrounded evals on early exit.
trap 'kill 0 2>/dev/null || true' EXIT

declare -a PIDS=()
declare -a PID_PLATES=()
declare -a PID_LOGS=()
declare -a FAILED_PLATES=()

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
for idx in "${!LEAVES[@]}"; do
  plate=${PLATES[$idx]}
  cmd_text=${CMDS[$idx]}
  log="$LOG_ROOT/slurm/local_${TS}_eval_${ORGANELLE}_${MODEL}_${plate}.log"
  echo "  [start] plate=$plate log=$(basename "$log")"
  # Pass the tokens to bash via xargs so the printed command runs as one
  # process (xargs handles word-splitting more safely than `eval`).
  printf '%s\n' "$cmd_text" | xargs -d '\n' -I {} echo {} > /dev/null  # validate parse
  (
    # shellcheck disable=SC2086
    args=()
    while IFS= read -r line; do args+=("$line"); done <<< "$cmd_text"
    exec "${args[@]}"
  ) >"$log" 2>&1 &
  PIDS+=($!)
  PID_PLATES+=("$plate")
  PID_LOGS+=("$log")
  i=$((i + 1))
  if [ $((i % PARALLEL)) -eq 0 ]; then
    flush_batch
  fi
done

flush_batch
trap - EXIT

if [ ${#FAILED_PLATES[@]} -gt 0 ]; then
  echo "[fail] ${#FAILED_PLATES[@]}/${#LEAVES[@]} plate(s) failed:" >&2
  for f in "${FAILED_PLATES[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi
echo "[done] all evals complete; logs: $LOG_ROOT/slurm/local_${TS}_eval_${ORGANELLE}_${MODEL}_*.log"

# Optional cross-condition probe (a549 test only; requires all 3 plates).
if [ "$CROSS_PROBE" -eq 1 ]; then
  if [ "$TEST_SET" != "a549_mantis" ]; then
    echo "[cross-probe] skipped: requires test_set=a549_mantis (got $TEST_SET)" >&2
  elif [ ${#SAVE_DIRS[@]} -lt 3 ]; then
    echo "[cross-probe] skipped: need >= 3 save_dirs (mock/denv/zikv); got ${#SAVE_DIRS[@]}" >&2
  else
    out_csv="$(dirname "${SAVE_DIRS[0]}")/cross_condition_probe.csv"
    echo "[cross-probe] running on ${#SAVE_DIRS[@]} eval dirs -> $out_csv"
    uv run python -m dynacell.evaluation.cross_condition_probe \
      --eval_dirs "${SAVE_DIRS[@]}" \
      --out "$out_csv"
  fi
fi
