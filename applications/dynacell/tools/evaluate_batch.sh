#!/bin/bash
# Submit ONE sbatch job that runs `dynacell evaluate` for every per-plate
# predict leaf for a given (organelle, model, train_set, test_set) tuple,
# in series. Path-1 batching: amortizes queue submission + GPU allocation.
#
# Usage:
#   evaluate_batch.sh <organelle> <model> <train_set> <test_set> \
#       [submit_evaluation_batch.py args...]
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
# Examples:
#   evaluate_batch.sh er    fnet3d_paper             ipsc  a549
#   evaluate_batch.sh er    fnet3d_paper             ipsc  a549 --dry-run
#   evaluate_batch.sh nucleus fcmae_vscyto3d_scratch a549  ipsc
#   evaluate_batch.sh nucleus fcmae_vscyto3d_scratch joint a549 --overwrite
#
# Extra args are forwarded verbatim to submit_evaluation_batch.py —
# including --overwrite (force_recompute.all=true on every leaf),
# --cross-condition-probe (handled by the python tool / runner), and
# --dry-run.

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 <organelle> <model> <train_set> <test_set> [submit_evaluation_batch.py args...]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
TRAIN_RAW=$3
TEST_RAW=$4
shift 4

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

echo "[evaluate_batch] organelle=$ORGANELLE model=$MODEL train=$TRAIN_SET test=$TEST_SET leaves=${#LEAVES[@]}"
for leaf in "${LEAVES[@]}"; do
  echo "  - $(basename "$leaf")"
done

# Job name is composed inside submit_evaluation_batch.py via
# `_composite_job_name`, which applies `paper_key()` to the code-side model
# (e.g. fcmae_vscyto3d_scratch -> UNEXT2). Don't override here — the bash
# shorthand `${MODEL^^}` would keep the raw config key and produce a
# different SLURM job name than the Python tool's default.
cd "$VISCY_ROOT"
exec uv run python applications/dynacell/tools/submit_evaluation_batch.py \
  "${LEAVES[@]}" \
  "$@"
