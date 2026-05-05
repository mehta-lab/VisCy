#!/bin/bash
# Submit one sbatch job that runs every per-plate predict leaf for a given
# (organelle, model, train_set, test_set) tuple, in series. Path-1 batching:
# amortizes queue submission + GPU allocation; total compute is the same as
# N per-plate jobs.
#
# Usage:
#   predict_batch.sh <organelle> <model> <train_set> <test_set> [submit_benchmark_batch.py args...]
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
#   predict_batch.sh er    fnet3d_paper             ipsc  a549             # iPSC-trained → A549
#   predict_batch.sh er    fnet3d_paper             ipsc  a549  --dry-run  # render only
#   predict_batch.sh nucleus fcmae_vscyto3d_scratch a549  ipsc             # A549-trained → iPSC
#   predict_batch.sh nucleus fcmae_vscyto3d_scratch joint a549             # joint-trained → A549
#   predict_batch.sh er    fnet3d_paper             ipsc  a549  --overwrite
#
# Extra args are forwarded verbatim to submit_benchmark_batch.py — including
# --overwrite (alias for HCSPredictionWriter.overwrite=True on every leaf;
# required to re-run a plate whose output store already has predictions) and
# --override KEY.PATH=VALUE (dict-key dotlist, deep-merged after compose).
#
# Discovers leaves under
#   configs/benchmarks/virtual_staining/<organelle>/<model>/<train_set>/
# matching `predict__<test_set>.yml` (1 leaf for ipsc) or
# `predict__<test_set>_*.yml` (per-plate leaves for a549, typically 3:
# mock, denv, zikv).

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 <organelle> <model> <train_set> <test_set> [submit_benchmark_batch.py args...]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
TRAIN_RAW=$3
TEST_RAW=$4
shift 4

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

JOB_NAME="${MODEL}_PRED_${ORGANELLE}_${TRAIN_SET}_ON_${TEST_SET}"

echo "[predict_batch] organelle=$ORGANELLE model=$MODEL train=$TRAIN_SET test=$TEST_SET leaves=${#LEAVES[@]}"
for leaf in "${LEAVES[@]}"; do
  echo "  - $(basename "$leaf")"
done
echo "[predict_batch] composite job_name=$JOB_NAME"

cd "$VISCY_ROOT"
exec uv run python applications/dynacell/tools/submit_benchmark_batch.py \
  "${LEAVES[@]}" \
  --job-name "$JOB_NAME" \
  "$@"
