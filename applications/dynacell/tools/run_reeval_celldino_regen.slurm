#!/bin/bash
# CELL-DINO-only regen: re-run grouped eval on the 13 leaves whose
# already-cached conditions were computed under the OLD min/max CELL-DINO
# preprocessing (pre-commit e648c4ce). Phase B (commit 6c234714) added
# preprocess_version auto-invalidation, but pre-version manifests are
# treated as lenient (no auto-trigger) — operator-driven invalidation
# via Phase A's existing per-extractor force_recompute flags handles
# this bootstrap transition.
#
# Strategy: 4 array tasks, one per organelle, bundling that organelle's
# (a549_trained / ipsc_trained / joint / *_a549_iterative_rerun) leaves.
# Each leaf is invoked with Hydra CLI overrides setting
# force_recompute.{pred,gt}_celldino=true so the per-FOV CELL-DINO cache
# is bypassed and features get re-extracted with the
# self_normalize_v1 recipe. Cellpose masks, CP regionprops, DINOv3 and
# DynaCLR features all stay cached.
#
# Dependency: afterok:33237646 — waits for the missing-condition eval
# array to finish so we don't race on the 3 partially-done leaves
# (mitochondria_ipsc_trained / nucleus_ipsc_trained / mitochondria_joint).
# A small overlap exists where this regen also re-computes CELL-DINO
# for those leaves' just-filled missing conditions; that's bounded
# (~37 conditions × ~2 min CELL-DINO each) and not worth the extra
# config-surface complexity to avoid.
#
# Submit with:
#   sbatch --dependency=afterok:33237646 \
#     applications/dynacell/tools/run_reeval_celldino_regen.slurm
#
#SBATCH --job-name=reeval-celldino-regen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

# Per-task leaf bundles: every organelle's leaves with any already-cached
# CELL-DINO features. Space-separated within a bundle; iterated below.
LEAVES_PER_TASK=(
  "grouped/er_a549_trained/eval_grouped grouped/er_ipsc_trained/eval_grouped grouped/er_joint/eval_grouped"
  "grouped/membrane_a549_trained/eval_grouped grouped/membrane_ipsc_trained/eval_grouped grouped/membrane_joint/eval_grouped"
  "grouped/mitochondria_a549_trained/eval_grouped grouped/mitochondria_ipsc_trained/eval_grouped grouped/mitochondria_joint/eval_grouped"
  "grouped/nucleus_a549_trained/eval_grouped grouped/nucleus_ipsc_trained/eval_grouped grouped/nucleus_joint/eval_grouped grouped/nucleus_celldiff_r2_iterative_a549_rerun/eval_grouped"
)

LEAVES="${LEAVES_PER_TASK[${SLURM_ARRAY_TASK_ID}]}"
echo "[celldino-regen] task ${SLURM_ARRAY_TASK_ID}: leaves=${LEAVES}"
echo "[celldino-regen] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[celldino-regen] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

export DYNACELL_THREADS_PER_WORKER=32

cd /hpc/mydata/alex.kalinin/VisCy

for LEAF in $LEAVES; do
  echo "[celldino-regen] >>> ${LEAF}"
  uv run dynacell evaluate-grouped \
    "leaf=${LEAF}" \
    "force_recompute.pred_celldino=true" \
    "force_recompute.gt_celldino=true" \
    "force_recompute.final_metrics=true"
done
