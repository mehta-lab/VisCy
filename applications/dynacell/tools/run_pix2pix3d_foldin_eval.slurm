#!/bin/bash
# Fold-in eval: compute ONLY the newly-added pix2pix3d_unetvit conditions in the
# ER + membrane ipsc_trained grouped buckets, leaving every other model's results
# untouched.
#
# The iPSC-trained pix2pix3d ER (sec61b) + membrane (caax) A549 predicts landed
# after the focus / no-carve campaigns, so regeneration folded 4 pix2pix3d
# conditions (mock/denv/zikv + iPSC) into er_ipsc_trained and membrane_ipsc_trained.
# This runs each bucket with force_recompute.final_metrics=false: the grouped
# driver skips every condition whose final CSV/NPY already exist (all the other
# models, freshly computed by the focus campaign / no-carve gtwarm) and computes
# only pix2pix3d, which has no cache yet. The membrane GT instance cache is already
# warm + no-carve from gtwarm 33977978, so pix2pix3d membrane hits it (no race, no
# two-phase needed); ER has no instance interpretation.
#
# Focus overrides identical to the campaign (define HOW pix2pix3d is scored, so it
# is comparable to the rest of the bucket); NO force_recompute.pred_cp /
# final_metrics so the cached conditions are loaded, not recomputed.
#
# Usage:
#   sbatch --array=0-1 tools/run_pix2pix3d_foldin_eval.slurm
#
#SBATCH --job-name=pix2pix3d_foldin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/er_ipsc_trained/eval_grouped
  grouped/membrane_ipsc_trained/eval_grouped
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID unset — launch via sbatch --array=0-1." >&2
  exit 2
fi

LEAF="${LEAVES[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$LEAF" ]]; then
  echo "No leaf at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

export DYNACELL_THREADS_PER_WORKER=${SLURM_CPUS_PER_TASK:-32}

echo "[pix2pix3d_foldin] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[pix2pix3d_foldin] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1) threads=${DYNACELL_THREADS_PER_WORKER}"

cd /hpc/mydata/alex.kalinin/VisCy

exec uv run dynacell evaluate-grouped "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  segmentation.slice_selection=focus \
  force_recompute.final_metrics=false
