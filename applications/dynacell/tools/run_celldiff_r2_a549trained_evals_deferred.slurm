#!/bin/bash
# Eval campaign: celldiff_r2 a549-trained on A549 (membrane/nucleus/mito) — deferred.
# Scheduled with --dependency afterok:33213763:33213764 to wait for all
# CELLDIFF_R2_A549TR_g0 + g1 predict arrays to complete.
#
# Array index → grouped leaf mapping:
#   0: membrane_celldiff_r2_a549trained_later     (A549 mock + zikv)
#   1: nucleus_celldiff_r2_a549trained_later      (A549 denv/mock/zikv)
#   2: mitochondria_celldiff_r2_a549trained_later (A549 denv/mock/zikv)
#
#SBATCH --job-name=celldiff-r2-a549tr-eval-def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/membrane_celldiff_r2_a549trained_later/eval_grouped
  grouped/nucleus_celldiff_r2_a549trained_later/eval_grouped
  grouped/mitochondria_celldiff_r2_a549trained_later/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[celldiff-r2-a549tr-eval-def] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[celldiff-r2-a549tr-eval-def] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# Derive the thread cap from the allocation (never hardcode — matches the #429 fix).
export DYNACELL_THREADS_PER_WORKER=${SLURM_CPUS_PER_TASK:-32}

cd /hpc/mydata/alex.kalinin/VisCy
# GLCM + focus (compute-at-eval-time; no .ozx repackaging needed — see
# run_celldiff_r2_a549trained_evals.slurm). slice_selection=focus = 2D instance
# seg on the in-focus plane. NOTE: superseded by tools/run_focus_campaign.slurm.
exec uv run dynacell evaluate-grouped "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  segmentation.slice_selection=focus \
  force_recompute.final_metrics=true
