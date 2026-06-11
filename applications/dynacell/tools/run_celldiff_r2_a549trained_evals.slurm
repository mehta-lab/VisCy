#!/bin/bash
# Eval campaign: celldiff_r2 a549-trained model evals + nucleus celldiff_r2_iterative A549 rerun.
# Runs immediately — all prediction zarrs are present.
#
# Array index → grouped leaf mapping:
#   0: er_celldiff_r2_a549trained           (iPSC + A549 sec61b denv/mock/zikv)
#   1: membrane_celldiff_r2_a549trained     (iPSC + A549 membrane denv)
#   2: nucleus_celldiff_r2_a549trained      (iPSC only)
#   3: mitochondria_celldiff_r2_a549trained (iPSC only)
#   4: nucleus_celldiff_r2_iterative_a549_rerun (A549 nucleus denv/mock/zikv — rerun of failed campaign task 6)
#
#SBATCH --job-name=celldiff-r2-a549tr-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/er_celldiff_r2_a549trained/eval_grouped
  grouped/membrane_celldiff_r2_a549trained/eval_grouped
  grouped/nucleus_celldiff_r2_a549trained/eval_grouped
  grouped/mitochondria_celldiff_r2_a549trained/eval_grouped
  grouped/nucleus_celldiff_r2_iterative_a549_rerun/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[celldiff-r2-a549tr-eval] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[celldiff-r2-a549tr-eval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# Derive the thread cap from the allocation (never hardcode — matches the #429 fix).
export DYNACELL_THREADS_PER_WORKER=${SLURM_CPUS_PER_TASK:-32}

cd /hpc/mydata/alex.kalinin/VisCy
# GLCM + focus (compute-at-eval-time: the in-focus plane is read from focus_slice
# zattrs when present, else computed from Phase3D and cached in io.gt_cache_dir, so
# no .ozx repackaging is needed). focus_slab=deep-feature/per-cell ±2 max-proj;
# slice_selection=focus=2D instance seg on the in-focus plane (was frac=0.30).
# NOTE: superseded by tools/run_focus_campaign.slurm (these leaves run in its phase 2).
exec uv run dynacell evaluate-grouped "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  segmentation.slice_selection=focus \
  force_recompute.final_metrics=true
