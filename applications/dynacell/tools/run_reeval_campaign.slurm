#!/bin/bash
# Re-eval campaign: 12 grouped jobs, one per (organelle, train_set) bucket.
# See applications/dynacell/configs/benchmarks/virtual_staining/_internal/leaf/grouped/README.md
# and /hpc/mydata/alex.kalinin/.claude/plans/vectorized-sleeping-clock.md.
#
# Eval-time GPU load: cellpose (~1 GB) + 4 ViT backbones (DINOv3 / DynaCLR /
# CellDINO / cp) — fits comfortably on A100/H100/H200 80GB. The
# pix2pix3d_needs_h200 memory applies to predict-time DDP only.
#
# --array=0-11%4 caps concurrency at 4 simultaneous jobs to ease VAST
# metadata pressure. Hold mitochondria_joint (index 4) until the 3 tomm20
# F-net joint A549 predict jobs land (see Step 1 of the campaign plan);
# submit indices 0-3,5-11 first, then index 4 with
# --dependency=afterok:<predict_jobids>.
#
#SBATCH --job-name=reeval-campaign
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-11%4
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/er_ipsc_trained/eval_grouped
  grouped/er_joint/eval_grouped
  grouped/er_a549_trained/eval_grouped
  grouped/mitochondria_ipsc_trained/eval_grouped
  grouped/mitochondria_joint/eval_grouped
  grouped/mitochondria_a549_trained/eval_grouped
  grouped/nucleus_ipsc_trained/eval_grouped
  grouped/nucleus_joint/eval_grouped
  grouped/nucleus_a549_trained/eval_grouped
  grouped/membrane_ipsc_trained/eval_grouped
  grouped/membrane_joint/eval_grouped
  grouped/membrane_a549_trained/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[reeval] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[reeval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# Pre-set BLAS / OpenBLAS / OMP thread caps before any C-extension loads.
# Matches dynacell/CLAUDE.md "Eval runtime / parallelism" guidance.
export DYNACELL_THREADS_PER_WORKER=32

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped "leaf=${LEAF}"
