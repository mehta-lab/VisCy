#!/bin/bash
# Grouped eval for the A549-trained VSCyto3D (fcmae_vscyto3d_pretrained) models
# that were trained + predicted after the 12-bucket re-eval campaign:
#   task 0: membrane  (A549 mock/denv/zikv + iPSC confocal) — 4 conditions
#   task 1: mitochondria (A549 mock/denv/zikv) — 3 conditions; iPSC has no mito GT
#
# Mirrors tools/run_reeval_campaign.slurm directives. Eval-time GPU load is
# cellpose (~1 GB) + 4 ViT backbones (DINOv3 / DynaCLR / CellDINO / cp), which
# fits on A100/H100/H200 80 GB. evaluate-grouped loads the models once per task
# and loops the conditions serially.
#
#SBATCH --job-name=eval-vscyto3d-a549tr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/membrane_vscyto3d_a549trained/eval_grouped
  grouped/mitochondria_vscyto3d_a549trained/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[eval] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[eval] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# Pre-set BLAS / OpenBLAS / OMP thread caps before any C-extension loads.
# Matches dynacell/CLAUDE.md "Eval runtime / parallelism" guidance.
export DYNACELL_THREADS_PER_WORKER=32

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped "leaf=${LEAF}"
