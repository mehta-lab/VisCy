#!/bin/bash
# Deferred eval campaign: 3 "_later" celldiff_r2 a549-trained leaves that
# evaluate the in-flight predict array CELLDIFF_R2_A549TR_g0_RESUME (job
# 33232310, tasks 4-11 → membrane/mito/nucleus on mock/denv/zikv).
# Holds until ALL 8 predict tasks complete OK via afterok:33232310, then
# dispatches the three "_later" evals in parallel.
#
# Submit with: --dependency=afterok:33232310
#
#SBATCH --job-name=reeval-optimized-later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/membrane_celldiff_r2_a549trained_later/eval_grouped      # 2 conditions: a549 mock + zikv
  grouped/mitochondria_celldiff_r2_a549trained_later/eval_grouped  # 3 conditions: a549 denv/mock/zikv
  grouped/nucleus_celldiff_r2_a549trained_later/eval_grouped       # 3 conditions: a549 denv/mock/zikv
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[reeval-optimized-later] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[reeval-optimized-later] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[reeval-optimized-later] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

export DYNACELL_THREADS_PER_WORKER="${SLURM_CPUS_PER_TASK}"

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped "leaf=${LEAF}"
