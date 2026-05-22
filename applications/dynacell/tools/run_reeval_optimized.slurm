#!/bin/bash
# Re-eval campaign (optimized): 7 grouped leaves whose predicts are already
# on disk and whose pred_cache_dir collisions have been fixed by commit
# fdf71bb7 (organelle-namespaced iPSC cache dirs). Uses the GPU-microssim
# acceleration commits f02b9aed + 79f35b83 currently on the branch tip.
#
# Each task runs ONE leaf via `dynacell evaluate-grouped`. The pipeline
# amortizes model loading across conditions inside a leaf, so 7 tasks
# means 7 GPU allocations rather than (sum of conditions) allocations.
#
# No --array-concurrency cap — VAST/metadata can handle dozens.
#
#SBATCH --job-name=reeval-optimized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-6
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/mitochondria_ipsc_trained/eval_grouped       # 7 missing — non-celldiff ipsc + celldiff_r2_* ipsc
  grouped/nucleus_ipsc_trained/eval_grouped            # 7 missing — same shape
  grouped/mitochondria_joint/eval_grouped              # 4 missing — joint ipsc conditions
  grouped/er_celldiff_r2_a549trained/eval_grouped      # 4 missing — ipsc + 3 a549 mantis plates
  grouped/membrane_celldiff_r2_a549trained/eval_grouped       # 2 missing — ipsc + a549 denv
  grouped/mitochondria_celldiff_r2_a549trained/eval_grouped   # 1 missing — ipsc cross-domain
  grouped/nucleus_celldiff_r2_a549trained/eval_grouped        # 1 missing — ipsc cross-domain
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[reeval-optimized] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[reeval-optimized] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[reeval-optimized] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

export DYNACELL_THREADS_PER_WORKER=32

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped "leaf=${LEAF}"
