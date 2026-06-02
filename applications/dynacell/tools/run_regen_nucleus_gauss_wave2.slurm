#!/bin/bash
# Nucleus segmentation re-eval, WAVE 2 (depends on wave 1).
#
# Submit with `--dependency=afterany:<wave1_jobid>` so wave 2 starts
# after wave 1's `nucleus_ipsc_trained` job finishes — wave 1 warms all
# four GT nucleus mask caches (iPSC + h2b_{mock,denv,zikv}), so each of
# the 5 remaining leaves only pays per-condition pred-mask compute.
#
# Array task → leaf (4 leaves; nucleus_celldiff_r2_a549trained is being
# completed by the prior submission 33421587_1 — iPSC-only, unaffected by
# the A549 cp_features spacing wall — so it's not relisted here).
#   0: nucleus_a549_trained                    (9 a549 conds)
#   1: nucleus_celldiff_r2_a549trained_later   (3 a549 conds)
#   2: nucleus_celldiff_r2_iterative_a549_rerun (3 a549 conds)
#   3: nucleus_joint                           (12 conds: 3 ipsc + 9 a549)
#
#SBATCH --job-name=regen-nucleus-gauss-w2
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

LEAVES=(
  grouped/nucleus_a549_trained/eval_grouped
  grouped/nucleus_celldiff_r2_a549trained_later/eval_grouped
  grouped/nucleus_celldiff_r2_iterative_a549_rerun/eval_grouped
  grouped/nucleus_joint/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[regen-nucleus-w2] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[regen-nucleus-w2] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[regen-nucleus-w2] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

# See run_regen_nucleus_gauss_wave1.slurm for override rationale.
export DYNACELL_THREADS_PER_WORKER=$((SLURM_CPUS_PER_TASK / 2))

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped \
  "leaf=${LEAF}" \
  "force_recompute.final_metrics=true" \
  "force_recompute.gt_cp=true" \
  "force_recompute.pred_cp=true" \
  "runtime.executor=process" \
  "runtime.fov_workers=2" \
  "runtime.threads_per_worker=${DYNACELL_THREADS_PER_WORKER}"
