#!/bin/bash
# microSSIM-only regen across all leaves whose pixel_metrics.csv was written
# before a005f9be (2026-05-26 11:37:27 -0700) — i.e. under the OLD per-FOV
# alpha fit. After this sweep, every grouped leaf in the catalog will share
# leaf-level alpha semantics, making cross-leaf microSSIM comparisons valid.
#
# Inventory derivation:
#   1. Walked /hpc/projects/virtual_staining/training/dynacell/{ipsc,a549}/
#      for all pixel_metrics.csv (356 files total).
#   2. Partitioned by mtime vs a005f9be commit time.
#   3. Mapped each save_dir back to its eval_grouped.yaml leaf.
#   4. 18 unique grouped leaves had at least one pre-cutoff CSV. Three of
#      them (er_ipsc_trained, membrane_ipsc_trained, mitochondria_ipsc_trained)
#      have CSV mtimes straddling the cutoff — i.e. mixed semantics within a
#      single leaf — and must be regenerated for self-consistency.
#
# Scope deliberately excludes:
#   * er_celldiff_r2_a549trained — re-running now in 33399426 (ER smoke).
#   * membrane_celldiff_r2_a549trained — already cached under leaf-level α
#     by smoke task 33389044_1.
#   * 186 unmapped pre-cutoff save_dirs in legacy non-grouped pipelines
#     (evaluations/eval_celldiff_iterative_*, joint_evaluations*/, ...).
#     Those are outputs of an older eval framework that no longer has a
#     producing yaml in the grouped catalog; nothing in the current pipeline
#     re-creates them.
#
# Only force_recompute.final_metrics=true is set — embeddings, segmentations,
# CellPose, DINOv3, DynaCLR, CellDINO caches stay frozen (they are independent
# of the microSSIM fit semantics). This keeps each leaf's wall time tight.
#
# Submit (no array cap — let SLURM use as many GPUs as the partition has):
#   sbatch applications/dynacell/tools/run_regen_microssim_leaves.slurm
#
#SBATCH --job-name=regen-microssim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-17
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

LEAVES=(
  grouped/er_a549_trained/eval_grouped
  grouped/er_ipsc_trained/eval_grouped
  grouped/er_joint/eval_grouped
  grouped/membrane_a549_trained/eval_grouped
  grouped/membrane_celldiff_r2_a549trained_later/eval_grouped
  grouped/membrane_ipsc_trained/eval_grouped
  grouped/membrane_joint/eval_grouped
  grouped/mitochondria_a549_trained/eval_grouped
  grouped/mitochondria_celldiff_r2_a549trained/eval_grouped
  grouped/mitochondria_celldiff_r2_a549trained_later/eval_grouped
  grouped/mitochondria_ipsc_trained/eval_grouped
  grouped/mitochondria_joint/eval_grouped
  grouped/nucleus_a549_trained/eval_grouped
  grouped/nucleus_celldiff_r2_a549trained/eval_grouped
  grouped/nucleus_celldiff_r2_a549trained_later/eval_grouped
  grouped/nucleus_celldiff_r2_iterative_a549_rerun/eval_grouped
  grouped/nucleus_ipsc_trained/eval_grouped
  grouped/nucleus_joint/eval_grouped
)

LEAF="${LEAVES[${SLURM_ARRAY_TASK_ID}]}"
echo "[regen] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[regen] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[regen] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

export DYNACELL_THREADS_PER_WORKER="${SLURM_CPUS_PER_TASK}"

cd /hpc/mydata/alex.kalinin/VisCy
exec uv run dynacell evaluate-grouped \
  "leaf=${LEAF}" \
  "force_recompute.final_metrics=true"
