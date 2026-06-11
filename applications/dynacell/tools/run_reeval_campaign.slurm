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
# GLCM Haralick texture recompute: glcm.enabled flips the cp_features cache
# identity (cp_glcm_enabled false->true) so only the CP artifact auto-invalidates
# and recomputes on GPU; deep-feature / mask / instance caches are untouched.
# force_recompute.final_metrics re-aggregates the saved CSV/NPY from the enriched
# CP cache (it is reused on file-existence alone otherwise). final_metrics=true is
# also baked into the leaves' _BASE_OVERLAY; repeated here so the recompute intent
# is explicit at the launch site.
# focus_slab restricts the 2D deep-feature / per-cell projection to an in-focus
# slab centered on the GT phase focus plane (h=2 -> 5 planes), keeping the MIP off
# the out-of-focus caps. PREREQUISITE: focus_slice zattrs must already exist in
# every GT store this campaign touches -- write them with `dynacell precompute-gt
# build.focus=true` (writable .zarr) and repackage the A549 .ozx with the metadata.
# Without it the eval fails loud (no silent full-stack fallback).
exec uv run dynacell evaluate-grouped "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  force_recompute.final_metrics=true
