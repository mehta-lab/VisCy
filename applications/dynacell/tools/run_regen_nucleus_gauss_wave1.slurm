#!/bin/bash
# Nucleus segmentation re-eval, WAVE 1.
#
# Rationale: H2B SuperModel preprocessing now applies cubic-GPU
# Gaussian σ=1 smoothing to the input (segmentation.py
# `_smooth_nucleus_input`, landed via PR #440). The 4 GT and 47 pred
# `organelle_masks/nucleus.zarr` caches were renamed to
# `*.pre_gauss_20260528T011530Z` so the cache layer recomputes from
# scratch with the new preprocessing.
#
# Wave 1 runs `nucleus_ipsc_trained` first because its 22 conditions
# cover the broadest GT footprint: iPSC `aics-hipsc` + all three A549
# mantis plates (`h2b_mock` / `h2b_denv` / `h2b_zikv`). Completing this
# job populates all 4 GT mask caches, so wave 2 (the other 5 nucleus
# leaves) only pays per-condition pred-mask compute and hits a warm GT.
#
#SBATCH --job-name=regen-nucleus-gauss-w1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%j.out

set -euo pipefail

LEAF=grouped/nucleus_ipsc_trained/eval_grouped

echo "[regen-nucleus-w1] leaf=${LEAF}"
echo "[regen-nucleus-w1] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "[regen-nucleus-w1] commit=$(cd /hpc/mydata/alex.kalinin/VisCy && git rev-parse --short HEAD)"

# Per-FOV worker thread budget. With executor=process + fov_workers=2,
# split the SLURM cpu allocation in half so each worker gets a sane BLAS cap.
export DYNACELL_THREADS_PER_WORKER=$((SLURM_CPUS_PER_TASK / 2))

cd /hpc/mydata/alex.kalinin/VisCy
# Overrides:
#   force_recompute.final_metrics=true   — rewrite CSVs (mask metrics depend on the
#                                          newly recomputed nucleus masks).
#   force_recompute.gt_cp=true           — A549 GT cp_features manifests had
#                                          spacing=[0.174, 0.116, 0.116] stale from
#                                          before bd8bda71; manifest entries were
#                                          stripped, but the on-disk cp.zarr also
#                                          needs to be overwritten with re-spaced
#                                          features.
#   force_recompute.pred_cp=true         — same story for the 36 pred caches under
#                                          a549/eval_cache_pred/.../h2b_*.
#   runtime.executor=process,fov_workers=2 — 2-up per-FOV parallelism. fcntl GPU
#                                          lock serializes the GPU; CPU prep / iohub
#                                          reads overlap. ~2× wall on segmentation-
#                                          heavy evals. Per-condition pool re-loads
#                                          models (~30-90 s × n_conditions), small
#                                          tax relative to the FOV-time savings.
exec uv run dynacell evaluate-grouped \
  "leaf=${LEAF}" \
  "force_recompute.final_metrics=true" \
  "force_recompute.gt_cp=true" \
  "force_recompute.pred_cp=true" \
  "runtime.executor=process" \
  "runtime.fov_workers=2" \
  "runtime.threads_per_worker=${DYNACELL_THREADS_PER_WORKER}"
