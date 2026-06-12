#!/bin/bash
# Instance-AP re-eval of the nucleus & membrane grouped buckets (PR #468 +
# instance-AP unification follow-up).
#
# The grouped leaves now compute instance average-precision in the SAME pass as
# the pixel/feature/semantic metrics (generate_grouped_eval_configs.py wires
# compute_instance_ap + the Cellpose/watershed backend into nucleus & membrane).
# The original focus campaign predates that change, so its nucleus/membrane
# results are semantic-only (supermodel masks, no AP_*/mAP/instance_dice). This
# re-runs the six nucleus+membrane buckets to add instance metrics and switch the
# semantic Dice/IoU rows onto the instance masks (matching the paper methodology).
# ER/mito have no cell instances and are NOT re-run.
#
# Two-phase, like run_focus_campaign.slurm — the three nucleus buckets share ONE
# GT instance-mask cache (iPSC nucleus + the three A549 h2b conditions), and the
# three membrane buckets share another. Phase 1 (gtwarm) runs the two ipsc_trained
# buckets, each of which folds BOTH test sets, so it PRODUCES the full GT instance
# cache (Cellpose nucleus / GT-nuclei-seeded watershed whole-cell) for both
# organelles. Phase 2 (eval) runs the other four buckets read-only against the warm
# GT cache (afterany phase 1) so they fan out with no produce-then-hit write race.
#
# GT CP / deep-feature caches are already warm + clean from the focus campaign, so
# this does NOT force gt_cp (the instance MASK cache is a distinct namespace and
# misses fresh on the slice_selection=focus instance identity, computing on its
# own). Each task forces pred_cp fresh and re-aggregates final metrics.
#
# Focus overrides (identical to the campaign):
#   feature_metrics.cp.glcm.enabled=true
#   feature_metrics.focus_slab.enabled=true
#   feature_metrics.focus_slab.halfwidth=2
#   segmentation.slice_selection=focus
#   force_recompute.pred_cp=true
#   force_recompute.final_metrics=true
#
# Usage (via the wrapper, which wires the afterany dependency + array sizes):
#   tools/submit_instance_reeval.sh
# Or by hand:
#   J1=$(sbatch --parsable --array=0-1 tools/run_instance_reeval.slurm gtwarm)
#   sbatch --dependency=afterany:$J1 --array=0-3 tools/run_instance_reeval.slurm eval
#
#SBATCH --job-name=instance_reeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

PHASE=${1:?"usage: run_instance_reeval.slurm <gtwarm|eval> [nucleus|membrane|all]"}
ORGANELLE=${2:-all}  # restrict to one organelle (e.g. a membrane-only no-carve re-run)

# Phase 1 produces the GT instance caches; phase 2 reads them. The leaf list is
# built per (phase, organelle) so a single-organelle re-run gets a compact array
# (index 0.. with no gaps); submit_instance_reeval.sh sizes --array to match.
LEAVES=()
case "$PHASE" in
  gtwarm)
    [[ "$ORGANELLE" == all || "$ORGANELLE" == nucleus ]] && LEAVES+=(grouped/nucleus_ipsc_trained/eval_grouped)
    [[ "$ORGANELLE" == all || "$ORGANELLE" == membrane ]] && LEAVES+=(grouped/membrane_ipsc_trained/eval_grouped)
    ;;
  eval)
    [[ "$ORGANELLE" == all || "$ORGANELLE" == nucleus ]] &&
      LEAVES+=(grouped/nucleus_joint/eval_grouped grouped/nucleus_a549_trained/eval_grouped)
    [[ "$ORGANELLE" == all || "$ORGANELLE" == membrane ]] &&
      LEAVES+=(grouped/membrane_joint/eval_grouped grouped/membrane_a549_trained/eval_grouped)
    ;;
  *)
    echo "Unknown phase '$PHASE' (expected gtwarm|eval)" >&2
    exit 2
    ;;
esac

if [[ ${#LEAVES[@]} -eq 0 ]]; then
  echo "No leaves for phase '$PHASE' organelle '$ORGANELLE' (expected nucleus|membrane|all)" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID unset — launch via sbatch --array (see submit_instance_reeval.sh)." >&2
  exit 2
fi

LEAF="${LEAVES[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$LEAF" ]]; then
  echo "No leaf at array index ${SLURM_ARRAY_TASK_ID} for phase ${PHASE}" >&2
  exit 2
fi

export DYNACELL_THREADS_PER_WORKER=${SLURM_CPUS_PER_TASK:-32}

echo "[instance_reeval/$PHASE] task ${SLURM_ARRAY_TASK_ID}: leaf=${LEAF}"
echo "[instance_reeval/$PHASE] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1) threads=${DYNACELL_THREADS_PER_WORKER}"

cd /hpc/mydata/alex.kalinin/VisCy

exec uv run dynacell evaluate-grouped "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  segmentation.slice_selection=focus \
  force_recompute.pred_cp=true \
  force_recompute.final_metrics=true
