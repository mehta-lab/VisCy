#!/bin/bash
# Focus-aware re-eval campaign (PR #468). Two phases, submitted with a
# dependency by tools/submit_focus_campaign.sh:
#
#   PHASE 1 (gtwarm): the 4 *_ipsc_trained grouped leaves. Each tests its
#     organelle on all 3 A549 conditions (denv/mock/zikv) + iPSC, so the four
#     collectively PRODUCE every GT-side cache the campaign needs — focus planes
#     (computed from Phase3D, cached in io.gt_cache_dir), GLCM-enriched CP, the
#     focus-slab GT deep features, organelle masks, and the in-focus instance
#     plane. Different organelles use distinct gt_cache_dirs, so the four run
#     concurrently with no contention.
#   PHASE 2 (eval): the remaining 16 leaves (joint + a549_trained +
#     celldiff_r2_a549trained). They only READ the now-warm GT caches (writing
#     just their own per-model pred caches + final metrics), so they fan out
#     fully in parallel — the GT cache is produced once (phase 1) then hit.
#
# Each array task runs 2 evals on its single GPU (backgrounded), splitting
# cpus-per-task between them. Time limit is 12 h to clear the cluster
# maintenance window; fault isolation is per task.
#
# Focus overrides applied to every leaf (compute-at-eval-time, Option A):
#   feature_metrics.cp.glcm.enabled=true        — Haralick texture (CP recompute)
#   feature_metrics.focus_slab.enabled=true     — deep-feature crops + per-cell
#   feature_metrics.focus_slab.halfwidth=2       — max-proj over the in-focus ±2 slab
#   segmentation.slice_selection=focus           — 2D instance seg on the in-focus
#                                                  plane (was frac=0.30)
#   force_recompute.final_metrics=true           — re-aggregate the CSV/NPY
# The focus plane is read from focus_slice zattrs when present (iPSC) else computed
# from Phase3D and cached in io.gt_cache_dir (A549 published .ozx stay byte-identical).
#
# Usage (via the wrapper, which wires the afterok dependency + array sizes):
#   tools/submit_focus_campaign.sh
# Or by hand:
#   J1=$(sbatch --parsable --array=0-1%2 tools/run_focus_campaign.slurm gtwarm)
#   sbatch --dependency=afterok:$J1 --array=0-7%8 tools/run_focus_campaign.slurm eval
#
#SBATCH --job-name=focus_reeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -uo pipefail

PHASE=${1:?"usage: run_focus_campaign.slurm <gtwarm|eval>"}

# Phase 1 produces all GT caches; phase 2 reads them. Leaf order within a phase
# is paired (2 per task); pairs mix organelles so co-scheduled evals touch
# distinct gt_cache dirs.
case "$PHASE" in
  gtwarm)
    LEAVES=(
      grouped/er_ipsc_trained/eval_grouped
      grouped/mitochondria_ipsc_trained/eval_grouped
      grouped/nucleus_ipsc_trained/eval_grouped
      grouped/membrane_ipsc_trained/eval_grouped
    )
    ;;
  eval)
    LEAVES=(
      grouped/er_joint/eval_grouped
      grouped/mitochondria_joint/eval_grouped
      grouped/er_a549_trained/eval_grouped
      grouped/mitochondria_a549_trained/eval_grouped
      grouped/nucleus_joint/eval_grouped
      grouped/membrane_joint/eval_grouped
      grouped/nucleus_a549_trained/eval_grouped
      grouped/membrane_a549_trained/eval_grouped
      grouped/er_celldiff_r2_a549trained/eval_grouped
      grouped/mitochondria_celldiff_r2_a549trained/eval_grouped
      grouped/nucleus_celldiff_r2_a549trained/eval_grouped
      grouped/membrane_celldiff_r2_a549trained/eval_grouped
      grouped/nucleus_celldiff_r2_iterative_a549_rerun/eval_grouped
      grouped/mitochondria_celldiff_r2_a549trained_later/eval_grouped
      grouped/nucleus_celldiff_r2_a549trained_later/eval_grouped
      grouped/membrane_celldiff_r2_a549trained_later/eval_grouped
    )
    ;;
  *)
    echo "Unknown phase '$PHASE' (expected gtwarm|eval)" >&2
    exit 2
    ;;
esac

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID unset — launch via sbatch --array (see submit_focus_campaign.sh)." >&2
  exit 2
fi

# Two leaves per task (this GPU runs both concurrently). Split CPUs so the two
# backgrounded processes don't oversubscribe BLAS/OMP. Matches the #429 fix:
# derive threads from SLURM_CPUS_PER_TASK, never hardcode.
I0=$((SLURM_ARRAY_TASK_ID * 2))
I1=$((I0 + 1))
LEAF_A="${LEAVES[$I0]:-}"
LEAF_B="${LEAVES[$I1]:-}"
export DYNACELL_THREADS_PER_WORKER=$(( ${SLURM_CPUS_PER_TASK:-32} / 2 ))

echo "[focus_reeval/$PHASE] task ${SLURM_ARRAY_TASK_ID}: A=${LEAF_A} B=${LEAF_B}"
echo "[focus_reeval/$PHASE] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1) threads/eval=${DYNACELL_THREADS_PER_WORKER}"

cd /hpc/mydata/alex.kalinin/VisCy

run_leaf() {
  local leaf="$1"
  [[ -z "$leaf" ]] && return 0
  uv run dynacell evaluate-grouped "leaf=${leaf}" \
    feature_metrics.cp.glcm.enabled=true \
    feature_metrics.focus_slab.enabled=true \
    feature_metrics.focus_slab.halfwidth=2 \
    segmentation.slice_selection=focus \
    force_recompute.final_metrics=true
}

rc=0
run_leaf "$LEAF_A" & PID_A=$!
run_leaf "$LEAF_B" & PID_B=$!
wait $PID_A || { echo "[focus_reeval] leaf A failed: ${LEAF_A}" >&2; rc=1; }
[[ -n "$LEAF_B" ]] && { wait $PID_B || { echo "[focus_reeval] leaf B failed: ${LEAF_B}" >&2; rc=1; }; }
exit $rc
