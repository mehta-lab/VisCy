#!/bin/bash
# Focus-aware re-eval of the vscyto3d-ablations track (PR #468 follow-up).
#
# The grouped focus campaign (run_focus_campaign.slurm) deliberately SKIPS the
# ablation models — the grouped-config generator carves them out
# (_is_ablation_track_zarr). This launcher re-evaluates those 32 ablation leaves
# with the SAME focus overrides so they're comparable to the main campaign:
#
#   RandInit            er/mito/nucleus/membrane  (Track A, untrained)
#   Cytoland            nucleus/membrane          (Track B1, public ckpt, no FT)
#   InfectionFT         nucleus/membrane          (Track B2, A549-infection ckpt, no FT)
#   CytolandFT          nucleus/membrane × {a549,ipsc}-trained  (Track C1, + dynacell FT)
#   InfectionFT-DynacellFT  same grid             (Track C2)
#
# Each (org, model, train_set) has an eval__a549_mantis leaf (grouped, 3 A549
# conditions -> evaluate-grouped) and an eval__ipsc_confocal leaf (single iPSC
# -> evaluate). 16 grouped + 16 single = 32 leaves, one per array task.
#
# GT IS ALREADY WARM. The ablations resolve the SAME per-dataset gt_cache_dir
# (eval_cache/{h2b,caax,sec61b,tomm20}_*, ipsc/eval_cache/*) the main campaign
# warmed — focus planes, GLCM-CP (22-col), focus-tagged GT deep features, GT
# masks, and the in-focus GT instance plane. So this launcher does NOT force
# gt_cp (forcing it on 32 concurrent leaves would re-trigger the produce-then-hit
# contention on the shared GT cache). Each task computes only its PRED side
# (pred CP forced fresh to avoid any stale-schema pred cache; pred deep features
# auto-invalidate via the focus tag). Gate this on the main campaign with
# afterany (see submit_ablations_focus.sh) so GT is fully built before we read it.
#
# Focus overrides (compute-at-eval-time, Option A — identical to the campaign):
#   feature_metrics.cp.glcm.enabled=true
#   feature_metrics.focus_slab.enabled=true
#   feature_metrics.focus_slab.halfwidth=2
#   segmentation.slice_selection=focus
#   force_recompute.pred_cp=true
#   force_recompute.final_metrics=true
#
# Usage (via the wrapper, which wires the afterany dependency + array size):
#   tools/submit_ablations_focus.sh
# Or by hand:
#   sbatch --array=0-31 tools/run_ablations_focus.slurm
#
#SBATCH --job-name=ablations_focus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=/hpc/mydata/alex.kalinin/logs/reeval/%x_%A_%a.out

set -euo pipefail

# One leaf per array task. eval__a549_mantis -> grouped (3 conditions);
# eval__ipsc_confocal -> single. Index = SLURM_ARRAY_TASK_ID.
LEAVES=(
  er/fcmae_vscyto3d_pretrained_randinit/randinit/eval__a549_mantis
  er/fcmae_vscyto3d_pretrained_randinit/randinit/eval__ipsc_confocal
  mito/fcmae_vscyto3d_pretrained_randinit/randinit/eval__a549_mantis
  mito/fcmae_vscyto3d_pretrained_randinit/randinit/eval__ipsc_confocal
  nucleus/fcmae_vscyto3d_pretrained_randinit/randinit/eval__a549_mantis
  nucleus/fcmae_vscyto3d_pretrained_randinit/randinit/eval__ipsc_confocal
  membrane/fcmae_vscyto3d_pretrained_randinit/randinit/eval__a549_mantis
  membrane/fcmae_vscyto3d_pretrained_randinit/randinit/eval__ipsc_confocal
  nucleus/fcmae_vscyto3d_pretrained_cytoland/cytoland/eval__a549_mantis
  nucleus/fcmae_vscyto3d_pretrained_cytoland/cytoland/eval__ipsc_confocal
  membrane/fcmae_vscyto3d_pretrained_cytoland/cytoland/eval__a549_mantis
  membrane/fcmae_vscyto3d_pretrained_cytoland/cytoland/eval__ipsc_confocal
  nucleus/fcmae_vscyto3d_pretrained_infectionft/infectionft/eval__a549_mantis
  nucleus/fcmae_vscyto3d_pretrained_infectionft/infectionft/eval__ipsc_confocal
  membrane/fcmae_vscyto3d_pretrained_infectionft/infectionft/eval__a549_mantis
  membrane/fcmae_vscyto3d_pretrained_infectionft/infectionft/eval__ipsc_confocal
  nucleus/vscyto3d_cytolandft/a549_mantis/eval__a549_mantis
  nucleus/vscyto3d_cytolandft/a549_mantis/eval__ipsc_confocal
  nucleus/vscyto3d_cytolandft/ipsc_confocal/eval__a549_mantis
  nucleus/vscyto3d_cytolandft/ipsc_confocal/eval__ipsc_confocal
  membrane/vscyto3d_cytolandft/a549_mantis/eval__a549_mantis
  membrane/vscyto3d_cytolandft/a549_mantis/eval__ipsc_confocal
  membrane/vscyto3d_cytolandft/ipsc_confocal/eval__a549_mantis
  membrane/vscyto3d_cytolandft/ipsc_confocal/eval__ipsc_confocal
  nucleus/vscyto3d_infectionft_dynacellft/a549_mantis/eval__a549_mantis
  nucleus/vscyto3d_infectionft_dynacellft/a549_mantis/eval__ipsc_confocal
  nucleus/vscyto3d_infectionft_dynacellft/ipsc_confocal/eval__a549_mantis
  nucleus/vscyto3d_infectionft_dynacellft/ipsc_confocal/eval__ipsc_confocal
  membrane/vscyto3d_infectionft_dynacellft/a549_mantis/eval__a549_mantis
  membrane/vscyto3d_infectionft_dynacellft/a549_mantis/eval__ipsc_confocal
  membrane/vscyto3d_infectionft_dynacellft/ipsc_confocal/eval__a549_mantis
  membrane/vscyto3d_infectionft_dynacellft/ipsc_confocal/eval__ipsc_confocal
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID unset — launch via sbatch --array (see submit_ablations_focus.sh)." >&2
  exit 2
fi

LEAF="${LEAVES[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$LEAF" ]]; then
  echo "No leaf at array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# eval__a549_mantis carries a conditions: list -> grouped; iPSC leaf is single.
case "$(basename "$LEAF")" in
  eval__a549_mantis) SUBCMD=evaluate-grouped ;;
  *) SUBCMD=evaluate ;;
esac

export DYNACELL_THREADS_PER_WORKER=${SLURM_CPUS_PER_TASK:-16}

echo "[ablations_focus] task ${SLURM_ARRAY_TASK_ID}: subcmd=${SUBCMD} leaf=${LEAF}"
echo "[ablations_focus] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1) threads=${DYNACELL_THREADS_PER_WORKER}"

cd /hpc/mydata/alex.kalinin/VisCy

exec uv run dynacell "${SUBCMD}" "leaf=${LEAF}" \
  feature_metrics.cp.glcm.enabled=true \
  feature_metrics.focus_slab.enabled=true \
  feature_metrics.focus_slab.halfwidth=2 \
  segmentation.slice_selection=focus \
  force_recompute.pred_cp=true \
  force_recompute.final_metrics=true
