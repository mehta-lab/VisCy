#!/usr/bin/env bash
# A549 UNetViT3D evaluation against the cropped 512x512 OZX test corpus.
# Pixel + segmentation tracks only (no cell_segmentation for A549 yet).

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/evaluations

V1_SPACING="[0.174,0.1494,0.1494]"

run_eval () {
    local target=$1 plate_short=$2 plate_dir=$3 gt_basename=$4 \
          pred_zarr=$5 pred_chan=$6 gt_chan=$7 spacing=$8
    local save_dir="${OUT_ROOT}/eval_unetvit3d_${target}_${plate_short}"
    echo ">>> unetvit3d ${target} ${plate_short}"
    uv run dynacell evaluate \
        target_name="${target}" \
        io.pred_path="${PRED_ROOT}/${pred_zarr}" \
        io.pred_channel_name="${pred_chan}" \
        io.gt_path="${GT_ROOT}/${plate_dir}/test/${gt_basename}.ozx" \
        io.gt_channel_name="${gt_chan}" \
        pixel_metrics.spacing="${spacing}" \
        save.save_dir="${save_dir}" \
        compute_feature_metrics=false \
        force_recompute.all=true
}

# SEC61B (ER) — mantis_v1
run_eval er 2024_10_31 2024_10_31_A549_SEC61_ZIKV_DENV SEC61B sec61b_unetvit3d__2024_10_31.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er 2024_11_07 2024_11_07_A549_SEC61_DENV      SEC61B sec61b_unetvit3d__2024_11_07.zarr Structure_prediction Structure "${V1_SPACING}"
