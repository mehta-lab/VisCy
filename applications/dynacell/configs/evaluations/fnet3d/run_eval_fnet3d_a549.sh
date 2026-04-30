#!/usr/bin/env bash
# A549 FNet3D evaluation against the cropped 512x512 OZX test corpus.
# Pixel + segmentation tracks only; feature track (compute_feature_metrics=true)
# requires io.cell_segmentation_path which is not yet authored for A549.
# force_recompute.all=true is mandatory: the OZX rebuild changed shape,
# so any cached gt_masks / final_metrics from prior runs are stale.

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/evaluations

V1_SPACING="[0.174,0.1494,0.1494]"
V2_SPACING="[0.174,0.116,0.116]"

# Layout: pred__<model>__<organelle>__<plate>__<plate_dir>__<gt_basename>__<channel_pred>__<gt_channel>__<spacing>
run_eval () {
    local target=$1 plate_short=$2 plate_dir=$3 gt_basename=$4 \
          pred_zarr=$5 pred_chan=$6 gt_chan=$7 spacing=$8
    local save_dir="${OUT_ROOT}/eval_fnet3d_${target}_${plate_short}"
    echo ">>> fnet3d ${target} ${plate_short}"
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
run_eval er 2024_10_31 2024_10_31_A549_SEC61_ZIKV_DENV          SEC61B sec61b_fnet3d_paper__2024_10_31.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er 2024_11_07 2024_11_07_A549_SEC61_DENV               SEC61B sec61b_fnet3d_paper__2024_11_07.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er 2025_07_24 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV  SEC61B sec61b_fnet3d_paper__2025_07_24.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er 2025_08_26 2025_08_26_A549_SEC61_TOMM20_ZIKV        SEC61B sec61b_fnet3d_paper__2025_08_26.zarr Structure_prediction Structure "${V1_SPACING}"

# TOMM20 (mitochondria) — mantis_v1
run_eval mitochondria 2024_11_05 2024_11_05_A549_TOMM20_ZIKV_DENV          TOMM20 tomm20_fnet3d_paper__2024_11_05.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval mitochondria 2024_11_21 2024_11_21_A549_TOMM20_DENV               TOMM20 tomm20_fnet3d_paper__2024_11_21.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval mitochondria 2025_07_24 2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV   TOMM20 tomm20_fnet3d_paper__2025_07_24.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval mitochondria 2025_08_26 2025_08_26_A549_SEC61_TOMM20_ZIKV         TOMM20 tomm20_fnet3d_paper__2025_08_26.zarr Structure_prediction Structure "${V1_SPACING}"

# CAAX (membrane) — mantis_v2
run_eval membrane 2026_03_26 2026_03_26_A549_CAAX_H2B_DENV_ZIKV CAAX memb_fnet3d_paper__2026_03_26.zarr Membrane_prediction Membrane "${V2_SPACING}"

# H2B (nucleus) — mantis_v2
run_eval nucleus  2026_03_26 2026_03_26_A549_CAAX_H2B_DENV_ZIKV H2B  nucl_fnet3d_paper__2026_03_26.zarr Nuclei_prediction   Nuclei   "${V2_SPACING}"
