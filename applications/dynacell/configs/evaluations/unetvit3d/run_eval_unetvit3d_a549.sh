#!/usr/bin/env bash
# A549 UNetViT3D evaluation — 4 organelles × 3 infection conditions.

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/evaluations

V1_SPACING="[0.174,0.1494,0.1494]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

run_eval () {
    local target=$1 infection=$2 gt_basename=$3 \
          pred_zarr=$4 pred_chan=$5 gt_chan=$6 spacing=$7
    local save_dir="${OUT_ROOT}/eval_unetvit3d_${target}_${infection}"
    echo ">>> unetvit3d ${target} ${infection}"
    uv run dynacell evaluate \
        target_name="${target}" \
        io.pred_path="${PRED_ROOT}/${pred_zarr}" \
        io.pred_channel_name="${pred_chan}" \
        io.gt_path="${GT_ROOT}/test/${gt_basename}.ozx" \
        io.gt_channel_name="${gt_chan}" \
        io.cell_segmentation_path="${GT_ROOT}/test/${gt_basename}_seg_cleaned.zarr" \
        pixel_metrics.spacing="${spacing}" \
        save.save_dir="${save_dir}" \
        compute_feature_metrics=true \
        "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
        force_recompute.all=true
}

# SEC61B (ER)
run_eval er   mock SEC61B_mock sec61b_unetvit3d__sec61b_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er   denv SEC61B_DENV sec61b_unetvit3d__sec61b_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval er   zikv SEC61B_ZIKV sec61b_unetvit3d__sec61b_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# CAAX (membrane)
run_eval memb mock CAAX_mock memb_unetvit3d_mock.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval memb denv CAAX_DENV memb_unetvit3d_denv.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval memb zikv CAAX_ZIKV memb_unetvit3d_zikv.zarr Membrane_prediction Membrane "${V1_SPACING}"

# H2B (nucleus)
run_eval nucleus mock H2B_mock nucleus_unetvit3d_mock.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval nucleus denv H2B_DENV nucleus_unetvit3d_denv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval nucleus zikv H2B_ZIKV nucleus_unetvit3d_zikv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"

# TOMM20 (mitochondria)
run_eval mito mock TOMM20_mock tomm20_unetvit3d__tomm20_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval mito denv TOMM20_DENV tomm20_unetvit3d__tomm20_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval mito zikv TOMM20_ZIKV tomm20_unetvit3d__tomm20_zikv.zarr Structure_prediction Structure "${V1_SPACING}"
