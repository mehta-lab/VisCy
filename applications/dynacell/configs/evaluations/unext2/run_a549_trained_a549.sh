#!/usr/bin/env bash
# UNeXt2 (fcmae_vscyto3d_scratch) A549-trained — evaluate on A549 test set (nucleus + membrane × 3 infections).

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/evaluations_a549trained

V1_SPACING="[0.174,0.1494,0.1494]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

run_eval () {
    local target=$1 infection=$2 gt_basename=$3 \
          pred_zarr=$4 pred_chan=$5 gt_chan=$6 spacing=$7
    local save_dir="${OUT_ROOT}/eval_unext2_a549trained_${target}_${infection}"
    echo ">>> unext2 a549trained ${target} ${infection}"
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

# H2B (nucleus)
run_eval nucleus mock H2B_mock nucl_fcmae_vscyto3d_scratch_a549trained_mock.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval nucleus denv H2B_DENV nucl_fcmae_vscyto3d_scratch_a549trained_denv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval nucleus zikv H2B_ZIKV nucl_fcmae_vscyto3d_scratch_a549trained_zikv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"

# CAAX (membrane)
run_eval membrane mock CAAX_mock memb_fcmae_vscyto3d_scratch_a549trained_mock.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval membrane denv CAAX_DENV memb_fcmae_vscyto3d_scratch_a549trained_denv.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval membrane zikv CAAX_ZIKV memb_fcmae_vscyto3d_scratch_a549trained_zikv.zarr Membrane_prediction Membrane "${V1_SPACING}"
