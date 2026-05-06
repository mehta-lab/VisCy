#!/usr/bin/env bash
# UNeXt2 (fcmae_vscyto3d_scratch) A549-trained — evaluate on iPSC test set (nucleus + membrane).

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations_a549trained

IPSC_SPACING="[0.29,0.108,0.108]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

# Nucleus (H2B)
echo ">>> unext2 a549trained nucleus (iPSC)"
uv run dynacell evaluate \
    target_name=nucleus \
    io.pred_path="${PRED_ROOT}/nucl_fcmae_vscyto3d_scratch_a549trained.zarr" \
    io.pred_channel_name=Nuclei_prediction \
    io.gt_path="${GT_ROOT}/cell.zarr" \
    io.gt_channel_name=Nuclei \
    io.cell_segmentation_path="${GT_ROOT}/cell_segmented_cleaned.zarr" \
    pixel_metrics.spacing="${IPSC_SPACING}" \
    save.save_dir="${OUT_ROOT}/eval_unext2_a549trained_nucleus" \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
    force_recompute.all=true

# Membrane (CAAX)
echo ">>> unext2 a549trained membrane (iPSC)"
uv run dynacell evaluate \
    target_name=membrane \
    io.pred_path="${PRED_ROOT}/memb_fcmae_vscyto3d_scratch_a549trained.zarr" \
    io.pred_channel_name=Membrane_prediction \
    io.gt_path="${GT_ROOT}/cell.zarr" \
    io.gt_channel_name=Membrane \
    io.cell_segmentation_path="${GT_ROOT}/cell_segmented_cleaned.zarr" \
    pixel_metrics.spacing="${IPSC_SPACING}" \
    save.save_dir="${OUT_ROOT}/eval_unext2_a549trained_membrane" \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
    force_recompute.all=true
