#!/usr/bin/env bash
# UNeXt2 (fcmae_vscyto3d_scratch) joint-trained (iPSC + A549 mantis) —
# evaluate on iPSC test set (membrane). Companion to the existing
# joint membrane evals (eval_{fnet3d,vscyto3d,celldiff}_joint_membrane).

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/ipsc/joint_evaluations

IPSC_SPACING="[0.29,0.108,0.108]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

mkdir -p "${OUT_ROOT}"

echo ">>> unext2 joint membrane (iPSC)"
uv run dynacell evaluate \
    target_name=membrane \
    io.pred_path="${PRED_ROOT}/memb_fcmae_vscyto3d_scratch_jointtrained.zarr" \
    io.pred_channel_name=Membrane_prediction \
    io.gt_path="${GT_ROOT}/cell.zarr" \
    io.gt_channel_name=Membrane \
    io.cell_segmentation_path="${GT_ROOT}/cell_segmented_cleaned.zarr" \
    pixel_metrics.spacing="${IPSC_SPACING}" \
    save.save_dir="${OUT_ROOT}/eval_unext2_joint_membrane" \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
    force_recompute.all=true
