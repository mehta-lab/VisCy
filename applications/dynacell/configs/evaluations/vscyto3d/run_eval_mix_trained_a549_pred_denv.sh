#!/usr/bin/env bash
# VSCyto3D joint (iPSC+A549) model — membrane prediction on A549 DENV test set.

set -euo pipefail
ml uv
source ".envrc"

DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

uv run dynacell evaluate \
    target_name=membrane \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/a549/joint_predictions/memb_fcmae_vscyto3d_pretrained_jointtrained_denv.zarr \
    io.pred_channel_name=Membrane_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/test/CAAX_DENV.ozx \
    io.gt_channel_name=Membrane \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/test/CAAX_DENV_seg_cleaned.zarr \
    pixel_metrics.spacing=[0.174,0.1494,0.1494] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/eval_vscyto3d_joint_membrane_denv \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
    force_recompute.all=true
