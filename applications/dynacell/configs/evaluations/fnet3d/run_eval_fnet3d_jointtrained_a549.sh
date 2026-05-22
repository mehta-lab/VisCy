#!/usr/bin/env bash
# FNet3D joint-trained (iPSC + A549 mantis) — evaluate on A549 test set (nucleus × 3 infections).
# Membrane already evaluated under joint_evaluations/eval_fnet3d_joint_membrane_<cond>.

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations

V1_SPACING="[0.174,0.1494,0.1494]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

mkdir -p "${OUT_ROOT}"

run_eval () {
    local infection=$1 gt_basename=$2 pred_zarr=$3
    local save_dir="${OUT_ROOT}/eval_fnet3d_joint_nucleus_${infection}"
    echo ">>> fnet3d joint nucleus ${infection}"
    uv run dynacell evaluate \
        target_name=nucleus \
        io.pred_path="${PRED_ROOT}/${pred_zarr}" \
        io.pred_channel_name=Nuclei_prediction \
        io.gt_path="${GT_ROOT}/test/${gt_basename}.ozx" \
        io.gt_channel_name=Nuclei \
        io.cell_segmentation_path="${GT_ROOT}/test/${gt_basename}_seg_cleaned.zarr" \
        pixel_metrics.spacing="${V1_SPACING}" \
        save.save_dir="${save_dir}" \
        compute_feature_metrics=true \
        "feature_extractor.dynaclr.checkpoint='${DYNACLR_CKPT}'" \
        force_recompute.all=true
}

run_eval mock H2B_mock     nucl_fnet3d_paper_jointtrained_mock.zarr
run_eval denv H2B_DENV     nucl_fnet3d_paper_jointtrained_denv.zarr
run_eval zikv H2B_ZIKV     nucl_fnet3d_paper_jointtrained_zikv.zarr
