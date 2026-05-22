#!/usr/bin/env bash
# FNet3D joint (iPSC+A549) model — nucleus prediction on A549 mock test set.

set -euo pipefail
ml uv
source ".envrc"

uv run dynacell evaluate \
    target_name=nucleus \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/a549/predictions/nucl_fnet3d_paper_jointtrained_mock.zarr \
    io.pred_channel_name=Nuclei_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/test/H2B_mock.ozx \
    io.gt_channel_name=Nuclei \
    pixel_metrics.spacing=[0.174,0.1494,0.1494] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/eval_fnet3d_joint_nucleus_mock \
    compute_feature_metrics=true \
    force_recompute.all=true
