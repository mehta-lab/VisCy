#!/usr/bin/env bash
# A549 CellDiff evaluation — 3 model variants × 4 organelles × 3 infection conditions.
# fcmae_vscyto3d_scratch has no nucleus predictions; those 3 calls are omitted.

set -euo pipefail
ml uv
source ".envrc"

PRED_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/predictions
GT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1
OUT_ROOT=/hpc/projects/virtual_staining/training/dynacell/a549/evaluations

V1_SPACING="[0.174,0.1494,0.1494]"
DYNACLR_CKPT='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'

run_eval () {
    local model=$1 target=$2 infection=$3 gt_basename=$4 \
          pred_zarr=$5 pred_chan=$6 gt_chan=$7 spacing=$8
    local save_dir="${OUT_ROOT}/eval_${model}_${target}_${infection}"
    echo ">>> ${model} ${target} ${infection}"
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

# ── celldiff_iterative ────────────────────────────────────────────────────────

# SEC61B (ER)
run_eval celldiff_iterative er   mock SEC61B_mock sec61b_celldiff_iterative__sec61b_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval celldiff_iterative er   denv SEC61B_DENV sec61b_celldiff_iterative__sec61b_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval celldiff_iterative er   zikv SEC61B_ZIKV sec61b_celldiff_iterative__sec61b_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# CAAX (membrane)
run_eval celldiff_iterative memb mock CAAX_mock memb_celldiff_iterative_mock.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval celldiff_iterative memb denv CAAX_DENV memb_celldiff_iterative_denv.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval celldiff_iterative memb zikv CAAX_ZIKV memb_celldiff_iterative_zikv.zarr Membrane_prediction Membrane "${V1_SPACING}"

# H2B (nucleus)
run_eval celldiff_iterative nucleus mock H2B_mock nucl_celldiff_iterative_mock.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval celldiff_iterative nucleus denv H2B_DENV nucl_celldiff_iterative_denv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval celldiff_iterative nucleus zikv H2B_ZIKV nucl_celldiff_iterative_zikv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"

# TOMM20 (mitochondria)
run_eval celldiff_iterative mito mock TOMM20_mock tomm20_celldiff_iterative__tomm20_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval celldiff_iterative mito denv TOMM20_DENV tomm20_celldiff_iterative__tomm20_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval celldiff_iterative mito zikv TOMM20_ZIKV tomm20_celldiff_iterative__tomm20_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# ── fcmae_vscyto3d_pretrained ─────────────────────────────────────────────────

# SEC61B (ER)
run_eval fcmae_pretrained er   mock SEC61B_mock sec61b_fcmae_vscyto3d_pretrained__sec61b_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_pretrained er   denv SEC61B_DENV sec61b_fcmae_vscyto3d_pretrained__sec61b_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_pretrained er   zikv SEC61B_ZIKV sec61b_fcmae_vscyto3d_pretrained__sec61b_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# CAAX (membrane)
run_eval fcmae_pretrained memb mock CAAX_mock memb_fcmae_vscyto3d_pretrained_mock.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval fcmae_pretrained memb denv CAAX_DENV memb_fcmae_vscyto3d_pretrained_denv.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval fcmae_pretrained memb zikv CAAX_ZIKV memb_fcmae_vscyto3d_pretrained_zikv.zarr Membrane_prediction Membrane "${V1_SPACING}"

# H2B (nucleus)
run_eval fcmae_pretrained nucleus mock H2B_mock nucl_fcmae_vscyto3d_pretrained_mock.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval fcmae_pretrained nucleus denv H2B_DENV nucl_fcmae_vscyto3d_pretrained_denv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"
run_eval fcmae_pretrained nucleus zikv H2B_ZIKV nucl_fcmae_vscyto3d_pretrained_zikv.zarr Nuclei_prediction Nuclei "${V1_SPACING}"

# TOMM20 (mitochondria)
run_eval fcmae_pretrained mito mock TOMM20_mock tomm20_fcmae_vscyto3d_pretrained__tomm20_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_pretrained mito denv TOMM20_DENV tomm20_fcmae_vscyto3d_pretrained__tomm20_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_pretrained mito zikv TOMM20_ZIKV tomm20_fcmae_vscyto3d_pretrained__tomm20_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# ── fcmae_vscyto3d_scratch ────────────────────────────────────────────────────
# nucleus predictions not available for this variant

# SEC61B (ER)
run_eval fcmae_scratch er   mock SEC61B_mock sec61b_fcmae_vscyto3d_scratch__sec61b_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_scratch er   denv SEC61B_DENV sec61b_fcmae_vscyto3d_scratch__sec61b_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_scratch er   zikv SEC61B_ZIKV sec61b_fcmae_vscyto3d_scratch__sec61b_zikv.zarr Structure_prediction Structure "${V1_SPACING}"

# CAAX (membrane)
run_eval fcmae_scratch memb mock CAAX_mock memb_fcmae_vscyto3d_scratch_mock.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval fcmae_scratch memb denv CAAX_DENV memb_fcmae_vscyto3d_scratch_denv.zarr Membrane_prediction Membrane "${V1_SPACING}"
run_eval fcmae_scratch memb zikv CAAX_ZIKV memb_fcmae_vscyto3d_scratch_zikv.zarr Membrane_prediction Membrane "${V1_SPACING}"

# TOMM20 (mitochondria)
run_eval fcmae_scratch mito mock TOMM20_mock tomm20_fcmae_vscyto3d_scratch__tomm20_mock.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_scratch mito denv TOMM20_DENV tomm20_fcmae_vscyto3d_scratch__tomm20_denv.zarr Structure_prediction Structure "${V1_SPACING}"
run_eval fcmae_scratch mito zikv TOMM20_ZIKV tomm20_fcmae_vscyto3d_scratch__tomm20_zikv.zarr Structure_prediction Structure "${V1_SPACING}"
