ml uv

source "$(dirname "$0")/../.envrc"

# CELL-Diff iterative — ER (SEC61B)
uv run dynacell evaluate \
    target_name=er \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/sec61b_celldiff_iterative.zarr \
    io.pred_channel_name=Structure_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B.zarr \
    io.gt_channel_name=Structure \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/SEC61B_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_iterative_sec61b \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true

# CELL-Diff iterative — Membrane
uv run dynacell evaluate \
    target_name=membrane \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/memb_celldiff_iterative.zarr \
    io.pred_channel_name=Membrane_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
    io.gt_channel_name=Membrane \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_iterative_membrane \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true

# CELL-Diff iterative — Mitochondria (TOMM20)
uv run dynacell evaluate \
    target_name=mitochondria \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/tomm20_celldiff_iterative.zarr \
    io.pred_channel_name=Structure_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20.zarr \
    io.gt_channel_name=Structure \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_iterative_tomm20 \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true

# CELL-Diff iterative — Nucleus
uv run dynacell evaluate \
    target_name=nucleus \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/nucl_celldiff_iterative.zarr \
    io.pred_channel_name=Nuclei_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
    io.gt_channel_name=Nuclei \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_iterative_nucleus \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true
