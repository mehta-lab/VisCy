ml uv

source .envrc

# This script runs the evaluation of the CELL-Diff predictions for the TOMM20 structure in the iPSC dataset.
# Evaluate the CELL-Diff denoise predictions
uv run dynacell evaluate \
    target_name=mitochondria \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/tomm20_celldiff_denoise.zarr \
    io.pred_channel_name=Structure_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20.zarr \
    io.gt_channel_name=Structure \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_denoise_tomm20 \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true \

# Evaluate the CELL-Diff sliding window predictions
uv run dynacell evaluate \
    target_name=mitochondria \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/tomm20_celldiff_sliding_window.zarr \
    io.pred_channel_name=Structure_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20.zarr \
    io.gt_channel_name=Structure \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_sliding_window_tomm20 \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true \

# Evaluate the CELL-Diff iterative predictions
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
    force_recompute.all=true \

# This script runs the evaluation of the UNetVit3D predictions for the TOMM20 structure in the iPSC dataset.
uv run dynacell evaluate \
    target_name=mitochondria \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/tomm20_unetvit3d.zarr \
    io.pred_channel_name=Structure_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20.zarr \
    io.gt_channel_name=Structure \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/TOMM20_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_unetvit3d_tomm20 \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
    force_recompute.all=true \