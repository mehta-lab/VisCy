ml uv

source "$(dirname "$0")/.envrc"

# This script runs the evaluation of the CELL-Diff predictions for the Membrane in the iPSC dataset.
# # Evaluate the CELL-Diff denoise predictions
# dynacell evaluate \
#     target_name=membrane \
#     io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/memb_celldiff_denoise.zarr \
#     io.pred_channel_name=Membrane_prediction \
#     io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
#     io.gt_channel_name=Membrane \
#     io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
#     pixel_metrics.spacing=[0.29,0.108,0.108] \
#     save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_denoise_membrane \
#     compute_feature_metrics=true \
#     "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \

# # Evaluate the CELL-Diff sliding window predictions
# dynacell evaluate \
#     target_name=membrane \
#     io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/memb_celldiff_sliding_window.zarr \
#     io.pred_channel_name=Membrane_prediction \
#     io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
#     io.gt_channel_name=Membrane \
#     io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
#     pixel_metrics.spacing=[0.29,0.108,0.108] \
#     save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_sliding_window_membrane \
#     compute_feature_metrics=true \
#     "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \

# # Evaluate the CELL-Diff iterative predictions
# dynacell evaluate \
#     target_name=membrane \
#     io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/memb_celldiff_iterative.zarr \
#     io.pred_channel_name=Membrane_prediction \
#     io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
#     io.gt_channel_name=Membrane \
#     io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
#     pixel_metrics.spacing=[0.29,0.108,0.108] \
#     save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_celldiff_iterative_membrane \
#     compute_feature_metrics=true \
#     "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \

# This script runs the evaluation of the UNetVit3D predictions for the Membrane in the iPSC dataset.
dynacell evaluate \
    target_name=membrane \
    io.pred_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/predictions/memb_unetvit3d.zarr \
    io.pred_channel_name=Membrane_prediction \
    io.gt_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell.zarr \
    io.gt_channel_name=Membrane \
    io.cell_segmentation_path=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/test_cropped/cell_segmented_cleaned.zarr \
    pixel_metrics.spacing=[0.29,0.108,0.108] \
    save.save_dir=/hpc/projects/virtual_staining/training/dynacell/ipsc/evaluations/eval_unetvit3d_membrane \
    compute_feature_metrics=true \
    "feature_extractor.dynaclr.checkpoint='/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/epoch=104-step=53760.ckpt'" \
