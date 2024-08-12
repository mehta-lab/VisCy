from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from viscy.data.triplet import TripletDataModule, TripletDataset
from viscy.light.engine import ContrastiveModule
import os 
from torch.multiprocessing import Manager
from viscy.transforms import (
    NormalizeSampled,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)
from monai.transforms import NormalizeIntensityd, ScaleIntensityRangePercentilesd

# Updated normalizations
normalizations = [
    NormalizeIntensityd(
        keys=["Phase3D"],
        subtrahend=None,
        divisor=None,
        nonzero=False,  
        channel_wise=False,  
        dtype=None,  
        allow_missing_keys=False  
    ),
    ScaleIntensityRangePercentilesd(
        keys=["MultiCam_GFP_mCherry_BF-Prime BSI Express"],
        lower=50,  
        upper=99,  
        b_min=0.0,
        b_max=1.0,
        clip=False,  
        relative=False, 
        channel_wise=False,  
        dtype=None, 
        allow_missing_keys=False  
    ),
]

def main(hparams):
    # Set paths
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/expanded_final_track_timesteps.csv
    # /hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/uninfected_cells.csv
    # /hpc/mydata/alishba.imran/VisCy/viscy/applications/contrastive_phenotyping/expanded_transitioning_cells_metadata.csv
    checkpoint_path = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/test_tb/lightning_logs/copy/contrastive_model-test-epoch=24-val_loss=0.00.ckpt"
    
    # non-rechunked data 
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/5-infection_pred/infection_score.zarr
    # /hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2.1-register/registered.zarr
    data_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/5-infection_pred/infection_score.zarr"

    # updated tracking data
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.1-tracking/test_tracking_4.zarr 
    # /hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr
    tracks_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.1-tracking/test_tracking_4.zarr"
    
    # MultiCam_GFP_mCherry_BF-Prime BSI Express for June dataset
    source_channel = ["MultiCam_GFP_mCherry_BF-Prime BSI Express", "Phase3D"]
    z_range = (28, 43)
    batch_size = 1 
   
    # infected cells - JUNE
    # include_fov_names = ['/0/8/001001', '/0/8/001001', '/0/8/000001', '/0/6/002002', '/0/6/002002', '/0/6/00200']
    # include_track_ids = [31, 8, 21, 4, 2, 21]

    # # uninfected cells - JUNE
    # include_fov_names = ['/0/1/000000', '/0/1/000000', '/0/1/000000', '/0/1/000000', '/0/8/000002', '/0/8/000002']
    # include_track_ids = [25, 36, 37, 48, 16, 17]

    # # dividing cells - JUNE
    # include_fov_names = ['/0/1/000000', '/0/1/000000', '/0/1/000000']
    # include_track_ids = [18, 21, 50]

    # uninfected cells - FEB
    # include_fov_names = ['/A/3/0', 'B/3/5', 'B/3/5', 'B/3/5', 'B/3/5', '/A/4/14', '/A/4/14']
    # include_track_ids = [15, 34, 32, 31, 26, 33, 30]

    # # infected cells - FEB
    # include_fov_names = ['/A/4/13', '/A/4/14', '/B/4/4', '/B/4/5', '/B/4/6', '/B/4/6']
    # include_track_ids = [25, 19, 68, 11, 29, 35]

    # # dividing cells - FEB
    # include_fov_names = ['/B/4/4', '/B/3/5']
    # include_track_ids = [71, 42]

    # Initialize the data module for prediction
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        source_channel=source_channel,
        z_range=z_range,
        initial_yx_patch_size=(224, 224),
        final_yx_patch_size=(224, 224),
        batch_size=batch_size,
        num_workers=hparams.num_workers,
        normalizations=normalizations,
        # predict_cells = True,
        # include_fov_names=include_fov_names,
        # include_track_ids=include_track_ids,
    )

    data_module.setup(stage="predict")

    print(f"Total prediction dataset size: {len(data_module.predict_dataset)}")
    
    # Load the model from checkpoint
    backbone = "convnext_tiny"
    in_stack_depth = 15
    stem_kernel_size = (5, 3, 3)
    model = ContrastiveModule.load_from_checkpoint(
    str(checkpoint_path), 
    predict=True, 
    backbone=backbone,
    in_channels=len(source_channel),
    in_stack_depth=in_stack_depth,
    stem_kernel_size=stem_kernel_size,
    tracks_path = tracks_path,
    )
    
    model.eval()

    # Initialize the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[TQDMProgressBar(refresh_rate=1)],
    )

    # Run prediction
    trainer.predict(model, datamodule=data_module)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", type=str, default="Constant")
    parser.add_argument("--log_steps_per_epoch", type=int, default=10)
    parser.add_argument("--embedding_len", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)