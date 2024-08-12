# %% Imports and paths.
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from viscy.transforms import (
    NormalizeSampled,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)

from viscy.data.triplet import TripletDataModule, TripletDataset
from viscy.light.engine import ContrastiveModule
from viscy.representation.contrastive import ContrastiveEncoder
import pandas as pd
from pathlib import Path
from monai.transforms import NormalizeIntensityd, ScaleIntensityRangePercentilesd
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import LearningRateMonitor


top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
model_dir = top_dir / "infection_classification/models/infection_score"

# Data parameters
# 15 for covnext backbone, 12 for resnet (z slices)
# (28, 43) for covnext backbone, (26, 38) for resnet

# rechunked data 
data_path = "/hpc/projects/virtual_staining/2024_02_04_A549_DENV_ZIKV_timelapse/registered_chunked.zarr"

# updated tracking data
tracks_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/7.1-seg_track/tracking_v1.zarr"
source_channel = ["RFP", "Phase3D"]
z_range = (28, 43)
batch_size = 64

# normalizations = [
#             # Normalization for Phase3D using mean and std
#             NormalizeSampled(
#                 keys=["Phase3D"],
#                 level="fov_statistics",
#                 subtrahend="mean",
#                 divisor="std",
#             ),
#             # Normalization for RFP using median and IQR
#             NormalizeSampled(
#                 keys=["RFP"],
#                 level="fov_statistics",
#                 subtrahend="median",
#                 divisor="iqr",
#             ),
# ]

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
        keys=["RFP"],
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

augmentations = [
            # Apply rotations and scaling together to both channels
            RandAffined(
                keys=source_channel,
                rotate_range=[3.14, 0.0, 0.0], 
                scale_range=[0.0, 0.2, 0.2],  
                prob=0.8,
                padding_mode="zeros",
                shear_range=[0.0, 0.01, 0.01],
            ),
            # Apply contrast adjustment separately for each channel
            RandAdjustContrastd(keys=["RFP"], prob=0.5, gamma=(0.7, 1.3)),  # Broader range for RFP
            RandAdjustContrastd(keys=["Phase3D"], prob=0.5, gamma=(0.8, 1.2)),  # Moderate range for Phase
            # Apply intensity scaling separately for each channel
            RandScaleIntensityd(keys=["RFP"], factors=0.7, prob=0.5),  # Broader scaling for RFP
            RandScaleIntensityd(keys=["Phase3D"], factors=0.5, prob=0.5),  # Moderate scaling for Phase
            # Apply Gaussian smoothing to both channels together
            RandGaussianSmoothd(
                keys=source_channel,
                sigma_x=(0.25, 0.75),
                sigma_y=(0.25, 0.75),
                sigma_z=(0.0, 0.0),
                prob=0.5,
            ),
            # Apply Gaussian noise separately for each channel
            RandGaussianNoised(keys=["RFP"], prob=0.5, mean=0.0, std=0.5),  # Higher noise for RFP
            RandGaussianNoised(keys=["Phase3D"], prob=0.5, mean=0.0, std=0.2),  # Moderate noise for Phase
        ]

torch.set_float32_matmul_precision("medium")


# %% Define the main function for training
def main(hparams):
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    print("Starting data module..")
    # Initialize the data module
    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        source_channel=source_channel,
        z_range=z_range,
        initial_yx_patch_size=(512, 512),
        final_yx_patch_size=(224, 224),
        batch_size=batch_size,
        num_workers=hparams.num_workers,
        normalizations=normalizations,
        augmentations=augmentations,
    )

    print("data module set up!")

    # Setup the data module for training, val and testing
    data_module.setup(stage="fit")

    print(
        f"Total dataset size: {len(data_module.train_dataset) + len(data_module.val_dataset)}"
    )
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")

    # Initialize the model
    model = ContrastiveModule(
        backbone=hparams.backbone,
        loss_function=torch.nn.TripletMarginLoss(),
        margin=hparams.margin,
        lr=hparams.lr,
        schedule=hparams.schedule,
        log_batches_per_epoch=1, # total 2 images per epoch are logged
        log_samples_per_batch=2,
        in_channels=len(source_channel),
        in_stack_depth=z_range[1] - z_range[0],
        stem_kernel_size=(5, 4, 4),
        embedding_len=hparams.embedding_len,
    )
    print("Model initialized!")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        # limit_train_batches=2,
        # limit_val_batches=2,
        callbacks=[ModelCheckpoint(), lr_monitor],
        logger=TensorBoardLogger(
            "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/test_tb",
            log_graph=True,
            default_hp_metric=True,
        ),
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        num_nodes=hparams.num_nodes,
        strategy=DDPStrategy(),
        log_every_n_steps=hparams.log_every_n_steps,
        num_sanity_val_steps=0,
    )


    print("Trainer initialized!")

    trainer.fit(model, datamodule=data_module)

    # # Validate the model
    trainer.validate(model, datamodule=data_module)

# Argument parser for command-line options
# to-do: need to clean up to always use the same args
parser = ArgumentParser()
parser.add_argument("--backbone", type=str, default="convnext_tiny")
parser.add_argument("--margin", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--schedule", type=str, default="CosineAnnealingWarmRestarts")
parser.add_argument("--log_steps_per_epoch", type=int, default=10)
parser.add_argument("--embedding_len", type=int, default=256)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)  # 4 GPUs
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--log_every_n_steps", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=15)
args = parser.parse_args()

main(args)

