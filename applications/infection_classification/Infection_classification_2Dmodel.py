# %%
import lightning.pytorch as pl
import torch
import torch.nn as nn
from applications.infection_classification.classify_infection_2D import (
    SemanticSegUNet2D,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from viscy.data.hcs import HCSDataModule
from viscy.preprocessing.pixel_ratio import sematic_class_weights
from viscy.transforms import (
    NormalizeSampled,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)

# %% calculate the ratio of background, uninfected and infected pixels in the input dataset

# Set the path to the dataset
dataset_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/4-human_annotation/train_data.zarr"

# %% Create an instance of HCSDataModule

data_module = HCSDataModule(
    dataset_path,
    source_channel=["TXR_Density3D", "Phase3D"],
    target_channel=["Inf_mask"],
    yx_patch_size=[128, 128],
    split_ratio=0.7,
    z_window_size=1,
    architecture="2D",
    num_workers=1,
    batch_size=256,
    normalizations=[
        NormalizeSampled(
            keys=["Phase3D", "TXR_Density3D"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
    augmentations=[
        RandWeightedCropd(
            num_samples=16,
            spatial_size=[-1, 128, 128],
            keys=["TXR_Density3D", "Phase3D", "Inf_mask"],
            w_key="Inf_mask",
        ),
        RandScaleIntensityd(
            keys=["TXR_Density3D", "Phase3D"],
            factors=[0.5, 0.5],
            prob=0.5,
        ),
        RandGaussianSmoothd(
            keys=["TXR_Density3D", "Phase3D"],
            prob=0.5,
            sigma_x=[0.5, 1.0],
            sigma_y=[0.5, 1.0],
            sigma_z=[0.5, 1.0],
        ),
    ],
)
pixel_ratio = sematic_class_weights(dataset_path, target_channel="Inf_mask")

# Prepare the data
data_module.prepare_data()

# Setup the data
data_module.setup(stage="fit")

# Create a dataloader
train_dm = data_module.train_dataloader()

val_dm = data_module.val_dataloader()

# %% Set up for training

# define the logger
logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/",
    name="logs",
)

# Pass the logger to the Trainer
trainer = pl.Trainer(
    logger=logger,
    max_epochs=500,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs/",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="/hpc/projects/intracellular_dashboard/viral-sensor/2024_04_25_BJ5a_DENV_TimeCourse/5-infection_classifier/0-model_training/logs/",
    filename="checkpoint_{epoch:02d}",
    save_top_k=-1,
    verbose=True,
    monitor="loss/validate",
    mode="min",
)

# Add the checkpoint callback to the trainer
trainer.callbacks.append(checkpoint_callback)

# Fit the model
model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor(pixel_ratio)),
)

# visualize the model
print(model)

# %% Run training.

trainer.fit(model, data_module)

# %%
