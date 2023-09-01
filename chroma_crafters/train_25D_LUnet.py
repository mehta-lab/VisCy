# %%
import viscy

viscy.__file__

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.light.data import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSTrainer, VSUNet

seed_everything(42, workers=True)


# %%
# Paths to data and log directory
data_path = Path(Path("/home/eduardoh/cropped_dataset_v3.zarr"))
assert data_path.exists()

log_dir = Path("/home/eduardoh/vs_data/training/LUnet/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# %%
GPU_ID = 0
YX_PATCH_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_WORKERS = 8

# Dictionary that specifies key parameters of the model.
phase2fluor_25D_LUnet_config = {
    "num_filters": [24, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 5,
    "stem_kernel_size": (5, 3, 3),
    "decoder_conv_blocks": 2,
    "drop_path_rate": 0.1,
}
phase2fluor_25D_LUnet_model = VSUNet(
    "25D_LUnet",
    model_config=phase2fluor_25D_LUnet_config.copy(),
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=3,  # Number of samples from each batch to log to tensorboard.
    example_input_yx_shape=YX_PATCH_SIZE,
    lr=2e-4,
)
# Reinitialize the data module.
phase2fluor_data = HCSDataModule(
    data_path,
    source_channel="Phase3D",
    target_channel=["GFP EX488 EM525-45", "mCherry EX561 EM600-37"],
    z_window_size=5,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    architecture="25D_LUnet",
    yx_patch_size=YX_PATCH_SIZE,
    normalize_source=True,
    augment=True,
    train_z_scale_range=[-0.2, 0.5],
    train_patches_per_stack=2,
    train_noise_std=2.0,
)
print(phase2fluor_data)
phase2fluor_data.setup("fit")
train_dataloader = phase2fluor_data.train_dataloader()
val_dataloader = phase2fluor_data.val_dataloader()

# %%
N_EPOCH = 50
N_SAMPLES = len(phase2fluor_data.train_dataset)
STEPS_PER_EPOCH = N_SAMPLES // BATCH_SIZE  # steps per epoch.

trainer = VSTrainer(
    accelerator="gpu",
    devices=[GPU_ID],
    max_epochs=N_EPOCH,
    log_every_n_steps=STEPS_PER_EPOCH,
    logger=TensorBoardLogger(
        save_dir=log_dir,
        # lightning trainer transparently saves logs and model checkpoints in this directory.
        name="phase2fluor_25D_LUnet",
        log_graph=True,
    ),
)
# Log graph
trainer.logger.log_graph(
    phase2fluor_25D_LUnet_model, phase2fluor_data.train_dataset[0]["source"]
)
# Launch training.
trainer.fit(phase2fluor_25D_LUnet_model, datamodule=phase2fluor_data)

# %%
