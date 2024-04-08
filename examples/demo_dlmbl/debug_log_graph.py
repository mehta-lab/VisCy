
# %% 
# %% Imports and paths

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchview
import torchvision
from iohub import open_ome_zarr
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

# pytorch lightning wrapper for Tensorboard.
from tensorboard import notebook  # for viewing tensorboard in notebook
from torch.utils.tensorboard import SummaryWriter  # for logging to tensorboard

# HCSDataModule makes it easy to load data during training.
from viscy.data.hcs import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import VSUNet
from viscy.light.trainer import VSTrainer

seed_everything(42, workers=True)

# Paths to data and log directory
data_path = Path(
    Path("~/data/04_image_translation/HEK_nuclei_membrane_pyramid.zarr/")
).expanduser()

log_dir = Path("~/data/04_image_translation/logs/").expanduser()

# Create log directory if needed, and launch tensorboard
log_dir.mkdir(parents=True, exist_ok=True)

# fmt: off
%reload_ext tensorboard
%tensorboard --logdir {log_dir} --port 6007 --bind_all
# fmt: on

# %%  The entire training loop is contained in this cell.

GPU_ID = 0
BATCH_SIZE = 10
YX_PATCH_SIZE = (512, 512)


# Dictionary that specifies key parameters of the model.
phase2fluor_config = {
    "architecture": "2D",
    "num_filters": [24, 48, 96, 192, 384],
    "in_channels": 1,
    "out_channels": 2,
    "residual": True,
    "dropout": 0.1,  # dropout randomly turns off weights to avoid overfitting of the model to data.
    "task": "reg",  # reg = regression task.
}

phase2fluor_model = VSUNet(
    model_config=phase2fluor_config.copy(),
    batch_size=BATCH_SIZE,
    loss_function=torch.nn.functional.l1_loss,
    schedule="WarmupCosine",
    log_num_samples=10,  # Number of samples from each batch to log to tensorboard.
    example_input_yx_shape=YX_PATCH_SIZE,
)

# Reinitialize the data module.
phase2fluor_data = HCSDataModule(
    data_path,
    source_channel="Phase",
    target_channel=["Nuclei", "Membrane"],
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=YX_PATCH_SIZE,
    augmentations=None,
)
phase2fluor_data.setup("fit")


# Train for 3 epochs to see if you can log graph.
trainer = VSTrainer(accelerator="gpu", devices=[GPU_ID], max_epochs=3, default_root_dir=log_dir)

# trainer class takes the model and the data module as inputs.
trainer.fit(phase2fluor_model, datamodule=phase2fluor_data)

# %% Is exmple_input_array present?
print(f'{phase2fluor_model.example_input_array.shape},{phase2fluor_model.example_input_array.dtype}')
trainer.logger.log_graph(phase2fluor_model, phase2fluor_model.example_input_array)
# %%
