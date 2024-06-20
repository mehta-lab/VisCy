# %% [markdown]
"""
# 2D Virtual Staining of A549 Cells
---
This example shows how to virtually stain A549 cells using the _VSCyto2D_ model.

First we import the necessary libraries and set the random seed for reproducibility.
"""
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

# from rich.pretty import pprint #TODO: add pretty print(?)

from napari.utils.notebook_display import nbscreenshot
import napari

# %% Imports and paths
from viscy.data.hcs import HCSDataModule

# Trainer class and UNet.
from viscy.light.engine import FcmaeUNet
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.data.hcs import HCSDataModule

# %% [markdown]
"""
## Prediction using the 2D U-Net model to predict nuclei and membrane from phase.

### Construct a 2D U-Net
See ``viscy.unet.networks.Unet2D.Unet2d`` ([source code](https://github.com/mehta-lab/VisCy/blob/7c5e4c1d68e70163cf514d22c475da8ea7dc3a88/viscy/unet/networks/Unet2D.py#L7)) for configuration details.
"""

# %%
input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/test/cell_types_20x/a549_sliced/a549_hoechst_cellmask_test.zarr/0/0/0"
model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/models/hek-a549-bj5a-20x/lightning_logs/tiny-2x2-finetune-e2e-amp-hek-a549-bj5a-nucleus-membrane-400ep/checkpoints/last.ckpt"
output_path = "./test_a549_demo.zarr"

# %%
# Create a the VSCyto2D

GPU_ID = 0
BATCH_SIZE = 10
YX_PATCH_SIZE = (384, 384)
phase_channel_name = "Phase3D"


# %%
# Setup the data module.
data_module = HCSDataModule(
    data_path=input_data_path,
    source_channel=phase_channel_name,
    target_channel=["Membrane", "Nuclei"],
    z_window_size=1,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=8,
    architecture="2D",
    yx_patch_size=YX_PATCH_SIZE,
    normalizations=[
        NormalizeSampled(
            [phase_channel_name],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)
data_module.prepare_data()
data_module.setup(stage="predict")
# %%
# Setup the model.
# Dictionary that specifies key parameters of the model.
config_VSCyto2D = {
    "in_channels": 1,
    "out_channels": 2,
    "encoder_blocks": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768],
    "decoder_conv_blocks": 2,
    "stem_kernel_size": [1, 2, 2],
    "in_stack_depth": 1,
    "pretraining": False,
}

model_VSCyto2D = FcmaeUNet.load_from_checkpoint(
    model_ckpt_path, model_config=config_VSCyto2D
)
model_VSCyto2D.eval()

# %%
trainer = VSTrainer(
    accelerator="gpu",
    callbacks=[HCSPredictionWriter(output_path)],
)

# Start the predictions
trainer.predict(
    model=model_VSCyto2D,
    datamodule=data_module,
    return_predictions=False,
)

# %%
