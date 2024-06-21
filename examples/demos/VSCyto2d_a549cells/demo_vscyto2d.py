# %% [markdown]
"""
# 2D Virtual Staining of A549 Cells
---
## Prediction using the VSCyto2D to predict nuclei and membrane from phase.
This example shows how to virtually stain A549 cells using the _VSCyto2D_ model.
The model is trained to predict the membrane and nuclei channels from the phase channel.
"""
# %% Imports and paths
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from iohub import open_ome_zarr

from viscy.data.hcs import HCSDataModule

# %% Imports and paths
# Viscy classes for the trainer and model
from viscy.light.engine import FcmaeUNet
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled
from skimage.exposure import rescale_intensity

# %% [markdown]

# %%
input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/test/cell_types_20x/a549_sliced/a549_hoechst_cellmask_test.zarr"
model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/models/hek-a549-bj5a-20x/lightning_logs/tiny-2x2-finetune-e2e-amp-hek-a549-bj5a-nucleus-membrane-400ep/checkpoints/last.ckpt"
output_path = "./test_a549_demo.zarr"
fov = "0/0/0"  # NOTE: FOV of interest

input_data_path = Path(input_data_path) / fov
# %%
# Create a the VSCyto2D

# NOTE: Change the following parameters as needed.
GPU_ID = 0
BATCH_SIZE = 10
YX_PATCH_SIZE = (384, 384)
phase_channel_name = "Phase3D"

# %%[markdown]
"""
For this example we will use the following parameters:
### For more information on the VSCyto2D model:
See ``viscy.unet.networks.fcmae`` ([source code](https://github.com/mehta-lab/VisCy/blob/6a3457ec8f43ecdc51b1760092f1a678ed73244d/viscy/unet/networks/fcmae.py#L398)) for configuration details.
"""
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
# Setup the Trainer
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
# Open the output_zarr store and inspect the output
colormap_1 = [0.1254902, 0.6784314, 0.972549]  # bop blue
colormap_2 = [0.972549, 0.6784314, 0.1254902]  # bop orange

# Show the individual channels and the fused in a 1x3 plot
output_path = Path(output_path) / fov
# %%

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
with open_ome_zarr(output_path, mode="r") as store:

    # Get the 2D images
    vs_nucleus = store[0][0, 0, 0]  # (t,c,z,y,x)
    vs_membrane = store[0][0, 1, 0]  # (t,c,z,y,x)
    # Rescale the intensity
    vs_nucleus = rescale_intensity(vs_nucleus, out_range=(0, 1))
    vs_membrane = rescale_intensity(vs_membrane, out_range=(0, 1))
    # VS Nucleus RGB
    vs_nucleus_rgb = np.zeros((*store.data.shape[-2:], 3))
    vs_nucleus_rgb[:, :, 0] = vs_nucleus * colormap_1[0]
    vs_nucleus_rgb[:, :, 1] = vs_nucleus * colormap_1[1]
    vs_nucleus_rgb[:, :, 2] = vs_nucleus * colormap_1[2]
    # VS Membrane RGB
    vs_membrane_rgb = np.zeros((*store.data.shape[-2:], 3))
    vs_membrane_rgb[:, :, 0] = vs_membrane * colormap_2[0]
    vs_membrane_rgb[:, :, 1] = vs_membrane * colormap_2[1]
    vs_membrane_rgb[:, :, 2] = vs_membrane * colormap_2[2]
    # Merge the two channels
    merged_image = np.zeros((*store.data.shape[-2:], 3))
    merged_image[:, :, 0] = vs_nucleus * colormap_1[0] + vs_membrane * colormap_2[0]
    merged_image[:, :, 1] = vs_nucleus * colormap_1[1] + vs_membrane * colormap_2[1]
    merged_image[:, :, 2] = vs_nucleus * colormap_1[2] + vs_membrane * colormap_2[2]

    # Plot
    ax[0].imshow(vs_nucleus_rgb)
    ax[0].set_title("VS Nucleus")
    ax[1].imshow(vs_membrane_rgb)
    ax[1].set_title("VS Membrane")
    ax[2].imshow(merged_image)
    ax[2].set_title("VS Nucleus+Membrane")
    for a in ax:
        a.axis("off")
    plt.margins(0, 0)
    plt.show()
# %%
