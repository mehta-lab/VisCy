# %% [markdown]
"""
# 3D Virtual Staining of HEK293T Cells
---
## Prediction using the VSCyto3D to predict nuclei and membrane from phase.
This example shows how to virtually stain HEK293T cells using the _VSCyto3D_ model.
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
from viscy.light.engine import VSUNet
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled
from skimage.exposure import rescale_intensity

# %% [markdown]

# %%
# TODO: change paths to respective locations
input_data_path = "/hpc/projects/comp.micro/virtual_staining/datasets/test/2022_04_19_HEK_ImagingVariations_torch/no_pertubation_Phase1e-3_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50.zarr"
model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/models/viscy-0.1.0/VSCyto3D/epoch=48-step=18130.ckpt"
output_path = "./test_hek3d_demo.zarr"
fov = "plate/0/0"  # NOTE: FOV of interest

input_data_path = Path(input_data_path) / fov
# %%
# Create a the VSCyto3D

# NOTE: Change the following parameters as needed.
GPU_ID = 0
BATCH_SIZE = 2
YX_PATCH_SIZE = (384, 384)
NUM_WORKERS = 8
phase_channel_name = "Phase3D"

# %%[markdown]
"""
For this example we will use the following parameters:
### For more information on the VSCyto3D model:
See ``viscy.unet.networks.fcmae`` ([source code](https://github.com/mehta-lab/VisCy/blob/6a3457ec8f43ecdc51b1760092f1a678ed73244d/viscy/unet/networks/unext2.py#L252)) for configuration details.
"""
# %%
# Setup the data module.
data_module = HCSDataModule(
    data_path=input_data_path,
    source_channel=phase_channel_name,
    target_channel=["Membrane", "Nuclei"],
    z_window_size=5,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    architecture="UNeXt2",
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
config_VSCyto3D = {
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 5,
    "backbone": "convnextv2_tiny",
    "stem_kernel_size": (5, 4, 4),
    "decoder_mode": "pixelshuffle",
    "head_expansion_ratio": 4,
    "head_pool": True,
}

model_VSCyto3D = VSUNet.load_from_checkpoint(
    model_ckpt_path, architecture="UNeXt2", model_config=config_VSCyto3D
)
model_VSCyto3D.eval()

# %%
# Setup the Trainer
trainer = VSTrainer(
    accelerator="gpu",
    callbacks=[HCSPredictionWriter(output_path)],
)

# Start the predictions
trainer.predict(
    model=model_VSCyto3D,
    datamodule=data_module,
    return_predictions=False,
)

# %%
# Open the output_zarr store and inspect the output
colormap_1 = [0.1254902, 0.6784314, 0.972549]  # bop blue
colormap_2 = [0.972549, 0.6784314, 0.1254902]  # bop orange
colormap_3 = [0, 1, 0]  # green
colormap_4 = [1, 0, 1]  # magenta
# Show the individual channels and the fused in a 1x3 plot
output_path = Path(output_path) / fov
# %%
# Open the predicted data
vs_store = open_ome_zarr(output_path, mode="r")
T, C, Z, Y, X = vs_store.data.shape

# Get a z-slice
z_slice = Z // 2  # NOTE: using the middle slice of the stack. Change as needed.
vs_nucleus = vs_store[0][0, 0, z_slice]  # (t,c,z,y,x)
vs_membrane = vs_store[0][0, 1, z_slice]  # (t,c,z,y,x)
# Rescale the intensity
vs_nucleus = rescale_intensity(vs_nucleus, out_range=(0, 1))
vs_membrane = rescale_intensity(vs_membrane, out_range=(0, 1))
# VS Nucleus RGB
vs_nucleus_rgb = np.zeros((*vs_store.data.shape[-2:], 3))
vs_nucleus_rgb[:, :, 0] = vs_nucleus * colormap_1[0]
vs_nucleus_rgb[:, :, 1] = vs_nucleus * colormap_1[1]
vs_nucleus_rgb[:, :, 2] = vs_nucleus * colormap_1[2]
# VS Membrane RGB
vs_membrane_rgb = np.zeros((*vs_store.data.shape[-2:], 3))
vs_membrane_rgb[:, :, 0] = vs_membrane * colormap_2[0]
vs_membrane_rgb[:, :, 1] = vs_membrane * colormap_2[1]
vs_membrane_rgb[:, :, 2] = vs_membrane * colormap_2[2]
# Merge the two channels
merged_vs = np.zeros((*vs_store.data.shape[-2:], 3))
merged_vs[:, :, 0] = vs_nucleus * colormap_1[0] + vs_membrane * colormap_2[0]
merged_vs[:, :, 1] = vs_nucleus * colormap_1[1] + vs_membrane * colormap_2[1]
merged_vs[:, :, 2] = vs_nucleus * colormap_1[2] + vs_membrane * colormap_2[2]

# Open the experimental fluorescence
fluor_store = open_ome_zarr(input_data_path, mode="r")
# Get the 2D images
# NOTE: Channel indeces hardcoded for this dataset
fluor_nucleus = fluor_store[0][0, 2, z_slice]  # (t,c,z,y,x)
fluor_membrane = fluor_store[0][0, 1, z_slice]  # (t,c,z,y,x)
# Rescale the intensity
fluor_nucleus = rescale_intensity(fluor_nucleus, out_range=(0, 1))
fluor_membrane = rescale_intensity(fluor_membrane, out_range=(0, 1))
# fluor Nucleus RGB
fluor_nucleus_rgb = np.zeros((*fluor_store.data.shape[-2:], 3))
fluor_nucleus_rgb[:, :, 0] = fluor_nucleus * colormap_3[0]
fluor_nucleus_rgb[:, :, 1] = fluor_nucleus * colormap_3[1]
fluor_nucleus_rgb[:, :, 2] = fluor_nucleus * colormap_3[2]
# fluor Membrane RGB
fluor_membrane_rgb = np.zeros((*fluor_store.data.shape[-2:], 3))
fluor_membrane_rgb[:, :, 0] = fluor_membrane * colormap_4[0]
fluor_membrane_rgb[:, :, 1] = fluor_membrane * colormap_4[1]
fluor_membrane_rgb[:, :, 2] = fluor_membrane * colormap_4[2]
# Merge the two channels
merged_fluor = np.zeros((*fluor_store.data.shape[-2:], 3))
merged_fluor[:, :, 0] = fluor_nucleus * colormap_3[0] + fluor_membrane * colormap_4[0]
merged_fluor[:, :, 1] = fluor_nucleus * colormap_3[1] + fluor_membrane * colormap_4[1]
merged_fluor[:, :, 2] = fluor_nucleus * colormap_3[2] + fluor_membrane * colormap_4[2]

# %%
# Plot
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Virtual staining plots
ax[0, 0].imshow(vs_nucleus_rgb)
ax[0, 0].set_title("VS Nuclei")
ax[0, 1].imshow(vs_membrane_rgb)
ax[0, 1].set_title("VS Membrane")
ax[0, 2].imshow(merged_vs)
ax[0, 2].set_title("VS Nuclei+Membrane")

# Experimental fluorescence plots
ax[1, 0].imshow(fluor_nucleus_rgb)
ax[1, 0].set_title("Experimental Fluorescence Nuclei")
ax[1, 1].imshow(fluor_membrane_rgb)
ax[1, 1].set_title("Experimental Fluorescence Membrane")
ax[1, 2].imshow(merged_fluor)
ax[1, 2].set_title("Experimental Fluorescence Nuclei+Membrane")

# turnoff axis
for a in ax.flatten():
    a.axis("off")
plt.margins(0, 0)
plt.show()

vs_store.close()
fluor_store.close()

# %%
