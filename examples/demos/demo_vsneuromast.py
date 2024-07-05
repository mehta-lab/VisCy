# %% [markdown]
"""
# 3D Virtual Staining of Neuromast
---
## Prediction using the VSNeuromast to predict nuclei and membrane from phase.
This example shows how to virtually stain zebrafish neuromast cells using the _VSNeuromast_ model.
The model is trained to predict the membrane and nuclei channels from the phase channel.
"""
# %% Imports and paths
from pathlib import Path

from iohub import open_ome_zarr
from plot import plot_vs_n_fluor

from viscy.data.hcs import HCSDataModule

# Viscy classes for the trainer and model
from viscy.light.engine import VSUNet
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled

# %% [markdown]
"""
## Data and Model Paths

The dataset and model checkpoint files need to be downloaded before running this example.
"""

# %%
# Download from
# https://public.czbiohub.org/comp.micro/viscy/VSNeuromast/test/20230803_fish2_60x_1_cropped_zyx_resampled_clipped_2.zarr/
input_data_path = "VSNeuromast/test/20230803_fish2_60x_1_cropped_zyx_resampled_clipped_2.zarr"
# Download from GitHub release page of v0.1.0
model_ckpt_path = "VisCy-0.1.0-VS-models/VSNeuromast/timelapse_finetine_1hr_dT_downsample_lr1e-4_45epoch_clahe_v5/epoch=44-step=1215.ckpt"
# Zarr store to save the predictions
output_path = "./test_neuromast_demo.zarr"
# FOV of interest
fov = "0/3/0"

input_data_path = Path(input_data_path) / fov
# %%
# Create the VSNeuromast model

# Reduce the batch size if encountering out-of-memory errors
BATCH_SIZE = 2
# NOTE: Set the number of workers to 0 for Windows and macOS
# since multiprocessing only works with a
# `if __name__ == '__main__':` guard.
# On Linux, set it to the number of CPU cores to maximize performance.
NUM_WORKERS = 0
phase_channel_name = "Phase3D"

# %%[markdown]
"""
For this example we will use the following parameters:
### For more information on the VSNeuromast model:
See ``viscy.unet.networks.fcmae`` ([source code](https://github.com/mehta-lab/VisCy/blob/6a3457ec8f43ecdc51b1760092f1a678ed73244d/viscy/unet/networks/unext2.py#L252)) for configuration details.
"""
# %%
# Setup the data module.
data_module = HCSDataModule(
    data_path=input_data_path,
    source_channel=phase_channel_name,
    target_channel=["Membrane", "Nuclei"],
    z_window_size=21,
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    architecture="UNeXt2",
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
config_VSNeuromast = {
    "in_channels": 1,
    "out_channels": 2,
    "in_stack_depth": 21,
    "backbone": "convnextv2_tiny",
    "stem_kernel_size": (7, 4, 4),
    "decoder_mode": "pixelshuffle",
    "head_expansion_ratio": 4,
    "head_pool": True,
}

model_VSNeuromast = VSUNet.load_from_checkpoint(
    model_ckpt_path, architecture="UNeXt2", model_config=config_VSNeuromast
)
model_VSNeuromast.eval()

# %%
# Setup the Trainer
trainer = VSTrainer(
    accelerator="gpu",
    callbacks=[HCSPredictionWriter(output_path)],
)

# Start the predictions
trainer.predict(
    model=model_VSNeuromast,
    datamodule=data_module,
    return_predictions=False,
)

# %%
# Open the output_zarr store and inspect the output
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

# Open the experimental fluorescence
fluor_store = open_ome_zarr(input_data_path, mode="r")
# Get the 2D images
# NOTE: Channel indeces hardcoded for this dataset
fluor_nucleus = fluor_store[0][0, 1, z_slice]  # (t,c,z,y,x)
fluor_membrane = fluor_store[0][0, 2, z_slice]  # (t,c,z,y,x)

# Plot
plot_vs_n_fluor(vs_nucleus, vs_membrane, fluor_nucleus, fluor_membrane)

vs_store.close()
fluor_store.close()
