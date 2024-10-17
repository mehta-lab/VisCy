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

from iohub import open_ome_zarr
from plot import plot_vs_n_fluor
from viscy.data.hcs import HCSDataModule

# Viscy classes for the trainer and model
from viscy.translation.engine import VSUNet
from viscy.translation.predict_writer import HCSPredictionWriter
from viscy.trainer import VisCyTrainer
from viscy.transforms import NormalizeSampled

# %% [markdown]
"""
## Data and Model Paths

The dataset and model checkpoint files need to be downloaded before running this example.
"""

# %% [markdown] tags=[]
#
# <div class="alert alert-block alert-info">
#
# # Download the dataset and checkpoints VSCyto3D
#
# - Download the VSCyto3D test dataset and model checkpoint from here: <br>
# https://public.czbiohub.org/comp.micro/viscy
# - Update the `input_data_path` and `model_ckpt_path` variables with the path to the downloaded files.
# - Select a FOV (i.e plate/0/0).
# - Set an output path for the predictions.
#
# </div>
# %%
# TODO: modify the path to the downloaded dataset
input_data_path = "/no_pertubation_Phase1e-3_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50.zarr"

# TODO: modify the path to the downloaded checkpoint
model_ckpt_path = "/epoch=48-step=18130.ckpt"

# TODO: modify the path
# Zarr store to save the predictions
output_path = "./hek_prediction_3d.zarr"

# TODO: Choose an FOV
# FOV of interest
fov = "plate/0/0"

input_data_path = Path(input_data_path) / fov

# %%
# Create the VSCyto3D model

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
### For more information on the VSCyto3D model:
See ``viscy.unet.networks.unext2``
([source code](https://github.com/mehta-lab/VisCy/blob/main/viscy/unet/networks/unext2.py))
for configuration details.
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
trainer = VisCyTrainer(
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
fluor_nucleus = fluor_store[0][0, 2, z_slice]  # (t,c,z,y,x)
fluor_membrane = fluor_store[0][0, 1, z_slice]  # (t,c,z,y,x)

# Plot
plot_vs_n_fluor(vs_nucleus, vs_membrane, fluor_nucleus, fluor_membrane)

# Close stores
vs_store.close()
fluor_store.close()
