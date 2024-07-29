# %% [markdown]
"""
# 2D Virtual Staining of A549 Cells
---
## Prediction using the VSCyto2D to predict nuclei and plasma membrane from phase.
This example shows how to virtually stain A549 cells using the _VSCyto2D_ model.
The model is trained to predict the membrane and nuclei channels from the phase channel.
"""
# %% Imports and paths
from pathlib import Path

from iohub import open_ome_zarr
from plot import plot_vs_n_fluor

# Viscy classes for the trainer and model
from viscy.data.hcs import HCSDataModule
from viscy.light.engine import FcmaeUNet
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.light.trainer import VSTrainer
from viscy.transforms import NormalizeSampled

# %% [markdown]
"""
## Data and Model Paths

The dataset and model checkpoint files need to be downloaded before running this example.
"""

# %%
# Set download paths
root_dir = Path("")
# Download from
# https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/
input_data_path = (
    root_dir / "VSCyto2D/test/a549_hoechst_cellmask_test.zarr"
)
# Download from GitHub release page of v0.1.0
model_ckpt_path = (
    root_dir / "VisCy-0.1.0-VS-models/VSCyto2D/epoch=399-step=23200.ckpt"
)
# Zarr store to save the predictions
output_path = root_dir / "./a549_prediction.zarr"
# FOV of interest
fov = "0/0/0"

input_data_path = input_data_path / fov

# %%
# Create the VSCyto2D network

# Reduce the batch size if encountering out-of-memory errors
BATCH_SIZE = 8
# NOTE: Set the number of workers to 0 for Windows and macOS
# since multiprocessing only works with a
# `if __name__ == '__main__':` guard.
# On Linux, set it to the number of CPU cores to maximize performance.
NUM_WORKERS = 0
phase_channel_name = "Phase3D"

# %%[markdown]
"""
For this example we will use the following parameters:
For more information on the VSCyto2D model,
see ``viscy.unet.networks.fcmae``
([source code](https://github.com/mehta-lab/VisCy/blob/6a3457ec8f43ecdc51b1760092f1a678ed73244d/viscy/unet/networks/fcmae.py#L398))
for configuration details.
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
    num_workers=NUM_WORKERS,
    architecture="fcmae",
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
# Show the individual channels and the fused in a 1x3 plot
output_path = Path(output_path) / fov

# %%
# Open the predicted data
vs_store = open_ome_zarr(output_path, mode="r")
# Get the 2D images
vs_nucleus = vs_store[0][0, 0, 0]  # (t,c,z,y,x)
vs_membrane = vs_store[0][0, 1, 0]  # (t,c,z,y,x)
# Open the experimental fluorescence
fluor_store = open_ome_zarr(input_data_path, mode="r")
# Get the 2D images
# NOTE: Channel indeces hardcoded for this dataset
fluor_nucleus = fluor_store[0][0, 1, 0]  # (t,c,z,y,x)
fluor_membrane = fluor_store[0][0, 2, 0]  # (t,c,z,y,x)

# Plot
plot_vs_n_fluor(vs_nucleus, vs_membrane, fluor_nucleus, fluor_membrane)

vs_store.close()
fluor_store.close()
