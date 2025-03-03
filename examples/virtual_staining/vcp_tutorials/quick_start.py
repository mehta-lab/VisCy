# %% [markdown]
"""
# Quick Start: 2D Virtual Staining of A549 Cells

**Estimated time to complete:** 15 minutes
"""

# %% [markdown]
"""
# Learning Goals

* Download the VSCyto2D model.
* Predict nuclei and plasma membrane from quantitative phase.
"""

# %% [markdown]
"""
# Prerequisites
Python>=3.10

"""

# %% [markdown]
"""
# Introduction

## Model

The VSCyto2D model is a 2D U-Net that predicts and cell nuclei and plasma membrane
from quantitative label-free images such as quantitative phase, Zernike phase, and brightfield.
It has been trained on A549, HEK293T, and BJ-5ta cells.

## Example Dataset

The example dataset contains quantitative phase and paired nuclei and plasma membrane fluorescence images of A549 cells.
It is stored in OME-Zarr format and can be downloaded from
[here](https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/a549_hoechst_cellmask_test.zarr).
It has pre-computed statistics for normalization, generated using the `viscy preprocess` CLI.

Refer to our [preprint](https://doi.org/10.1101/2024.05.31.596901) for more details
about how the dataset and model were generated.
"""

# %% [markdown]
"""
# Setup

The commands below will install the required packages and download the example dataset and model checkpoint.
It may take a few minutes to download all the files.
Assuming a Unix-like shell.
"""

# %%
# Install VisCy with the optional dependencies for this example
# See the [repository](https://github.com/mehta-lab/VisCy) for more details
# !pip install "viscy[metrics,visual]==0.3.0rc2"

# %%
# Validate installation
# viscy --help

# %%
# Download the example dataset
# !wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"
# Download the model checkpoint
# !wget https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt

# %% [markdown]
"""
# Run Model Inference

The following code will run inference on a single field of view (FOV) of the example dataset.
This can also be achieved by using the VisCy CLI.
"""

# %%
from pathlib import Path

from iohub import open_ome_zarr
from torchview import draw_graph

from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.transforms import NormalizeSampled
from viscy.translation.engine import FcmaeUNet
from viscy.translation.predict_writer import HCSPredictionWriter

# %%
# TODO: Set download paths, by default the working directory is used
root_dir = Path()
# TODO: modify the path to the downloaded dataset
input_data_path = root_dir / "a549_hoechst_cellmask_test.zarr"
# TODO: modify the path to the downloaded checkpoint
model_ckpt_path = root_dir / "epoch=399-step=23200.ckpt"
# TODO: modify the path
# Zarr store to save the predictions
output_path = root_dir / "a549_prediction.zarr"
# TODO: Choose an FOV
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

# %%
# See API documentation for more details. For example:
# ?HCSDataModule

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
model_VSCyto2D = model_VSCyto2D.eval()

# %%
# Visualize the model graph
model_graph = draw_graph(
    model_VSCyto2D,
    (model_VSCyto2D.example_input_array),
    graph_name="VSCyto2D",
    roll=True,
    depth=3,
    expand_nested=True,
)

fcmae = model_graph.visual_graph
fcmae

# %%
# Setup the Trainer
trainer = VisCyTrainer(callbacks=[HCSPredictionWriter(output_path)])

# Start the predictions
trainer.predict(
    model=model_VSCyto2D,
    datamodule=data_module,
    return_predictions=False,
)

# %% [markdown]
"""
# Model Outputs

The model outputs are also stored in an OME-Zarr store.
It can be visualized in an image viewer such as [napari](https://napari.org/).
Below we show a snapshot in the notebook.
"""

# %%
# Open the output_zarr store and inspect the output
# Show the individual channels and the fused in a 1x3 plot

# Open the predicted data
vs_store = open_ome_zarr(output_path / fov, mode="r")
# Get the 2D images
vs_nucleus = vs_store[0][0, 0, 0]  # (t,c,z,y,x)
vs_membrane = vs_store[0][0, 1, 0]  # (t,c,z,y,x)
# Open the experimental fluorescence dataset
fluor_store = open_ome_zarr(input_data_path, mode="r")
# Get the 2D images
# NOTE: Channel indeces hardcoded for this dataset
fluor_nucleus = fluor_store[0][0, 1, 0]  # (t,c,z,y,x)
fluor_membrane = fluor_store[0][0, 2, 0]  # (t,c,z,y,x)

# Plot
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from skimage.exposure import rescale_intensity


def render_rgb(image: np.ndarray, colormap: Colormap):
    image = rescale_intensity(image, out_range=(0, 1))
    image = colormap(image)
    return image


vs_nucleus_rgb = render_rgb(vs_nucleus, Colormap("bop_blue"))
vs_membrane_rgb = render_rgb(vs_membrane, Colormap("bop_orange"))
merged_vs = (vs_nucleus_rgb + vs_membrane_rgb).clip(0, 1)

fluor_nucleus_rgb = render_rgb(fluor_nucleus, Colormap("green"))
fluor_membrane_rgb = render_rgb(fluor_membrane, Colormap("magenta"))
merged_fluor = (fluor_nucleus_rgb + fluor_membrane_rgb).clip(0, 1)

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
plt.tight_layout()
plt.show()


vs_store.close()
fluor_store.close()
