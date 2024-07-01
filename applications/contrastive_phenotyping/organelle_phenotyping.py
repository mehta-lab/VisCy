# %% Imports and paths.
import os
import torch
from viscy.light.engine import ContrastiveLearningModel
from viscy.unet.networks.unext2 import UNeXt2Stem
from pathlib import Path
import torchview

top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
input_zarr = top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/patch_final.zarr"
model_dir = top_dir / "infection_classification/models/infection_score"

# %% Initialize the model and log the graph.

# %% Initialize the data module and view the data.

# %% Train the model.


# %% Playground
import timm

available_models = timm.list_models(pretrained=True)
encoder = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    features_only=False,
    drop_path_rate=0.2,
)

encoder.stem[0].Conv
encoder_graph = torchview.draw_graph(
    encoder,
    torch.randn(1, 3, 512, 512),
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
encoder_graph.visual_graph

# %%
