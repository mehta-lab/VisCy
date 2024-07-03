# %% Imports and paths.
import os
import torch
from viscy.light.engine import ContrastiveLearningModel
from viscy.unet.networks.unext2 import UNeXt2Stem
from viscy.unet.networks.embedding import ContrastiveConvNext
from pathlib import Path
import torchview

top_dir = Path("/hpc/projects/intracellular_dashboard/viral-sensor/")
input_zarr = top_dir / "2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/patch_final.zarr"
model_dir = top_dir / "infection_classification/models/infection_score"

%load_ext autoreload
%autoreload 2
# %% Initialize the model and log the graph.
contra_model = ContrastiveConvNext()
print(contra_model)

model_graph = torchview.draw_graph(
    contra_model,
    torch.randn(1, 2, 15, 200, 200),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph

# %% Initiatlize the lightning module and view the model.
contrastive_module = ContrastiveLearningModel()
print(contrastive_module.model)
model_graph = torchview.draw_graph(
    contrastive_module.model,
    torch.randn(1, 2, 15, 200, 200),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph
# %% Initialize the data module and view the data.

# %% Train the model.


# %% Playground
import timm

available_models = timm.list_models(pretrained=True)

stem = UNeXt2Stem(
    in_channels=2, out_channels=96, kernel_size=(5, 2, 2), in_stack_depth=15
)
print(stem)
stem_graph = torchview.draw_graph(
    stem,
    torch.randn(1, 2, 15, 256, 256),
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
stem_graph.visual_graph
# %%
encoder = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    features_only=False,
    num_classes=200,
)

print(encoder)

# %%

encoder.stem = stem

model_graph = torchview.draw_graph(
    encoder,
    torch.randn(1, 2, 15, 256, 256),
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph
# %%
encoder.stem = torch.nn.Identity()

encoder_graph = torchview.draw_graph(
    encoder,
    torch.randn(1, 96, 128, 128),
    depth=2,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
encoder_graph.visual_graph

# %%
