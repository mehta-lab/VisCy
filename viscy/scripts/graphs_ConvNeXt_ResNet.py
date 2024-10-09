# %% Imports and paths.
import timm
import torch
import torchview

from viscy.representation.contrastive import ContrastiveEncoder, StemDepthtoChannels
from viscy.representation.engine import ContrastiveModule

# %load_ext autoreload
# %autoreload 2
# %% Initialize the model and log the graph.
contra_model = ContrastiveEncoder(
    backbone="convnextv2_tiny",
    in_stack_depth=15,
    in_channels=2,
)  # other options: convnext_tiny resnet50
print(contra_model)

model_graph = torchview.draw_graph(
    contra_model,
    torch.randn(1, 2, 15, 224, 224),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.resize_graph(scale=2.5)
model_graph.visual_graph

# %% Initialize a resent50 model and log the graph.
contra_model = ContrastiveEncoder(
    backbone="resnet50", in_stack_depth=16, stem_kernel_size=(4, 3, 3)
)  # note that the resnet first layer takes 64 channels (so we can't have multiples of 3)
print(contra_model)
contra_model(torch.randn(1, 2, 16, 224, 224))
model_graph = torchview.draw_graph(
    contra_model,
    torch.randn(1, 2, 16, 224, 224),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.resize_graph(scale=2.5)
model_graph.visual_graph


# %% Initiatlize the lightning module and view the model.
contrastive_module = ContrastiveModule()
print(contrastive_module.encoder)

# %%
model_graph = torchview.draw_graph(
    contrastive_module.encoder,
    torch.randn(1, 2, 15, 200, 200),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph

# %% Playground

available_models = timm.list_models(pretrained=True)

stem = StemDepthtoChannels(
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
