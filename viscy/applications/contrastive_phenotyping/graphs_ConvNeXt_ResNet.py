# %% Imports and paths.
import os
import torch
from viscy.representation.contrastive import ContrastiveEncoder
import torchview
import timm

# uncomment if you are using jupyter and want to autoreload the updated code.
# %load_ext autoreload
# %autoreload 2

# %% Explore model graphs returned by timm

convnextv1 = timm.create_model(
    "convnext_tiny", pretrained=False, features_only=False, num_classes=200
)
print(convnextv1)
output = convnextv1(torch.randn(1, 3, 256, 256))
print(output.shape)
# %% Initialize the model and log the graph: convnext.
in_channels = 1
in_stack_depth = 15

contrastive_convnext1 = ContrastiveEncoder(
    backbone="convnext_tiny", in_channels=in_channels, in_stack_depth=in_stack_depth
)
print(contrastive_convnext1)


projections, embedding = contrastive_convnext1(
    torch.randn(1, in_channels, in_stack_depth, 256, 256)
)
print(
    f"shape of projections:{projections.shape}, shape of embedding: {embedding.shape}"
)
# %%

in_channels = 3
in_stack_depth = 18

contrastive_convnext2 = ContrastiveEncoder(
    backbone="convnextv2_tiny", in_channels=in_channels, in_stack_depth=in_stack_depth
)
print(contrastive_convnext2)
embedding, projections = contrastive_convnext2(
    torch.randn(1, in_channels, in_stack_depth, 256, 256)
)
print(
    f"shape of projections:{projections.shape}, shape of embedding: {embedding.shape}"
)

# %%
in_channels = 10
in_stack_depth = 11
contrastive_resnet = ContrastiveEncoder(
    backbone="resnet50", in_channels=in_channels, in_stack_depth=in_stack_depth
)
print(contrastive_resnet)
embedding, projections = contrastive_resnet(
    torch.randn(1, in_channels, in_stack_depth, 256, 256)
)
print(
    f"shape of projections:{projections.shape}, shape of embedding: {embedding.shape}"
)

# %%
plot_model = contrastive_resnet
model_graph = torchview.draw_graph(
    plot_model,
    input_size=(20, in_channels, in_stack_depth, 224, 224),
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
    torch.randn(1, in_channels, in_stack_depth, 200, 200),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph

# %% Playground

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
