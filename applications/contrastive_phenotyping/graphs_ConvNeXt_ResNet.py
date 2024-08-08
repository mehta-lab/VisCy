# %% Imports and paths.
import os
import torch
from viscy.representation.contrastive_copy1 import ContrastiveEncoder
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

example_input = torch.randn(2, 3, 256, 256)
output = convnextv1(example_input)
print(output.shape)
# %% Initialize the model and log the graph: convnext.
in_channels = 1
in_stack_depth = 15

contrastive_convnext1 = ContrastiveEncoder(
    backbone="convnext_tiny", in_channels=in_channels, in_stack_depth=in_stack_depth
)
print(contrastive_convnext1)
example_input = torch.randn(1, in_channels, in_stack_depth, 256, 256)
projections = contrastive_convnext1(example_input)
print(
    f"shape of embedding: {projections.shape}"
)

model_graph = torchview.draw_graph(
    contrastive_convnext1,
    example_input,
    depth=3,
    device="cpu",
)
model_graph.visual_graph
# %% Try convnextv2 backbone

in_channels = 3
in_stack_depth = 15

contrastive_convnext2 = ContrastiveEncoder(
    backbone="convnextv2_tiny", in_channels=in_channels, in_stack_depth=in_stack_depth
)
print(contrastive_convnext2)

example_input = torch.randn(1, in_channels, in_stack_depth, 256, 256)
embedding, projections = contrastive_convnext2(example_input)
print(
    f"shape of projections:{projections.shape}, shape of embedding: {embedding.shape}"
)

# %% ResNet backbone.
in_channels = 10
in_stack_depth = 12
contrastive_resnet = ContrastiveEncoder(
    backbone="resnet50",
    in_channels=in_channels,
    in_stack_depth=in_stack_depth,
    embedding_len=256,
)
print(contrastive_resnet)
example_input = torch.randn(1, in_channels, in_stack_depth, 256, 256)
embedding, projections = contrastive_resnet(example_input)
print(
    f"shape of projections:{projections.shape}, shape of embedding: {embedding.shape}"
)

# %%
