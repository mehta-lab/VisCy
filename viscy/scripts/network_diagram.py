# %%
from torchview import draw_graph

from viscy.light.engine import VSUNet

# %% 2D UNet
model = VSUNet(
    architecture="2D",
    model_config={"in_channels": 1, "out_channels": 2},
)
# %%
model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2D UNet",
    roll=True,
    depth=4,
    # graph_dir="LR",
    # save_graph=True,
)

graph2d = model_graph.visual_graph
graph2d

# %% 2.5D UNet
model = VSUNet(
    architecture="2.5D",
    model_config={
        "in_channels": 1,
        "out_channels": 3,
        "in_stack_depth": 9,
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2.5D UNet",
    roll=True,
    depth=2,
)

graph25d = model_graph.visual_graph
graph25d

# %%
# 3D->2D
model = VSUNet(
    architecture="UNext2",
    model_config={
        "in_channels": 2,
        "out_channels": 3,
        "in_stack_depth": 5,
        "out_stack_depth": 1,
        "backbone": "convnextv2_tiny",
        "decoder_mode": "pixelshuffle",
        "stem_kernel_size": (5, 4, 4),
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="UNext2",
    roll=True,
    depth=3,
)

model_graph.visual_graph

# %%
# 3D->3D
model = VSUNet(
    architecture="UNext2",
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 9,
        "backbone": "convnextv2_tiny",
        "decoder_mode": "pixelshuffle",
        "stem_kernel_size": (3, 4, 4),
    },
)

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="UNext2",
    roll=True,
    depth=3,
)

model_graph.visual_graph
# %% If you want to save the graphs as SVG files:
# model_graph.visual_graph.render(format="svg")
