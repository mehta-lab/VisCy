# %%
from torchview import draw_graph

from viscy.light.engine import VSUNet

# %%
model = VSUNet(
    architecture="2.2D",
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 5,
        "backbone": "convnextv2_tiny",
        "stem_kernel_size": (5, 4, 4),
        "decoder_mode": "pixelshuffle",
    },
)
# %%
model_graph = draw_graph(
    model,
    model.example_input_array,
    # model.example_input_array,
    graph_name="2.1D UNet",
    roll=True,
    depth=4,
    # graph_dir="LR",
    directory="/hpc/projects/comp.micro/virtual_staining/models/HEK_phase_to_nuc_mem/",
    # save_graph=True,
)

graph = model_graph.visual_graph
graph
# %%
model_graph.visual_graph.render(format="svg")
