# %%
from torchview import draw_graph

from viscy.light.engine import VSUNet

# %%
model = VSUNet(
    model_config={
        "architecture": "2.5D",
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 5,
        "residual": True,
        "task": "reg",
        "dropout": 0.1,
    },
    batch_size=32,
)
# %%

model_graph = draw_graph(
    model,
    model.example_input_array,
    graph_name="2.5D UNet",
    roll=True,
    depth=4,
    # graph_dir="LR",
    directory="/hpc/projects/comp.micro/virtual_staining/models/HEK_phase_to_nuc_mem/",
    save_graph=True,
)

graph = model_graph.visual_graph
graph
# %%
model_graph.visual_graph.render(format="svg")
