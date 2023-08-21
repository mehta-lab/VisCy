# %%
import torch
from ptflops import get_model_complexity_info

from viscy.light.engine import VSUNet

# %%
model = VSUNet(
    model_config={
        "architecture": "2.1D",
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 9,
        "backbone": "convnextv2_tiny",
        "stem_kernel_size": (9, 4, 4)
    },
)

# %%
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model, (1, 9, 2048, 2048), print_per_layer_stat=False
    )
print(macs, params)
# %%
