# %%
import torch
from ptflops import get_model_complexity_info

from viscy.light.engine import VSUNet

# %%
model = VSUNet(
    architecture="UNeXt2",
    model_config={
        "in_channels": 1,
        "out_channels": 2,
        "in_stack_depth": 5,
        "backbone": "convnextv2_tiny",
        "stem_kernel_size": (5, 4, 4),
        "decoder_mode": "pixelshuffle",
        "head_expansion_ratio": 4,
    },
)

# %%
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model,
        (1, 5, 2048, 2048),  # print_per_layer_stat=False
    )
print(macs, params)
# %%
