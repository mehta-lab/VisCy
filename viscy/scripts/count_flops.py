# %%
import torch
from ptflops import get_model_complexity_info

from viscy.representation.contrastive import ContrastiveEncoder, ResNet3dEncoder


# %%
def print_flops(model):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model,
            (1, 32, 128, 128),  # print_per_layer_stat=False
        )
    print(macs, params)


# %%
resnet_3d = ResNet3dEncoder(
    "resnet10", in_channels=1, embedding_dim=512, projection_dim=32
)

print_flops(resnet_3d)

# %%
with torch.inference_mode():
    features = resnet_3d(torch.rand(1, 1, 32, 128, 128))

for f in features:
    print(f.shape)

# %%
convnext_2d = ContrastiveEncoder(
    backbone="convnext_tiny",
    in_channels=1,
    in_stack_depth=32,
    stem_kernel_size=(4, 4, 4),
    stem_stride=(4, 4, 4),
    embedding_dim=768,
    projection_dim=32,
    drop_path_rate=0.0,
)

print_flops(convnext_2d)

# %%
