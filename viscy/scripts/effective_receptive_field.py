# %%
"""
Script to visualize the effective receptive field.
Original paper:
https://doi.org/10.48550/arXiv.1701.04128 (Luo et al., 2017).
"""

# %%
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

# from dlmbl_unet.unet import UNet
from viscy.light.engine import VSUNet

device = "cuda"


# %%
def plot_response(model: torch.nn.Module, title: str, ndim: Literal[2, 3] = 3) -> None:
    impulse_pattern = torch.zeros(1, 1, 800, 800, device=device)
    impulse_pattern[0, 0, 400, 400] = torch.finfo(torch.float32).eps
    if ndim == 3:
        impulse_pattern = impulse_pattern.unsqueeze(0)
    impulse = impulse_pattern.clone().requires_grad_(True)

    model.zero_grad()
    input_image = torch.zeros_like(impulse, requires_grad=True)
    fake_response = model(input_image)
    fake_loss = (fake_response * impulse_pattern.clone()).sum()
    fake_loss.backward()
    response_backwards = input_image.grad.detach().cpu().numpy()
    if ndim == 3:
        response_backwards = response_backwards.squeeze(0)
    scaled_response = response_backwards[0, 0] / response_backwards.max()

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    spectrum = np.log1p(np.abs(scaled_response))
    im = ax[0].imshow(spectrum, cmap="nipy_spectral")
    _ = plt.colorbar(im, label="log power", shrink=0.5, location="bottom")
    ax[1].plot(np.log1p(scaled_response.std(axis=(0))), label="rows", alpha=0.5)
    ax[1].plot(np.log1p(scaled_response.std(axis=(1))), label="columns", alpha=0.5)
    ax[1].legend()
    ax[1].set_ylabel("log std of power")
    f.suptitle(title)
    f.tight_layout()


# %%
# unet = UNet(depth=4, in_channels=1).to(device)

# plot_response(unet, "DLMBL U-Net (initialization)", ndim=2)

# %%
model_kwargs = dict(
    architecture="fcmae",
    model_config=dict(
        in_channels=1,
        out_channels=2,
        encoder_blocks=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        decoder_conv_blocks=2,
        stem_kernel_size=(1, 2, 2),
        in_stack_depth=1,
        pretraining=False,
    ),
)

initial_model = VSUNet(**model_kwargs).to(device)
trained_model = VSUNet.load_from_checkpoint(
    "/hpc/websites/public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt",
    **model_kwargs,
).to(device)

plot_response(initial_model, "UNeXt2 (initialization)")
plot_response(trained_model, "UNeXt2 (trained)")

# %%
