"""Flow-matching 3D U-Net backbone with Vision Transformer bottleneck.

Velocity field predictor for flow-matching virtual staining.  Architecture:
CNN encoder with skip connections, timestep-conditioned transformer bottleneck,
CNN decoder with skip connections.

This module contains only the backbone network (``CELLDiffNet``).
The flow-matching training wrapper (``CELLDiff3DVS``) belongs in the
application layer and is not part of this package.
"""

import torch
from torch import Tensor

from viscy_models.celldiff.vit_bottleneck import ViTBottleneck3D
from viscy_models.unet.unet3d_base import UNet3DBase

__all__ = ["CELLDiffNet"]


class CELLDiffNet(UNet3DBase):
    """3D U-Net with ViT bottleneck for flow-matching virtual staining.

    Takes a noisy target, a phase contrast conditioning image, and a diffusion
    timestep, and predicts the velocity field for flow-matching training.

    Spatial downsampling uses stride ``(1, 2, 2)`` so the Z dimension is
    preserved through the encoder/decoder.

    Parameters
    ----------
    input_spatial_size : list[int]
        Expected input spatial size ``[D, H, W]``.  Used for positional
        embedding computation and forward-pass assertions.
    in_channels : int
        Number of input/output channels.
    dims : list[int]
        Channel widths at each encoder level, length ``L``.
        Must satisfy ``len(dims) == len(num_res_block) + 1``.
    num_res_block : list[int]
        Number of residual blocks at each encoder/decoder level, length ``L-1``.
    hidden_size : int
        Transformer hidden dimension.
    num_heads : int
        Number of self-attention heads.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Attention dropout rate.
    final_dropout : float
        Feed-forward output dropout rate.
    num_hidden_layers : int
        Number of transformer blocks in the bottleneck.
    patch_size : int
        Cubic patch size for the 3D patch embedding.
    """

    def __init__(
        self,
        input_spatial_size: list[int] = [8, 512, 512],
        in_channels: int = 1,
        dims: list[int] = [32, 64, 128],
        num_res_block: list[int] = [2, 2],
        hidden_size: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        num_hidden_layers: int = 2,
        patch_size: int = 4,
    ) -> None:
        bottleneck = ViTBottleneck3D(
            in_channels=dims[-1],
            input_spatial_size=input_spatial_size,
            num_downsamples=len(num_res_block),
            downsample_z=False,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            final_dropout=final_dropout,
            num_hidden_layers=num_hidden_layers,
            patch_size=patch_size,
            time_embed_dim=hidden_size,
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            dims=dims,
            num_res_block=num_res_block,
            bottleneck=bottleneck,
            downsample_z=False,
            time_embed_dim=hidden_size,
            cond_channels=1,
        )
        self.input_spatial_size = input_spatial_size

    def forward(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        """Predict velocity field for flow-matching.

        Parameters
        ----------
        x : Tensor
            Noisy target volume of shape ``(B, in_channels, D, H, W)``.
        cond : Tensor
            Phase contrast conditioning of shape ``(B, 1, D, H, W)``.
        t : Tensor
            Diffusion timesteps of shape ``(B,)``.

        Returns
        -------
        Tensor
            Predicted velocity field of shape ``(B, in_channels, D, H, W)``.
        """
        if x.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(f"x spatial size {list(x.shape[2:])} does not match expected {self.input_spatial_size}")
        if cond.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(
                f"cond spatial size {list(cond.shape[2:])} does not match expected {self.input_spatial_size}"
            )
        if cond.shape[1] != 1:
            raise ValueError(f"cond must have 1 channel, got {cond.shape[1]}")
        return super().forward(x, cond=cond, t=t)
