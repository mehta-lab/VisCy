"""End-to-end 3D U-Net with Vision Transformer bottleneck for virtual staining.

Deterministic (no diffusion) model mapping label-free phase contrast to
fluorescence virtual staining.  Architecture: CNN encoder with skip connections,
transformer bottleneck, CNN decoder with skip connections.
"""

import torch
from torch import Tensor

from viscy_models.celldiff.vit_bottleneck import ViTBottleneck3D
from viscy_models.unet.unet3d_base import UNet3DBase

__all__ = ["UNetViT3D"]


class UNetViT3D(UNet3DBase):
    """3D U-Net with Vision Transformer bottleneck for end-to-end virtual staining.

    Takes a 3D label-free input (e.g. phase contrast) and directly predicts
    fluorescence virtual staining.  No diffusion / timestep conditioning.

    Spatial downsampling uses stride ``(1, 2, 2)`` so the Z dimension is
    preserved through the encoder/decoder (``downsamples_z = False``).

    Parameters
    ----------
    input_spatial_size : list[int]
        Expected input spatial size ``[D, H, W]``.  Used for positional
        embedding computation and forward-pass assertions.
    in_channels : int
        Number of input channels (label-free modality).
    out_channels : int
        Number of output channels (fluorescence targets).
    dims : list[int]
        Channel widths at each encoder level, length ``L``.
        Must have ``len(dims) == len(num_res_block) + 1`` so the last entry
        is the bottleneck channel count.
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
        input_spatial_size: list[int] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        dims: list[int] | None = None,
        num_res_block: list[int] | None = None,
        hidden_size: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        num_hidden_layers: int = 2,
        patch_size: int = 4,
    ) -> None:
        if input_spatial_size is None:
            input_spatial_size = [8, 512, 512]
        if dims is None:
            dims = [32, 64, 128]
        if num_res_block is None:
            num_res_block = [2, 2]
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
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dims=dims,
            num_res_block=num_res_block,
            bottleneck=bottleneck,
            downsample_z=False,
        )
        self.input_spatial_size = input_spatial_size

    def forward(self, x: Tensor) -> Tensor:
        """Run end-to-end virtual staining prediction.

        Parameters
        ----------
        x : Tensor
            Input volume of shape ``(B, in_channels, D, H, W)``.
            Spatial dimensions must match ``input_spatial_size``.

        Returns
        -------
        Tensor
            Predicted staining of shape ``(B, out_channels, D, H, W)``.
        """
        if x.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(f"x spatial size {list(x.shape[2:])} does not match expected {self.input_spatial_size}")
        return super().forward(x)
