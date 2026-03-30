"""Vision Transformer bottleneck for 3D U-Net architectures.

Encapsulates PatchEmbed3D, sinusoidal positional embedding,
TransformerBlock stack, FinalLayer, and unpatchify into a single
module with the unified bottleneck interface
``forward(x, time_embeds=None)``.

Requires ``diffusers`` (for Attention and FeedForward).
"""

import torch
import torch.nn as nn
from torch import Tensor

from viscy_models.celldiff.modules.patch_embed_3d import PatchEmbed3D
from viscy_models.celldiff.modules.positional_embedding import get_3d_sincos_pos_embed
from viscy_models.celldiff.modules.transformer import (
    FinalLayer,
    TransformerBlock,
    unpatchify,
)

__all__ = ["ViTBottleneck3D"]


class ViTBottleneck3D(nn.Module):
    """Vision Transformer bottleneck for 3D U-Net architectures.

    Patches the input, adds sinusoidal positional embeddings, runs
    through transformer blocks, then projects and unpatchifies back
    to a 3D volume.

    Parameters
    ----------
    in_channels : int
        Number of input (and output) channels from the encoder.
    input_spatial_size : list[int]
        Original input spatial size ``[D, H, W]`` before any encoding.
    num_downsamples : int
        Number of encoder downsample stages applied before bottleneck.
    downsample_z : bool
        Whether the encoder downsamples the Z dimension.
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
        Number of transformer blocks.
    patch_size : int
        Cubic patch size for the 3D patch embedding.
    time_embed_dim : int | None
        Timestep embedding dimension for adaLN-Zero conditioning.
        Pass None for unconditional (deterministic) models.
    """

    def __init__(
        self,
        in_channels: int,
        input_spatial_size: list[int],
        num_downsamples: int,
        downsample_z: bool,
        hidden_size: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        num_hidden_layers: int = 2,
        patch_size: int = 4,
        time_embed_dim: int | None = None,
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._patch_size = patch_size

        # ── Compute latent spatial size after encoder downsamples ────────
        if downsample_z:
            latent_size = [s // (2**num_downsamples) for s in input_spatial_size]
        else:
            latent_size = input_spatial_size[:1] + [s // (2**num_downsamples) for s in input_spatial_size[1:]]

        # ── Validate patch divisibility ─────────────────────────────────
        dim_names = ["D", "H", "W"]
        for dim_val, name, orig in zip(latent_size, dim_names, input_spatial_size):
            if dim_val == 0:
                raise ValueError(
                    f"Latent {name} dimension is 0 (from input {name}={orig}). "
                    f"input_spatial_size is too small for {num_downsamples} downsamples."
                )
            if dim_val % patch_size != 0:
                raise ValueError(
                    f"Latent {name} dimension {dim_val} (from input {name}={orig}) "
                    f"is not divisible by patch_size={patch_size}. "
                    f"Each spatial dimension after {num_downsamples} encoder downsamples "
                    f"must be divisible by patch_size."
                )

        self.latent_grid_size = [s // patch_size for s in latent_size]

        # ── Patch embedding ─────────────────────────────────────────────
        self.img_embedding = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # ── Fixed sinusoidal positional embedding ───────────────────────
        img_pos_embed = (
            torch.from_numpy(get_3d_sincos_pos_embed(hidden_size, self.latent_grid_size)).float().unsqueeze(0)
        )
        self.img_pos_embed = nn.Parameter(img_pos_embed, requires_grad=False)

        # ── Transformer blocks ──────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    final_dropout=final_dropout,
                    time_embed_dim=time_embed_dim,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # ── Output projection ───────────────────────────────────────────
        self.proj_out = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=in_channels,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x: Tensor, time_embeds: Tensor | None = None) -> Tensor:
        """Forward pass through the ViT bottleneck.

        Parameters
        ----------
        x : Tensor
            Encoder output of shape ``(B, in_channels, D', H', W')``.
        time_embeds : Tensor | None
            Timestep embeddings of shape ``(B, time_embed_dim)``.

        Returns
        -------
        Tensor
            Reconstructed volume of shape ``(B, in_channels, D', H', W')``.
        """
        h = self.img_embedding(x) + self.img_pos_embed
        for block in self.blocks:
            h = block(h, time_embeds)
        h = self.proj_out(h, time_embeds)
        return unpatchify(h, self._in_channels, self.latent_grid_size, self._patch_size)
