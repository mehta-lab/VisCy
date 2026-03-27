"""Flow-matching 3D U-Net backbone with Vision Transformer bottleneck.

Velocity field predictor for flow-matching virtual staining.  Architecture:
CNN encoder with skip connections, timestep-conditioned transformer bottleneck,
CNN decoder with skip connections.

This module contains only the backbone network (``CELLDiffNet``).
The flow-matching training wrapper (``CELLDiff3DVS``) belongs in the
application layer and is not part of this package.
"""

import math

import torch
import torch.nn as nn

from viscy_models.celldiff.modules import TimestepEmbedder
from viscy_models.celldiff.modules.patch_embed_3d import PatchEmbed3D
from viscy_models.celldiff.modules.positional_embedding import get_3d_sincos_pos_embed
from viscy_models.celldiff.modules.simple_diffusion import ResnetBlock
from viscy_models.celldiff.modules.transformer import (
    FinalLayer,
    TransformerBlock,
    unpatchify,
)


class CELLDiffNet(nn.Module):
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
        super().__init__()

        assert len(dims) == len(num_res_block) + 1, (
            f"len(dims)={len(dims)} must equal len(num_res_block)+1={len(num_res_block) + 1}"
        )

        self.input_spatial_size = input_spatial_size
        self.num_res_block = num_res_block
        self._dims = dims
        self._patch_size = patch_size

        # ── Input projections ───────────────────────────────────────────────
        self.inconv = nn.Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1)
        self.cond_inconv = nn.Conv3d(1, dims[0], kernel_size=3, stride=1, padding=1)

        # ── Timestep embedding ──────────────────────────────────────────────
        self.t_embedding = TimestepEmbedder(hidden_size=hidden_size)

        # ── Encoder (downsampling) ───────────────────────────────────────────
        downs: dict[str, nn.Module] = {}
        for i_level in range(len(num_res_block)):
            for i_block in range(num_res_block[i_level]):
                downs[f"{i_level}{i_block}"] = ResnetBlock(dims[i_level], dims[i_level], time_emb_dim=hidden_size)
            downs[f"down{i_level}"] = nn.Conv3d(
                dims[i_level],
                dims[i_level + 1],
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            )
        self.downs = nn.ModuleDict(downs)

        # ── Transformer bottleneck ──────────────────────────────────────────
        self.img_embedding = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=dims[-1],
            embed_dim=hidden_size,
            bias=True,
        )

        n_down = len(num_res_block)
        latent_size = input_spatial_size[:1] + [s // (2**n_down) for s in input_spatial_size[1:]]
        self.latent_grid_size = [s // patch_size for s in latent_size]
        assert math.prod(self.latent_grid_size) > 0, (
            f"latent_grid_size {self.latent_grid_size} contains a zero; "
            "check that input_spatial_size is divisible by 2^n_down * patch_size"
        )

        img_pos_embed = (
            torch.from_numpy(get_3d_sincos_pos_embed(hidden_size, self.latent_grid_size)).float().unsqueeze(0)
        )
        self.img_pos_embed = nn.Parameter(img_pos_embed, requires_grad=False)

        self.mids = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    final_dropout=final_dropout,
                    time_embed_dim=hidden_size,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.img_proj_out = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=dims[-1],
            time_embed_dim=hidden_size,
        )

        # ── Decoder (upsampling) ────────────────────────────────────────────
        ups: dict[str, nn.Module] = {}
        for i_level in reversed(range(len(num_res_block))):
            ups[f"up{i_level}"] = nn.ConvTranspose3d(
                dims[i_level + 1],
                dims[i_level],
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            )
            for i_block in range(num_res_block[i_level]):
                ups[f"{i_level}{i_block}"] = ResnetBlock(dims[i_level] * 2, dims[i_level], time_emb_dim=hidden_size)
        self.ups = nn.ModuleDict(ups)

        # ── Output projection ───────────────────────────────────────────────
        self.outconv = nn.Conv3d(dims[0], in_channels, kernel_size=3, stride=1, padding=1)

    @property
    def num_blocks(self) -> int:
        """Number of spatial downsampling stages (each halves H and W by 2)."""
        return len(self.num_res_block)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for flow-matching.

        Parameters
        ----------
        x : torch.Tensor
            Noisy target volume of shape ``(B, in_channels, D, H, W)``.
        cond : torch.Tensor
            Phase contrast conditioning of shape ``(B, 1, D, H, W)``.
        t : torch.Tensor
            Diffusion timesteps of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Predicted velocity field of shape ``(B, in_channels, D, H, W)``.
        """
        if x.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(f"x spatial size {list(x.shape[2:])} does not match expected {self.input_spatial_size}")
        if cond.shape[2:] != torch.Size(self.input_spatial_size):
            raise ValueError(
                f"cond spatial size {list(cond.shape[2:])} does not match expected {self.input_spatial_size}"
            )

        time_embeds = self.t_embedding(t)
        h = self.inconv(x) + self.cond_inconv(cond)

        # ── Encode ──────────────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for i_level in range(len(self.num_res_block)):
            for i_block in range(self.num_res_block[i_level]):
                h = self.downs[f"{i_level}{i_block}"](h, time_embeds)
                skips.append(h)
            h = self.downs[f"down{i_level}"](h)

        # ── Transformer bottleneck ──────────────────────────────────────────
        h_embeds = self.img_embedding(h) + self.img_pos_embed
        for block in self.mids:
            h_embeds = block(h_embeds, time_embeds)
        h = self.img_proj_out(h_embeds, time_embeds)
        h = unpatchify(h, self._dims[-1], self.latent_grid_size, self._patch_size)

        # ── Decode ──────────────────────────────────────────────────────────
        for i_level in reversed(range(len(self.num_res_block))):
            h = self.ups[f"up{i_level}"](h)
            for i_block in range(self.num_res_block[i_level]):
                h = torch.cat((h, skips.pop()), dim=1)
                h = self.ups[f"{i_level}{i_block}"](h, time_embeds)

        return self.outconv(h)
