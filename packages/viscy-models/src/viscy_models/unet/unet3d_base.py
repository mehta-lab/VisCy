"""Parametrized 3D U-Net base class.

Iterative encoder-decoder with skip connections and an injected bottleneck
module. All three VisCy 3D U-Net variants (Unet3d / FNet, UNetViT3D,
CELLDiffNet) are expressed as thin wrappers that configure this base.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from viscy_models.unet.blocks import ResnetBlock, TimestepEmbedder

__all__ = ["UNet3DBase"]


class UNet3DBase(nn.Module):
    """Parametrized 3D U-Net with injected bottleneck.

    Encoder → bottleneck → decoder with concatenation skip connections.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dims : list[int]
        Channel widths at each encoder level. Length ``L``.
        ``len(dims) == len(num_res_block) + 1`` so the last entry is the
        bottleneck channel count.
    num_res_block : list[int]
        Number of conv blocks at each encoder/decoder level. Length ``L-1``.
    bottleneck : nn.Module
        Bottleneck module with signature
        ``forward(x: Tensor, time_embeds: Tensor | None) -> Tensor``.
    downsample_z : bool
        If True, downsample/upsample all three spatial dims (stride 2,2,2).
        If False, only H and W (stride 1,2,2).
    residual : bool
        Whether encoder/decoder blocks use residual connections.
    norm : {"group", "batch"}
        Normalization type for conv blocks.
    activation : {"silu", "relu"}
        Activation type for conv blocks.
    groups : int
        Number of groups for GroupNorm (ignored for BatchNorm).
    time_embed_dim : int | None
        If provided, adds a ``TimestepEmbedder`` and passes timestep
        embeddings to all conv blocks and the bottleneck.
    cond_channels : int | None
        If provided, adds a conditioning input convolution that is summed
        with the main input projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dims: list[int],
        num_res_block: list[int],
        bottleneck: nn.Module,
        downsample_z: bool = False,
        residual: bool = True,
        norm: Literal["group", "batch"] = "group",
        activation: Literal["silu", "relu"] = "silu",
        groups: int = 8,
        time_embed_dim: int | None = None,
        cond_channels: int | None = None,
    ) -> None:
        super().__init__()

        if len(dims) != len(num_res_block) + 1:
            raise ValueError(f"len(dims)={len(dims)} must equal len(num_res_block)+1={len(num_res_block) + 1}")

        self._num_res_block = list(num_res_block)
        self.downsamples_z: bool = downsample_z

        block_kwargs = dict(norm=norm, activation=activation, groups=groups)

        # ── Timestep embedding ──────────────────────────────────────────
        self._time_embedder: TimestepEmbedder | None = None
        if time_embed_dim is not None:
            self._time_embedder = TimestepEmbedder(hidden_size=time_embed_dim)

        # ── Input projections ───────────────────────────────────────────
        self.inconv = nn.Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1)
        self._cond_inconv: nn.Conv3d | None = None
        if cond_channels is not None:
            self._cond_inconv = nn.Conv3d(cond_channels, dims[0], kernel_size=3, stride=1, padding=1)

        # ── Stride configuration ────────────────────────────────────────
        if downsample_z:
            down_stride = (2, 2, 2)
            up_kwargs = dict(kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=1)
        else:
            down_stride = (1, 2, 2)
            up_kwargs = dict(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))

        # ── Encoder ─────────────────────────────────────────────────────
        n_levels = len(num_res_block)
        self._encoder_blocks = nn.ModuleList()
        self._downsamples = nn.ModuleList()
        for i_level in range(n_levels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_block[i_level]):
                level_blocks.append(
                    ResnetBlock(
                        dims[i_level], dims[i_level], time_emb_dim=time_embed_dim, residual=residual, **block_kwargs
                    )
                )
            self._encoder_blocks.append(level_blocks)
            self._downsamples.append(
                nn.Conv3d(dims[i_level], dims[i_level + 1], kernel_size=3, stride=down_stride, padding=1)
            )

        # ── Bottleneck ──────────────────────────────────────────────────
        self.bottleneck = bottleneck

        # ── Decoder ─────────────────────────────────────────────────────
        self._upsamples = nn.ModuleList()
        self._decoder_blocks = nn.ModuleList()
        for i_level in reversed(range(n_levels)):
            self._upsamples.append(nn.ConvTranspose3d(dims[i_level + 1], dims[i_level], **up_kwargs))
            level_blocks = nn.ModuleList()
            for _ in range(num_res_block[i_level]):
                level_blocks.append(
                    ResnetBlock(
                        dims[i_level] * 2, dims[i_level], time_emb_dim=time_embed_dim, residual=residual, **block_kwargs
                    )
                )
            self._decoder_blocks.append(level_blocks)

        # ── Output projection ───────────────────────────────────────────
        self.outconv = nn.Conv3d(dims[0], out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def num_blocks(self) -> int:
        """Number of spatial downsampling stages."""
        return len(self._num_res_block)

    def forward(self, x: Tensor, cond: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, D, H, W)``.
        cond : Tensor | None
            Optional conditioning input of shape ``(B, cond_channels, D, H, W)``.
        t : Tensor | None
            Optional timesteps of shape ``(B,)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, out_channels, D, H, W)``.
        """
        # ── Timestep embedding ──────────────────────────────────────────
        time_embeds: Tensor | None = None
        if self._time_embedder is not None and t is not None:
            time_embeds = self._time_embedder(t)

        # ── Input projection ────────────────────────────────────────────
        h = self.inconv(x)
        if self._cond_inconv is not None and cond is not None:
            h = h + self._cond_inconv(cond)

        # ── Encode ──────────────────────────────────────────────────────
        skips: list[Tensor] = []
        for level_blocks, downsample in zip(self._encoder_blocks, self._downsamples):
            for block in level_blocks:
                h = block(h, time_embeds)
                skips.append(h)
            h = downsample(h)

        # ── Bottleneck ──────────────────────────────────────────────────
        h = self.bottleneck(h, time_embeds=time_embeds)

        # ── Decode ──────────────────────────────────────────────────────
        for upsample, level_blocks in zip(self._upsamples, self._decoder_blocks):
            h = upsample(h)
            for block in level_blocks:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, time_embeds)

        return self.outconv(h)
