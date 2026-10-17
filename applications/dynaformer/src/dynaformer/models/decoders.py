"""Image decoders that can be conditioned on a temporally-mixed token.

Every decoder here reconstructs one frame at a time from that frame's spatial
features, plus an optional conditioning vector carrying temporal context. The
conditioning projection is **zero-initialized**: at step 0 the decoder output is
bit-identical to the unconditioned per-frame decoder, and training only ever adds
temporal corrections on top of a working per-frame model.

That detail is not cosmetic. In the study this application generalizes, a decoder
that saw no temporal context produced frame-to-frame jitter and scored 0.303
per-track Pearson r; adding exactly this zero-init edge took it to 0.581.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from viscy_models.components.blocks import UNeXt2Decoder
from viscy_models.components.heads import PixelToVoxelHead

__all__ = ["MAEImageDecoder", "UNeXt2ImageDecoder", "ZeroInitConditioning"]


class ZeroInitConditioning(nn.Module):
    """Zero-initialized projection of a conditioning vector into a feature space.

    Returns exact zeros until trained, making the conditioned model identical to
    the unconditioned one at initialization.

    Parameters
    ----------
    cond_dim : int
        Width of the conditioning vector.
    out_dim : int
        Width of the target feature space.
    """

    def __init__(self, cond_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: Tensor) -> Tensor:
        """Project ``(N, cond_dim)`` conditioning to ``(N, out_dim)``."""
        return self.proj(cond)


class _DecoderBlock(nn.Module):
    """Pre-norm transformer block used by the MAE decoder."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Run one attention and one MLP residual block."""
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MAEImageDecoder(nn.Module):
    """Shallow transformer decoder over patch tokens, unpatchified to an image.

    Dense (no mask tokens), so every patch of the frame is reconstructed. Pairs
    with a transformer frame encoder such as
    :class:`~dynaformer.models.encoders.MorphEmFrameEncoder`.

    The reconstruction happens on the encoder's patch grid and is then resampled
    to ``out_size``, so output detail is bounded by the patch grid, not by
    ``out_size``.

    Parameters
    ----------
    embed_dim : int
        Width of the incoming patch tokens.
    out_channels : int
        Number of output channels per frame.
    grid_size : int
        Number of patches per side in the encoder's grid.
    patch_size : int
        Pixels per patch side in the decoder's own reconstruction, by default 16.
    out_size : tuple[int, int] | None
        ``(Y, X)`` size the reconstruction is resampled to. ``None`` keeps the
        decoder's native ``grid_size * patch_size``, by default ``None``.
    out_stack_depth : int
        Z depth of the output, by default 1. The decoder is 2D, so depths above 1
        are produced by repeating the plane.
    dec_dim : int
        Decoder token width, by default 256.
    dec_depth : int
        Number of decoder blocks, by default 4.
    dec_heads : int
        Number of decoder attention heads, by default 8.
    cond_dim : int
        Width of the temporal conditioning vector. ``0`` disables conditioning,
        by default 0.
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        grid_size: int,
        patch_size: int = 16,
        out_size: tuple[int, int] | None = None,
        out_stack_depth: int = 1,
        dec_dim: int = 256,
        dec_depth: int = 4,
        dec_heads: int = 8,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.out_size = tuple(out_size) if out_size is not None else None
        self.out_stack_depth = out_stack_depth
        self.embed = nn.Linear(embed_dim, dec_dim)
        self.pos = nn.Parameter(torch.zeros(1, grid_size**2, dec_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cond_proj = ZeroInitConditioning(cond_dim, dec_dim) if cond_dim else None
        self.blocks = nn.ModuleList([_DecoderBlock(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_size * patch_size * out_channels)

    def forward(self, spatial: Tensor, cond: Tensor | None = None) -> Tensor:
        """Reconstruct frames from patch tokens.

        Parameters
        ----------
        spatial : Tensor
            Patch tokens of shape ``(N, grid_size ** 2, embed_dim)``.
        cond : Tensor | None
            Temporal conditioning of shape ``(N, cond_dim)``, broadcast over
            patches. Ignored when the decoder was built with ``cond_dim=0``.

        Returns
        -------
        Tensor
            ``(N, out_channels, out_stack_depth, Y, X)``.
        """
        x = self.embed(spatial) + self.pos
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = self.pred(self.norm(x))
        n, g, p, c = x.shape[0], self.grid_size, self.patch_size, self.out_channels
        x = x.reshape(n, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(n, c, g * p, g * p)
        if self.out_size is not None and x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x.unsqueeze(2).expand(-1, -1, self.out_stack_depth, -1, -1)


class UNeXt2ImageDecoder(nn.Module):
    """UNeXt2 decoder with skip connections, conditioned at the bottleneck.

    Pairs with :class:`~dynaformer.models.encoders.UNeXt2FrameEncoder`. Conditioning
    is added to the deepest feature map through a zero-initialized 1x1 projection
    before decoding, so the skip connections still carry full per-frame spatial
    detail while the bottleneck carries temporal context.

    Parameters
    ----------
    feature_channels : list[int]
        Channel counts of the encoder's multi-scale features, deepest first.
    out_channels : int
        Number of output channels per frame.
    out_stack_depth : int
        Z depth of the output, by default 1.
    stem_stride : int
        Spatial stride of the encoder stem. Must be 4 — see Notes, by default 4.
    decoder_mode : Literal["deconv", "pixelshuffle"]
        Upsampling mode, by default ``"pixelshuffle"``.
    decoder_conv_blocks : int
        Convolution blocks per decoder stage, by default 2.
    decoder_norm_layer : str
        Normalization layer name, by default ``"instance"``.
    decoder_upsample_pre_conv : bool
        Apply a pre-convolution before pixel-shuffle upsampling, by default ``False``.
    head_expansion_ratio : int
        Channel expansion in the output head, by default 4.
    head_pool : bool
        Apply pad-pool smoothing in the output head, by default ``False``.
    cond_dim : int
        Width of the temporal conditioning vector. ``0`` disables conditioning,
        by default 0.

    Notes
    -----
    The decoder runs ``len(feature_channels) - 1`` stages at stride 2 and the
    output head upsamples by 4, so it magnifies by ``2 ** (n - 1) * 4``. A
    matching encoder shrinks by ``stem_stride * 2 ** (n - 1)``. The two agree
    only when the stem stride is 4, so any other value is rejected rather than
    silently returning an image of the wrong size.
    """

    def __init__(
        self,
        feature_channels: list[int],
        out_channels: int,
        out_stack_depth: int = 1,
        stem_stride: int = 4,
        decoder_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
        decoder_upsample_pre_conv: bool = False,
        head_expansion_ratio: int = 4,
        head_pool: bool = False,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if stem_stride != 4:
            raise ValueError(
                f"UNeXt2ImageDecoder restores the input resolution only with a stem stride of 4, got {stem_stride}."
            )
        # UNeXt2UpStage assumes each skip has half the channels of its stage input,
        # and PixelToVoxelHead fixes the final channel count; both follow UNeXt2.
        decoder_channels = list(feature_channels)
        decoder_channels[-1] = (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        self.decoder = UNeXt2Decoder(
            decoder_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(feature_channels) - 1),
            upsample_pre_conv="default" if decoder_upsample_pre_conv else None,
        )
        self.head = PixelToVoxelHead(
            decoder_channels[-1], out_channels, out_stack_depth, head_expansion_ratio, pool=head_pool
        )
        self.cond_proj = ZeroInitConditioning(cond_dim, feature_channels[0]) if cond_dim else None
        self.out_stack_depth = out_stack_depth

    def forward(self, spatial: list[Tensor], cond: Tensor | None = None) -> Tensor:
        """Reconstruct frames from multi-scale features.

        Parameters
        ----------
        spatial : list[Tensor]
            Bottleneck-first feature maps ``(N, C_i, H_i, W_i)``.
        cond : Tensor | None
            Temporal conditioning of shape ``(N, cond_dim)``, broadcast over the
            bottleneck's spatial positions. Ignored when built with ``cond_dim=0``.

        Returns
        -------
        Tensor
            ``(N, out_channels, out_stack_depth, Y, X)``.
        """
        features = list(spatial)
        if self.cond_proj is not None and cond is not None:
            features[0] = features[0] + self.cond_proj(cond)[..., None, None]
        return self.head(self.decoder(features))
