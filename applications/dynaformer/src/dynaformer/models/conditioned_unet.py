"""Virtual-staining UNet g(p_t, ê_t): a phase frame in, conditioned on that frame's temporal embedding.

The UNet is the UNeXt2 layout used by VSCyto2D (stride-4 stem, ConvNeXt V2 encoder, pixel-shuffle decoder with skip
connections). The temporal embedding enters through feature-wise linear modulation (FiLM) of the bottleneck and of
every decoder stage, ``h * (1 + gamma) + beta``, with ``gamma`` and ``beta`` projected from the embedding by
**zero-initialized** layers: a freshly built conditioned UNet is exactly the per-frame UNet, and training only adds
what the temporal context is worth (the same rule as :class:`~dynaformer.models.decoders.ZeroInitConditioning`).

The embedding describes the tracked cell, which sits at the centre of the crop, not its neighbours. With
``cond_sigma`` set, the modulation is weighted by a Gaussian around the crop centre (``cond_sigma`` in input pixels),
so it acts on the tracked nucleus and fades over the neighbours. Without it the same modulation reaches every pixel.
"""

import timm
import torch
from torch import Tensor, nn

from viscy_models.components.blocks import UNeXt2Decoder
from viscy_models.components.heads import PixelToVoxelHead

__all__ = ["ConditionedUNet", "ZeroInitFiLM"]


class ZeroInitFiLM(nn.Module):
    """Zero-initialized FiLM of a feature map by a conditioning vector, optionally gated around the centre.

    Parameters
    ----------
    cond_dim : int
        Width of the conditioning vector.
    channels : int
        Channels of the modulated feature map.
    sigma : float | None
        Standard deviation of the centred Gaussian gate, in input pixels. ``None`` applies the modulation uniformly.
    """

    def __init__(self, cond_dim: int, channels: int, sigma: float | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.sigma = sigma

    def gate(self, height: int, width: int, stride: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Centred Gaussian over a ``(height, width)`` feature map whose pixels are ``stride`` input pixels apart."""
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride - height * stride / 2
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride - width * stride / 2
        squared = y[:, None] ** 2 + x[None, :] ** 2
        return torch.exp(-squared / (2 * self.sigma**2)).to(dtype)[None, None]

    def forward(self, h: Tensor, cond: Tensor, stride: int) -> Tensor:
        """Modulate ``(N, C, H, W)`` features by ``(N, cond_dim)`` conditioning."""
        gamma, beta = self.proj(cond).to(h.dtype)[..., None, None].chunk(2, dim=1)
        if self.sigma is not None:
            gate = self.gate(h.shape[-2], h.shape[-1], stride, h.device, h.dtype)
            gamma, beta = gamma * gate, beta * gate
        return h * (1 + gamma) + beta


class ConditionedUNet(nn.Module):
    """UNeXt2 virtual-staining network conditioned on a per-frame embedding.

    Parameters
    ----------
    in_channels : int
        Input channels (phase), by default 1.
    out_channels : int
        Output channels (Geminin-mKO, PIP-mVenus), by default 2.
    cond_dim : int
        Width of the conditioning embedding; 0 builds a plain per-frame UNet, by default 0.
    cond_sigma : float | None
        Standard deviation of the centred gate in input pixels; ``None`` conditions every pixel equally, by
        default 48 (the tracked nucleus is about 64 px across).
    backbone : str
        timm ConvNeXt encoder, by default ``"convnextv2_tiny"`` (as VSCyto2D).
    decoder_conv_blocks : int
        ConvNeXt blocks per decoder stage, by default 2.
    head_expansion_ratio : int
        Channel expansion of the output head, by default 4.
    drop_path_rate : float
        Stochastic depth of the encoder, by default 0.1.

    Notes
    -----
    The stem and every encoder stage after the first halve the resolution to a total stride of 32, so the image
    side must be divisible by 32.
    """

    STRIDES = (32, 16, 8, 4)  # bottleneck, then the output of each decoder stage

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        cond_dim: int = 0,
        cond_sigma: float | None = 48.0,
        backbone: str = "convnextv2_tiny",
        decoder_conv_blocks: int = 2,
        head_expansion_ratio: int = 4,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        encoder = timm.create_model(backbone, pretrained=False, features_only=True, drop_path_rate=drop_path_rate)
        channels = encoder.feature_info.channels()
        if len(channels) != len(self.STRIDES):
            raise ValueError(f"Expected a 4-stage encoder, {backbone} has {len(channels)} stages.")
        # The stride-4 stem replaces the backbone's own patchify convolution, as in UNeXt2.
        encoder.stem_0 = nn.Identity()
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=4, stride=4)
        self.encoder = encoder
        decoder_channels = list(reversed(channels))
        # PixelToVoxelHead fixes the last decoder width; out_stack_depth 1 gives a single plane.
        decoder_channels[-1] = 3 * out_channels * 2**2 * head_expansion_ratio
        self.decoder = UNeXt2Decoder(
            decoder_channels,
            norm_name="instance",
            mode="pixelshuffle",
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(channels) - 1),
            upsample_pre_conv=None,
        )
        self.head = PixelToVoxelHead(decoder_channels[-1], out_channels, 1, head_expansion_ratio, pool=False)
        self.cond_dim = cond_dim
        self.film = (
            nn.ModuleList([ZeroInitFiLM(cond_dim, c, cond_sigma) for c in decoder_channels]) if cond_dim else None
        )

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Predict the stain from ``(N, in_channels, Y, X)`` frames and optional ``(N, cond_dim)`` conditioning.

        Returns
        -------
        Tensor
            ``(N, out_channels, Y, X)``.
        """
        if x.shape[-1] % self.STRIDES[0] or x.shape[-2] % self.STRIDES[0]:
            raise ValueError(f"Image side must be divisible by {self.STRIDES[0]}, got {tuple(x.shape[-2:])}.")
        if self.film is not None and cond is None:
            raise ValueError("This UNet was built with conditioning; pass cond.")
        features = self.encoder(self.stem(x))[::-1]  # deepest first
        h = features[0]
        if self.film is not None:
            h = self.film[0](h, cond, self.STRIDES[0])
        for i, (skip, stage) in enumerate(zip(features[1:], self.decoder.decoder_stages, strict=True)):
            h = stage(h, skip)
            if self.film is not None:
                h = self.film[i + 1](h, cond, self.STRIDES[i + 1])
        return self.head(h)[:, :, 0]
