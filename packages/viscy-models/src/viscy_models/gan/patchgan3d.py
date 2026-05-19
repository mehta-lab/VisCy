"""3D PatchGAN discriminators for paired virtual-staining adversarial training.

Single-scale ``PatchGAN3D`` is a 5-layer convolutional 3D discriminator with
anisotropic strides (preserving Z early, downsampling YX aggressively), inspired
by vox2vox / pix2pix 3D medical-imaging conventions. ``MultiScalePatchGAN3D``
stacks several ``PatchGAN3D`` instances operating on progressively
YX-downsampled inputs (pix2pixHD-style; Wang et al. 2018).
"""

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

__all__ = ["MultiScalePatchGAN3D", "PatchGAN3D"]


def _maybe_spectral_norm(module: nn.Module, use_spectral_norm: bool) -> nn.Module:
    """Wrap ``module`` in spectral normalization if requested."""
    return spectral_norm(module) if use_spectral_norm else module


class PatchGAN3D(nn.Module):
    """Single-scale 5-layer 3D PatchGAN discriminator.

    Five strided 3D convolutions with anisotropic strides
    ``(1, 2, 2) -> (1, 2, 2) -> (2, 2, 2) -> (2, 2, 2) -> (1, 1, 1)``. The
    first four convs use ``kernel_size=4`` and ``padding=1``; the final conv
    uses ``kernel_size=(1, 4, 4)`` and ``padding=(0, 1, 1)`` so it remains
    valid when the Z dimension has already collapsed to 1. Inner layers 2-4
    receive ``InstanceNorm3d`` followed by ``LeakyReLU(0.2)``; the first conv
    has only ``LeakyReLU(0.2)`` and the final conv emits raw logits.

    Spectral normalization (Miyato et al. 2018) is applied to every conv when
    ``use_spectral_norm`` is True (default and recommended).

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels for the discriminator. For a paired
        conditional PatchGAN this is the channel count of
        ``concat([source, target], dim=1)``. Default is 2.
    base_channels : int, optional
        Channel count of the first conv. Subsequent convs double the channels
        up to ``base_channels * 8`` before projecting to a single logit
        channel. Default is 64.
    num_layers : int, optional
        Number of conv layers. Only ``num_layers=5`` is supported in v1; the
        argument exists so future ablations can extend the network without an
        API break.
    use_spectral_norm : bool, optional
        If True, apply ``spectral_norm`` to every conv. Default is True.

    Returns
    -------
    Tensor
        Raw logits of shape ``(B, 1, Z', H', W')``; spatial size depends on
        the input. ``MultiScalePatchGAN3D.forward`` wraps this as
        ``list[Tensor]``.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 5,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers != 5:
            raise ValueError(f"PatchGAN3D only supports num_layers=5 in v1, got {num_layers}.")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Layer 1: stride (1, 2, 2), no norm, LeakyReLU
        self.layer1 = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv3d(in_channels, c1, kernel_size=4, stride=(1, 2, 2), padding=1),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 2: stride (1, 2, 2), InstanceNorm + LeakyReLU
        self.layer2 = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv3d(c1, c2, kernel_size=4, stride=(1, 2, 2), padding=1),
                use_spectral_norm,
            ),
            nn.InstanceNorm3d(c2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 3: stride (2, 2, 2), InstanceNorm + LeakyReLU
        self.layer3 = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv3d(c2, c3, kernel_size=4, stride=(2, 2, 2), padding=1),
                use_spectral_norm,
            ),
            nn.InstanceNorm3d(c3, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 4: stride (2, 2, 2), InstanceNorm + LeakyReLU
        self.layer4 = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv3d(c3, c4, kernel_size=4, stride=(2, 2, 2), padding=1),
                use_spectral_norm,
            ),
            nn.InstanceNorm3d(c4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 5: stride 1, kernel (1, 4, 4) so it stays valid when Z=1.
        # Raw logits — no norm, no activation.
        self.layer5 = _maybe_spectral_norm(
            nn.Conv3d(c4, 1, kernel_size=(1, 4, 4), stride=1, padding=(0, 1, 1)),
            use_spectral_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute raw discriminator logits.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, in_channels, Z, H, W)``. For paired
            conditional PatchGAN this is ``concat([source, target], dim=1)``.

        Returns
        -------
        Tensor
            Raw logits of shape ``(B, 1, Z', H', W')``.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class MultiScalePatchGAN3D(nn.Module):
    """Multi-scale 3D PatchGAN discriminator (pix2pixHD-style).

    Wraps ``num_scales`` independent ``PatchGAN3D`` instances. Scale 0 sees
    the input at its native resolution; each subsequent scale operates on a
    YX-downsampled (``F.avg_pool3d(kernel=(1, 2, 2), stride=(1, 2, 2))``)
    version of the previous scale's input. The forward returns one logit
    tensor per scale, with no aggregation — losses (LSGAN) average across
    scales separately.

    Parameters
    ----------
    in_channels : int, optional
        Channel count of the input to scale 0. Default is 2 (paired source +
        target / pred).
    base_channels : int, optional
        First-conv channel count for each ``PatchGAN3D``. Default is 64.
    num_layers : int, optional
        Number of conv layers per ``PatchGAN3D``. Default is 5.
    num_scales : int, optional
        Number of independent discriminators. ``num_scales=2`` is the v1
        default; ``num_scales=1`` is supported as a single-scale ablation.
        Default is 2.
    use_spectral_norm : bool, optional
        If True, apply spectral normalization to every conv in each scale.
        Default is True.

    Returns
    -------
    list of Tensor
        One logit tensor per scale, ordered from highest to lowest resolution.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 5,
        num_scales: int = 2,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}.")
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [
                PatchGAN3D(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    num_layers=num_layers,
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Compute discriminator logits at every scale.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, in_channels, Z, H, W)``.

        Returns
        -------
        list of Tensor
            ``num_scales`` logit tensors, ordered from highest to lowest
            resolution. Each entry has shape ``(B, 1, Z', H', W')``; spatial
            size shrinks by roughly 2x in YX between consecutive scales.
        """
        outputs: list[Tensor] = []
        current = x
        for scale_idx, disc in enumerate(self.discriminators):
            outputs.append(disc(current))
            if scale_idx + 1 < self.num_scales:
                current = F.avg_pool3d(current, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return outputs
