"""Per-frame spatial encoders.

A frame encoder maps one frame to :class:`FrameFeatures`: a single token that the
temporal transformer consumes, and optional spatial features that an image decoder
consumes. Nothing here is aware of time — the caller folds the time axis into the
batch before calling and unfolds it afterwards.
"""

from dataclasses import dataclass

import timm
from torch import Tensor, nn

from viscy_models.components.stems import UNeXt2Stem
from viscy_models.foundation.morphem import MORPHEM_DEFAULT, MorphEmModel

__all__ = ["FrameFeatures", "MorphEmFrameEncoder", "UNeXt2FrameEncoder"]


@dataclass
class FrameFeatures:
    """Output of a per-frame encoder.

    Attributes
    ----------
    token : Tensor
        ``(N, token_dim)``, one vector per frame, fed to the temporal module.
    spatial : Tensor | list[Tensor] | None
        Features an image decoder consumes. Patch tokens ``(N, P, D)`` for a
        transformer decoder, or a bottleneck-first list of feature maps
        ``(N, C_i, H_i, W_i)`` for a convolutional decoder. ``None`` when the
        encoder is used without an image decoder.
    """

    token: Tensor
    spatial: Tensor | list[Tensor] | None = None


class MorphEmFrameEncoder(nn.Module):
    """Pretrained MorphEm ViT-S as a per-frame encoder.

    The class token becomes the frame token and the patch tokens become the
    spatial features, so a single backbone pass serves both the temporal path and
    the image decoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, by default ``"CaicedoLab/MorphEm"``.
    freeze : bool
        Freeze the backbone and keep it in eval mode, by default ``False``.
        Fine-tuning is worth substantially more than frozen features here, but
        costs roughly ten times the compute.
    img_size : int
        Side length the backbone runs at, by default 224.

    Attributes
    ----------
    token_dim : int
        Width of the frame token.
    spatial_dim : int
        Width of each patch token.
    grid_size : int
        Number of patches per side.
    """

    def __init__(self, model_name: str = MORPHEM_DEFAULT, freeze: bool = False, img_size: int = 224) -> None:
        super().__init__()
        self.backbone = MorphEmModel(model_name=model_name, freeze=freeze, img_size=img_size)
        self.token_dim = self.backbone.embed_dim
        self.spatial_dim = self.backbone.embed_dim
        self.grid_size = self.backbone.grid_size

    def forward(self, x: Tensor) -> FrameFeatures:
        """Encode a batch of single frames.

        Parameters
        ----------
        x : Tensor
            ``(N, 1, Z, Y, X)`` or ``(N, 1, Y, X)``.

        Returns
        -------
        FrameFeatures
            ``token`` of shape ``(N, token_dim)`` and ``spatial`` patch tokens of
            shape ``(N, grid_size ** 2, spatial_dim)``.
        """
        class_token, patch_tokens = self.backbone(self.backbone.preprocess_2d(x))
        return FrameFeatures(token=class_token, spatial=patch_tokens)


class UNeXt2FrameEncoder(nn.Module):
    """UNeXt2 stem plus a timm multi-scale backbone as a per-frame encoder.

    Mirrors the encoder half of :class:`viscy_models.unet.unext2.UNeXt2`, so the
    spatial features line up with :class:`~dynaformer.models.decoders.UNeXt2ImageDecoder`.

    The frame token is built by reducing the deepest feature map to ``token_channels``
    with a 1x1 convolution and flattening it, rather than by global average pooling.
    Keeping the bottleneck's spatial layout lets the temporal model reason about
    where signal sits within the frame, which matters when the crop is centered on a
    tracked object.

    Parameters
    ----------
    in_channels : int
        Number of input channels per frame.
    in_stack_depth : int
        Z depth of the input, by default 1.
    backbone : str
        timm model name, by default ``"convnextv2_tiny"``.
    pretrained : bool
        Load ImageNet weights for the backbone, by default ``False``.
    stem_kernel_size : tuple[int, int, int]
        Stem convolution kernel and stride, by default ``(1, 4, 4)`` (2D input).
    yx_patch_size : tuple[int, int]
        YX size of the frames the encoder will see. Only used to size the token
        projection, by default ``(256, 256)``.
    token_dim : int
        Width of the frame token, by default 512.
    token_channels : int
        Channel width the deepest feature map is reduced to before flattening,
        by default 64.
    drop_path_rate : float
        Stochastic depth rate for the backbone, by default 0.

    Attributes
    ----------
    feature_channels : list[int]
        Channel counts of the multi-scale features, deepest first.
    bottleneck_yx_size : tuple[int, int]
        Spatial size of the deepest feature map, derived from ``yx_patch_size``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_stack_depth: int = 1,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (1, 4, 4),
        yx_patch_size: tuple[int, int] = (256, 256),
        token_dim: int = 512,
        token_channels: int = 64,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"in_stack_depth {in_stack_depth} is not divisible by stem kernel depth {stem_kernel_size[0]}."
            )
        encoder_stages = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, drop_path_rate=drop_path_rate
        )
        channels = encoder_stages.feature_info.channels()
        # The stem replaces the backbone's own downsampling patchify layer.
        encoder_stages.stem_0 = nn.Identity()
        self.encoder_stages = encoder_stages
        self.stem = UNeXt2Stem(in_channels, channels[0], stem_kernel_size, in_stack_depth)
        self.feature_channels = list(reversed(channels))
        self.stem_stride = stem_kernel_size[-1]

        # The stem downsamples by its own stride; every stage after the first
        # halves again, so the deepest map is this much smaller than the input.
        total_stride = self.stem_stride * 2 ** (len(channels) - 1)
        if any(size % total_stride for size in yx_patch_size):
            raise ValueError(
                f"yx_patch_size {tuple(yx_patch_size)} is not divisible by the total stride {total_stride}."
            )
        self.bottleneck_yx_size = tuple(size // total_stride for size in yx_patch_size)

        self.token_channels = token_channels
        self.token_dim = token_dim
        self.reduce = nn.Conv2d(self.feature_channels[0], token_channels, kernel_size=1)
        self.token_proj = nn.Linear(token_channels * self.bottleneck_yx_size[0] * self.bottleneck_yx_size[1], token_dim)

    def forward(self, x: Tensor) -> FrameFeatures:
        """Encode a batch of single frames.

        Parameters
        ----------
        x : Tensor
            ``(N, C, Z, Y, X)``.

        Returns
        -------
        FrameFeatures
            ``token`` of shape ``(N, token_dim)`` and ``spatial`` as the
            bottleneck-first list of multi-scale feature maps.
        """
        features: list[Tensor] = self.encoder_stages(self.stem(x))
        features.reverse()
        bottleneck = features[0]
        if bottleneck.shape[-2:] != self.bottleneck_yx_size:
            raise ValueError(
                f"Encoder built for a {self.bottleneck_yx_size} bottleneck but got "
                f"{tuple(bottleneck.shape[-2:])}; set yx_patch_size to the frame size the model receives."
            )
        token = self.token_proj(self.reduce(bottleneck).flatten(1))
        return FrameFeatures(token=token, spatial=features)
