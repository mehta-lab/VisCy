from collections.abc import Callable, Sequence
from typing import Literal

import timm
import torch
from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.utils import normal_init
from torch import Tensor, nn


def icnr_init(
    conv: nn.Module,
    upsample_factor: int,
    upsample_dims: int,
    init: Callable = nn.init.kaiming_normal_,
):
    """ICNR initialization for 2D/3D kernels.

    Adapted from Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".
    Adapted from MONAI v1.2.0, added support for upsampling dimensions
    that are not the same as the kernel dimension.

    Parameters
    ----------
    conv : nn.Module
        Convolution layer to initialize.
    upsample_factor : int
        Upsample factor.
    upsample_dims : int
        Upsample dimensions, 2 or 3.
    init : Callable, optional
        Initialization function, by default nn.init.kaiming_normal_.
    """
    out_channels, in_channels, *dims = conv.weight.shape
    scale_factor = upsample_factor**upsample_dims

    oc2 = int(out_channels / scale_factor)

    kernel = torch.zeros([oc2, in_channels] + dims)
    kernel = init(kernel)
    kernel = kernel.transpose(0, 1)
    kernel = kernel.reshape(oc2, in_channels, -1)
    kernel = kernel.repeat(1, 1, scale_factor)
    kernel = kernel.reshape([in_channels, out_channels] + dims)
    kernel = kernel.transpose(0, 1)
    conv.weight.data.copy_(kernel)


def _get_convnext_stage(
    in_channels: int,
    out_channels: int,
    depth: int,
    upsample_factor: int | None = None,
) -> nn.Module:
    stage = timm.models.convnext.ConvNeXtStage(
        in_chs=in_channels,
        out_chs=out_channels,
        stride=1,
        depth=depth,
        ls_init_value=None,
        conv_mlp=True,
        use_grn=True,
        norm_layer=timm.layers.LayerNorm2d,
        norm_layer_cl=timm.layers.LayerNorm,
    )
    stage.apply(timm.models.convnext._init_weights)
    if upsample_factor:
        icnr_init(stage.blocks[-1].mlp.fc2, upsample_factor, upsample_dims=2)
    return stage


class UNeXt2Stem(nn.Module):
    """Stem for UNeXt2 and ContrastiveEncoder networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        in_stack_depth: int,
    ) -> None:
        super().__init__()
        ratio = in_stack_depth // kernel_size[0]
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // ratio,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: Tensor):
        """Forward pass through UNeXt2 stem with depth-to-channel projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) where D is the stack depth.

        Returns
        -------
        torch.Tensor
            Output tensor with depth projected to channels, shape (B, C*D', H', W')
            where D' = D // kernel_size[0] after 3D convolution.
        """
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)


class StemDepthtoChannels(nn.Module):
    """Stem with 3D convolution that maps depth to channels."""

    def __init__(
        self,
        in_channels: int,
        in_stack_depth: int,
        in_channels_encoder: int,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        stem_stride: tuple[int, int, int] = (5, 4, 4),  # stride for the kernel
    ) -> None:
        super().__init__()
        stem3d_out_channels = self.compute_stem_channels(
            in_stack_depth, stem_kernel_size, stem_stride[0], in_channels_encoder
        )

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=stem3d_out_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
        )

    def compute_stem_channels(
        self, in_stack_depth, stem_kernel_size, stem_stride_depth, in_channels_encoder
    ):
        """Compute required 3D stem output channels for encoder compatibility.

        Parameters
        ----------
        in_stack_depth : int
            Input stack depth dimension.
        stem_kernel_size : tuple[int, int, int]
            3D convolution kernel size.
        stem_stride_depth : int
            Stride in the depth dimension.
        in_channels_encoder : int
            Required input channels for the encoder after depth projection.

        Returns
        -------
        int
            Required output channels for the 3D stem convolution.

        Raises
        ------
        ValueError
            If channel dimensions cannot be matched with current configuration.
        """
        stem3d_out_depth = (
            in_stack_depth - stem_kernel_size[0]
        ) // stem_stride_depth + 1
        stem3d_out_channels = in_channels_encoder // stem3d_out_depth
        channel_mismatch = in_channels_encoder - stem3d_out_depth * stem3d_out_channels
        if channel_mismatch != 0:
            raise ValueError(
                f"Stem needs to output {channel_mismatch} more channels to match the encoder. Adjust the in_stack_depth."
            )
        return stem3d_out_channels

    def forward(self, x: Tensor):
        """Forward pass with 3D convolution and depth-to-channel mapping.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) where D is the input stack depth.

        Returns
        -------
        torch.Tensor
            Output tensor with depth projected to channels, maintaining spatial
            dimensions after strided 3D convolution.
        """
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)


class UNeXt2UpStage(nn.Module):
    """UNeXt2 decoder upsampling stage with skip connection fusion.

    Implements hierarchical feature upsampling using either deconvolution or
    pixel shuffle, followed by ConvNeXt blocks for feature refinement. Combines
    low-resolution features with high-resolution skip connections for multi-scale
    feature fusion.

    # TODO: MANUAL_REVIEW - ConvNeXt block integration with skip connections
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        norm_name: str,
        upsample_pre_conv: Literal["default"] | Callable | None,
    ) -> None:
        super().__init__()
        spatial_dims = 2
        if mode == "deconv":
            self.upsample = (
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=scale_factor,
                    kernel_size=scale_factor,
                    norm=norm_name,
                    is_transposed=True,
                ),
            )
            self.conv = nn.Sequential(
                ResidualUnit(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm=norm_name,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            )
        elif mode == "pixelshuffle":
            mid_channels = in_channels // scale_factor**2
            self.upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=mid_channels,
                scale_factor=scale_factor,
                mode=mode,
                pre_conv=upsample_pre_conv,
                apply_pad_pool=False,
            )
            conv_weight_init_factor = None if upsample_pre_conv else scale_factor
            self.conv = _get_convnext_stage(
                mid_channels + skip_channels,
                out_channels,
                conv_blocks,
                upsample_factor=conv_weight_init_factor,
            )

    def forward(self, inp: Tensor, skip: Tensor) -> Tensor:
        """Forward pass with upsampling and skip connection fusion.

        Parameters
        ----------
        inp : torch.Tensor
            Low resolution input features from deeper decoder stage.
        skip : torch.Tensor
            High resolution skip connection features from encoder.

        Returns
        -------
        torch.Tensor
            Upsampled and refined features combining both inputs through
            ConvNeXt blocks or residual units.
        """
        inp = self.upsample(inp)
        inp = torch.cat([inp, skip], dim=1)
        return self.conv(inp)


class PixelToVoxelHead(nn.Module):
    """Head module for converting 2D features to 3D voxel output.

    Performs 2D-to-3D reconstruction using pixel shuffle upsampling and 3D
    convolutions. Applies depth channel expansion and spatial upsampling to
    generate volumetric outputs from 2D feature representations.

    # TODO: MANUAL_REVIEW - 2D to 3D reconstruction mechanism
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int,
        expansion_ratio: int,
        pool: bool,
    ) -> None:
        super().__init__()
        first_scale = 2
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels // first_scale**2,
            scale_factor=first_scale,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )
        mid_channels = out_channels * expansion_ratio * 2**2
        self.conv = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_channels // first_scale**2 // (out_stack_depth + 2),
                out_channels=mid_channels,
                kernel_size=3,
                padding=(0, 1, 1),
            ),
            nn.Conv3d(mid_channels, out_channels * 2**2, 1),
        )
        normal_init(self.conv[0])
        icnr_init(self.conv[-1], 2, upsample_dims=2)
        self.out = nn.PixelShuffle(2)
        self.out_stack_depth = out_stack_depth

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D to 3D voxel reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Input 2D feature tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output 3D voxel tensor with upsampled spatial dimensions and
            reconstructed depth, shape (B, out_channels, out_stack_depth, H', W').
        """
        x = self.upsample(x)
        d = self.out_stack_depth + 2
        b, c, h, w = x.shape
        x = x.reshape((b, c // d, d, h, w))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.out(x)
        return x.transpose(1, 2)


class UnsqueezeHead(nn.Module):
    """Unsqueeze 2D (B, C, H, W) feature map to 3D (B, C, 1, H, W) output"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass adding singleton depth dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input 2D tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output 3D tensor with singleton depth dimension, shape (B, C, 1, H, W).
        """
        x = x.unsqueeze(2)
        return x


class UNeXt2Decoder(nn.Module):
    """UNeXt2 hierarchical decoder with multi-stage upsampling.

    Implements progressive upsampling through multiple UNeXt2UpStage modules,
    combining features from different encoder scales through skip connections.
    Each stage performs feature upsampling and refinement using ConvNeXt blocks.

    # TODO: MANUAL_REVIEW - Multi-scale feature fusion strategy
    """

    def __init__(
        self,
        num_channels: list[int],
        norm_name: str,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        strides: list[int],
        upsample_pre_conv: Literal["default"] | Callable | None,
    ) -> None:
        super().__init__()
        self.decoder_stages = nn.ModuleList([])
        stages = len(num_channels) - 1
        for i in range(stages):
            stage = UNeXt2UpStage(
                in_channels=num_channels[i],
                skip_channels=num_channels[i] // 2,
                out_channels=num_channels[i + 1],
                scale_factor=strides[i],
                mode=mode,
                conv_blocks=conv_blocks,
                norm_name=norm_name,
                upsample_pre_conv=upsample_pre_conv,
            )
            self.decoder_stages.append(stage)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Forward pass through hierarchical decoder stages.

        Parameters
        ----------
        features : Sequence[torch.Tensor]
            List of multi-scale encoder features, ordered from lowest to highest
            resolution. First element is the bottleneck feature.

        Returns
        -------
        torch.Tensor
            Decoded high-resolution features after progressive upsampling and
            skip connection fusion through all decoder stages.
        """
        feat = features[0]
        # padding
        features.append(None)
        for skip, stage in zip(features[1:], self.decoder_stages):
            feat = stage(feat, skip)
        return feat


class UNeXt2(nn.Module):
    """UNeXt2: ConvNeXt-based U-Net for 3D-to-2D-to-3D processing.

    Advanced transformer-inspired U-Net architecture using ConvNeXt backbones
    for hierarchical feature extraction. Performs 3D-to-2D projection via stem,
    2D multi-scale processing through ConvNeXt encoder-decoder, and 2D-to-3D
    reconstruction via specialized head modules.

    # TODO: MANUAL_REVIEW - ConvNeXt transformer integration patterns
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        in_stack_depth: int = 5,
        out_stack_depth: int = None,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        decoder_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
        decoder_upsample_pre_conv: bool = False,
        head_pool: bool = False,
        head_expansion_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"Input stack depth {in_stack_depth} is not divisible "
                f"by stem kernel depth {stem_kernel_size[0]}."
            )
        if out_stack_depth is None:
            out_stack_depth = in_stack_depth
        multi_scale_encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        num_channels = multi_scale_encoder.feature_info.channels()
        # replace first convolution layer with a projection tokenizer
        multi_scale_encoder.stem_0 = nn.Identity()
        self.encoder_stages = multi_scale_encoder
        self.stem = UNeXt2Stem(
            in_channels, num_channels[0], stem_kernel_size, in_stack_depth
        )
        decoder_channels = num_channels
        decoder_channels.reverse()
        decoder_channels[-1] = (
            (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        )
        self.decoder = UNeXt2Decoder(
            decoder_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(num_channels) - 1) + [stem_kernel_size[-1]],
            upsample_pre_conv="default" if decoder_upsample_pre_conv else None,
        )
        self.head = PixelToVoxelHead(
            decoder_channels[-1],
            out_channels,
            out_stack_depth,
            head_expansion_ratio,
            pool=head_pool,
        )
        self.out_stack_depth = out_stack_depth

    @property
    def num_blocks(self) -> int:
        """2-times downscaling factor of the smallest feature map"""
        return 6

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through complete UNeXt2 architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) where D is the input stack depth.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, out_stack_depth, H', W')
            after 3D-to-2D-to-3D processing through ConvNeXt backbone.
        """
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        return self.head(x)
