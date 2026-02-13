"""Shared encoder/decoder building blocks and initialization utilities."""

from typing import Callable, Literal, Sequence

import timm
import torch
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch import Tensor, nn

__all__ = ["icnr_init", "UNeXt2UpStage", "UNeXt2Decoder"]


def icnr_init(
    conv: nn.Module,
    upsample_factor: int,
    upsample_dims: int,
    init: Callable = nn.init.kaiming_normal_,
):
    """ICNR initialization for 2D/3D kernels.

    Adapted from Aitken et al., 2017,
    "Checkerboard artifact free sub-pixel convolution".

    Adapted from MONAI v1.2.0, added support for upsampling dimensions
    that are not the same as the kernel dimension.

    :param conv: convolution layer
    :param upsample_factor: upsample factor
    :param upsample_dims: upsample dimensions, 2 or 3
    :param init: initialization function
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


class UNeXt2UpStage(nn.Module):
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
        """Forward pass combining upsampled features with skip connection.

        :param Tensor inp: Low resolution features
        :param Tensor skip: High resolution skip connection features
        :return Tensor: High resolution features
        """
        inp = self.upsample(inp)
        inp = torch.cat([inp, skip], dim=1)
        return self.conv(inp)


class UNeXt2Decoder(nn.Module):
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
        feat = features[0]
        # padding
        features.append(None)
        for skip, stage in zip(features[1:], self.decoder_stages):
            feat = stage(feat, skip)
        return feat
