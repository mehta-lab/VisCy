from typing import Literal, Sequence

import timm
import torch
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch import nn


def _get_convnext_stage(in_channels: int, out_channels: int, depth: int):
    return timm.models.convnext.ConvNeXtStage(
        in_chs=in_channels,
        out_chs=out_channels,
        stride=1,
        depth=depth,
        ls_init_value=None,
        use_grn=True,
        norm_layer=timm.layers.LayerNorm2d,
        norm_layer_cl=timm.layers.LayerNorm,
    )


class Conv21dStem(nn.Module):
    """Stem for 2.1D networks."""

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

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)


class Unet2dUpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        norm_name: str,
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
            self.upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=scale_factor,
                mode=mode,
                pre_conv="default",
                apply_pad_pool=True,
            )
            self.conv = _get_convnext_stage(
                out_channels + out_channels, out_channels, conv_blocks
            )
            self.conv.apply(timm.models.convnext._init_weights)

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor inp: Low resolution features
        :param torch.Tensor skip: High resolution skip connection features
        :return torch.Tensor: High resolution features
        """
        inp = self.upsample(inp)
        inp = torch.cat([inp, skip], dim=1)
        return self.conv(inp)


class PixelToVoxelHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int,
        conv_blocks: int,
    ) -> None:
        super().__init__()
        self.norm = timm.layers.LayerNorm2d(num_channels=in_channels)
        self.gelu = nn.GELU()
        self.conv = nn.Conv3d(
            in_channels // out_stack_depth,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_stack_depth = out_stack_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gelu(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c // self.out_stack_depth, self.out_stack_depth, h, w))
        x = self.conv(x)
        return x


class UnsqueezeHead(nn.Module):
    """Unsqueeze 2D (B, C, H, W) feature map to 3D (B, C, 1, H, W) output"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        return x


class Unet2dDecoder(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        out_channels: int,
        norm_name: str,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        strides: list[int],
    ) -> None:
        super().__init__()
        self.decoder_stages = nn.ModuleList([])
        stages = len(num_channels)
        num_channels.append(num_channels[-1])
        for i in range(stages):
            stride = strides[i]
            stage = Unet2dUpStage(
                in_channels=num_channels[i],
                out_channels=num_channels[i + 1],
                scale_factor=stride,
                mode=mode,
                conv_blocks=conv_blocks,
                norm_name=norm_name,
            )
            self.decoder_stages.append(stage)
        self.head = UpSample(
            spatial_dims=2,
            in_channels=num_channels[-1],
            out_channels=out_channels,
            scale_factor=strides[-1],
            mode=mode,
            pre_conv="default",
            apply_pad_pool=False,
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        feat = features[0]
        # padding
        features.append(None)
        for skip, stage in zip(features[1:], self.decoder_stages[:-1]):
            feat = stage(feat, skip)
        return self.head(feat)


class Unet21d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        in_stack_depth: int = 5,
        out_stack_depth: int = 1,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        decoder_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
    ) -> None:
        super().__init__()
        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"Input stack depth {in_stack_depth} is not divisible "
                f"by stem kernel depth {stem_kernel_size[0]}."
            )
        if not (in_stack_depth == out_stack_depth or out_stack_depth == 1):
            raise ValueError(
                "`out_stack_depth` must be either 1 or "
                f"the same as `input_stack_depth` ({in_stack_depth}), "
                f"but got {out_stack_depth}."
            )
        multi_scale_encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        num_channels = multi_scale_encoder.feature_info.channels()
        # replace first convolution layer with a projection tokenizer
        multi_scale_encoder.stem_0 = nn.Identity()
        self.encoder_stages = multi_scale_encoder
        self.stem = Conv21dStem(
            in_channels, num_channels[0], stem_kernel_size, in_stack_depth
        )
        decoder_channels = num_channels
        decoder_channels.reverse()
        if out_stack_depth == 1:
            decoder_out_channels = out_channels
            self.head = UnsqueezeHead()
        else:
            decoder_out_channels = (
                out_stack_depth * decoder_channels[-1] // stem_kernel_size[-1] ** 2
            )
            self.head = PixelToVoxelHead(
                decoder_out_channels,
                out_channels,
                out_stack_depth,
                conv_blocks=decoder_conv_blocks,
            )
        self.decoder = Unet2dDecoder(
            decoder_channels,
            decoder_out_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * len(num_channels) + [stem_kernel_size[-1]],
        )
        self.out_stack_depth = out_stack_depth

    @property
    def num_blocks(self) -> int:
        """2-times downscaling factor of the smallest feature map"""
        return 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        return self.head(x)
