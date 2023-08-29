from typing import Callable, Literal, Optional, Sequence, Union

import timm
import torch
from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.utils import icnr_init, normal_init
from torch import nn


def _get_convnext_stage(
    in_channels: int,
    out_channels: int,
    depth: int,
    upsample_factor: Optional[int] = None,
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
        icnr_init(stage.blocks[-1].mlp.fc2, upsample_factor)
    return stage


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
        skip_channels: int,
        out_channels: int,
        scale_factor: int,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        norm_name: str,
        upsample_pre_conv: Optional[Union[Literal["default"], Callable]],
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
        icnr_init(self.conv[-1], 2)
        self.out = nn.PixelShuffle(2)
        self.out_stack_depth = out_stack_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        return x


class Unet2dDecoder(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        norm_name: str,
        mode: Literal["deconv", "pixelshuffle"],
        conv_blocks: int,
        strides: list[int],
        upsample_pre_conv: Optional[Union[Literal["default"], Callable]],
    ) -> None:
        super().__init__()
        self.decoder_stages = nn.ModuleList([])
        stages = len(num_channels) - 1
        for i in range(stages):
            stage = Unet2dUpStage(
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

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        feat = features[0]
        # padding
        features.append(None)
        for skip, stage in zip(features[1:], self.decoder_stages):
            feat = stage(feat, skip)
        return feat


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
        if not (in_stack_depth == out_stack_depth or out_stack_depth == 1):
            raise ValueError(
                "`out_stack_depth` must be either 1 or "
                f"the same as `input_stack_depth` ({in_stack_depth}), "
                f"but got {out_stack_depth}."
            )
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
        self.stem = Conv21dStem(
            in_channels, num_channels[0], stem_kernel_size, in_stack_depth
        )
        decoder_channels = num_channels
        decoder_channels.reverse()
        decoder_channels[-1] = (
            (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        )
        self.decoder = Unet2dDecoder(
            decoder_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(num_channels) - 1) + [stem_kernel_size[-1]],
            upsample_pre_conv="default" if decoder_upsample_pre_conv else None,
        )
        if out_stack_depth == 1:
            self.head = UnsqueezeHead()
        else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        return self.head(x)
