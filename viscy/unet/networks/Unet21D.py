from typing import Sequence, Union

import timm
import torch
from monai.networks.blocks import ResidualUnit, UnetrUpBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch import nn


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


class Unet2dDecoder(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        out_channels: int,
        res_block: bool,
        norm_name: str,
        kernel_size: Union[int, tuple[int, int]],
        last_kernel_size: Union[int, tuple[int, int]],
        dropout: float = 0,
    ) -> None:
        super().__init__()
        decoder_stages = []
        stages = len(num_channels)
        num_channels.append(out_channels)
        stride = 2
        for i in range(stages):
            stage = UnetrUpBlock(
                spatial_dims=2,
                in_channels=num_channels[i],
                out_channels=num_channels[i + 1],
                kernel_size=kernel_size,
                upsample_kernel_size=stride,
                norm_name=norm_name,
                res_block=res_block,
            )
            decoder_stages.append(stage)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.head = nn.Sequential(
            get_conv_layer(
                spatial_dims=2,
                in_channels=num_channels[-2],
                out_channels=num_channels[-2],
                stride=last_kernel_size,
                kernel_size=last_kernel_size,
                norm=norm_name,
                is_transposed=True,
            ),
            ResidualUnit(
                spatial_dims=2,
                in_channels=num_channels[-2],
                out_channels=num_channels[-2],
                kernel_size=kernel_size,
                norm=norm_name,
                dropout=dropout,
            ),
            nn.Conv2d(
                num_channels[-2],
                out_channels,
                kernel_size=(1, 1),
            ),
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
        in_stack_depth: int = 9,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (3, 4, 4),
        decoder_res_block: bool = True,
        decoder_norm_layer: str = "instance",
    ) -> None:
        super().__init__()
        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"Input stack depth {in_stack_depth} is not divisible "
                f"by stem kernel depth {stem_kernel_size[0]}."
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
        self.decoder = Unet2dDecoder(
            decoder_channels,
            out_channels,
            res_block=decoder_res_block,
            norm_name=decoder_norm_layer,
            kernel_size=3,
            last_kernel_size=stem_kernel_size[-2:],
        )
        # shape compatibility
        self.num_blocks = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        # add Z/depth back
        return x.unsqueeze(2)
