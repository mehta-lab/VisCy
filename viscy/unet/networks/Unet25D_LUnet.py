from typing import Literal, Sequence

import timm
import torch
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch import nn

# TODO: modify the encoder with vanilla Conv2D as mode: 'deconv' is broken
# TODO: Modify the multiscale encoder and replace the ConvNext backbone
# TODO: check the decoder is the same as in 25D Unet

# Questions
# TODO: what does self.head in torch do?


class Conv25d_LUnetStem(nn.Module):
    """Stem for 2.5D_LUnet networks."""

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
            # stride=kernel_size,
            padding=(0, kernel_size[1] // 2, kernel_size[2] // 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)


class Unet2dDownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        mode: Literal["vanilla"],
        conv_blocks: int,
        norm_name: str,
        down_mode: Literal["maxpool", "avgpool", None] = "avgpool",
    ) -> None:
        super().__init__()
        spatial_dims = 2
        if mode == "vanilla":
            if down_mode == "maxpool":
                self.down_mode = nn.MaxPool2d(kernel_size=2)
            elif down_mode == "avgpool":
                self.down_mode = nn.AvgPool2d(kernel_size=2)
            else:
                self.down_mode = None

            self.conv = nn.Sequential(
                ResidualUnit(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm=norm_name,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                # self.down_mode,
            )

        else:
            NotImplementedError("only vanilla encoder is implemented")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        skip = self.conv(inp)
        if self.down_mode is not None:
            out = self.down_mode(skip)
            return out, skip
        return skip, None


class Unet2dUpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        mode: Literal["deconv"],
        conv_blocks: int,
        norm_name: str,
    ) -> None:
        super().__init__()
        spatial_dims = 2
        if mode == "deconv":
            self.upsample = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=scale_factor,
                kernel_size=scale_factor,
                norm=norm_name,
                is_transposed=True,
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
        else:
            raise NotImplementedError("only deconv method is implemented")

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


class Unet2dEncoder(nn.Module):
    def __init__(
        self,
        num_filters: list[int],
        norm_name: str,
        mode: Literal["vanilla"],
        conv_blocks: int,
        strides: list[int],
    ) -> None:
        super().__init__()
        self.encoder_stages = nn.ModuleList([])
        stages = len(num_filters)
        for i in range(stages - 1):
            stride = strides[i]
            stage = Unet2dDownStage(
                in_channels=num_filters[i],
                out_channels=num_filters[i + 1],
                scale_factor=stride,
                mode=mode,
                conv_blocks=conv_blocks,
                norm_name=norm_name,
                down_mode="avgpool" if i != (stages - 2) else None,
            )
            self.encoder_stages.append(stage)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # padding
        features = []
        for stage in self.encoder_stages:
            inp, skip = stage(inp)
            features.append(skip)
        return inp, features


class Unet2dDecoder(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        out_channels: int,
        norm_name: str,
        mode: Literal["deconv"],
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

    def forward(
        self, feature, skip_connections: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        feat = feature
        # padding
        # skip_connections.append(None)
        for skip, stage in zip(skip_connections[1:], self.decoder_stages[:-1]):
            feat = stage(feat, skip)
            # print(feat.shape)
        return self.head(feat)


class Unet25d_LUnet(nn.Module):
    def __name__():
        return "Unet25d_LUnet"

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        in_stack_depth: int = 5,
        out_stack_depth: int = 1,
        pretrained: bool = False,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        decoder_mode: Literal["deconv"] = "deconv",
        decoder_conv_blocks: int = 2,
        decoder_norm_layer: str = "instance",
        drop_path_rate: float = 0.0,
        num_filters: list[int] = [],
    ) -> None:
        super(Unet25d_LUnet, self).__init__()
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

        # 3D Conv block to tokenized
        self.stem = Conv25d_LUnetStem(
            in_channels, num_filters[0], stem_kernel_size, in_stack_depth
        )
        # self.stem_out = nn.Conv2d(
        #     num_filters[0] * in_stack_depth, num_filters[1], kernel_size=(1, 1)
        # )
        self.encoder_stages = Unet2dEncoder(
            num_filters=[num_filters[0]] + num_filters,
            norm_name=decoder_norm_layer,
            mode="vanilla",
            conv_blocks=decoder_conv_blocks - 1,
            strides=[2] * (len(num_filters)),
        )

        decoder_channels = num_filters
        decoder_channels.reverse()
        if out_stack_depth == 1:
            decoder_out_channels = out_channels
            self.head = UnsqueezeHead()
        else:
            decoder_out_channels = (
                out_stack_depth * decoder_channels[-1] // stem_kernel_size[-1] ** 2
            )
            self.head = PixelToVoxelHead(
                decoder_out_channels, out_channels, out_stack_depth
            )

        self.decoder = Unet2dDecoder(
            decoder_channels[:-1],
            decoder_out_channels,
            norm_name=decoder_norm_layer,
            mode=decoder_mode,
            conv_blocks=decoder_conv_blocks,
            strides=[2] * (len(num_filters) - 1),
        )
        self.out_stack_depth = out_stack_depth

    @property
    def num_blocks(self) -> int:
        """2-times downscaling factor of the smallest feature map"""
        return 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skip_first = x
        x, skip_connections = self.encoder_stages(x)
        skip_connections.reverse()
        x = self.decoder(x, skip_connections)
        return self.head(x)


# if __name__ == "__main__":
#     import torch

#     x = torch.rand((4, 1, 5, 256, 256))
#     model = Unet25d_LUnet(
#         in_channels=x.shape[1],
#         out_channels=2,
#         in_stack_depth=x.shape[2],
#         out_stack_depth=1,
#         pretrained=False,
#         stem_kernel_size=(5, 3, 3),
#         decoder_mode="deconv",
#         decoder_conv_blocks=2,
#         decoder_norm_layer="instance",
#         drop_path_rate=0.1,
#         num_filters=[24, 48, 96, 192, 384],
#     )
#     a = model(x)
#     print(a.shape)
