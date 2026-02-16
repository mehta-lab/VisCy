"""UNeXt2 architecture composing timm encoder with custom stem, decoder, and head."""

from typing import Literal

import timm
from torch import Tensor, nn

from viscy_models._components.blocks import UNeXt2Decoder
from viscy_models._components.heads import PixelToVoxelHead
from viscy_models._components.stems import UNeXt2Stem


class UNeXt2(nn.Module):
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
        """2-times downscaling factor of the smallest feature map."""
        return 6

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x: list = self.encoder_stages(x)
        x.reverse()
        x = self.decoder(x)
        return self.head(x)
