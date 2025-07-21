from types import SimpleNamespace
from typing import Callable, Literal

import timm
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch import Tensor, nn

from viscy.unet.networks.unext2 import (
    PixelToVoxelHead,
    StemDepthtoChannels,
)


class VaeUpStage(nn.Module):
    """VAE upsampling stage without skip connections."""

    def __init__(
        self,
        in_channels: int,
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
            self.upsample = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=scale_factor,
                kernel_size=scale_factor,
                norm=norm_name,
                is_transposed=True,
            )
            # Simple conv blocks for deconv mode
            self.conv = nn.Sequential(
                ResidualUnit(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm=norm_name,
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
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
            conv_layers = []
            current_channels = mid_channels

            for i in range(conv_blocks):
                block_out_channels = out_channels
                conv_layers.extend(
                    [
                        nn.Conv2d(
                            current_channels,
                            block_out_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        (
                            nn.BatchNorm2d(block_out_channels)
                            if norm_name == "batch"
                            else nn.InstanceNorm2d(block_out_channels)
                        ),
                        nn.ReLU(inplace=True),
                    ]
                )
                current_channels = block_out_channels

            self.conv = nn.Sequential(*conv_layers)

    def forward(self, inp: Tensor) -> Tensor:
        """
        :param Tensor inp: Low resolution features
        :return Tensor: High resolution features
        """
        inp = self.upsample(inp)
        return self.conv(inp)


class VaeEncoder(nn.Module):
    """VAE encoder for microscopy data with 3D to 2D conversion."""

    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 2,
        in_stack_depth: int = 32,
        embedding_dim: int = 128,
        stem_kernel_size: tuple[int, int, int] = (8, 4, 4),
        stem_stride: tuple[int, int, int] = (8, 2, 2),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim

        encoder = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )

        if "resnet" in backbone:
            in_channels_encoder = encoder.conv1.out_channels
            # remove the original 3D stem for rgb imges to support the multichannel 3D input
            encoder.conv1 = nn.Identity()
            out_channels_encoder = encoder.feature_info.channels()[-1]
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # Stem for 3d multichannel and to convert 3D to 2D
        self.stem = StemDepthtoChannels(
            in_channels=in_channels,
            in_stack_depth=in_stack_depth,
            in_channels_encoder=in_channels_encoder,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
        )
        self.encoder = encoder

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_mu = nn.Linear(out_channels_encoder, embedding_dim)
        self.fc_logvar = nn.Linear(out_channels_encoder, embedding_dim)

    def forward(self, x: Tensor) -> dict:
        """Forward pass returning VAE encoder outputs."""
        x = self.stem(x)

        features = self.encoder(x)

        # Take highest resolution features
        x = features[-1]
        x = self.global_pool(x)
        x = x.flatten(1)

        # VAE outputs
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return SimpleNamespace(embedding=mu, log_covariance=logvar)


class VaeDecoder(nn.Module):
    """VAE decoder for microscopy data with 2D to 3D conversion."""

    def __init__(
        self,
        decoder_channels: list[int] = [1024, 512, 256, 128],
        latent_dim: int = 128,
        out_channels: int = 2,
        out_stack_depth: int = 20,
        latent_spatial_size: int = 8,
        head_expansion_ratio: int = 4,
        head_pool: bool = False,
        upsample_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        conv_blocks: int = 2,
        norm_name: str = "batch",
        upsample_pre_conv: Literal["default"] | Callable | None = None,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth
        self.latent_spatial_size = latent_spatial_size

        head_channels = (
            (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        )

        # Copy decoder_channels to avoid modifying the original list
        decoder_channels_with_head = decoder_channels.copy() + [head_channels]

        # Set optimal default strides for ResNet50 if not provided
        num_stages = len(decoder_channels_with_head) - 1
        if strides is None:
            if (
                num_stages == 4
            ):  # Default [1024, 512, 256, 128] + head = 5 channels, 4 stages
                strides = [2, 2, 2, 4]  # 8→16→32→64→256 (32x total upsampling)
            else:
                strides = [2] * num_stages  # Fallback to uniform 2x upsampling
        elif len(strides) != num_stages:
            raise ValueError(
                f"Length of strides ({len(strides)}) must match number of stages ({num_stages})"
            )

        # Project latent vector to first feature map
        self.latent_proj = nn.Linear(
            latent_dim,
            decoder_channels_with_head[0] * latent_spatial_size * latent_spatial_size,
        )

        # Build the decoder stages
        self.decoder_stages = nn.ModuleList()

        for i in range(num_stages):
            in_channels = decoder_channels_with_head[i]
            out_channels_stage = decoder_channels_with_head[i + 1]
            stride = strides[i]

            stage = VaeUpStage(
                in_channels=in_channels,
                out_channels=out_channels_stage,
                scale_factor=stride,
                mode=upsample_mode,
                conv_blocks=conv_blocks,
                norm_name=norm_name,
                upsample_pre_conv=upsample_pre_conv,
            )
            self.decoder_stages.append(stage)

        # Head to convert back to 3D (no final_conv needed - last stage outputs head_channels)
        self.head = PixelToVoxelHead(
            in_channels=head_channels,
            out_channels=self.out_channels,
            out_stack_depth=self.out_stack_depth,
            expansion_ratio=head_expansion_ratio,
            pool=head_pool,
        )

    def forward(self, z: Tensor) -> dict:
        """Forward pass converting latent to 3D output."""
        batch_size = z.shape[0]

        # Project latent to feature map
        x = self.latent_proj(z)
        x = x.view(batch_size, -1, self.latent_spatial_size, self.latent_spatial_size)

        for stage in self.decoder_stages:
            x = stage(x)

        # Last stage outputs head_channels directly - no final_conv needed
        output = self.head(x)

        return output
