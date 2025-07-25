from types import SimpleNamespace
from typing import Callable, Literal

import timm
import torch
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

    # TODO: roll back the Conv2d to AveragePooling and linear layer to global pooling
    # TODO: embedding dim
    # TODO: check the OG VAE compression rate
    # TODO do log grid search for the best embedding dim

    def __init__(
        self,
        backbone: str = "resnet50",  # [64, 256, 512, 1024, 2048] channels
        in_channels: int = 2,
        in_stack_depth: int = 16,
        latent_dim: int = 1024,
        input_spatial_size: tuple[int, int] = (256, 256),
        stem_kernel_size: tuple[int, int, int] = (4, 5, 5),
        stem_stride: tuple[int, int, int] = (4, 5, 5),  # same as kernel size
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim

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

        # Calculate spatial dimensions after encoder and initialize linear layers
        self.out_channels_encoder = out_channels_encoder

        if "resnet50" in backbone:
            # Calculate spatial size after stem, then ResNet50 downsampling
            stem_spatial_h = (
                input_spatial_size[0] - stem_kernel_size[1]
            ) // stem_stride[1] + 1
            stem_spatial_w = (
                input_spatial_size[1] - stem_kernel_size[2]
            ) // stem_stride[2] + 1

            # ResNet50 downsamples by 32x total, but stem already downsampled
            total_downsample_factor = 32
            stem_downsample_factor = stem_stride[1]  # Spatial downsampling from stem
            resnet_downsample_factor = total_downsample_factor // stem_downsample_factor
            final_h = stem_spatial_h // resnet_downsample_factor
            final_w = stem_spatial_w // resnet_downsample_factor
            flattened_size = out_channels_encoder * final_h * final_w
        else:
            raise ValueError(
                f"Backbone {backbone} not supported for analytical calculation"
            )

        # Multi-layer perceptron for better representation learning
        self.fc = nn.Linear(flattened_size, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> SimpleNamespace:
        """Forward pass returning VAE encoder outputs."""
        x = self.stem(x)

        features = self.encoder(x)

        # Take highest resolution features and flatten
        x = features[-1]  # [B, C, H, W]
        x_flat = x.flatten(1)  # [B, C*H*W] - flatten from dim 1 onwards

        # Apply intermediate FC layer
        x_intermediate = self.fc(x_flat)  # [B, intermediate_dim]

        # Apply linear layers to get 1D embeddings
        mu = self.fc_mu(x_intermediate)  # [B, latent_dim]
        logvar = self.fc_logvar(x_intermediate)  # [B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        return SimpleNamespace(mean=mu, log_covariance=logvar, z=z)


class VaeDecoder(nn.Module):
    """VAE decoder for microscopy data with 2D to 3D conversion."""

    def __init__(
        self,
        decoder_channels: list[int] = [1024, 512, 256, 128],
        latent_dim: int = 1024,
        out_channels: int = 2,
        out_stack_depth: int = 16,
        head_expansion_ratio: int = 2,
        head_pool: bool = False,
        upsample_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        conv_blocks: int = 2,
        norm_name: str = "batch",
        upsample_pre_conv: Literal["default"] | Callable | None = None,
        strides: list[int] | None = None,
        input_spatial_size: tuple[int, int] = (
            128,
            128,
        ),  # Input size to calculate spatial dimensions
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth

        head_channels = (
            (out_stack_depth + 2) * out_channels * 2**2 * head_expansion_ratio
        )

        decoder_channels_with_head = decoder_channels.copy() + [head_channels]

        num_stages = len(decoder_channels_with_head) - 1
        if strides is None:
            if (
                num_stages == 4
            ):  # Default [1024, 512, 256, 128] + head = 5 channels, 4 stages
                strides = [
                    2,
                    2,
                    2,
                    1,
                ]  # Reduce to account for PixelToVoxelHead's 4x upsampling
            else:
                strides = [2] * num_stages  # Fallback to uniform 2x upsampling
        elif len(strides) != num_stages:
            raise ValueError(
                f"Length of strides ({len(strides)}) must match number of stages ({num_stages})"
            )
        # Calculate spatial size based on input dimensions and ResNet50 32x downsampling
        self.spatial_size = input_spatial_size[0] // 32  # ResNet50 downsamples by 32x
        self.spatial_channels = latent_dim // (self.spatial_size * self.spatial_size)

        # Project 1D latent to spatial format, then to first decoder channels
        self.latent_reshape = nn.Linear(
            latent_dim, self.spatial_channels * self.spatial_size * self.spatial_size
        )
        self.latent_proj = nn.Conv2d(
            self.spatial_channels, decoder_channels_with_head[0], kernel_size=1
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

        # Head to convert back to 3D
        self.head = PixelToVoxelHead(
            in_channels=head_channels,
            out_channels=self.out_channels,
            out_stack_depth=self.out_stack_depth,
            expansion_ratio=head_expansion_ratio,
            pool=head_pool,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass converting latent to 3D output."""

        batch_size = z.size(0)

        # Reshape 1D latent back to spatial format so we can reconstruct the 2.5D image
        z_spatial = self.latent_reshape(z)  # [batch, spatial_channels * H * W]
        z_spatial = z_spatial.view(
            batch_size, self.spatial_channels, self.spatial_size, self.spatial_size
        )

        # Project spatial latent to first decoder channels using 1x1 conv
        x = self.latent_proj(
            z_spatial
        )  # [batch, decoder_channels[0], spatial_H, spatial_W]

        for stage in self.decoder_stages:
            x = stage(x)

        # Last stage outputs head_channels directly - no final_conv needed
        output = self.head(x)

        return output
