from collections.abc import Sequence
from types import SimpleNamespace
from typing import Callable, Literal

import timm
import torch
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.factories import Norm
from monai.networks.nets import VarAutoEncoder
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
        norm_name: Literal["batch", "instance"],
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
                conv_layers.append(
                    ResidualUnit(
                        spatial_dims=spatial_dims,
                        in_channels=current_channels,
                        out_channels=block_out_channels,
                        norm=norm_name,
                    )
                )
                current_channels = block_out_channels

            self.conv = nn.Sequential(*conv_layers)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : Tensor
            Low resolution features

        Returns
        -------
        Tensor
            High resolution features
        """
        inp = self.upsample(inp)
        return self.conv(inp)


class VaeEncoder(nn.Module):
    """VAE encoder for microscopy data with 3D to 2D conversion."""

    def __init__(
        self,
        backbone: Literal["resnet50", "convnext_tiny"] = "resnet50",
        in_channels: int = 2,
        in_stack_depth: int = 16,
        latent_dim: int = 1024,
        input_spatial_size: tuple[int, int] = (256, 256),
        stem_kernel_size: tuple[int, int, int] = (2, 4, 4),
        stem_stride: tuple[int, int, int] = (2, 4, 4),
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim

        encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        num_channels = encoder.feature_info.channels()
        in_channels_encoder = num_channels[0]
        out_channels_encoder = num_channels[-1]

        if "convnext" in backbone:
            num_channels = encoder.feature_info.channels()
            encoder.stem_0 = nn.Identity()
        elif "resnet" in backbone:
            encoder.conv1 = nn.Identity()
            out_channels_encoder = num_channels[-1]
        else:
            raise ValueError(
                f"Backbone {backbone} not supported. Use 'resnet50', 'convnext_tiny', or 'convnextv2_tiny'"
            )

        # Stem for 3d multichannel and to convert 3D to 2D
        self.stem = StemDepthtoChannels(
            in_channels=in_channels,
            in_stack_depth=in_stack_depth,
            in_channels_encoder=in_channels_encoder,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
        )
        self.encoder = encoder
        self.num_channels = num_channels
        self.in_channels_encoder = in_channels_encoder
        self.out_channels_encoder = out_channels_encoder

        # Calculate spatial size after stem
        stem_spatial_size_h = input_spatial_size[0] // stem_stride[1]
        stem_spatial_size_w = input_spatial_size[1] // stem_stride[2]

        # Spatial size after backbone
        backbone_reduction = 2 ** (len(num_channels) - 1)
        final_spatial_size_h = stem_spatial_size_h // backbone_reduction
        final_spatial_size_w = stem_spatial_size_w // backbone_reduction

        flattened_size = (
            out_channels_encoder * final_spatial_size_h * final_spatial_size_w
        )

        self.fc = nn.Linear(flattened_size, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Store final spatial size for decoder (assuming square for simplicity)
        self.encoder_spatial_size = final_spatial_size_h  # Assuming square output

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> SimpleNamespace:
        """Forward pass returning VAE encoder outputs."""
        x = self.stem(x)

        features = self.encoder(x)

        # NOTE: taking the highest semantic features and flatten
        # When features_only=False, encoder returns single tensor, not list
        if isinstance(features, list):
            x = features[-1]  # [B, C, H, W]
        else:
            x = features  # [B, C, H, W]
        x_flat = x.flatten(1)  # [B, C*H*W] - flatten from dim 1 onwards

        x_intermediate = self.fc(x_flat)

        mu = self.fc_mu(x_intermediate)
        logvar = self.fc_logvar(x_intermediate)
        z = self.reparameterize(mu, logvar)

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
        strides: list[int] = [2, 2, 2, 1],
        encoder_spatial_size: int = 16,
        head_pool: bool = False,
        upsample_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        conv_blocks: int = 2,
        norm_name: Literal["batch", "instance"] = "batch",
        upsample_pre_conv: Literal["default"] | Callable | None = None,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth

        self.spatial_size = encoder_spatial_size
        self.spatial_channels = latent_dim // (self.spatial_size * self.spatial_size)

        self.latent_reshape = nn.Linear(
            latent_dim, self.spatial_channels * self.spatial_size * self.spatial_size
        )
        self.latent_proj = nn.Conv2d(
            self.spatial_channels, decoder_channels[0], kernel_size=1
        )

        # Build the decoder stages
        self.decoder_stages = nn.ModuleList()
        num_stages = len(self.decoder_channels) - 1
        for i in range(num_stages):
            stage = VaeUpStage(
                in_channels=self.decoder_channels[i],
                out_channels=self.decoder_channels[i + 1],
                scale_factor=strides[i],
                mode=upsample_mode,
                conv_blocks=conv_blocks,
                norm_name=norm_name,
                upsample_pre_conv=upsample_pre_conv,
            )
            self.decoder_stages.append(stage)

        # Head to convert back to 3D
        self.head = PixelToVoxelHead(
            in_channels=decoder_channels[-1],
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

        output = self.head(x)

        return output


class BetaVae25D(nn.Module):
    """2.5D Beta-VAE combining VaeEncoder and VaeDecoder."""

    def __init__(
        self,
        backbone: Literal["resnet50", "convnext_tiny"] = "resnet50",
        in_channels: int = 2,
        in_stack_depth: int = 16,
        out_stack_depth: int = 16,
        latent_dim: int = 1024,
        input_spatial_size: tuple[int, int] = (256, 256),
        stem_kernel_size: tuple[int, int, int] = (2, 4, 4),
        stem_stride: tuple[int, int, int] = (2, 4, 4),
        drop_path_rate: float = 0.0,
        decoder_stages: int = 4,
        head_expansion_ratio: int = 2,
        head_pool: bool = False,
        upsample_mode: Literal["deconv", "pixelshuffle"] = "pixelshuffle",
        conv_blocks: int = 2,
        norm_name: Literal["batch", "instance"] = "batch",
        upsample_pre_conv: Literal["default"] | Callable | None = None,
    ):
        super().__init__()

        self.encoder = VaeEncoder(
            backbone=backbone,
            in_channels=in_channels,
            in_stack_depth=in_stack_depth,
            latent_dim=latent_dim,
            input_spatial_size=input_spatial_size,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            drop_path_rate=drop_path_rate,
        )

        base_channels = self.encoder.num_channels[-1]
        decoder_channels = [base_channels]
        for i in range(decoder_stages - 1):
            decoder_channels.append(base_channels // (2 ** (i + 1)))
        decoder_channels.append(
            (out_stack_depth + 2) * in_channels * 2**2 * head_expansion_ratio
        )

        strides = [2] * decoder_stages + [1]

        self.decoder = VaeDecoder(
            decoder_channels=decoder_channels,
            latent_dim=latent_dim,
            out_channels=in_channels,
            out_stack_depth=out_stack_depth,
            head_expansion_ratio=head_expansion_ratio,
            head_pool=head_pool,
            upsample_mode=upsample_mode,
            conv_blocks=conv_blocks,
            norm_name=norm_name,
            upsample_pre_conv=upsample_pre_conv,
            strides=strides,
            encoder_spatial_size=self.encoder.encoder_spatial_size,
        )

    def forward(self, x: Tensor) -> SimpleNamespace:
        """Forward pass returning VAE outputs."""
        encoder_output = self.encoder(x)
        recon_x = self.decoder(encoder_output.z)

        return SimpleNamespace(
            recon_x=recon_x,
            mean=encoder_output.mean,
            logvar=encoder_output.log_covariance,
            z=encoder_output.z,
        )


class BetaVaeMonai(nn.Module):
    """Beta-VAE with Monai architecture."""

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int] | Sequence[Sequence[int]],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        use_sigmoid: bool = False,
        norm: Literal["batch", "instance"] = "instance",
        **kwargs,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.use_sigmoid = use_sigmoid
        self.norm = norm
        if self.norm not in ["batch", "instance"]:
            raise ValueError("norm must be 'batch' or 'instance'")
        if self.norm == "batch":
            self.norm = Norm.BATCH
        else:
            self.norm = Norm.INSTANCE

        self.model = VarAutoEncoder(
            spatial_dims=self.spatial_dims,
            in_shape=self.in_shape,
            out_channels=self.out_channels,
            latent_size=self.latent_size,
            channels=self.channels,
            strides=self.strides,
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            num_res_units=self.num_res_units,
            use_sigmoid=self.use_sigmoid,
            norm=self.norm,
            **kwargs,
        )

    def forward(self, x: Tensor) -> SimpleNamespace:
        """Forward pass returning VAE encoder outputs."""
        recon_x, mu, logvar, z = self.model(x)
        return SimpleNamespace(recon_x=recon_x, mean=mu, logvar=logvar, z=z)
