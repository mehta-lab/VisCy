"""Head modules that project 2D feature maps back to 3D output volumes."""

from monai.networks.blocks import Convolution, UpSample
from monai.networks.utils import normal_init
from torch import Tensor, nn

from viscy_models._components.blocks import icnr_init

__all__ = ["PixelToVoxelHead", "UnsqueezeHead", "PixelToVoxelShuffleHead"]


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
        normal_init(self.conv[0])
        icnr_init(self.conv[-1], 2, upsample_dims=2)
        self.out = nn.PixelShuffle(2)
        self.out_stack_depth = out_stack_depth

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        d = self.out_stack_depth + 2
        b, c, h, w = x.shape
        x = x.reshape((b, c // d, d, h, w))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.out(x)
        return x.transpose(1, 2)


class UnsqueezeHead(nn.Module):
    """Unsqueeze 2D (B, C, H, W) feature map to 3D (B, C, 1, H, W) output."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2)
        return x


class PixelToVoxelShuffleHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_stack_depth: int = 5,
        xy_scaling: int = 4,
        pool: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.out_stack_depth = out_stack_depth
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_stack_depth * out_channels,
            scale_factor=xy_scaling,
            mode="pixelshuffle",
            pre_conv=None,
            apply_pad_pool=pool,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        b, _, h, w = x.shape
        x = x.reshape(b, self.out_channels, self.out_stack_depth, h, w)
        return x
