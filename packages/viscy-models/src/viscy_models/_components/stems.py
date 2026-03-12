"""Stem modules that project 3D input volumes into 2D feature maps."""

from torch import Tensor, nn

__all__ = ["UNeXt2Stem", "StemDepthtoChannels"]


class UNeXt2Stem(nn.Module):
    """Stem for UNeXt2 and ContrastiveEncoder networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        in_stack_depth: int,
    ) -> None:
        super().__init__()
        if in_stack_depth < kernel_size[0]:
            raise ValueError(f"in_stack_depth ({in_stack_depth}) must be >= kernel_size[0] ({kernel_size[0]})")
        ratio = in_stack_depth // kernel_size[0]
        if out_channels % ratio != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by in_stack_depth // kernel_size[0] ({ratio})"
            )
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // ratio,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: Tensor):
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)


class StemDepthtoChannels(nn.Module):
    """Stem with 3D convolution that maps depth to channels."""

    def __init__(
        self,
        in_channels: int,
        in_stack_depth: int,
        in_channels_encoder: int,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        stem_stride: tuple[int, int, int] = (5, 4, 4),  # stride for the kernel
    ) -> None:
        super().__init__()
        stem3d_out_channels = self.compute_stem_channels(
            in_stack_depth, stem_kernel_size, stem_stride[0], in_channels_encoder
        )

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=stem3d_out_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
        )

    def compute_stem_channels(
        self,
        in_stack_depth,
        stem_kernel_size,
        stem_stride_depth,
        in_channels_encoder,
    ):
        stem3d_out_depth = (in_stack_depth - stem_kernel_size[0]) // stem_stride_depth + 1
        stem3d_out_channels = in_channels_encoder // stem3d_out_depth
        channel_mismatch = in_channels_encoder - stem3d_out_depth * stem3d_out_channels
        if channel_mismatch != 0:
            raise ValueError(
                f"Stem needs to output {channel_mismatch} more channels "
                "to match the encoder. Adjust the in_stack_depth."
            )
        return stem3d_out_channels

    def forward(self, x: Tensor):
        x = self.conv(x)
        b, c, d, h, w = x.shape
        # project Z/depth into channels
        # return a view when possible (contiguous)
        return x.reshape(b, c * d, h, w)
