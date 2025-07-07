from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.normalization import RMSNorm
from diffusers.utils import is_torch_version
from diffusers.models.activations import get_activation

class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = out_channels
        self.conv2 = nn.Conv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)
        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        norm_type: Optional[str] = None,
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.kernel_size = kernel_size
        stride = 2  # Downsampling stride is fixed to 2

        # Initialize normalization
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(self.channels, eps=eps, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Choose between convolutional or pooling downsampling
        if use_conv:
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels, "out_channels must match channels when using pooling"
            self.conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling block.

        Args:
            hidden_states (torch.Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Downsampled feature map.
        """
        assert hidden_states.shape[1] == self.channels, \
            f"Expected input channels {self.channels}, but got {hidden_states.shape[1]}"

        # Apply normalization if specified
        if self.norm is not None:
            # LayerNorm expects (B, C, D, H, W), but normalizes over C. Permute to (B, D, H, W, C)
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 4, 1))
            hidden_states = hidden_states.permute(0, 4, 1, 2, 3)  # Back to (B, C, D, H, W)

        # Apply padding if using conv downsampling and no padding was specified
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 0, 1)  # Padding for 3D tensor: (D, H, W)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        # Apply downsampling
        hidden_states = self.conv(hidden_states)

        return hidden_states

class Upsample3D(nn.Module):
    """A 3D upsampling layer with a convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(self.channels, eps=eps, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if kernel_size is None:
            kernel_size = 3
        conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels, f"Expected {self.channels} channels, got {hidden_states.shape[1]}"

        # Apply normalization if specified
        if self.norm is not None:
            # LayerNorm expects (B, C, D, H, W), but normalizes over C. Permute to (B, D, H, W, C)
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 4, 1))
            hidden_states = hidden_states.permute(0, 4, 1, 2, 3)  # Back to (B, C, D, H, W)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if output_size is None:
            B, C, D, H, W = hidden_states.shape
            if B*C*D*H*W*8 > 2_100_000_000:
                x1 = F.interpolate(hidden_states[:, :, :D//2], scale_factor=2, mode="nearest")
                x2 = F.interpolate(hidden_states[:, :, D//2:], scale_factor=2, mode="nearest")
                hidden_states = torch.cat([x1, x2], dim=2)
            else:
                hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="nearest")
            # hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states

class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        if attn_groups is None:
            attn_groups = resnet_groups

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
            ), 
        ]

        for _ in range(num_layers):
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states)

        return hidden_states