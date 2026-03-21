"""3D U-Net following Ounkomol et al. 2018 (F-Net).

Faithful port of the recursive encoder-decoder architecture from
``pytorch_fnet`` for label-free prediction of 3D fluorescence images
from transmitted-light microscopy.

Reference
---------
Ounkomol, C., Seshamani, S., Maleckar, M.M. et al. Label-free prediction
of three-dimensional fluorescence images from transmitted-light microscopy.
Nat Methods 15, 917-920 (2018). https://doi.org/10.1038/s41592-018-0111-2
"""

import torch
from torch import Tensor, nn

__all__ = ["Unet3d"]


def _fnet_weights_init(m: nn.Module) -> None:
    """Apply F-Net weight initialization.

    Parameters
    ----------
    m : nn.Module
        Module to initialize.
    """
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class _DoubleConv3d(nn.Module):
    """Two consecutive Conv3d(3x3x3) → BatchNorm3d → ReLU blocks."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class _FNetRecurse(nn.Module):
    """Recursive U-Net level.

    At the top level (``depth == depth_parent``), output channels equal
    ``mult_chan``. At deeper levels, channels double via ``mult_chan=2``.

    Parameters
    ----------
    n_in_channels : int
        Input channels to this level.
    mult_chan : int
        Channel multiplier. 32 at top level, 2 for recursive calls.
    depth_parent : int
        Total depth of the network (used to detect the top level).
    depth : int
        Remaining recursion depth. 0 = leaf (double conv only).
    """

    def __init__(self, n_in_channels: int, mult_chan: int = 2, depth_parent: int = 0, depth: int = 0) -> None:
        super().__init__()
        self.depth = depth

        if self.depth == depth_parent:
            n_out_channels = mult_chan
        else:
            n_out_channels = n_in_channels * mult_chan

        self.sub_2conv_more = _DoubleConv3d(n_in_channels, n_out_channels)
        if depth > 0:
            self.sub_2conv_less = _DoubleConv3d(2 * n_out_channels, n_out_channels)
            self.conv_down = nn.Conv3d(n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn0 = nn.BatchNorm3d(n_out_channels)
            self.relu0 = nn.ReLU(inplace=True)
            self.convt = nn.ConvTranspose3d(2 * n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = nn.BatchNorm3d(n_out_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.sub_u = _FNetRecurse(n_out_channels, mult_chan=2, depth_parent=depth_parent, depth=depth - 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through one recursive level."""
        if self.depth == 0:
            return self.sub_2conv_more(x)
        x_2conv = self.sub_2conv_more(x)
        x_down = self.relu0(self.bn0(self.conv_down(x_2conv)))
        x_sub = self.sub_u(x_down)
        x_up = self.relu1(self.bn1(self.convt(x_sub)))
        x_cat = torch.cat((x_2conv, x_up), dim=1)
        return self.sub_2conv_less(x_cat)


class Unet3d(nn.Module):
    """3D U-Net following Ounkomol et al. 2018 (F-Net).

    Recursive encoder-decoder with concatenation skip connections.
    Uses strided ``Conv3d`` for downsampling and ``ConvTranspose3d`` for
    upsampling in all three spatial dimensions.

    All spatial dimensions (Z, Y, X) must be divisible by ``2**depth``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    depth : int
        Recursion depth (number of downsampling levels).
    mult_chan : int
        Base channel count at the first encoder level.
    in_stack_depth : int or None
        Z-window size. Stored for engine compatibility
        (``example_input_array``, ``DivisiblePad``, sliding window prediction).
        The model itself handles arbitrary Z as long as it is divisible
        by ``2**depth``.
    """

    downsamples_z: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        mult_chan: int = 32,
        in_stack_depth: int | None = None,
    ) -> None:
        super().__init__()
        self.in_stack_depth = in_stack_depth
        self.num_blocks = depth
        self.out_stack_depth = in_stack_depth
        self._divisor = 2**depth

        self.net_recurse = _FNetRecurse(
            n_in_channels=in_channels,
            mult_chan=mult_chan,
            depth_parent=depth,
            depth=depth,
        )
        self.conv_out = nn.Conv3d(mult_chan, out_channels, kernel_size=3, padding=1)
        self.apply(_fnet_weights_init)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C, Z, Y, X)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, out_channels, Z, Y, X)``.

        Raises
        ------
        ValueError
            If any spatial dimension is not divisible by ``2**depth``.
        """
        for dim, name in zip(x.shape[2:], ("Z", "Y", "X")):
            if dim % self._divisor != 0:
                raise ValueError(
                    f"{name} dimension {dim} is not divisible by 2**depth={self._divisor}. "
                    f"All spatial dimensions must be divisible by {self._divisor}."
                )
        return self.conv_out(self.net_recurse(x))
