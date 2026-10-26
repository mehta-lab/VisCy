"""3D U-Net following Ounkomol et al. 2018 (F-Net).

FNet-configured preset of the unified 3D U-Net base.  Uses BatchNorm + ReLU,
non-residual double-conv blocks, and a convolutional bottleneck.  Downsamples
all three spatial dimensions (Z, Y, X).

Reference
---------
Ounkomol, C., Seshamani, S., Maleckar, M.M. et al. Label-free prediction
of three-dimensional fluorescence images from transmitted-light microscopy.
Nat Methods 15, 917-920 (2018). https://doi.org/10.1038/s41592-018-0111-2
"""

from torch import nn

from viscy_models.unet.blocks import ConvBottleneck3D
from viscy_models.unet.unet3d_base import UNet3DBase

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


class Unet3d(UNet3DBase):
    """3D U-Net following Ounkomol et al. 2018 (F-Net).

    FNet-configured preset of the unified 3D U-Net base with BatchNorm,
    ReLU activations, non-residual double-conv blocks, and a convolutional
    bottleneck.  Downsamples Y and X, and (when ``downsample_z=True``) Z.

    Each downsampled spatial dimension must be divisible by ``2**depth``: Y and
    X always, and Z only when ``downsample_z=True``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    depth : int
        Number of downsampling levels.
    mult_chan : int
        Base channel count at the first encoder level.
    in_stack_depth : int or None
        Z-window size. Stored for engine compatibility
        (``example_input_array``, ``DivisiblePad``, sliding window prediction).
        The model itself handles arbitrary Z; when ``downsample_z=True`` the Z
        size must be divisible by ``2**depth`` (no constraint when ``False``).
    downsample_z : bool
        Whether to downsample the Z axis in the encoder (``True`` preserves the
        FNet3D behavior). Set ``False`` for a Z-preserving 2D-style network
        (stride ``(1, 2, 2)``, ``ConvTranspose3d`` kernel ``(1, 3, 3)``), which
        is required to run at ``Z=1`` without tripping the base
        ``D % 2**depth`` divisibility check.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        mult_chan: int = 32,
        in_stack_depth: int | None = None,
        downsample_z: bool = True,
    ) -> None:
        dims = [mult_chan * (2**i) for i in range(depth + 1)]
        bottleneck = ConvBottleneck3D(dims[-1], residual=False, norm="batch", activation="relu")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dims=dims,
            num_res_block=[1] * depth,
            bottleneck=bottleneck,
            downsample_z=downsample_z,
            residual=False,
            norm="batch",
            activation="relu",
        )
        self.in_stack_depth = in_stack_depth
        self.out_stack_depth = in_stack_depth
        self.apply(_fnet_weights_init)
