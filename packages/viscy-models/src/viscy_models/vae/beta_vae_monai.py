"""Beta-VAE with MONAI VarAutoEncoder backend."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Literal

from monai.networks.layers.factories import Norm
from monai.networks.nets import VarAutoEncoder
from torch import Tensor, nn


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
