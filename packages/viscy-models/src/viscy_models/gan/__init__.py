"""3D PatchGAN discriminators and GAN losses for adversarial virtual staining."""

from viscy_models.gan.losses import (
    lsgan_d_loss,
    lsgan_g_loss,
    nonsat_d_loss,
    nonsat_g_loss,
    r1_penalty,
    r2_penalty,
    rpgan_d_loss,
    rpgan_g_loss,
)
from viscy_models.gan.patchgan3d import MultiScalePatchGAN3D, PatchGAN3D

__all__ = [
    "MultiScalePatchGAN3D",
    "PatchGAN3D",
    "lsgan_d_loss",
    "lsgan_g_loss",
    "nonsat_d_loss",
    "nonsat_g_loss",
    "r1_penalty",
    "r2_penalty",
    "rpgan_d_loss",
    "rpgan_g_loss",
]
