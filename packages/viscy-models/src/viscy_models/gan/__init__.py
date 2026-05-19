"""3D PatchGAN discriminators and LSGAN losses for adversarial virtual staining."""

from viscy_models.gan.losses import lsgan_d_loss, lsgan_g_loss
from viscy_models.gan.patchgan3d import MultiScalePatchGAN3D, PatchGAN3D

__all__ = ["MultiScalePatchGAN3D", "PatchGAN3D", "lsgan_d_loss", "lsgan_g_loss"]
