"""LSGAN losses for multi-scale 3D PatchGAN discriminators.

Both losses always operate on lists of per-scale logit tensors and average
the per-scale loss across scales (Mao et al. 2017 "Least Squares GANs",
adapted to the multi-scale pix2pixHD convention).
"""

import torch
from torch import Tensor

__all__ = ["lsgan_d_loss", "lsgan_g_loss"]


def lsgan_d_loss(d_real: list[Tensor], d_fake: list[Tensor]) -> Tensor:
    """Multi-scale LSGAN discriminator loss.

    For each scale, computes ``0.5 * (mean((d_real - 1) ** 2) + mean(d_fake ** 2))``,
    then averages across scales.

    Parameters
    ----------
    d_real : list of Tensor
        Per-scale discriminator logits on real (source + target) pairs.
    d_fake : list of Tensor
        Per-scale discriminator logits on fake (source + generator-output)
        pairs. These should come from a detached generator output so the loss
        does not backprop into the generator.

    Returns
    -------
    Tensor
        Scalar loss, mean across scales.
    """
    if len(d_real) != len(d_fake):
        raise ValueError(f"Number of scales must match: len(d_real)={len(d_real)} vs len(d_fake)={len(d_fake)}.")
    if len(d_real) == 0:
        raise ValueError("Discriminator outputs must contain at least one scale.")

    per_scale: list[Tensor] = []
    for real_logits, fake_logits in zip(d_real, d_fake, strict=True):
        real_loss = torch.mean((real_logits - 1.0) ** 2)
        fake_loss = torch.mean(fake_logits**2)
        per_scale.append(0.5 * (real_loss + fake_loss))
    return torch.stack(per_scale).mean()


def lsgan_g_loss(d_fake: list[Tensor]) -> Tensor:
    """Multi-scale LSGAN generator loss.

    For each scale, computes ``mean((d_fake - 1) ** 2)``, then averages
    across scales. ``d_fake`` here is the discriminator's response to the
    *current* (non-detached) generator output, so the loss backpropagates
    through both the discriminator and the generator parameters along the
    chain; the discriminator side is typically frozen via ``requires_grad``
    in the training step.

    Parameters
    ----------
    d_fake : list of Tensor
        Per-scale discriminator logits on fake pairs built from the live
        generator output.

    Returns
    -------
    Tensor
        Scalar loss, mean across scales.
    """
    if len(d_fake) == 0:
        raise ValueError("Discriminator outputs must contain at least one scale.")
    per_scale = [torch.mean((logits - 1.0) ** 2) for logits in d_fake]
    return torch.stack(per_scale).mean()
