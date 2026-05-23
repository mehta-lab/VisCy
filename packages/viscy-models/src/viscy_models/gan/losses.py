"""GAN losses and gradient penalties for multi-scale 3D PatchGAN discriminators.

Includes three loss families and two gradient penalties:

- ``lsgan_*``: Mao et al. 2017 "Least Squares GANs" (legacy default).
- ``nonsat_*``: non-saturating softplus loss (StyleGAN2 default; verified
  verbatim from ``stylegan2-ada-pytorch/training/loss.py``).
- ``rpgan_*``: relativistic pairing loss from Huang et al. 2024 (R3GAN,
  NeurIPS 2024), verified verbatim from ``brownvc/R3GAN/Trainer.py``.
- ``r1_penalty`` / ``r2_penalty``: zero-centered gradient penalties on real
  / fake inputs, Mescheder et al. 2018, with the multi-scale per-scale
  aggregation extended for ``MultiScalePatchGAN3D``.

All losses operate on lists of per-scale logit tensors and average per-scale
loss across scales, matching the multi-scale pix2pixHD convention.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "lsgan_d_loss",
    "lsgan_g_loss",
    "nonsat_d_loss",
    "nonsat_g_loss",
    "r1_penalty",
    "r2_penalty",
    "rpgan_d_loss",
    "rpgan_g_loss",
]


def _validate_scales(d_real: Sequence[Tensor], d_fake: Sequence[Tensor]) -> None:
    """Validate that real and fake scale lists have matching, non-empty length."""
    if len(d_real) != len(d_fake):
        raise ValueError(f"Number of scales must match: len(d_real)={len(d_real)} vs len(d_fake)={len(d_fake)}.")
    if len(d_real) == 0:
        raise ValueError("Discriminator outputs must contain at least one scale.")


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
    _validate_scales(d_real, d_fake)
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


def nonsat_d_loss(d_real: list[Tensor], d_fake: list[Tensor]) -> Tensor:
    """Multi-scale non-saturating (softplus) discriminator loss.

    For each scale, computes ``softplus(-d_real).mean() + softplus(d_fake).mean()``,
    then averages across scales. Verified verbatim from
    ``stylegan2-ada-pytorch/training/loss.py`` (D-real ``softplus(-real)``,
    D-fake ``softplus(fake)``).

    Parameters
    ----------
    d_real : list of Tensor
        Per-scale discriminator logits on real pairs.
    d_fake : list of Tensor
        Per-scale discriminator logits on fake pairs from a detached generator
        output.

    Returns
    -------
    Tensor
        Scalar loss, mean across scales.
    """
    _validate_scales(d_real, d_fake)
    per_scale = [F.softplus(-real).mean() + F.softplus(fake).mean() for real, fake in zip(d_real, d_fake, strict=True)]
    return torch.stack(per_scale).mean()


def nonsat_g_loss(d_fake: list[Tensor]) -> Tensor:
    """Multi-scale non-saturating (softplus) generator loss.

    For each scale, computes ``softplus(-d_fake).mean()``, then averages
    across scales. Verified verbatim from ``stylegan2-ada-pytorch/training/loss.py``
    (``loss_Gmain = softplus(-gen_logits)``).

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
    per_scale = [F.softplus(-fake).mean() for fake in d_fake]
    return torch.stack(per_scale).mean()


def rpgan_d_loss(d_real: list[Tensor], d_fake: list[Tensor]) -> Tensor:
    """Multi-scale RpGAN (relativistic pairing) discriminator loss.

    For each scale, computes ``softplus(-(d_real - d_fake)).mean()``, then
    averages across scales. Verified verbatim from R3GAN
    ``Trainer.py.AccumulateDiscriminatorGradients`` (NeurIPS 2024).

    The relativistic loss pushes D to assign higher scores to real than fake
    at the *pairwise* level; minimizing pushes ``(d_real - d_fake)`` large.
    RpGAN requires R1 + R2 penalties for local convergence on sharp
    distributions (R3GAN Theorem 3.1) — caller must enable both gradient
    penalties when using this loss.

    Parameters
    ----------
    d_real : list of Tensor
        Per-scale discriminator logits on real pairs.
    d_fake : list of Tensor
        Per-scale discriminator logits on fake pairs from a detached generator
        output.

    Returns
    -------
    Tensor
        Scalar loss, mean across scales.
    """
    _validate_scales(d_real, d_fake)
    per_scale = [F.softplus(-(real - fake)).mean() for real, fake in zip(d_real, d_fake, strict=True)]
    return torch.stack(per_scale).mean()


def rpgan_g_loss(d_real: list[Tensor], d_fake: list[Tensor]) -> Tensor:
    """Multi-scale RpGAN (relativistic pairing) generator loss.

    For each scale, computes ``softplus(d_real - d_fake).mean()`` —
    equivalently ``softplus(-(d_fake - d_real))``. Verified verbatim from
    R3GAN ``Trainer.py.AccumulateGeneratorGradients``.

    Both ``d_real`` and ``d_fake`` must be computed against the *current*
    (post-D-update) discriminator. Reusing ``d_real`` from the D step would
    use stale logits — verified against R3GAN convention.

    Parameters
    ----------
    d_real : list of Tensor
        Per-scale discriminator logits on real pairs, freshly computed in the
        G step against the post-update discriminator.
    d_fake : list of Tensor
        Per-scale discriminator logits on fake pairs built from the live
        generator output.

    Returns
    -------
    Tensor
        Scalar loss, mean across scales.
    """
    _validate_scales(d_real, d_fake)
    per_scale = [F.softplus(real - fake).mean() for real, fake in zip(d_real, d_fake, strict=True)]
    return torch.stack(per_scale).mean()


def _zero_centered_grad_penalty(discriminator: nn.Module, sample_input: Tensor) -> Tensor:
    """Multi-scale Mescheder zero-centered gradient penalty.

    Computes per-scale ``||∇x D_scale(x)||²`` (squared L2 of grad-of-logits
    w.r.t. input, summed over channel + spatial dims, averaged over batch)
    and averages across scales.

    NOT ``||∇x (sum over scales D_scale(x))||²`` — that would mix cross-scale
    derivatives because scale-N's input is derived from scale-0's input via
    avg-pool in ``MultiScalePatchGAN3D``. Per-scale grads keep magnitudes
    independent and match the multi-scale convention used by ``lsgan_*`` /
    ``nonsat_*`` (mean over per-scale terms).

    Caller MUST wrap this in ``torch.amp.autocast(..., enabled=False)`` when
    running under Lightning's bf16-mixed precision. The grad-of-grad path
    (``create_graph=True``) is numerically fragile under bf16; references like
    StyleGAN2-ADA / StyleGAN3 / R3GAN don't AMP at all so they don't need the
    wrapper, but Lightning's global autocast injects bf16 into D forwards.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator. Called as ``discriminator(sample_input)`` and
        expected to return a list of per-scale logit tensors.
    sample_input : Tensor
        Input pair (already detached from any prior graph). Will be
        ``.requires_grad_(True)`` internally.

    Returns
    -------
    Tensor
        Scalar penalty, mean across scales.
    """
    sample_input = sample_input.detach().requires_grad_(True)
    d_out = discriminator(sample_input)
    per_scale: list[Tensor] = []
    # retain_graph=True throughout: subsequent per-scale grad calls reuse the
    # D forward's graph, and the caller's `.backward()` on the returned penalty
    # backprops through `grads -> D's parameters` which also needs the graph.
    for d_scale in d_out:
        grads = torch.autograd.grad(
            outputs=d_scale.sum(),
            inputs=sample_input,
            create_graph=True,
            retain_graph=True,
        )[0]
        per_scale.append(grads.flatten(1).pow(2).sum(1).mean())
    return torch.stack(per_scale).mean()


def r1_penalty(discriminator: nn.Module, real_input: Tensor) -> Tensor:
    """R1 penalty: zero-centered gradient penalty on real input.

    Multi-scale version of Mescheder et al. 2018 ``R1 = E[||∇x D(x)||²]``
    on real samples. See :func:`_zero_centered_grad_penalty` for implementation
    details and the bf16-AMP gotcha.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator, returns a list of per-scale logits.
    real_input : Tensor
        Real pair (e.g. ``cat([source, target], dim=1)``). Will be detached.

    Returns
    -------
    Tensor
        Scalar penalty, mean across scales.
    """
    return _zero_centered_grad_penalty(discriminator, real_input)


def r2_penalty(discriminator: nn.Module, fake_input: Tensor) -> Tensor:
    """R2 penalty: zero-centered gradient penalty on fake input.

    Same form as :func:`r1_penalty` but on fake samples. Introduced in R3GAN
    (NeurIPS 2024) for local convergence of RpGAN on sharp distributions.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator, returns a list of per-scale logits.
    fake_input : Tensor
        Fake pair (e.g. ``cat([source, pred.detach()], dim=1)``).

    Returns
    -------
    Tensor
        Scalar penalty, mean across scales.
    """
    return _zero_centered_grad_penalty(discriminator, fake_input)
