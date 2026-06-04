"""Tests for LSGAN losses."""

import math

import torch

from viscy_models.gan import lsgan_d_loss, lsgan_g_loss


def test_lsgan_d_loss_single_scale():
    """Single-scale D loss returns a finite scalar."""
    d_real = [torch.ones(2, 1, 2, 8, 8)]
    d_fake = [torch.zeros(2, 1, 2, 8, 8)]
    loss = lsgan_d_loss(d_real, d_fake)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    # D_real perfectly classifies (1 -> target 1) and D_fake perfectly
    # classifies (0 -> target 0), so the loss is 0.
    assert loss.item() == 0.0


def test_lsgan_d_loss_multi_scale():
    """Multi-scale D loss with different YX per scale returns a finite scalar."""
    d_real = [
        torch.randn(2, 1, 2, 8, 8),
        torch.randn(2, 1, 2, 4, 4),
    ]
    d_fake = [
        torch.randn(2, 1, 2, 8, 8),
        torch.randn(2, 1, 2, 4, 4),
    ]
    loss = lsgan_d_loss(d_real, d_fake)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_lsgan_g_loss_single_and_multi():
    """G loss works for both single- and multi-scale inputs."""
    d_fake_single = [torch.ones(2, 1, 2, 8, 8)]
    loss_single = lsgan_g_loss(d_fake_single)
    assert loss_single.ndim == 0
    assert math.isfinite(loss_single.item())
    # All ones means D classifies fake as real -> G loss is 0.
    assert loss_single.item() == 0.0

    d_fake_multi = [
        torch.randn(2, 1, 2, 8, 8),
        torch.randn(2, 1, 2, 4, 4),
    ]
    loss_multi = lsgan_g_loss(d_fake_multi)
    assert loss_multi.ndim == 0
    assert math.isfinite(loss_multi.item())
    assert loss_multi.item() > 0.0


def test_losses_gradient_flow():
    """Both losses propagate non-zero gradients back into the input logits."""
    real = torch.randn(2, 1, 2, 8, 8, requires_grad=True)
    fake = torch.randn(2, 1, 2, 8, 8, requires_grad=True)
    d_loss = lsgan_d_loss([real], [fake])
    d_loss.backward()
    assert real.grad is not None and real.grad.abs().sum().item() > 0.0
    assert fake.grad is not None and fake.grad.abs().sum().item() > 0.0

    fake_g = torch.randn(2, 1, 2, 8, 8, requires_grad=True)
    g_loss = lsgan_g_loss([fake_g])
    g_loss.backward()
    assert fake_g.grad is not None and fake_g.grad.abs().sum().item() > 0.0
