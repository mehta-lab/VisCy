"""Tests for non-saturating, RpGAN, and R1/R2 gradient penalties."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from viscy_models.gan import (
    MultiScalePatchGAN3D,
    nonsat_d_loss,
    nonsat_g_loss,
    r1_penalty,
    r2_penalty,
    rpgan_d_loss,
    rpgan_g_loss,
)

# ---------------------------------------------------------------------------
# Non-saturating losses
# ---------------------------------------------------------------------------


def test_nonsat_d_loss_matches_closed_form():
    """D loss equals softplus(-real).mean() + softplus(fake).mean() per scale."""
    real = torch.tensor([[[[[1.0, -1.0]]]]])  # logits
    fake = torch.tensor([[[[[2.0, -2.0]]]]])
    loss = nonsat_d_loss([real], [fake])
    expected = F.softplus(-real).mean() + F.softplus(fake).mean()
    assert torch.allclose(loss, expected)


def test_nonsat_g_loss_matches_closed_form():
    """G loss equals softplus(-fake).mean() per scale."""
    fake = torch.tensor([[[[[2.0, -2.0]]]]])
    loss = nonsat_g_loss([fake])
    expected = F.softplus(-fake).mean()
    assert torch.allclose(loss, expected)


def test_nonsat_d_loss_multi_scale_averages():
    """Multi-scale D loss averages per-scale loss across scales."""
    real_a = torch.zeros(1, 1, 1, 4, 4)
    fake_a = torch.zeros(1, 1, 1, 4, 4)
    real_b = torch.ones(1, 1, 1, 2, 2)
    fake_b = -torch.ones(1, 1, 1, 2, 2)
    loss = nonsat_d_loss([real_a, real_b], [fake_a, fake_b])
    per_scale_a = F.softplus(-real_a).mean() + F.softplus(fake_a).mean()
    per_scale_b = F.softplus(-real_b).mean() + F.softplus(fake_b).mean()
    expected = torch.stack([per_scale_a, per_scale_b]).mean()
    assert torch.allclose(loss, expected)


def test_nonsat_d_loss_validation_empty():
    """Empty scale list raises ValueError."""
    with pytest.raises(ValueError, match="at least one scale"):
        nonsat_d_loss([], [])


def test_nonsat_d_loss_validation_mismatch():
    """Mismatched scale counts raise ValueError."""
    real = [torch.zeros(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 1, 1)]
    fake = [torch.zeros(1, 1, 1, 2, 2)]
    with pytest.raises(ValueError, match="Number of scales"):
        nonsat_d_loss(real, fake)


def test_nonsat_losses_gradient_flow():
    """Both losses propagate gradients into the input logits."""
    real = torch.randn(2, 1, 2, 4, 4, requires_grad=True)
    fake = torch.randn(2, 1, 2, 4, 4, requires_grad=True)
    d_loss = nonsat_d_loss([real], [fake])
    d_loss.backward()
    assert real.grad is not None and real.grad.abs().sum().item() > 0.0
    assert fake.grad is not None and fake.grad.abs().sum().item() > 0.0

    fake_g = torch.randn(2, 1, 2, 4, 4, requires_grad=True)
    g_loss = nonsat_g_loss([fake_g])
    g_loss.backward()
    assert fake_g.grad is not None and fake_g.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# RpGAN losses
# ---------------------------------------------------------------------------


def test_rpgan_d_loss_matches_closed_form():
    """D loss equals softplus(-(d_real - d_fake)).mean() per scale."""
    real = torch.tensor([[[[[2.0, -1.0]]]]])
    fake = torch.tensor([[[[[1.0, 0.5]]]]])
    loss = rpgan_d_loss([real], [fake])
    expected = F.softplus(-(real - fake)).mean()
    assert torch.allclose(loss, expected)


def test_rpgan_g_loss_matches_closed_form():
    """G loss equals softplus(d_real - d_fake).mean() per scale.

    Equivalent to softplus(-(d_fake - d_real)) — matches R3GAN convention.
    """
    real = torch.tensor([[[[[2.0, -1.0]]]]])
    fake = torch.tensor([[[[[1.0, 0.5]]]]])
    loss = rpgan_g_loss([real], [fake])
    expected = F.softplus(real - fake).mean()
    assert torch.allclose(loss, expected)


def test_rpgan_g_loss_pushes_fake_above_real():
    """Minimizing G loss should push d_fake > d_real (the relativistic objective)."""
    real = torch.zeros(1, 1, 1, 2, 2)
    # Two scenarios; G prefers higher d_fake.
    fake_low = -torch.ones(1, 1, 1, 2, 2)
    fake_high = torch.ones(1, 1, 1, 2, 2)
    g_loss_low = rpgan_g_loss([real], [fake_low])
    g_loss_high = rpgan_g_loss([real], [fake_high])
    # G loss is lower when d_fake > d_real.
    assert g_loss_high.item() < g_loss_low.item()


def test_rpgan_d_loss_pushes_real_above_fake():
    """Minimizing D loss should push d_real > d_fake."""
    real_low = -torch.ones(1, 1, 1, 2, 2)
    real_high = torch.ones(1, 1, 1, 2, 2)
    fake = torch.zeros(1, 1, 1, 2, 2)
    d_loss_low = rpgan_d_loss([real_low], [fake])
    d_loss_high = rpgan_d_loss([real_high], [fake])
    # D loss is lower when d_real > d_fake.
    assert d_loss_high.item() < d_loss_low.item()


# ---------------------------------------------------------------------------
# R1 / R2 gradient penalties
# ---------------------------------------------------------------------------


def _tiny_d() -> MultiScalePatchGAN3D:
    """A small multi-scale D for fast unit tests."""
    return MultiScalePatchGAN3D(in_channels=2, base_channels=8, num_scales=2, use_spectral_norm=False)


def test_r1_penalty_finite_and_scalar():
    """R1 returns a non-negative scalar tensor."""
    d = _tiny_d()
    real = torch.randn(1, 2, 8, 64, 64)
    pen = r1_penalty(d, real)
    assert pen.ndim == 0
    assert math.isfinite(pen.item())
    assert pen.item() >= 0.0


def test_r1_penalty_zero_for_constant_d():
    """R1 = 0 when D is identically zero (gradient w.r.t. input is 0)."""

    class ZeroD(nn.Module):
        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            zeros = x.sum() * 0.0  # zero, but with a grad path
            return [zeros.expand(1, 1, 1, 1, 1).clone()]

    real = torch.randn(1, 2, 8, 64, 64)
    pen = r1_penalty(ZeroD(), real)
    assert pen.item() == 0.0


def test_r1_penalty_grad_flows_to_d_params():
    """Backward through R1 populates D's parameter gradients."""
    d = _tiny_d()
    for p in d.parameters():
        p.grad = None
    real = torch.randn(1, 2, 8, 64, 64)
    pen = r1_penalty(d, real)
    pen.backward()
    grads = [p.grad for p in d.parameters() if p.grad is not None]
    assert grads, "expected at least one D parameter with gradient"
    assert any(g.abs().sum().item() > 0.0 for g in grads)


def test_r1_penalty_multi_scale_aggregation():
    """R1 is mean of per-scale ||grad||², NOT ||grad of sum of scales||²."""
    d = _tiny_d()
    real = torch.randn(1, 2, 8, 64, 64)
    pen = r1_penalty(d, real)
    # Recompute per-scale by hand and compare.
    real_manual = real.detach().requires_grad_(True)
    d_out = d(real_manual)
    per_scale = []
    for d_scale in d_out:
        grads = torch.autograd.grad(
            outputs=d_scale.sum(),
            inputs=real_manual,
            create_graph=False,
            retain_graph=True,
        )[0]
        per_scale.append(grads.flatten(1).pow(2).sum(1).mean())
    expected = torch.stack(per_scale).mean()
    assert torch.allclose(pen, expected, rtol=1e-5)


def test_r2_penalty_finite_and_scalar():
    """R2 returns a non-negative scalar tensor on fake input."""
    d = _tiny_d()
    fake = torch.randn(1, 2, 8, 64, 64)
    pen = r2_penalty(d, fake)
    assert pen.ndim == 0
    assert math.isfinite(pen.item())
    assert pen.item() >= 0.0


def test_r1_penalty_under_autocast_disabled_wrapper():
    """Inside autocast(enabled=False) the penalty stays in fp32 even when
    a surrounding autocast context is active.

    This mirrors the Lightning bf16-mixed gotcha: D forwards under autocast
    use bf16 for matmul, but the grad-of-grad path needs fp32. CPU autocast
    uses bf16, so the dtype check works without CUDA.
    """
    d = _tiny_d()
    real = torch.randn(1, 2, 8, 64, 64)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        with torch.amp.autocast(device_type="cpu", enabled=False):
            pen = r1_penalty(d, real)
    assert pen.dtype == torch.float32
    assert math.isfinite(pen.item())
