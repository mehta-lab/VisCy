"""Unit tests for SpotlightLoss."""

import pytest
import torch

from viscy_utils.losses.spotlight import SpotlightLoss, _otsu_threshold, _tunable_sigmoid


def test_tunable_sigmoid_range():
    """Output is clamped to [0, 1]."""
    x = torch.linspace(-3, 3, 100)
    y = _tunable_sigmoid(x, k=-0.95)
    assert y.min() >= 0, f"min={y.min()}"
    assert y.max() <= 1, f"max={y.max()}"
    # Negative inputs map to 0 (clamped)
    assert _tunable_sigmoid(torch.tensor(-2.0), k=-0.95).item() == 0.0
    # Positive inputs map to ~1
    assert _tunable_sigmoid(torch.tensor(2.0), k=-0.95).item() == 1.0


def test_tunable_sigmoid_sharpness():
    """Sharper k produces output closer to a step function."""
    x = torch.linspace(-1, 1, 1000)
    soft = _tunable_sigmoid(x, k=-0.5)
    sharp = _tunable_sigmoid(x, k=-0.99)
    # Sharp sigmoid should have more values near 0 or 1
    soft_extreme = ((soft < 0.1) | (soft > 0.9)).float().mean()
    sharp_extreme = ((sharp < 0.1) | (sharp > 0.9)).float().mean()
    assert sharp_extreme > soft_extreme


def test_tunable_sigmoid_monotonic():
    """Tunable sigmoid is monotonically non-decreasing (clamped at 0 and 1)."""
    x = torch.linspace(-5, 5, 1000)
    y = _tunable_sigmoid(x, k=-0.95)
    assert (y[1:] >= y[:-1] - 1e-6).all()


def test_otsu_threshold_bimodal():
    """Otsu finds the correct threshold on a bimodal distribution."""
    rng = torch.Generator().manual_seed(42)
    bg = torch.randn(1000, generator=rng) * 0.2  # mode at ~0
    fg = torch.randn(1000, generator=rng) * 0.2 + 2.0  # mode at ~2
    x = torch.cat([bg, fg])
    threshold = _otsu_threshold(x)
    assert 0.5 < threshold.item() < 1.5, f"threshold={threshold.item()}"


def test_otsu_threshold_constant():
    """Otsu on constant input returns that constant."""
    x = torch.full((100,), 5.0)
    threshold = _otsu_threshold(x)
    assert threshold.item() == 5.0


def test_masked_mse_ignores_background():
    """Masked MSE computes loss only on foreground voxels."""
    # Target with clear FG (1.0) and BG (0.0)
    target = torch.zeros(1, 1, 4, 8, 8)
    target[:, :, :, 4:, :] = 1.0  # right half is foreground

    # Prediction matches FG perfectly but differs in BG
    pred = target.clone()
    pred[:, :, :, :4, :] = 99.0  # large BG error

    # lambda=0.99 so masked MSE dominates (Dice weight ~0)
    loss_fn_mse = SpotlightLoss(lambda_mse=0.99, sigmoid_k=-0.95)
    loss = loss_fn_mse(pred, target)
    # Loss should be near 0 since FG prediction matches target
    assert loss.item() < 0.5, f"loss={loss.item()}, expected near 0 (FG matches)"


def test_dice_loss_good_match():
    """Dice term is low when prediction FG matches target FG."""
    target = torch.zeros(1, 1, 4, 8, 8)
    target[:, :, :, 4:, :] = 1.0

    pred = target.clone()  # perfect match
    loss_fn = SpotlightLoss(lambda_mse=0.01, sigmoid_k=-0.95)
    loss = loss_fn(pred, target)
    assert loss.item() < 0.5


def test_spotlight_loss_forward_shape():
    """End-to-end forward on 5D tensors returns a scalar."""
    loss_fn = SpotlightLoss()
    pred = torch.randn(2, 1, 4, 8, 8)
    target = torch.randn(2, 1, 4, 8, 8).abs()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_spotlight_loss_4d():
    """Works with 4D tensors (B, C, Y, X) for 2D models."""
    loss_fn = SpotlightLoss()
    pred = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32).abs()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0


def test_spotlight_loss_all_background():
    """Handles all-background target gracefully (no NaN/Inf)."""
    loss_fn = SpotlightLoss()
    pred = torch.randn(1, 1, 4, 8, 8)
    target = torch.zeros(1, 1, 4, 8, 8)  # all background
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)


def test_spotlight_loss_backward():
    """Gradients flow correctly through the loss."""
    loss_fn = SpotlightLoss()
    pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 4, 8, 8).abs()
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_spotlight_loss_invalid_params():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="sigmoid_k"):
        SpotlightLoss(sigmoid_k=0.5)
    with pytest.raises(ValueError, match="lambda_mse"):
        SpotlightLoss(lambda_mse=0.0)
    with pytest.raises(ValueError, match="lambda_mse"):
        SpotlightLoss(lambda_mse=1.0)
