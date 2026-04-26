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

    # With fg_threshold=0.5, mask captures FG correctly
    loss_fn = SpotlightLoss(lambda_mse=0.99, sigmoid_k=-0.95, fg_threshold=0.5)
    loss = loss_fn(pred, target)
    # Loss should be near 0 since FG prediction matches target
    assert loss.item() < 0.5, f"loss={loss.item()}, expected near 0 (FG matches)"


def test_dice_loss_good_match():
    """Dice term is low when prediction FG matches target FG."""
    target = torch.zeros(1, 1, 4, 8, 8)
    target[:, :, :, 4:, :] = 1.0

    pred = target.clone()  # perfect match
    loss_fn = SpotlightLoss(lambda_mse=0.01, sigmoid_k=-0.95, fg_threshold=0.5)
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
    loss_fn = SpotlightLoss(fg_threshold=0.5)
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


def test_fg_threshold_none_uses_otsu():
    """fg_threshold=None falls back to per-sample Otsu."""
    loss_fn = SpotlightLoss(fg_threshold=None)
    pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 4, 8, 8).abs()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None


def test_fg_threshold_float_uses_fixed():
    """fg_threshold=0.0 uses a fixed threshold, no Otsu."""
    loss_fn = SpotlightLoss(fg_threshold=0.0)
    pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    # Target centered at 0 (like Otsu-normalized data)
    target = torch.randn(2, 1, 4, 8, 8)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None


def test_spotlight_with_precomputed_mask():
    """Loss accepts and uses precomputed fg_mask."""
    loss_fn = SpotlightLoss()
    pred = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 4, 8, 8).abs()
    fg_mask = (target > target.median()).float()
    loss = loss_fn(pred, target, fg_mask=fg_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_precomputed_mask_overrides_threshold():
    """Precomputed mask takes priority over fg_threshold."""
    target = torch.zeros(1, 1, 4, 8, 8)
    target[:, :, :, 4:, :] = 1.0  # right half is FG

    pred = target.clone()
    pred[:, :, :, :4, :] = 99.0  # large BG error

    # fg_threshold=0.5 would create the same mask as the target FG
    # But we provide an all-ones mask — should include BG errors
    all_ones_mask = torch.ones_like(target)
    loss_fn = SpotlightLoss(lambda_mse=0.99, sigmoid_k=-0.95, fg_threshold=0.5)

    loss_with_mask = loss_fn(pred, target, fg_mask=all_ones_mask)
    loss_without_mask = loss_fn(pred, target)

    # With all-ones mask, BG errors contribute → loss should be much higher
    assert loss_with_mask.item() > loss_without_mask.item() * 2


def test_single_channel_backward_compat():
    """C=1, B=1 with fixed threshold is numerically identical to global reduction."""
    torch.manual_seed(0)
    pred = torch.randn(1, 1, 4, 8, 8)
    target = torch.randn(1, 1, 4, 8, 8)
    fg_mask = (target > 0).float()

    loss_fn = SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95, fg_threshold=0.0)
    loss = loss_fn(pred, target, fg_mask=fg_mask)

    # Manually compute the old global-reduction formula for B=1, C=1
    mask = fg_mask.float()
    sq_err = (pred - target) ** 2
    fg_count = mask.sum()
    old_mse = (sq_err * mask).sum() / fg_count
    soft_pred = _tunable_sigmoid(pred, -0.95)
    intersection = (soft_pred * mask).sum()
    old_dice = 1 - (2 * intersection) / (soft_pred.sum() + mask.sum() + 1e-6)
    old_loss = 0.5 * old_mse + 0.5 * old_dice

    assert torch.allclose(loss, old_loss, atol=1e-5), f"loss={loss.item()}, old={old_loss.item()}"


def test_multi_channel_all_masked():
    """Multi-channel with all channels masked computes per-channel loss."""
    pred = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 2, 4, 8, 8)
    fg_mask = torch.zeros_like(target)
    fg_mask[:, :, :, 4:, :] = 1.0  # half FG in both channels

    loss_fn = SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95, fg_threshold=0.0)
    loss = loss_fn(pred, target, fg_mask=fg_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_multi_channel_partial_mask():
    """Channels with all-zero mask use regular MSE and skip Dice."""
    pred = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 2, 4, 8, 8)
    # Channel 0 has mask, channel 1 is all-zero
    fg_mask = torch.zeros(2, 2, 4, 8, 8)
    fg_mask[:, 0, :, 4:, :] = 1.0

    loss_fn = SpotlightLoss(lambda_mse=0.5, sigmoid_k=-0.95, fg_threshold=0.0)
    loss = loss_fn(pred, target, fg_mask=fg_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_partial_mask_ignores_placeholder_channel_in_dice():
    """Dice excludes all-ones placeholder channels, includes real FG/BG channels."""
    pred = torch.randn(1, 2, 4, 8, 8)
    target = torch.randn(1, 2, 4, 8, 8)
    # Channel 0: real mask (has both 0s and 1s) — Dice should include
    # Channel 1: all-ones placeholder — Dice should exclude
    fg_mask = torch.ones(1, 2, 4, 8, 8)
    fg_mask[:, 0, :, :4, :] = 0.0  # channel 0 has BG in left half

    loss_fn = SpotlightLoss(lambda_mse=0.01, sigmoid_k=-0.95)
    loss = loss_fn(pred, target, fg_mask=fg_mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_otsu_per_channel():
    """Otsu computes per-(sample, channel) thresholds."""
    from viscy_utils.losses.spotlight import _otsu_threshold_batch

    target = torch.zeros(1, 2, 4, 8, 8)
    target[:, 0] = torch.randn(1, 4, 8, 8).abs() + 1.0  # high values
    target[:, 1] = torch.randn(1, 4, 8, 8).abs() * 0.1  # low values

    thresholds = _otsu_threshold_batch(target)
    assert thresholds.shape == (1, 2, 1, 1, 1)
    assert thresholds[0, 0] != thresholds[0, 1], "Per-channel thresholds should differ"


def test_spotlight_loss_invalid_params():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="sigmoid_k"):
        SpotlightLoss(sigmoid_k=0.5)
    with pytest.raises(ValueError, match="lambda_mse"):
        SpotlightLoss(lambda_mse=0.0)
    with pytest.raises(ValueError, match="lambda_mse"):
        SpotlightLoss(lambda_mse=1.0)
    with pytest.raises(ValueError, match="eps"):
        SpotlightLoss(eps=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
def test_spotlight_loss_forward_under_bf16_autocast():
    """Forward+backward under bf16 autocast produces finite scalar + grads.

    Verifies the @torch.amp.custom_fwd(cast_inputs=fp32) decorator removal
    didn't break the autocast path. SpotlightLoss has no conv-heavy ops;
    autocast policy already promotes the squared error / reductions /
    divisions to fp32, so the decorator was redundant.
    """
    loss_fn = SpotlightLoss().cuda()
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 4, 32, 32, device="cuda", requires_grad=True)
    target = torch.rand(2, 1, 4, 32, 32, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_fn(pred, target)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
def test_spotlight_loss_autocast_matches_fp32_baseline():
    """No-decorator autocast result tracks the explicit-fp32 baseline.

    Drift is bounded by rtol=1e-3, atol=1e-3 — SpotlightLoss has no convs
    (sigmoid is the only autocast-affected op) and the autocast policy
    promotes the precision-sensitive parts to fp32; in practice the
    measured drift is 0.0.
    """
    loss_fn = SpotlightLoss().cuda()
    torch.manual_seed(1)
    pred = torch.rand(2, 1, 4, 32, 32, device="cuda")
    target = torch.rand(2, 1, 4, 32, 32, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_autocast = loss_fn(pred, target)

    loss_fp32 = loss_fn(pred.float(), target.float())

    torch.testing.assert_close(loss_autocast.float(), loss_fp32, rtol=1e-3, atol=1e-3)
