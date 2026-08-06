"""Tests for TemporalStraighteningLoss (curvature loss on latent tracks)."""

import pytest
import torch

from viscy_models.contrastive.loss import TemporalStraighteningLoss


def test_straight_trajectory_zero_curvature():
    """A perfectly straight (constant-velocity) track has curvature ~0."""
    # z_t = t * direction: consecutive velocities are identical -> cos=1 -> loss=0
    t = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    direction = torch.randn(1, 1, 8)
    z_seq = t * direction  # (1, 5, 8), constant velocity = direction
    loss = TemporalStraighteningLoss()(z_seq)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_reversing_trajectory_max_curvature():
    """A track that reverses direction each step has curvature ~2 (cos=-1)."""
    # frames alternate between two points -> v_t and v_{t+1} are antiparallel
    a = torch.zeros(1, 8)
    b = torch.ones(1, 8)
    z_seq = torch.stack([a, b, a, b, a], dim=1)  # (1, 5, 8)
    loss = TemporalStraighteningLoss()(z_seq)
    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_random_iid_points_curvature_near_1p5():
    """I.i.d. random *points* give curvature ~1.5, not 1.0.

    Consecutive velocities v_t = z_{t+1}-z_t and v_{t+1} = z_{t+2}-z_{t+1} share
    the term z_{t+1} with opposite sign, so E[cos] = -0.5 -> E[1-cos] = 1.5.
    (A value < 1.5 on real data indicates residual straightness — useful as a
    reference: the DynaCLR probe measured ~1.29.)
    """
    torch.manual_seed(0)
    z_seq = torch.randn(512, 3, 128)
    loss = TemporalStraighteningLoss()(z_seq)
    assert loss.item() == pytest.approx(1.5, abs=0.1)


def test_valid_mask_drops_samples():
    """The valid mask selects which samples contribute to the mean."""
    straight = (torch.arange(3, dtype=torch.float32).view(1, 3, 1) * torch.ones(1, 1, 4)).repeat(2, 1, 1)
    reversing = torch.stack([torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 4)], dim=1)
    z_seq = torch.cat([straight, reversing], dim=0)  # (4, 3, 4): 2 straight, 2 reversing
    loss_fn = TemporalStraighteningLoss()
    # only the two straight samples counted -> ~0
    valid = torch.tensor([True, True, False, False])
    assert loss_fn(z_seq, valid=valid).item() == pytest.approx(0.0, abs=1e-5)
    # only the two reversing samples counted -> ~2
    valid = torch.tensor([False, False, True, True])
    assert loss_fn(z_seq, valid=valid).item() == pytest.approx(2.0, abs=1e-5)


def test_all_invalid_returns_graph_connected_zero():
    """When no sample is valid, return a zero that still carries grad (DDP-safe)."""
    z_seq = torch.randn(4, 3, 8, requires_grad=True)
    valid = torch.zeros(4, dtype=torch.bool)
    loss = TemporalStraighteningLoss()(z_seq, valid=valid)
    assert loss.item() == 0.0
    loss.backward()  # must not raise; grad graph is connected via new_zeros
    assert z_seq.grad is not None


def test_gradients_flow():
    """Loss is differentiable w.r.t. the embeddings."""
    z_seq = torch.randn(8, 3, 16, requires_grad=True)
    loss = TemporalStraighteningLoss()(z_seq)
    loss.backward()
    assert z_seq.grad is not None
    assert torch.isfinite(z_seq.grad).all()


def test_k_less_than_three_raises():
    """Curvature is undefined with fewer than 3 frames."""
    z_seq = torch.randn(4, 2, 8)
    with pytest.raises(ValueError, match="K >= 3"):
        TemporalStraighteningLoss()(z_seq)


def test_k_greater_than_three_sliding_window():
    """K>3 averages curvature over the sliding stencil; straight stays ~0."""
    z_seq = torch.arange(6, dtype=torch.float32).view(1, 6, 1) * torch.ones(1, 1, 4)
    loss = TemporalStraighteningLoss()(z_seq)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_step_is_noop():
    """step() exists (for the on_train_epoch_start hook) and does nothing."""
    loss_fn = TemporalStraighteningLoss()
    loss_fn.step(0)
    loss_fn.step(100)  # must not raise
