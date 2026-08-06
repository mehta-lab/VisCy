"""Tests for the shared latent Predictor (next-state forecaster)."""

import torch
import torch.nn.functional as F

from viscy_models.contrastive.predictor import Predictor


def test_output_shape():
    """Predictor maps (N, dim) -> (N, dim)."""
    pred = Predictor(dim=32)
    z = torch.randn(10, 32)
    assert pred(z).shape == (10, 32)


def test_custom_hidden_dim():
    """Hidden width is configurable; output dim still equals input dim."""
    pred = Predictor(dim=16, hidden_dim=64)
    assert pred(torch.randn(4, 16)).shape == (4, 16)


def test_learns_a_predictable_transition():
    """A shared predictor can fit a common linear transition z_{t+1} = z_t + v."""
    torch.manual_seed(0)
    dim = 8
    v = torch.randn(dim)  # one shared step common to all samples
    pred = Predictor(dim=dim)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    for _ in range(300):
        z_t = torch.randn(64, dim)
        z_next = z_t + v  # common transition
        opt.zero_grad()
        loss = F.mse_loss(pred(z_t), z_next.detach())
        loss.backward()
        opt.step()
    # after fitting, prediction error on the common transition is small
    z_t = torch.randn(256, dim)
    final = F.mse_loss(pred(z_t), z_t + v)
    assert final.item() < 0.05


def test_random_transition_stays_high():
    """A non-learnable (per-sample random) target cannot be fit -> loss stays high."""
    torch.manual_seed(0)
    dim = 8
    pred = Predictor(dim=dim)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    for _ in range(300):
        z_t = torch.randn(64, dim)
        z_next = torch.randn(64, dim)  # unrelated to z_t: no common rule
        opt.zero_grad()
        loss = F.mse_loss(pred(z_t), z_next.detach())
        loss.backward()
        opt.step()
    z_t = torch.randn(256, dim)
    z_next = torch.randn(256, dim)
    final = F.mse_loss(pred(z_t), z_next)
    # best a predictor can do for a zero-mean unit-var random target is ~var=1
    assert final.item() > 0.5


def test_stop_grad_target_no_grad_to_target():
    """With a detached target, no gradient flows into the target tensor."""
    dim = 8
    pred = Predictor(dim=dim)
    z_t = torch.randn(4, dim, requires_grad=True)
    z_next = torch.randn(4, dim, requires_grad=True)
    loss = F.mse_loss(pred(z_t), z_next.detach())
    loss.backward()
    assert z_next.grad is None  # target is stop-grad
    assert z_t.grad is not None  # input side still receives gradient
