"""Unit tests for the flow-matching transport module."""

import pytest

torchdiffeq = pytest.importorskip("torchdiffeq")

import torch  # noqa: E402

from viscy_models.celldiff.modules.transport import (  # noqa: E402
    ModelType,
    Sampler,
    WeightType,
    create_transport,
)
from viscy_models.celldiff.modules.transport.integrators import ODESolver  # noqa: E402
from viscy_models.celldiff.modules.transport.path import ICPlan  # noqa: E402


def test_create_transport_defaults():
    """Factory returns Transport with LINEAR path and VELOCITY model."""
    transport = create_transport()
    assert transport.model_type == ModelType.VELOCITY
    assert isinstance(transport.path_sampler, ICPlan)
    assert transport.loss_type == WeightType.NONE
    assert transport.train_eps == 0
    assert transport.sample_eps == 0


def test_create_transport_vp_path():
    """VP path sets non-zero eps automatically."""
    transport = create_transport(path_type="VP", prediction="noise")
    assert transport.train_eps == 1e-5
    assert transport.sample_eps == 1e-3


def test_transport_sample_shapes():
    """transport.sample(x1) returns (t, x0, x1) with correct shapes."""
    transport = create_transport()
    x1 = torch.randn(4, 1, 8, 16, 16)
    t, x0, x1_out = transport.sample(x1)
    assert t.shape == (4,)
    assert x0.shape == x1.shape
    assert x1_out is x1
    assert (t >= 0).all() and (t <= 1).all()


def test_training_losses_velocity():
    """Velocity loss has correct keys and is finite."""
    transport = create_transport()
    B, C, D, H, W = 2, 1, 4, 8, 8
    x1 = torch.randn(B, C, D, H, W)
    t, x0, x1 = transport.sample(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    # Simulate model output (velocity prediction).
    model_output = torch.randn_like(ut)
    losses = transport.training_losses(model_output, x0, x1, xt, ut, t)
    assert "loss" in losses
    assert "pred" in losses
    assert losses["loss"].shape == (B,)
    assert torch.isfinite(losses["loss"]).all()


def test_icplan_plan_shapes():
    """ICPlan.plan returns matching shapes."""
    plan = ICPlan()
    B, C, D, H, W = 3, 1, 4, 8, 8
    t = torch.rand(B)
    x0 = torch.randn(B, C, D, H, W)
    x1 = torch.randn(B, C, D, H, W)
    t_out, xt, ut = plan.plan(t, x0, x1)
    assert t_out.shape == (B,)
    assert xt.shape == (B, C, D, H, W)
    assert ut.shape == (B, C, D, H, W)


def test_ode_solver_integration():
    """ODESolver.sample() produces correct output shape."""
    num_steps = 5

    def dummy_drift(x, t, model, **kwargs):
        return torch.zeros_like(x)

    solver = ODESolver(
        drift=dummy_drift,
        t0=0.0,
        t1=1.0,
        sampler_type="euler",
        num_steps=num_steps,
        atol=1e-5,
        rtol=1e-3,
    )
    x_init = torch.randn(2, 1, 4, 4, 4)
    result = solver.sample(x_init, model=None)
    # odeint returns (num_steps, B, C, D, H, W).
    assert result.shape[0] == num_steps
    assert result.shape[1:] == x_init.shape


def test_sampler_sample_ode():
    """Sampler.sample_ode() returns a callable that produces sample trajectories."""
    transport = create_transport()
    sampler = Sampler(transport)

    def dummy_model(x, t):
        return torch.zeros_like(x)

    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=5, atol=1e-5, rtol=1e-3)
    x_init = torch.randn(2, 1, 4, 4, 4)
    result = sample_fn(x_init, dummy_model)
    # Result is a trajectory tensor from odeint.
    assert result.shape[0] == 5
    assert result.shape[1:] == x_init.shape
