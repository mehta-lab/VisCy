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


def test_ode_solver_cosine_schedule_packs_points_at_both_ends():
    """The cosine grid keeps the endpoints and spaces points densest next to t0 and t1."""
    kwargs = dict(drift=lambda x, t, model: x, t0=0.0, t1=1.0, sampler_type="euler", num_steps=9, atol=1e-5, rtol=1e-3)
    uniform = ODESolver(**kwargs).t
    cosine = ODESolver(**kwargs, time_schedule="cosine").t
    torch.testing.assert_close(uniform, torch.linspace(0, 1, 9))
    assert cosine[0] == 0.0 and cosine[-1] == 1.0
    gaps = cosine.diff()
    assert gaps[0] < gaps[4] and gaps[-1] < gaps[4]
    torch.testing.assert_close(gaps, gaps.flip(0))


def test_ode_solver_beta_schedule_spans_cosine_and_uniform():
    """Beta(0.5, 0.5) is the cosine grid, Beta(1, 1) the uniform one, and p < q packs points toward t0."""
    kwargs = dict(drift=lambda x, t, model: x, t0=0.0, t1=1.0, sampler_type="euler", num_steps=8, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(
        ODESolver(**kwargs, time_schedule=(0.5, 0.5)).t, ODESolver(**kwargs, time_schedule="cosine").t
    )
    torch.testing.assert_close(ODESolver(**kwargs, time_schedule=(1.0, 1.0)).t, ODESolver(**kwargs).t)
    skewed = ODESolver(**kwargs, time_schedule=(0.3, 0.6)).t
    assert skewed[0] == 0.0 and skewed[-1] == 1.0
    gaps = skewed.diff()
    assert (gaps > 0).all() and gaps[0] < gaps[-1]
    inner = ODESolver(**{**kwargs, "t0": 0.1, "t1": 0.9}, time_schedule=(0.3, 0.6)).t
    torch.testing.assert_close(inner, 0.1 + 0.8 * skewed)


def test_ode_solver_rejects_a_beta_grid_that_collapses_in_float32():
    """Beta(1, 0.1) at 8 points puts its last two quantiles within float32 rounding of 1.0, which odeint
    would only reject mid-predict."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ODESolver(
            drift=lambda x, t, model: x,
            t0=0.0,
            t1=1.0,
            sampler_type="euler",
            num_steps=8,
            atol=1e-5,
            rtol=1e-3,
            time_schedule=(1.0, 0.1),
        )


@pytest.mark.parametrize("schedule", ["cos", (0.0, 0.5), (0.5,), (0.5, float("nan")), (0.5, float("inf"))])
def test_ode_solver_rejects_unknown_schedule(schedule):
    """A misspelled schedule or a non-positive or non-finite Beta shape raises instead of building a bad grid."""
    with pytest.raises(ValueError, match="time_schedule"):
        ODESolver(
            drift=lambda x, t, model: x,
            t0=0.0,
            t1=1.0,
            sampler_type="euler",
            num_steps=5,
            atol=1e-5,
            rtol=1e-3,
            time_schedule=schedule,
        )


def test_sampler_euler_with_cosine_schedule_steps_on_the_grid():
    """A fixed-grid sampler evaluates the drift once per step, at the cosine grid's left points."""
    sampler = Sampler(create_transport())
    seen = []

    def model(x, t):
        seen.append(float(t[0]))
        return torch.zeros_like(x)

    sampler.sample_ode(sampling_method="euler", num_steps=5, time_schedule="cosine")(torch.randn(1, 1, 2, 2, 2), model)
    expected = (1 - torch.cos(torch.pi * torch.linspace(0, 1, 5, dtype=torch.float64))) / 2
    assert seen == pytest.approx(expected[:-1].tolist(), abs=1e-6)
