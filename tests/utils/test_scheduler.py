import pytest

from viscy.utils.scheduler import ParameterScheduler


def test_scheduler_constant():
    """Test constant schedule returns target value at all epochs."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.1,
        target_value=1.0,
        warmup_epochs=10,
        schedule_type="constant",
    )

    assert scheduler.get_value(0) == 1.0
    assert scheduler.get_value(5) == 1.0
    assert scheduler.get_value(10) == 1.0
    assert scheduler.get_value(100) == 1.0


def test_scheduler_linear_warmup():
    """Test linear schedule interpolates from initial to target."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.0,
        target_value=1.0,
        warmup_epochs=10,
        schedule_type="linear",
    )

    # At epoch 0, should be at initial
    assert abs(scheduler.get_value(0) - 0.0) < 0.01

    # At epoch 5 (halfway), should be approximately 0.5
    assert abs(scheduler.get_value(5) - 0.5) < 0.01

    # At epoch 10 (end), should be at target
    assert abs(scheduler.get_value(10) - 1.0) < 0.01

    # After warmup, should stay at target
    assert abs(scheduler.get_value(20) - 1.0) < 0.01


def test_scheduler_cosine_warmup():
    """Test cosine schedule increases smoothly from initial to target."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.0,
        target_value=1.0,
        warmup_epochs=10,
        schedule_type="cosine",
    )

    # At epoch 0, should be close to initial
    assert scheduler.get_value(0) < 0.1

    # At halfway through warmup
    val_halfway = scheduler.get_value(5)
    assert 0.0 < val_halfway < 1.0

    # At end of warmup, should be at target
    assert abs(scheduler.get_value(10) - 1.0) < 0.01

    # After warmup, should stay at target
    assert abs(scheduler.get_value(20) - 1.0) < 0.01


def test_scheduler_warmup_step():
    """Test warmup (step function) schedule."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.1,
        target_value=1.0,
        warmup_epochs=10,
        schedule_type="warmup",
    )

    # Before warmup completes, should be at initial
    assert abs(scheduler.get_value(0) - 0.1) < 0.01
    assert abs(scheduler.get_value(5) - 0.1) < 0.01
    assert abs(scheduler.get_value(9) - 0.1) < 0.01

    # At and after warmup, should jump to target
    assert abs(scheduler.get_value(10) - 1.0) < 0.01
    assert abs(scheduler.get_value(20) - 1.0) < 0.01


def test_scheduler_min_value_clipping():
    """Test that values are clipped to min_value."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.0,
        target_value=0.001,
        warmup_epochs=10,
        schedule_type="linear",
        min_value=0.01,  # Higher than target
    )

    # All values should be clipped to min_value
    assert scheduler.get_value(0) >= 0.01
    assert scheduler.get_value(5) >= 0.01
    assert scheduler.get_value(10) >= 0.01


def test_scheduler_negative_warmup_epochs_raises():
    """Test that negative warmup_epochs raises ValueError."""
    with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
        ParameterScheduler(
            param_name="test_param",
            initial_value=0.0,
            target_value=1.0,
            warmup_epochs=-5,
            schedule_type="linear",
        )


def test_scheduler_invalid_schedule_type_raises():
    """Test that invalid schedule_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid schedule_type"):
        ParameterScheduler(
            param_name="test_param",
            initial_value=0.0,
            target_value=1.0,
            warmup_epochs=10,
            schedule_type="invalid_schedule",  # type: ignore
        )


def test_scheduler_repr():
    """Test scheduler string representation."""
    scheduler = ParameterScheduler(
        param_name="beta",
        initial_value=0.1,
        target_value=1.0,
        warmup_epochs=50,
        schedule_type="linear",
    )

    repr_str = repr(scheduler)
    assert "beta" in repr_str
    assert "0.1" in repr_str
    assert "1.0" in repr_str
    assert "50" in repr_str
    assert "linear" in repr_str


def test_scheduler_zero_warmup_epochs():
    """Test scheduler with zero warmup epochs."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.0,
        target_value=1.0,
        warmup_epochs=0,
        schedule_type="linear",
    )

    # With zero warmup, should immediately be at target
    assert abs(scheduler.get_value(0) - 1.0) < 0.01
    assert abs(scheduler.get_value(10) - 1.0) < 0.01


def test_scheduler_decreasing_schedule():
    """Test scheduler can decrease from high to low value."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=1.0,
        target_value=0.1,
        warmup_epochs=10,
        schedule_type="linear",
    )

    # At epoch 0, should be at initial (high)
    assert abs(scheduler.get_value(0) - 1.0) < 0.01

    # At halfway, should be decreasing
    assert 0.1 < scheduler.get_value(5) < 1.0

    # At end, should be at target (low)
    assert abs(scheduler.get_value(10) - 0.1) < 0.01


def test_scheduler_linear_exact_values():
    """Test linear schedule produces exact expected values."""
    scheduler = ParameterScheduler(
        param_name="test_param",
        initial_value=0.0,
        target_value=10.0,
        warmup_epochs=10,
        schedule_type="linear",
    )

    # Test exact values at key points
    expected_values = [
        (0, 0.0),
        (1, 1.0),
        (2, 2.0),
        (5, 5.0),
        (9, 9.0),
        (10, 10.0),
        (15, 10.0),
    ]

    for epoch, expected in expected_values:
        actual = scheduler.get_value(epoch)
        assert abs(actual - expected) < 0.01, (
            f"At epoch {epoch}, expected {expected}, got {actual}"
        )
