"""Tests for prediction writer blending utilities."""

import numpy as np
import torch

from viscy_utils.callbacks.prediction_writer import _blend_in


def test_blend_in_consistency():
    """Verify _blend_in produces identical results for torch and numpy inputs."""
    depth = 5
    shape_4d = (2, depth, 8, 8)  # C, Z, Y, X (numpy from HCSPredictionWriter)

    rng = np.random.default_rng(42)
    old_np = rng.random(shape_4d).astype(np.float32)
    new_np = rng.random(shape_4d).astype(np.float32)
    old_torch = torch.from_numpy(old_np).unsqueeze(0)  # (1, C, Z, Y, X)
    new_torch = torch.from_numpy(new_np).unsqueeze(0)

    z_slice = slice(2, 2 + depth)

    result_np = _blend_in(old_np, new_np, z_slice)
    result_torch = _blend_in(old_torch, new_torch, z_slice)

    np.testing.assert_allclose(result_np, result_torch.squeeze(0).numpy(), rtol=1e-5, atol=1e-5)


def test_blend_in_zero_start():
    """Verify _blend_in returns new_stack unchanged when z_slice starts at 0."""
    old = np.ones((2, 5, 8, 8), dtype=np.float32)
    new = np.zeros((2, 5, 8, 8), dtype=np.float32)
    result = _blend_in(old, new, slice(0, 5))
    np.testing.assert_array_equal(result, new)


def test_blend_in_torch_preserves_dtype():
    """Verify _blend_in preserves torch tensor dtype."""
    old = torch.ones(1, 2, 5, 8, 8, dtype=torch.float32)
    new = torch.zeros(1, 2, 5, 8, 8, dtype=torch.float32)
    result = _blend_in(old, new, slice(2, 7))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
