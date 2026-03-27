"""Tests for CELLDiffNet flow-matching backbone."""

import pytest
import torch

pytest.importorskip("einops")
pytest.importorskip("diffusers")

from viscy_models.celldiff import CELLDiffNet  # noqa: E402


def test_forward(small_config):
    """Forward with (x, cond, t) -> (B, in_channels, D, H, W)."""
    model = CELLDiffNet(in_channels=1, **small_config)
    x = torch.randn(2, 1, 8, 64, 64)
    cond = torch.randn(2, 1, 8, 64, 64)
    t = torch.rand(2)
    y = model(x, cond, t)
    assert y.shape == (2, 1, 8, 64, 64)


def test_forward_multi_channel(small_config):
    """Multi-channel input: in_channels=2 produces matching output."""
    model = CELLDiffNet(in_channels=2, **small_config)
    x = torch.randn(1, 2, 8, 64, 64)
    cond = torch.randn(1, 1, 8, 64, 64)
    t = torch.rand(1)
    y = model(x, cond, t)
    assert y.shape == (1, 2, 8, 64, 64)
