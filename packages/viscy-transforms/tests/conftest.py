"""Pytest configuration for viscy-transforms tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set deterministic random seed for reproducible tests."""
    torch.manual_seed(42)
    return 42
