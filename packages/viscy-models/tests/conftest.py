"""Shared test fixtures for viscy-models."""

import pytest
import torch


@pytest.fixture
def device() -> str:
    """Return the best available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
