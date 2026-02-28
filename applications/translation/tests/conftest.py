"""Test fixtures for translation application tests."""

import pytest
import torch

# Synthetic data dimensions
SYNTH_B = 2  # batch size
SYNTH_C = 1  # input channels (phase)
SYNTH_D = 5  # depth (z-stack)
SYNTH_H = 64  # height
SYNTH_W = 64  # width


@pytest.fixture
def synthetic_batch():
    """Create a synthetic batch dict matching the Sample type."""
    return {
        "source": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
        "target": torch.randn(SYNTH_B, SYNTH_C, SYNTH_D, SYNTH_H, SYNTH_W),
        "index": (
            ["row/col/pos/0" for _ in range(SYNTH_B)],
            [torch.tensor(0) for _ in range(SYNTH_B)],
            [torch.tensor(0) for _ in range(SYNTH_B)],
        ),
    }
