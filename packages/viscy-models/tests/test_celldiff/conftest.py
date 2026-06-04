"""Shared test fixtures for CellDiff models."""

import pytest


@pytest.fixture
def small_config() -> dict:
    """Small model config for fast CPU tests.

    2 downsamples at stride (1,2,2): [8,64,64] -> [8,16,16]
    patch_size=4: grid [2,4,4] = 32 tokens, hidden_size=64, num_heads=4, dim_head=16
    """
    return dict(
        input_spatial_size=[8, 64, 64],
        dims=[16, 32, 64],
        num_res_block=[2, 2],
        hidden_size=64,
        num_heads=4,
        dim_head=16,
        num_hidden_layers=1,
        patch_size=4,
    )
