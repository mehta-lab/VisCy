"""Tests for image grid logging helpers."""

import torch

from viscy_utils.log_images import detach_sample, render_images


def test_detach_sample_logs_all_channels_per_view():
    """Each view contributes all of its own channels, even when counts differ.

    Regression: a single ``n_channels`` taken from the first view truncated
    multi-channel ``target``/``pred`` to the 1-channel ``source`` count,
    dropping the membrane channel in virtual-staining grids.
    """
    # phase2fluor shapes: source 1ch, target/pred 2ch, (B, C, Z, Y, X).
    source = torch.rand(2, 1, 5, 16, 16)
    target = torch.rand(2, 2, 5, 16, 16)
    pred = torch.rand(2, 2, 5, 16, 16)

    rows = detach_sample((source, target, pred), log_samples_per_batch=2)

    assert len(rows) == 2
    # 1 (source) + 2 (target) + 2 (pred) = 5 columns per sample.
    assert all(len(row) == 5 for row in rows)


def test_render_images_grid_shape_matches_channels():
    """Rendered grid width covers every channel of every view."""
    source = torch.rand(3, 1, 5, 64, 64)
    target = torch.rand(3, 2, 5, 64, 64)
    pred = torch.rand(3, 2, 5, 64, 64)

    rows = detach_sample((source, target, pred), log_samples_per_batch=3)
    grid = render_images(rows)

    # 3 samples * 64 px tall; 5 channels * 64 px wide; RGB.
    assert grid.shape == (192, 320, 3)
