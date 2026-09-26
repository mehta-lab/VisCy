"""Tests for :func:`dynacell.evaluation.segmentation.segment` input handling."""

from __future__ import annotations

import numpy as np
import pytest

from dynacell.evaluation.segmentation import segment


@pytest.mark.parametrize("target_name", ["er", "mitochondria"])
def test_classical_workflows_leave_the_input_untouched(target_name: str) -> None:
    """The aicssegmentation workflows clip their input in place; segment() must not pass that on.

    The eval segments the GT volume and then scores pixel metrics on the SAME array, so an
    in-place clip silently scores a clipped GT whenever the mask cache is cold (measured:
    SI_SSIM 0.4225 -> 0.3274 on one iPSC ER FOV).
    """
    rng = np.random.default_rng(0)
    img = rng.gamma(2.0, 1.0, size=(8, 64, 64)).astype(np.float32)
    img[4, 32, 32] = 1e3  # an outlier far above mean + 7.5 std, which the workflow clips
    before = img.copy()
    mask = segment(img, target_name)
    assert mask.shape == img.shape and mask.dtype == bool
    np.testing.assert_array_equal(img, before)
