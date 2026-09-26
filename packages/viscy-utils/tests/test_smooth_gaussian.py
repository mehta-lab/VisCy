"""Tests for the GPU-dispatching Gaussian blur behind the Otsu threshold.

Unlike the median, a Gaussian computes new values by weighted summation, so
the backends reassociate floating-point additions differently and agree only
to a tolerance -- these tests assert closeness, not equality. The blur only
ever picks a threshold, so a last-bit difference cannot change a mask except
for a voxel sitting exactly on the threshold.

Dispatch goes through ``cubic.skimage.filters``, whose proxies select the
backend from the input array's device. The GPU tests are skipped, not failed,
when cubic or a device is absent: viscy-utils does not depend on cubic, which
ships in dynacell's eval/preprocess extras.
"""

import numpy as np
import pytest
from skimage.filters import gaussian

from viscy_utils.meta_utils import _OTSU_BLUR_SIGMA, otsu_threshold_from_volume, smooth_gaussian

# The footprint otsu_threshold_from_volume uses: no blur across T, sigma in ZYX.
_SIGMA = (0,) + (_OTSU_BLUR_SIGMA,) * 3
_SHAPE = (2, 8, 16, 24)


def _volume(seed: int) -> np.ndarray:
    """Bimodal float32 array, so Otsu has two populations to separate."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(400.0, 20.0, size=_SHAPE).astype(np.float32)
    arr[..., 4:12, 6:18] += 300.0
    return arr


def test_matches_skimage_on_the_cpu_path() -> None:
    """Whichever backend runs, the result matches skimage's to float32 tolerance."""
    arr = _volume(seed=0)
    expected = gaussian(arr, sigma=_SIGMA, preserve_range=True)
    assert np.allclose(smooth_gaussian(arr, sigma=_SIGMA), expected, rtol=1e-5, atol=1e-4)


def test_preserve_range_is_honoured() -> None:
    """Without preserve_range skimage rescales via img_as_float, moving the threshold.

    The blurred values must stay in the input's intensity units, because the
    threshold Otsu picks from them is written to zattrs and later compared
    against raw voxel values.
    """
    arr = _volume(seed=1)
    out = smooth_gaussian(arr, sigma=_SIGMA)
    assert out.min() > 300.0
    assert out.max() < 800.0


def test_cubic_gpu_and_cpu_agree() -> None:
    """Pin cubic's GPU blur against skimage, the assumption the speedup rests on."""
    pytest.importorskip("cubic")
    torch = pytest.importorskip("torch")
    from cubic.cuda import ascupy, asnumpy
    from cubic.skimage import filters as cubic_filters

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    arr = _volume(seed=2)
    on_gpu = asnumpy(cubic_filters.gaussian(ascupy(arr), sigma=_SIGMA, preserve_range=True))
    assert np.allclose(on_gpu, gaussian(arr, sigma=_SIGMA, preserve_range=True), rtol=1e-4, atol=1e-3)


def test_threshold_separates_a_bimodal_volume() -> None:
    """The whole recipe, end to end: the threshold must land between the two modes."""
    arr = _volume(seed=3)
    threshold = otsu_threshold_from_volume(arr)
    assert 400.0 < threshold < 700.0


def test_constant_volume_returns_its_constant() -> None:
    """Otsu is undefined for a constant input; the recipe returns the value itself."""
    arr = np.full(_SHAPE, 42.0, dtype=np.float32)
    assert otsu_threshold_from_volume(arr) == pytest.approx(42.0)
