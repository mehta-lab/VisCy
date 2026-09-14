"""Tests for the GPU-dispatching median filter behind the fg_mask pass.

The filter is the whole cost of ``generate_fg_masks`` -- 3.68 s per
(44, 624, 924) float32 volume on the CPU versus 0.13 s on an A40 including
both transfers. That only makes it a safe swap if the two backends agree
exactly, which they should: a median filter SELECTS an existing element
rather than computing a new one, so there is no floating-point
reassociation for the backends to disagree about.

Dispatch goes through ``cubic.scipy.ndimage``, whose proxies select the
backend from the input array's device -- the same route
``dynacell.evaluation.metrics`` takes. The GPU test is skipped, not failed,
when cubic or a device is absent: viscy-utils does not depend on cubic, which
ships in dynacell's eval/preprocess extras.
"""

import numpy as np
import pytest
from scipy.ndimage import median_filter

from viscy_utils.meta_utils import smooth_median

# The two footprints the callers actually use: 4-D for the grid-sampled Otsu
# pass (T, C-collapsed, Y, X) and 3-D for the full-resolution mask pass (Z, Y, X).
_SIZES = [(1, 1, 3, 3), (1, 3, 3)]


def _volume(size: tuple[int, ...], seed: int) -> np.ndarray:
    """Bimodal float32 array of matching rank, so the filter has structure to act on."""
    rng = np.random.default_rng(seed)
    shape = (2, 3, 16, 24) if len(size) == 4 else (5, 16, 24)
    arr = rng.normal(400.0, 20.0, size=shape).astype(np.float32)
    arr[..., 4:12, 6:18] += 300.0
    return arr


@pytest.mark.parametrize("size", _SIZES)
def test_matches_scipy_on_the_cpu_path(size: tuple[int, ...]) -> None:
    """Whichever backend runs, the result equals SciPy's exactly."""
    arr = _volume(size, seed=0)
    assert np.array_equal(smooth_median(arr, size=size), median_filter(arr, size=size))


@pytest.mark.parametrize("size", _SIZES)
def test_cubic_gpu_and_cpu_agree_bitwise(size: tuple[int, ...]) -> None:
    """Pin cubic's GPU path == SciPy, the assumption the speedup rests on."""
    pytest.importorskip("cubic")
    torch = pytest.importorskip("torch")
    from cubic.cuda import ascupy, asnumpy
    from cubic.scipy import ndimage as cubic_ndimage

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    arr = _volume(size, seed=1)
    on_gpu = asnumpy(cubic_ndimage.median_filter(ascupy(arr), size=size))
    assert np.array_equal(on_gpu, median_filter(arr, size=size))


@pytest.mark.parametrize("size", _SIZES)
def test_cubic_cpu_path_agrees_bitwise(size: tuple[int, ...]) -> None:
    """A numpy input must route to cubic's CPU path and still match SciPy."""
    pytest.importorskip("cubic")
    from cubic.cuda import asnumpy
    from cubic.scipy import ndimage as cubic_ndimage

    arr = _volume(size, seed=3)
    on_cpu = asnumpy(cubic_ndimage.median_filter(arr, size=size))
    assert np.array_equal(on_cpu, median_filter(arr, size=size))


def test_preserves_dtype_and_shape() -> None:
    """The mask pass casts the result to uint8, so dtype must not drift to float64."""
    arr = _volume((1, 3, 3), seed=2)
    out = smooth_median(arr, size=(1, 3, 3))
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    assert isinstance(out, np.ndarray)
