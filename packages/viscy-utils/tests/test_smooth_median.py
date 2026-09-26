"""Tests for the GPU-dispatching median filter behind the fg_mask pass.

The filter is the whole cost of both Otsu passes -- 42.8 s of the 43.9 s a
(44, 624, 924) float32 volume takes on the CPU with the 5x5x5 cube, against
~0.1 s on an A40 including both transfers. That only makes it a safe swap if
the two backends agree exactly, which they should: a median filter SELECTS an
existing element rather than computing a new one, so there is no
floating-point reassociation for the backends to disagree about.

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

# The two footprints the callers actually use: the 4-D (1, 5, 5, 5) of
# otsu_threshold_from_volume over a (T, Z, Y, X) volume, and the bare 5 cube
# generate_fg_masks applies per timepoint to (Z, Y, X). Both span Z, which is
# where the accuracy comes from -- restricting them to the plane scores worse
# than the decimated (1, 3, 3) recipe they replaced.
_SIZES = [(1, 5, 5, 5), 5]
_SHAPES = {(1, 5, 5, 5): (2, 8, 16, 24), 5: (8, 16, 24)}


def _volume(size, seed: int) -> np.ndarray:
    """Bimodal float32 array of matching rank, so the filter has structure to act on."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(400.0, 20.0, size=_SHAPES[size]).astype(np.float32)
    arr[..., 4:12, 6:18] += 300.0
    return arr


@pytest.mark.parametrize("size", _SIZES)
def test_matches_scipy_on_the_cpu_path(size) -> None:
    """Whichever backend runs, the result equals SciPy's exactly."""
    arr = _volume(size, seed=0)
    assert np.array_equal(smooth_median(arr, size=size), median_filter(arr, size=size))


@pytest.mark.parametrize("size", _SIZES)
def test_cubic_gpu_and_cpu_agree_bitwise(size) -> None:
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
def test_cubic_cpu_path_agrees_bitwise(size) -> None:
    """A numpy input must route to cubic's CPU path and still match SciPy."""
    pytest.importorskip("cubic")
    from cubic.cuda import asnumpy
    from cubic.scipy import ndimage as cubic_ndimage

    arr = _volume(size, seed=3)
    on_cpu = asnumpy(cubic_ndimage.median_filter(arr, size=size))
    assert np.array_equal(on_cpu, median_filter(arr, size=size))


def test_preserves_dtype_and_shape() -> None:
    """The mask pass casts the result to uint8, so dtype must not drift to float64."""
    arr = _volume(5, seed=2)
    out = smooth_median(arr, size=5)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    assert isinstance(out, np.ndarray)
