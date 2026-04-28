"""CellProfiler-style measurements via the cp-measure library.

Wraps four core cp-measure groups for use on single-cell image patches:
- intensity   : MeasureObjectIntensity (per channel)
- sizeshape   : MeasureObjectSizeShape (channel-independent, mask only)
- texture     : MeasureTexture / Haralick (per channel)
- granularity : MeasureGranularity (per channel)

All functions take a 2-D single-channel image patch and/or a binary nuclear
mask and return a flat ``dict[str, float]`` with one value per feature.
The caller is responsible for adding channel prefixes.
"""

import warnings

import numpy as np

__all__ = [
    "cp_intensity_features",
    "cp_sizeshape_features",
    "cp_texture_features",
    "cp_granularity_features",
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from cp_measure.bulk import get_core_measurements

_CORE = get_core_measurements()


def _extract(result: dict, label_idx: int = 0) -> dict[str, float]:
    """Extract scalar at label_idx from each per-object result array."""
    return {k: float(v[label_idx]) for k, v in result.items()}


def cp_intensity_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """CellProfiler MeasureObjectIntensity features (per channel).

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask for the target cell.

    Returns
    -------
    dict[str, float]
        Keys follow CellProfiler naming: ``Intensity_MeanIntensity``,
        ``Intensity_StdIntensity``, etc.
    """
    cell_mask = mask.astype(np.int32)
    return _extract(_CORE["intensity"](cell_mask, image.astype(np.float64)))


def cp_sizeshape_features(mask: np.ndarray) -> dict[str, float]:
    """CellProfiler MeasureObjectSizeShape features (channel-independent).

    Parameters
    ----------
    mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask for the target cell.

    Returns
    -------
    dict[str, float]
        Keys follow CellProfiler naming: ``Area``, ``Perimeter``,
        ``Eccentricity``, ``FormFactor``, etc.
    """
    cell_mask = mask.astype(np.int32)
    return _extract(_CORE["sizeshape"](cell_mask, None))


def cp_texture_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """CellProfiler MeasureTexture (Haralick) features (per channel).

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask for the target cell.

    Returns
    -------
    dict[str, float]
        Keys follow CellProfiler naming: ``AngularSecondMoment_3_00_256``,
        ``Contrast_3_00_256``, ``Entropy_3_00_256``, etc.
    """
    cell_mask = mask.astype(np.int32)
    img = image.astype(np.float64)
    # cp_measure's texture module calls skimage.img_as_ubyte which requires [0, 1]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return _extract(_CORE["texture"](cell_mask, img))


def cp_granularity_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """CellProfiler MeasureGranularity features (per channel).

    Parameters
    ----------
    image : np.ndarray, shape (Y, X)
        Single-channel fluorescence patch.
    mask : np.ndarray, shape (Y, X), bool or int
        Binary nuclear mask for the target cell.

    Returns
    -------
    dict[str, float]
        Keys follow CellProfiler naming: ``Granularity_1`` … ``Granularity_16``.
    """
    cell_mask = mask.astype(np.int32)
    return _extract(_CORE["granularity"](cell_mask, image.astype(np.float64)))
