"""Data loading utilities for viscy-phenotyping."""

import numpy as np

__all__ = ["crop_2d"]


def crop_2d(array: np.ndarray, y: int, x: int, patch_yx: tuple[int, int]) -> np.ndarray:
    """Border-safe center crop along the last two axes of ``array``.

    Parameters
    ----------
    array : np.ndarray
        Array with shape ``(..., Y, X)``.
    y, x : int
        Requested crop center (cell centroid).
    patch_yx : tuple[int, int]
        Output patch height and width.

    Returns
    -------
    np.ndarray
        Cropped array with shape ``(..., patch_yx[0], patch_yx[1])``.
    """
    H, W = array.shape[-2], array.shape[-1]
    yh, xh = patch_yx[0] // 2, patch_yx[1] // 2
    yc = min(max(y, yh), H - yh)
    xc = min(max(x, xh), W - xh)
    return array[..., yc - yh : yc + yh, xc - xh : xc + xh]
