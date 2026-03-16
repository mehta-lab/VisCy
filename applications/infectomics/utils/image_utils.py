"""Shared image normalization and patch extraction utilities."""

import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an image to [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image
    return (image - min_val) / (max_val - min_val)


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """Z-score normalize an image (zero mean, unit variance)."""
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image
    return (image - mean) / std


def get_patch_2d(data: np.ndarray, x_centroid: int, y_centroid: int, patch_size: int) -> np.ndarray:
    """
    Extract a 2D patch (Y, X) centered on (x_centroid, y_centroid).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (Y, X).
    x_centroid, y_centroid : int
        Center of the patch in pixel coordinates.
    patch_size : int
        Side length of the square patch.

    Returns
    -------
    np.ndarray
        Cropped patch, potentially smaller near image borders.
    """
    x_start = max(0, x_centroid - patch_size // 2)
    x_end = min(data.shape[1], x_centroid + patch_size // 2)
    y_start = max(0, y_centroid - patch_size // 2)
    y_end = min(data.shape[0], y_centroid + patch_size // 2)
    return data[int(y_start):int(y_end), int(x_start):int(x_end)]


def get_patch_zyx(data: np.ndarray, x_centroid: int, y_centroid: int, patch_size: int) -> np.ndarray:
    """
    Extract a volumetric patch (Z, Y, X) centered on (x_centroid, y_centroid).

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (Z, Y, X).
    x_centroid, y_centroid : int
        Center of the patch in pixel coordinates (Z-axis is kept whole).
    patch_size : int
        Side length of the square XY patch.

    Returns
    -------
    np.ndarray
        Cropped patch of shape (Z, patch_size, patch_size) or smaller near borders.
    """
    x_start = max(0, x_centroid - patch_size // 2)
    x_end = min(data.shape[2], x_centroid + patch_size // 2)
    y_start = max(0, y_centroid - patch_size // 2)
    y_end = min(data.shape[1], y_centroid + patch_size // 2)
    return data[:, int(y_start):int(y_end), int(x_start):int(x_end)]


def get_patch_tcyx(
    data: np.ndarray, cell_centroid: tuple[float, float], patch_size: int
) -> np.ndarray:
    """
    Extract a (C, Y, X) patch from a (C, Y, X) array using an (x, y) centroid.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (C, Y, X).
    cell_centroid : tuple of (x, y)
        Center of the patch.
    patch_size : int
        Side length of the square patch.

    Returns
    -------
    np.ndarray
        Cropped patch of shape (C, patch_size, patch_size) or smaller at borders.
    """
    x_centroid, y_centroid = cell_centroid
    x_start = max(0, x_centroid - patch_size // 2)
    x_end = min(data.shape[2], x_centroid + patch_size // 2)
    y_start = max(0, y_centroid - patch_size // 2)
    y_end = min(data.shape[1], y_centroid + patch_size // 2)
    return data[:, :, int(y_start):int(y_end), int(x_start):int(x_end)]
