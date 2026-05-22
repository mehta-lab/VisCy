"""Utilities for CTC tracking accuracy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def seg_dir(dataset_dir: Path, sequence: str) -> Path:
    """Return path to the error-segmentation directory for a CTC sequence.

    Parameters
    ----------
    dataset_dir : Path
        CTC dataset root (e.g. .../BF-C2DL-HSC).
    sequence : str
        Sequence number (e.g. "01").
    """
    return dataset_dir / f"{sequence}_ERR_SEG"


def pad_to_shape(image: NDArray, shape: tuple[int, int], mode: str) -> NDArray:
    """Pad image symmetrically to at least the given spatial shape.

    Parameters
    ----------
    image : NDArray
        2-D array to pad.
    shape : tuple[int, int]
        Target (height, width). No-op if image is already large enough.
    mode : str
        Padding mode passed to ``np.pad``.
    """
    diff = np.asarray(shape) - np.asarray(image.shape)
    if diff.sum() == 0:
        return image
    left = diff // 2
    right = diff - left
    return np.pad(image, tuple(zip(left, right)), mode=mode)


def normalize_crop(crop: NDArray, frame_mean: float, frame_std: float) -> NDArray:
    """Z-score normalize a cell crop using whole-frame statistics.

    Matches the training normalization (``NormalizeSampled`` with
    ``level=timepoint_statistics``): mean/std are computed over the full
    frame, not the cell foreground, so the model sees the same intensity
    distribution it was trained on.

    Parameters
    ----------
    crop : NDArray
        Float32 2-D cell image.
    frame_mean : float
        Mean pixel intensity of the full frame at this timepoint.
    frame_std : float
        Std pixel intensity of the full frame at this timepoint.

    Returns
    -------
    NDArray
        Z-score normalized crop.
    """
    return (crop - frame_mean) / max(frame_std, 1e-8)
