"""FOV selection, focus detection, and depth cropping.

Note: ``find_focus_depth`` imports waveorder lazily inside its body.
This is a documented exception for optional heavyweight dependencies.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from numpy.typing import NDArray


def find_focus_depth(
    volume: NDArray,
    na_det: float,
    lambda_ill: float,
    pixel_size: float,
) -> int:
    """Find the best focal plane across timepoints using transverse band.

    Parameters
    ----------
    volume : NDArray
        Nuclei channel data of shape ``(T, D, H, W)``.
    na_det : float
        Detection numerical aperture.
    lambda_ill : float
        Illumination wavelength in micrometers.
    pixel_size : float
        Pixel size in micrometers.

    Returns
    -------
    int
        Median best-focus depth index across timepoints.
    """
    from waveorder.focus import focus_from_transverse_band

    T = volume.shape[0]
    best_ds = []
    for t in range(T):
        best_d = focus_from_transverse_band(
            volume[t],
            NA_det=na_det,
            lambda_ill=lambda_ill,
            pixel_size=pixel_size,
        )
        best_ds.append(best_d)
    return int(np.median(best_ds))


def crop_depth(
    data: NDArray,
    focus_depth: int,
    target_depth: int,
) -> NDArray:
    """Center-crop the depth dimension around a focal plane.

    Parameters
    ----------
    data : NDArray
        Array with depth at axis 2, shape ``(T, C, D, H, W)``.
    focus_depth : int
        Best-focus depth index.
    target_depth : int
        Desired output depth.

    Returns
    -------
    NDArray
        Cropped array of shape ``(T, C, target_depth, H, W)``.
    """
    D = data.shape[2]
    if target_depth > D:
        raise ValueError(f"target_depth ({target_depth}) exceeds data depth ({D})")
    d_start = max(0, focus_depth - target_depth // 2)
    if d_start + target_depth > D:
        d_start = D - target_depth
    return data[:, :, d_start : d_start + target_depth, :, :]


def process_position(
    row: str,
    col: str,
    fov: str,
    pos: Any,
    dataset: Any,
    channel_names: list[str],
    *,
    focus_channel: str = "Nuclei",
    na_det: float = 1.25,
    lambda_ill: float = 0.405,
    pixel_size: float = 0.108,
    depth: int = 44,
    output_chunks: tuple[int, ...] = (1, 1, 8, 512, 512),
    shards_ratio: tuple[int, ...] = (2, 1, 8, 8, 8),
) -> None:
    """Process one FOV: select channels, find focus, crop depth, write.

    Parameters
    ----------
    row : str
        Row identifier for output position.
    col : str
        Column identifier for output position.
    fov : str
        FOV identifier for output position.
    pos : Position
        Input iohub Position object (must be from an open store).
    dataset : Plate
        Output iohub Plate (must already be opened for writing).
    channel_names : list[str]
        Channel names to select from the input position.
    focus_channel : str
        Channel name for focus detection (default: ``"Nuclei"``).
    na_det : float
        Detection numerical aperture.
    lambda_ill : float
        Illumination wavelength in micrometers.
    pixel_size : float
        XY pixel size in micrometers.
    depth : int
        Target depth after cropping.
    output_chunks : tuple[int, ...]
        Chunk dimensions for the output array.
    shards_ratio : tuple[int, ...]
        Shard-to-chunk ratio for the output array.
    """
    new_position = dataset.create_position(row, col, fov)

    data = pos.data
    selected_channels = [pos.get_channel_index(name) for name in channel_names]
    data = data[:, selected_channels]

    # Use index within the re-indexed channel list, not the original store.
    if focus_channel not in channel_names:
        raise ValueError(f"Focus channel {focus_channel!r} not in channel_names: {channel_names}")
    nuclei_idx = channel_names.index(focus_channel)
    nuclei = data[:, nuclei_idx]
    T, D, H, W = nuclei.shape

    focus_d = find_focus_depth(
        nuclei,
        na_det=na_det,
        lambda_ill=lambda_ill,
        pixel_size=pixel_size,
    )

    cropped_data = crop_depth(data, focus_depth=focus_d, target_depth=depth)

    expected = (T, len(channel_names), depth, H, W)
    if cropped_data.shape != expected:
        raise ValueError(f"Position {row}/{col}/{fov}: shape {cropped_data.shape}, expected {expected}")

    new_position.create_image(
        "0",
        data=cropped_data,
        chunks=output_chunks,
        shards_ratio=shards_ratio,
        transform=pos.metadata.multiscales[0].datasets[0].coordinate_transformations,
    )


def split_fovs(
    fovs: list[Any],
    num_test: int,
    num_train: int,
    seed: int = 42,
) -> tuple[list[Any], list[Any]]:
    """Shuffle FOVs and split into train/test sets.

    Parameters
    ----------
    fovs : list
        List of FOV tuples or objects.
    num_test : int
        Number of test FOVs to select.
    num_train : int
        Maximum number of train FOVs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, list]
        ``(train_fovs, test_fovs)`` lists.
    """
    rng = random.Random(seed)
    shuffled = list(fovs)
    rng.shuffle(shuffled)
    test_fovs = shuffled[:num_test]
    train_fovs = shuffled[num_test : num_test + num_train]
    return train_fovs, test_fovs
