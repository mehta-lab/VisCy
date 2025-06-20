from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from pytest import TempPathFactory, fixture

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

channel_names = ["Phase", "Retardance", "GFP", "DAPI"]


def _build_hcs(
    path: Path,
    channel_names: list[str],
    zyx_shape: tuple[int, int, int],
    dtype: DTypeLike,
    max_value: int | float,
):
    dataset = open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    for row in ("A", "B"):
        for col in ("1", "2"):
            for fov in ("0", "1", "2", "3"):
                pos = dataset.create_position(row, col, fov)
                pos.create_image(
                    "0",
                    (
                        np.random.rand(2, len(channel_names), *zyx_shape) * max_value
                    ).astype(dtype),
                )


@fixture(scope="session")
def preprocessed_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("preprocessed.zarr")
    _build_hcs(dataset_path, channel_names, (12, 256, 256), np.float32, 1.0)
    # U[0, 1)
    expected = {"mean": 0.5, "std": 1 / np.sqrt(12), "median": 0.5, "iqr": 0.5}
    norm_meta = {channel: {"dataset_statistics": expected} for channel in channel_names}
    with open_ome_zarr(dataset_path, mode="r+") as dataset:
        dataset.zattrs["normalization"] = norm_meta
        for _, fov in dataset.positions():
            fov.zattrs["normalization"] = norm_meta
    return dataset_path


@fixture(scope="function")
def small_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("small.zarr")
    _build_hcs(dataset_path, channel_names, (12, 64, 64), np.uint16, 1)
    return dataset_path


@fixture(scope="function")
def small_hcs_labels(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset with labels."""
    dataset_path = tmp_path_factory.mktemp("small_with_labels.zarr")
    _build_hcs(
        dataset_path, ["nuclei_labels", "membrane_labels"], (12, 64, 64), np.uint16, 50
    )
    return dataset_path


@fixture(scope="function")
def labels_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("labels.zarr")
    _build_hcs(dataset_path, ["DAPI", "GFP"], (2, 16, 16), np.uint16, 3)
    return dataset_path


@fixture(scope="function")
def tracks_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a HCS OME-Zarr dataset with tracking CSV results."""
    dataset_path = tmp_path_factory.mktemp("tracks.zarr")
    _build_hcs(dataset_path, ["nuclei_labels"], (1, 256, 256), np.uint16, 3)
    for fov_name, _ in open_ome_zarr(dataset_path).positions():
        fake_tracks = pd.DataFrame(
            {
                "track_id": [0, 1],
                "t": [0, 1],
                "y": [100, 200],
                "x": [96, 160],
                "id": [0, 1],
                "parent_track_id": [-1, -1],
                "parent_id": [-1, -1],
            }
        )
        fake_tracks.to_csv(dataset_path / fov_name / "tracks.csv", index=False)
    return dataset_path
