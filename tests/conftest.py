from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from pytest import FixtureRequest, TempPathFactory, fixture

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

channel_names = ["Phase", "Retardance", "GFP", "DAPI"]


def _build_hcs(
    path: Path,
    channel_names: list[str],
    zyx_shape: tuple[int, int, int],
    dtype: DTypeLike,
    max_value: int | float,
    sharded: bool = False,
    multiscales: bool = False,
):
    dataset = open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
        version="0.4" if not sharded else "0.5",
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
                    chunks=(1, 1, 1, *zyx_shape[1:]),
                    shards_ratio=(2, len(channel_names), zyx_shape[0], 1, 1)
                    if sharded
                    else None,
                )
                if multiscales:
                    pos["1"] = pos["0"][::2, :, ::2, ::2, ::2]


@fixture(scope="session")
def preprocessed_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("preprocessed.zarr")
    _build_hcs(
        dataset_path, channel_names, (12, 256, 256), np.float32, 1.0, multiscales=True
    )
    # U[0, 1)
    expected = {"mean": 0.5, "std": 1 / np.sqrt(12), "median": 0.5, "iqr": 0.5}
    norm_meta = {channel: {"dataset_statistics": expected} for channel in channel_names}
    with open_ome_zarr(dataset_path, mode="r+") as dataset:
        dataset.zattrs["normalization"] = norm_meta
        for _, fov in dataset.positions():
            fov.zattrs["normalization"] = norm_meta
    return dataset_path


@fixture(scope="function", params=[False, True])
def small_hcs_dataset(
    tmp_path_factory: TempPathFactory, request: FixtureRequest
) -> Path:
    """Provides a small, not preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("small.zarr")
    _build_hcs(
        dataset_path, channel_names, (12, 64, 64), np.uint16, 1, sharded=request.param
    )
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


@fixture(scope="function")
def tracks_with_gaps_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a HCS OME-Zarr dataset with tracking results with gaps in time."""
    dataset_path = tmp_path_factory.mktemp("tracks_gaps.zarr")
    _build_hcs(dataset_path, ["nuclei_labels"], (1, 256, 256), np.uint16, 3)

    # Define different track patterns for different FOVs
    track_patterns = {
        "A/1/0": [
            # Track 0: complete sequence t=[0,1,2,3]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            {"track_id": 0, "t": 1, "y": 128, "x": 128, "id": 1},
            {"track_id": 0, "t": 2, "y": 128, "x": 128, "id": 2},
            {"track_id": 0, "t": 3, "y": 128, "x": 128, "id": 3},
            # Track 1: ends early t=[0,1]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 4},
            {"track_id": 1, "t": 1, "y": 100, "x": 100, "id": 5},
        ],
        "A/1/1": [
            # Track 0: gap at t=2, has t=[0,1,3]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            {"track_id": 0, "t": 1, "y": 128, "x": 128, "id": 1},
            {"track_id": 0, "t": 3, "y": 128, "x": 128, "id": 2},
            # Track 1: even timepoints only t=[0,2,4]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 3},
            {"track_id": 1, "t": 2, "y": 100, "x": 100, "id": 4},
            {"track_id": 1, "t": 4, "y": 100, "x": 100, "id": 5},
        ],
        "A/2/0": [
            # Track 0: single timepoint t=[0]
            {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            # Track 1: complete short sequence t=[0,1,2]
            {"track_id": 1, "t": 0, "y": 100, "x": 100, "id": 1},
            {"track_id": 1, "t": 1, "y": 100, "x": 100, "id": 2},
            {"track_id": 1, "t": 2, "y": 100, "x": 100, "id": 3},
        ],
    }

    for fov_name, _ in open_ome_zarr(dataset_path).positions():
        if fov_name in track_patterns:
            tracks_data = track_patterns[fov_name]
        else:
            # Default tracks for other FOVs
            tracks_data = [
                {"track_id": 0, "t": 0, "y": 128, "x": 128, "id": 0},
            ]

        tracks_df = pd.DataFrame(tracks_data)
        tracks_df["parent_track_id"] = -1
        tracks_df["parent_id"] = -1
        tracks_df.to_csv(dataset_path / fov_name / "tracks.csv", index=False)

    return dataset_path
