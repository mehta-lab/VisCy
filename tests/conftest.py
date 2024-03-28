from pathlib import Path

import numpy as np
from iohub import open_ome_zarr
from pytest import TempPathFactory, fixture

channel_names = ["Phase", "Retardance", "GFP", "DAPI"]


def _build_hcs(path: Path, channel_names: list[str], zyx_shape: tuple[int, int, int]):
    dataset = open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    for row in ("A", "B"):
        for col in ("1", "2"):
            for fov in ("0", "1"):
                pos = dataset.create_position(row, col, fov)
                pos.create_image(
                    "0",
                    np.random.rand(2, len(channel_names), *zyx_shape).astype(
                        np.float32
                    ),
                )


@fixture(scope="session")
def preprocessed_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a preprocessed HCS OME-Zarr dataset."""
    dataset_path = tmp_path_factory.mktemp("preprocessed.zarr")
    _build_hcs(dataset_path, channel_names, (12, 256, 256))
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
    _build_hcs(dataset_path, channel_names, (12, 64, 64))
    return dataset_path
