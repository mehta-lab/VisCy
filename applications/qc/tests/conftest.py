"""Test fixtures for QC metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from iohub import open_ome_zarr
from pytest import TempPathFactory, fixture

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

CHANNEL_NAMES = ["Phase", "Retardance"]
NUM_TIMEPOINTS = 5
ZYX_SHAPE = (10, 64, 64)


def _build_temporal_hcs(
    path: Path,
    channel_names: list[str],
    num_timepoints: int,
    zyx_shape: tuple[int, int, int],
    dtype: DTypeLike,
):
    dataset = open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    for row in ("A",):
        for col in ("1",):
            for fov in ("0", "1"):
                pos = dataset.create_position(row, col, fov)
                rng = np.random.default_rng(42)
                pos.create_image(
                    "0",
                    rng.random((num_timepoints, len(channel_names), *zyx_shape)).astype(dtype),
                    chunks=(1, 1, *zyx_shape),
                )
    dataset.close()


@fixture(scope="session")
def temporal_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a temporal HCS OME-Zarr dataset for QC tests."""
    dataset_path = tmp_path_factory.mktemp("temporal_qc.zarr")
    _build_temporal_hcs(
        dataset_path,
        CHANNEL_NAMES,
        NUM_TIMEPOINTS,
        ZYX_SHAPE,
        np.float32,
    )
    return dataset_path


MULTI_WELL_CHANNELS = ["Phase", "Fluorescence_405"]


@fixture(scope="session")
def multi_well_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Provides a multi-well HCS OME-Zarr dataset for annotation tests."""
    dataset_path = tmp_path_factory.mktemp("multi_well_qc.zarr")
    dataset = open_ome_zarr(
        dataset_path,
        layout="hcs",
        mode="w",
        channel_names=MULTI_WELL_CHANNELS,
    )
    for col in ("1", "2"):
        for fov in ("0",):
            pos = dataset.create_position("A", col, fov)
            rng = np.random.default_rng(42)
            pos.create_image(
                "0",
                rng.random(
                    (NUM_TIMEPOINTS, len(MULTI_WELL_CHANNELS), *ZYX_SHAPE)
                ).astype(np.float32),
                chunks=(1, 1, *ZYX_SHAPE),
            )
    dataset.close()
    return dataset_path
