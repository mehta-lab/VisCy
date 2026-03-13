from __future__ import annotations

from pathlib import Path

import numpy as np
from iohub import open_ome_zarr
from pytest import TempPathFactory, fixture

channel_names = ["Phase", "GFP"]


@fixture(scope="function")
def small_hcs_dataset(tmp_path_factory: TempPathFactory) -> Path:
    """Small HCS OME-Zarr with 2 FOVs, 2 timepoints, and distinct per-FOV data."""
    dataset_path = tmp_path_factory.mktemp("small.zarr")
    zyx_shape = (2, 64, 64)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=channel_names, version="0.5") as plate:
        for i, fov_id in enumerate(("0", "1")):
            pos = plate.create_position("A", "1", fov_id)
            rng = np.random.default_rng(seed=i)
            # Offset each FOV by i*10 so per-FOV statistics are clearly different
            data = (rng.random((2, len(channel_names), *zyx_shape)) + i * 10).astype(np.float32)
            pos.create_image("0", data, chunks=(1, 1, *zyx_shape))
    return dataset_path
