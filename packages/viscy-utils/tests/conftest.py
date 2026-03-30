from __future__ import annotations

import pandas as pd

# anndata 0.12.x zarr writer does not support pandas ArrowStringArray (default in pandas 2.x with PyArrow installed)
pd.options.future.infer_string = False

from pathlib import Path  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
from iohub import open_ome_zarr  # noqa: E402
from pytest import TempPathFactory, fixture  # noqa: E402

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


@fixture
def annotated_adata() -> ad.AnnData:
    """Synthetic AnnData with cell_death_state labels for classifier tests."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 16
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    fov_names = [f"A/{(i % 4) + 1}/0" for i in range(n_samples)]
    labels = (["alive"] * 20) + (["dead"] * 20) + (["apoptotic"] * 20)
    obs = pd.DataFrame(
        {
            "fov_name": fov_names,
            "id": np.arange(n_samples),
            "cell_death_state": labels,
        }
    )
    return ad.AnnData(X=X, obs=obs)


@fixture
def annotated_adata_zarr(annotated_adata, tmp_path) -> dict:
    """Write annotated_adata to zarr + CSV and return dataset dict."""
    zarr_path = tmp_path / "emb.zarr"
    annotated_adata.write_zarr(zarr_path)

    csv_path = tmp_path / "ann.csv"
    annotated_adata.obs[["fov_name", "id", "cell_death_state"]].to_csv(csv_path, index=False)

    return {"embeddings": str(zarr_path), "annotations": str(csv_path)}
