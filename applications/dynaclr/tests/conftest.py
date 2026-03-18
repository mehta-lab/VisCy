"""Shared fixtures and skip markers for DynaCLR integration tests."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from helpers import (
    _CHECKPOINT_PATH as CHECKPOINT_PATH,
)
from helpers import (
    _DATA_ZARR_PATH as DATA_ZARR_PATH,
)
from helpers import (
    _REFERENCE_ZARR_PATH as REFERENCE_ZARR_PATH,
)
from helpers import (
    _TRACKS_ZARR_PATH as TRACKS_ZARR_PATH,
)
from helpers import (
    SimpleEncoder,
    SyntheticTripletDataModule,
    requires_hpc_and_gpu,  # noqa: F401
)


def pytest_configure(config):
    # anndata 0.12.x zarr writer does not support pandas ArrowStringArray
    # (default in pandas 2.x with PyArrow installed)
    pd.options.future.infer_string = False
    config.addinivalue_line("markers", "hpc_integration: requires HPC paths and GPU")


@pytest.fixture
def checkpoint_path():
    return CHECKPOINT_PATH


@pytest.fixture
def reference_zarr_path():
    return REFERENCE_ZARR_PATH


@pytest.fixture
def data_zarr_path():
    return DATA_ZARR_PATH


@pytest.fixture
def tracks_zarr_path():
    return TRACKS_ZARR_PATH


@pytest.fixture
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


@pytest.fixture
def annotated_adata_zarr(annotated_adata, tmp_path) -> dict:
    """Write annotated_adata to zarr + CSV and return dataset dict."""
    zarr_path = tmp_path / "emb.zarr"
    annotated_adata.write_zarr(zarr_path)

    csv_path = tmp_path / "ann.csv"
    annotated_adata.obs[["fov_name", "id", "cell_death_state"]].to_csv(csv_path, index=False)

    return {"embeddings": str(zarr_path), "annotations": str(csv_path)}


@pytest.fixture
def simple_encoder():
    return SimpleEncoder()


@pytest.fixture
def synthetic_datamodule():
    return SyntheticTripletDataModule()
