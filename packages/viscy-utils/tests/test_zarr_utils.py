"""Writing AnnData zarr stores under pandas 3's default Arrow-backed strings."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata._io.specs.registry import IORegistryError

from viscy_utils.callbacks.embedding_writer import write_embedding_dataset
from viscy_utils.evaluation.zarr_utils import write_anndata_zarr


@pytest.fixture
def arrow_adata() -> ad.AnnData:
    """AnnData built under pandas' default, so its strings are Arrow-backed."""
    assert pd.options.future.infer_string, "pandas 3 default must be on for this test"
    obs = pd.DataFrame(
        {
            "fov_name": ["A/1/0", "B/2/0", "A/1/0"],
            "state": pd.Categorical(["alive", "dead", "alive"]),
            "id": [0, 1, 2],
        }
    )
    return ad.AnnData(X=np.zeros((3, 2), dtype=np.float32), obs=obs)


def test_plain_write_zarr_still_rejects_arrow_strings(arrow_adata, tmp_path) -> None:
    """Why the helper exists; fails once anndata writes Arrow strings natively."""
    with pytest.raises(IORegistryError, match="ArrowStringArray"):
        arrow_adata.write_zarr(tmp_path / "plain.zarr")


def test_write_anndata_zarr_round_trips_arrow_strings(arrow_adata, tmp_path) -> None:
    """Strings survive the write, and pandas' global option is left as it was."""
    write_anndata_zarr(arrow_adata, tmp_path / "emb.zarr")

    back = ad.read_zarr(tmp_path / "emb.zarr")
    assert list(back.obs["fov_name"]) == ["A/1/0", "B/2/0", "A/1/0"]
    assert list(back.obs["state"]) == ["alive", "dead", "alive"]
    assert list(back.obs_names) == ["0", "1", "2"]
    assert pd.options.future.infer_string


def test_write_embedding_dataset_leaves_pandas_default(tmp_path) -> None:
    """The embedding writer no longer switches string inference off process-wide."""
    index_df = pd.DataFrame({"fov_name": ["/A/1/0", "/B/2/0"], "track_id": [1, 2], "t": [0, 0], "id": [0, 1]})
    write_embedding_dataset(tmp_path / "emb.zarr", np.ones((2, 4), dtype=np.float32), index_df)

    back = ad.read_zarr(tmp_path / "emb.zarr")
    assert list(back.obs["fov_name"]) == ["A/1/0", "B/2/0"]
    assert pd.options.future.infer_string
