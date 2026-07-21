"""Tests for viscy_data.cell_index.preprocess_cell_index — zattrs vs CSV sidecar paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from iohub import open_ome_zarr

from viscy_data.cell_index import preprocess_cell_index, read_cell_index, write_cell_index
from viscy_data.meta_csv import metadata_to_rows, write_meta_rows_csv

NORM_META = {
    "fov_statistics": {"mean": 1.0, "std": 0.1, "median": 1.0, "iqr": 0.2, "max": 5.0, "min": 0.0},
    "timepoint_statistics": {
        "0": {"mean": 1.0, "std": 0.1, "median": 1.0, "iqr": 0.2, "max": 5.0, "min": 0.0},
        "1": {"mean": 1.2, "std": 0.15, "median": 1.1, "iqr": 0.25, "max": 5.5, "min": 0.0},
    },
}
FOCUS_META = {
    "fov_statistics": {"z_focus_mean": 4.0, "z_focus_std": 0.5},
    "per_timepoint": {"0": 4, "1": 5},
}


@pytest.fixture
def zarr_store(tmp_path):
    """Single-FOV, single-channel, 2-timepoint HCS store (no metadata written)."""
    dataset_path = tmp_path / "store.zarr"
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=["GFP"]) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_image("0", np.zeros((2, 1, 2, 4, 4), dtype=np.float32))
    return dataset_path


@pytest.fixture
def cell_index_df(zarr_store):
    """Minimal 2-row (one per timepoint) cell index for a single FOV/channel."""
    return pd.DataFrame(
        {
            "cell_id": ["c0", "c1"],
            "experiment": "exp",
            "store_path": str(zarr_store),
            "tracks_path": str(zarr_store),
            "fov": "A/1/0",
            "well": "A/1",
            "y": np.array([1.0, 2.0], dtype=np.float32),
            "x": np.array([1.0, 2.0], dtype=np.float32),
            "z": np.array([0, 0], dtype=np.int16),
            "perturbation": "none",
            "channel_name": "GFP",
            "microscope": "",
            "t": pd.array([0, 1], dtype="Int32"),
        }
    )


def _write_zattrs(zarr_store):
    with open_ome_zarr(zarr_store, mode="r+") as plate:
        for _, pos in plate.positions():
            pos.zattrs["normalization"] = {"GFP": NORM_META}
            pos.zattrs["focus_slice"] = {"GFP": FOCUS_META}


def _write_csv_sidecar(csv_dir, zarr_store):
    rows = [
        *metadata_to_rows(
            NORM_META, store_path=zarr_store, position_path="A/1/0", channel_name="GFP", field_name="normalization"
        ),
        *metadata_to_rows(
            FOCUS_META, store_path=zarr_store, position_path="A/1/0", channel_name="GFP", field_name="focus_slice"
        ),
    ]
    write_meta_rows_csv(csv_dir, zarr_store, rows)


def test_zattrs_path_adds_norm_and_focus_columns(cell_index_df, zarr_store, tmp_path):
    """Default (zattrs) path populates norm_* and z_focus_mean, and rewrites z from per_timepoint."""
    _write_zattrs(zarr_store)
    input_path = tmp_path / "cell_index.parquet"
    write_cell_index(cell_index_df, input_path)

    output_path = tmp_path / "out.parquet"
    preprocess_cell_index(input_path, output_path=output_path)
    result = read_cell_index(output_path)

    assert len(result) == 2
    row_t0 = result[result["t"] == 0].iloc[0]
    assert row_t0["norm_mean"] == pytest.approx(1.0)
    assert row_t0["z_focus_mean"] == pytest.approx(4.0)
    assert row_t0["z"] == 4


def test_csv_dir_path_matches_zattrs_path(cell_index_df, zarr_store, tmp_path):
    """csv_dir path produces the same norm_*/z_focus_mean/z columns as the zattrs path."""
    csv_dir = tmp_path / "csvs"
    _write_csv_sidecar(csv_dir, zarr_store)

    input_path = tmp_path / "cell_index.parquet"
    write_cell_index(cell_index_df, input_path)

    csv_output = tmp_path / "csv_out.parquet"
    preprocess_cell_index(input_path, output_path=csv_output, csv_dir=csv_dir)
    csv_result = read_cell_index(csv_output).sort_values("t").reset_index(drop=True)

    _write_zattrs(zarr_store)
    zattrs_output = tmp_path / "zattrs_out.parquet"
    preprocess_cell_index(input_path, output_path=zattrs_output)
    zattrs_result = read_cell_index(zattrs_output).sort_values("t").reset_index(drop=True)

    for col in ("norm_mean", "norm_std", "norm_median", "norm_iqr", "norm_max", "norm_min", "z_focus_mean", "z"):
        np.testing.assert_allclose(
            csv_result[col].to_numpy(dtype=float),
            zattrs_result[col].to_numpy(dtype=float),
        )


def test_csv_dir_missing_sidecar_raises(cell_index_df, zarr_store, tmp_path):
    """csv_dir set but no sidecar exists for the store -> ValueError, not a silent skip."""
    input_path = tmp_path / "cell_index.parquet"
    write_cell_index(cell_index_df, input_path)

    with pytest.raises(ValueError, match="no CSV sidecar"):
        preprocess_cell_index(input_path, csv_dir=tmp_path / "empty")
