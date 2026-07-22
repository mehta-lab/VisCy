"""Tests for viscy_data.meta_csv — CSV sidecar for metadata that would normally go to zattrs."""

from __future__ import annotations

import pandas as pd

from viscy_data.meta_csv import (
    build_provenance_fields,
    csv_path_for_store,
    metadata_to_rows,
    read_meta_rows_csv,
    write_meta_rows_csv,
)


def test_csv_path_for_store_uses_dataset_name():
    """The CSV is named after the store (zarr stem), not the full path."""
    p = csv_path_for_store("/csvs", "/bio/projects/x/datasets/exp_a/exp_a.zarr")
    assert p.name == "exp_a.csv"
    # trailing slash is handled the same way
    assert csv_path_for_store("/csvs", "/data/exp_a.zarr/").name == "exp_a.csv"


def test_csv_path_for_store_is_deterministic():
    """Same store path always resolves to the same CSV file."""
    p1 = csv_path_for_store("/csvs", "/data/exp_a.zarr")
    p2 = csv_path_for_store("/csvs", "/data/exp_a.zarr")
    assert p1 == p2
    assert p1.suffix == ".csv"


def test_build_provenance_fields_has_expected_keys():
    """Provenance dict always has written_at, git_commit, cli_invocation."""
    fields = build_provenance_fields()
    assert set(fields) == {"written_at", "git_commit", "cli_invocation"}
    assert isinstance(fields["written_at"], str)
    assert isinstance(fields["git_commit"], str)


def test_metadata_to_rows_normalization_shape():
    """dataset/fov/timepoint_statistics each produce the expected rows."""
    metadata = {
        "dataset_statistics": {"mean": 1.0, "std": 0.1},
        "fov_statistics": {"mean": 1.1, "std": 0.2},
        "timepoint_statistics": {
            "0": {"mean": 1.2, "std": 0.3},
            "1": {"mean": 1.3, "std": 0.4},
        },
    }
    rows = metadata_to_rows(
        metadata, store_path="/data/exp.zarr", position_path="A/1/0", channel_name="Phase", field_name="normalization"
    )
    scopes = {(r["scope"], r["timepoint"]) for r in rows}
    assert scopes == {("dataset", None), ("fov", None), ("timepoint", 0), ("timepoint", 1)}
    tp0 = next(r for r in rows if r["scope"] == "timepoint" and r["timepoint"] == 0)
    assert tp0["mean"] == 1.2
    assert tp0["position_path"] == "A/1/0"
    assert tp0["field_name"] == "normalization"


def test_metadata_to_rows_scalar_per_timepoint_uses_value_column():
    """focus_slice's per_timepoint (scalar ints) land under a 'value' column."""
    metadata = {
        "fov_statistics": {"z_focus_mean": 5.0, "z_focus_std": 0.5},
        "per_timepoint": {"0": 4, "1": 6},
    }
    rows = metadata_to_rows(
        metadata, store_path="/data/exp.zarr", position_path="A/1/0", channel_name="Phase", field_name="focus_slice"
    )
    tp_rows = {r["timepoint"]: r["value"] for r in rows if r["scope"] == "timepoint"}
    assert tp_rows == {0: 4, 1: 6}
    fov_row = next(r for r in rows if r["scope"] == "fov")
    assert fov_row["z_focus_mean"] == 5.0


def test_metadata_to_rows_skips_missing_keys():
    """Missing scopes (e.g. no dataset_statistics) produce no row for that scope."""
    rows = metadata_to_rows(
        {"fov_statistics": {"mean": 1.0}},
        store_path="/data/exp.zarr",
        position_path="A/1/0",
        channel_name="Phase",
        field_name="normalization",
    )
    assert len(rows) == 1
    assert rows[0]["scope"] == "fov"


def test_write_then_read_round_trip(tmp_path):
    """Rows written to the sidecar can be read back with the same values."""
    rows = [
        {
            "store_path": "/data/exp.zarr",
            "position_path": "A/1/0",
            "channel_name": "Phase",
            "field_name": "normalization",
            "scope": "timepoint",
            "timepoint": 0,
            "mean": 1.0,
            "std": 0.1,
        }
    ]
    write_meta_rows_csv(tmp_path, "/data/exp.zarr", rows)
    result = read_meta_rows_csv(tmp_path, "/data/exp.zarr")
    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["mean"] == 1.0


def test_read_missing_sidecar_returns_none(tmp_path):
    """No sidecar exists yet for this store -> None, not an error."""
    assert read_meta_rows_csv(tmp_path, "/data/never_written.zarr") is None


def test_write_upserts_on_matching_key(tmp_path):
    """A second write with the same key overwrites the row instead of duplicating it."""
    store_path = "/data/exp.zarr"
    row = {
        "store_path": store_path,
        "position_path": "A/1/0",
        "channel_name": "Phase",
        "field_name": "normalization",
        "scope": "timepoint",
        "timepoint": 0,
        "mean": 1.0,
    }
    write_meta_rows_csv(tmp_path, store_path, [dict(row)])
    updated_row = dict(row)
    updated_row["mean"] = 2.0
    write_meta_rows_csv(tmp_path, store_path, [updated_row])

    result = read_meta_rows_csv(tmp_path, store_path)
    assert len(result) == 1
    assert result.iloc[0]["mean"] == 2.0


def test_write_appends_new_key_without_dropping_existing(tmp_path):
    """Writing a different field_name (e.g. focus_slice after normalization) keeps both."""
    store_path = "/data/exp.zarr"
    norm_row = {
        "store_path": store_path,
        "position_path": "A/1/0",
        "channel_name": "Phase",
        "field_name": "normalization",
        "scope": "timepoint",
        "timepoint": 0,
        "mean": 1.0,
    }
    focus_row = {
        "store_path": store_path,
        "position_path": "A/1/0",
        "channel_name": "Phase",
        "field_name": "focus_slice",
        "scope": "fov",
        "timepoint": None,
        "z_focus_mean": 5.0,
    }
    write_meta_rows_csv(tmp_path, store_path, [norm_row])
    write_meta_rows_csv(tmp_path, store_path, [focus_row])

    result = read_meta_rows_csv(tmp_path, store_path)
    assert len(result) == 2
    assert set(result["field_name"]) == {"normalization", "focus_slice"}


def test_write_empty_rows_is_noop(tmp_path):
    """write_meta_rows_csv with an empty list does not create a file."""
    write_meta_rows_csv(tmp_path, "/data/exp.zarr", [])
    assert read_meta_rows_csv(tmp_path, "/data/exp.zarr") is None


def test_two_stores_get_separate_files(tmp_path):
    """Each store gets its own CSV; writing one doesn't affect the other."""
    row_a = {
        "store_path": "/data/exp_a.zarr",
        "position_path": "A/1/0",
        "channel_name": "Phase",
        "field_name": "normalization",
        "scope": "fov",
        "timepoint": None,
        "mean": 1.0,
    }
    row_b = {
        "store_path": "/data/exp_b.zarr",
        "position_path": "A/1/0",
        "channel_name": "Phase",
        "field_name": "normalization",
        "scope": "fov",
        "timepoint": None,
        "mean": 2.0,
    }
    write_meta_rows_csv(tmp_path, "/data/exp_a.zarr", [row_a])
    write_meta_rows_csv(tmp_path, "/data/exp_b.zarr", [row_b])

    result_a = read_meta_rows_csv(tmp_path, "/data/exp_a.zarr")
    result_b = read_meta_rows_csv(tmp_path, "/data/exp_b.zarr")
    assert result_a.iloc[0]["mean"] == 1.0
    assert result_b.iloc[0]["mean"] == 2.0
    assert isinstance(pd.concat([result_a, result_b]), pd.DataFrame)
