"""Tests for viscy_data.cell_index — schema, validation, I/O, and builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from iohub import open_ome_zarr

from viscy_data._typing import (
    CELL_INDEX_BIOLOGY_COLUMNS,
    CELL_INDEX_CORE_COLUMNS,
    CELL_INDEX_GROUPING_COLUMNS,
    CELL_INDEX_IMAGING_COLUMNS,
    CELL_INDEX_NORMALIZATION_COLUMNS,
    CELL_INDEX_OPS_COLUMNS,
    CELL_INDEX_TIMELAPSE_COLUMNS,
)
from viscy_data.cell_index import (
    CELL_INDEX_SCHEMA,
    _parse_bbox_min_size,
    _parse_bbox_to_centroid,
    build_timelapse_cell_index,
    convert_ops_parquet,
    read_cell_index,
    validate_cell_index,
    write_cell_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal valid cell index DataFrame."""
    return pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "experiment": "exp_a",
            "store_path": "/data/exp_a.zarr",
            "tracks_path": "/data/tracks",
            "fov": "A/1/0",
            "well": "A/1",
            "y": np.random.default_rng(0).random(n).astype(np.float32) * 256,
            "x": np.random.default_rng(1).random(n).astype(np.float32) * 256,
            "z": np.zeros(n, dtype=np.int16),
            "perturbation": "uninfected",
            "channel_name": "GFP",
            "microscope": "",
        }
    )


def _make_timelapse_df() -> pd.DataFrame:
    """Create a valid time-lapse cell index DataFrame."""
    df = _make_valid_df(4)
    df["t"] = pd.array([0, 1, 0, 1], dtype="Int32")
    df["track_id"] = pd.array([0, 0, 1, 1], dtype="Int32")
    df["global_track_id"] = ["exp_a_A/1/0_0", "exp_a_A/1/0_0", "exp_a_A/1/0_1", "exp_a_A/1/0_1"]
    df["lineage_id"] = df["global_track_id"]
    df["parent_track_id"] = pd.array([-1, -1, -1, -1], dtype="Int32")
    df["hours_post_perturbation"] = [0.0, 0.5, 0.0, 0.5]
    df["interval_minutes"] = 30.0
    return df


def _make_ops_df() -> pd.DataFrame:
    """Create a valid OPS cell index DataFrame."""
    df = _make_valid_df(3)
    df["gene_name"] = ["TP53", "NTC", "BRCA1"]
    df["reporter"] = ["GFP", "GFP", "mCherry"]
    df["sgRNA"] = ["sg1", "sg2", "sg3"]
    return df


# ---------------------------------------------------------------------------
# Schema + Validation (tests 1–4)
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for validate_cell_index."""

    def test_valid_df_passes(self):
        """1. Valid DataFrame passes validate_cell_index()."""
        df = _make_valid_df()
        warnings = validate_cell_index(df)
        assert isinstance(warnings, list)

    def test_missing_core_columns_raises(self):
        """2. Missing core columns raise ValueError."""
        df = _make_valid_df()
        df = df.drop(columns=["cell_id", "experiment"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_cell_index(df)

    def test_duplicate_cell_id_channel_name_raises(self):
        """3. Duplicate (cell_id, channel_name) raises ValueError."""
        df = _make_valid_df()
        df.loc[1, "cell_id"] = df.loc[0, "cell_id"]
        df.loc[1, "channel_name"] = df.loc[0, "channel_name"]
        with pytest.raises(ValueError, match="cell_id, channel_name.*must be unique"):
            validate_cell_index(df)

    def test_same_cell_id_different_channel_passes(self):
        """3b. Same cell_id with different channel_name is valid (flat parquet)."""
        df = _make_valid_df(2)
        df.loc[0, "cell_id"] = "shared_cell"
        df.loc[1, "cell_id"] = "shared_cell"
        df.loc[0, "channel_name"] = "Phase3D"
        df.loc[1, "channel_name"] = "GFP"
        warnings = validate_cell_index(df)
        assert isinstance(warnings, list)

    def test_strict_requires_all_columns(self):
        """4. strict=True requires all schema columns."""
        df = _make_valid_df()
        # Missing time-lapse and OPS columns
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_cell_index(df, strict=True)

    def test_strict_passes_with_all_columns(self):
        """4b. strict=True passes when all columns are present."""
        df = _make_valid_df()
        for col in (
            CELL_INDEX_BIOLOGY_COLUMNS
            + CELL_INDEX_TIMELAPSE_COLUMNS
            + CELL_INDEX_OPS_COLUMNS
            + CELL_INDEX_IMAGING_COLUMNS
            + CELL_INDEX_NORMALIZATION_COLUMNS
        ):
            df[col] = None
        warnings = validate_cell_index(df, strict=True)
        assert isinstance(warnings, list)

    def test_all_null_column_warns(self):
        """Nullable columns that are entirely null produce warnings."""
        df = _make_valid_df()
        for col in (
            CELL_INDEX_BIOLOGY_COLUMNS
            + CELL_INDEX_TIMELAPSE_COLUMNS
            + CELL_INDEX_OPS_COLUMNS
            + CELL_INDEX_IMAGING_COLUMNS
            + CELL_INDEX_NORMALIZATION_COLUMNS
        ):
            df[col] = None
        warnings = validate_cell_index(df, strict=True)
        assert any("all null" in w for w in warnings)


# ---------------------------------------------------------------------------
# I/O round-trip (test 5)
# ---------------------------------------------------------------------------


class TestIO:
    """Tests for write_cell_index and read_cell_index."""

    def test_round_trip_preserves_dtypes(self, tmp_path):
        """5. write + read preserves dtypes and nullability."""
        df = _make_timelapse_df()
        path = tmp_path / "cell_index.parquet"
        write_cell_index(df, path)
        result = read_cell_index(path)

        # Check all schema columns exist
        for field in CELL_INDEX_SCHEMA:
            assert field.name in result.columns

        # Core dtypes
        assert result["y"].dtype == np.float32
        assert result["x"].dtype == np.float32
        assert pd.api.types.is_string_dtype(result["cell_id"])

        # Nullable OPS columns should be null
        assert result["gene_name"].isna().all()

    def test_write_adds_missing_columns(self, tmp_path):
        """write_cell_index adds missing nullable columns as None."""
        df = _make_valid_df()
        path = tmp_path / "cell_index.parquet"
        write_cell_index(df, path)
        result = read_cell_index(path)
        assert "gene_name" in result.columns
        assert "t" in result.columns


# ---------------------------------------------------------------------------
# Time-lapse builder (tests 6–10)
# ---------------------------------------------------------------------------


def _create_collection_yaml(
    tmp_path: Path,
    dataset_path: Path,
    tracks_path: Path | None = None,
    channel_names: list[str] | None = None,
) -> Path:
    """Write a minimal collection YAML for testing the builder."""
    if channel_names is None:
        channel_names = ["nuclei_labels"]
    if tracks_path is None:
        tracks_path = dataset_path

    yaml_path = tmp_path / "collection.yml"
    config = {
        "name": "test_collection",
        "experiments": [
            {
                "name": "test_exp",
                "data_path": str(dataset_path),
                "tracks_path": str(tracks_path),
                "channels": [{"name": ch, "marker": ch} for ch in channel_names],
                "perturbation_wells": {"uninfected": ["A/1", "A/2"], "infected": ["B/1", "B/2"]},
                "interval_minutes": 30.0,
                "start_hpi": 0.0,
            }
        ],
    }
    yaml_path.write_text(yaml.dump(config))
    return yaml_path


class TestTimelapseBuilder:
    """Tests for build_timelapse_cell_index."""

    def test_produces_correct_schema(self, tracks_hcs_dataset, tmp_path):
        """6. Builder produces correct schema from mock experiment."""
        yaml_path = _create_collection_yaml(tmp_path, tracks_hcs_dataset)
        output = tmp_path / "output.parquet"
        df = build_timelapse_cell_index(yaml_path, output)

        assert len(df) > 0
        required = set(CELL_INDEX_CORE_COLUMNS + CELL_INDEX_GROUPING_COLUMNS)
        assert required.issubset(set(df.columns))

        # Round-trip via parquet
        result = read_cell_index(output)
        assert len(result) == len(df)

    def test_lineage_reconstruction(self, tmp_path):
        """7. Lineage reconstruction links daughters to root ancestor."""
        # Create a zarr with tracks that have parent relationships
        dataset_path = tmp_path / "lineage.zarr"
        dataset = open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=["nuclei_labels"])
        pos = dataset.create_position("A", "1", "0")
        rng = np.random.default_rng(42)
        pos.create_image("0", rng.random((4, 1, 1, 64, 64)).astype(np.float32))

        # Track 0 → root, Track 1 → child of 0, Track 2 → grandchild of 1
        tracks_df = pd.DataFrame(
            {
                "track_id": [0, 0, 1, 1, 2, 2],
                "t": [0, 1, 1, 2, 2, 3],
                "y": [32] * 6,
                "x": [32] * 6,
                "id": [0, 1, 2, 3, 4, 5],
                "parent_track_id": [-1, -1, 0, 0, 1, 1],
                "parent_id": [-1, -1, 1, 1, 3, 3],
            }
        )
        (dataset_path / "A" / "1" / "0").mkdir(parents=True, exist_ok=True)
        tracks_df.to_csv(dataset_path / "A/1/0" / "tracks.csv", index=False)

        yaml_path = _create_collection_yaml(tmp_path, dataset_path, channel_names=["nuclei_labels"])
        output = tmp_path / "lineage_output.parquet"
        df = build_timelapse_cell_index(yaml_path, output, include_wells=["A/1"])

        # All tracks in same lineage should share root's global_track_id
        root_gtid = "test_exp_A/1/0_0"
        assert (df["lineage_id"] == root_gtid).all()

    def test_well_filtering(self, tracks_hcs_dataset, tmp_path):
        """8. include_wells filters to specified wells only."""
        yaml_path = _create_collection_yaml(tmp_path, tracks_hcs_dataset)
        output = tmp_path / "filtered.parquet"
        df = build_timelapse_cell_index(yaml_path, output, include_wells=["A/1"])

        assert len(df) > 0
        assert (df["well"] == "A/1").all()

    def test_fov_exclusion(self, tracks_hcs_dataset, tmp_path):
        """9. exclude_fovs excludes specified FOVs."""
        yaml_path = _create_collection_yaml(tmp_path, tracks_hcs_dataset)
        output = tmp_path / "excluded.parquet"
        df = build_timelapse_cell_index(yaml_path, output, exclude_fovs=["A/1/0"])

        assert "A/1/0" not in df["fov"].to_numpy()

    def test_cell_id_unique(self, tracks_hcs_dataset, tmp_path):
        """10. cell_id is unique across all observations."""
        yaml_path = _create_collection_yaml(tmp_path, tracks_hcs_dataset)
        output = tmp_path / "unique.parquet"
        df = build_timelapse_cell_index(yaml_path, output)

        assert not df["cell_id"].duplicated().any()


# ---------------------------------------------------------------------------
# OPS builder helpers (tests 11–14)
# ---------------------------------------------------------------------------


class TestOPSHelpers:
    """Tests for OPS-specific helper functions and synthetic OPS data."""

    def test_bbox_to_centroid(self):
        """11. bbox string → centroid conversion correct."""
        y, x = _parse_bbox_to_centroid("(10, 20, 30, 40)")
        assert y == pytest.approx(20.0)
        assert x == pytest.approx(30.0)

    def test_small_bbox_filtering(self):
        """13. Small bbox filtering drops cells."""
        assert _parse_bbox_min_size("(10, 20, 12, 40)") == 2.0  # height=2, width=20
        assert _parse_bbox_min_size("(10, 20, 30, 40)") == 20.0  # both sides ≥ 5

    def test_perturbation_map_populates_perturbation(self):
        """14. perturbation_map populates perturbation column."""
        from viscy_data.cell_index import _resolve_perturbation

        perturbation_map = {"treated": ["A/1"], "control": ["B/1"]}
        assert _resolve_perturbation(perturbation_map, "A/1") == "treated"
        assert _resolve_perturbation(perturbation_map, "C/1") == "unknown"


# ---------------------------------------------------------------------------
# Cross-paradigm compatibility (tests 15–17)
# ---------------------------------------------------------------------------


class TestCrossParadigm:
    """Tests for schema compatibility between time-lapse and OPS indices."""

    def test_timelapse_has_null_ops_columns(self):
        """15. Time-lapse parquet has OPS columns as null."""
        df = _make_timelapse_df()
        for col in (
            CELL_INDEX_OPS_COLUMNS
            + CELL_INDEX_BIOLOGY_COLUMNS
            + CELL_INDEX_IMAGING_COLUMNS
            + CELL_INDEX_NORMALIZATION_COLUMNS
        ):
            df[col] = None
        warnings = validate_cell_index(df, strict=True)
        ops_warnings = [w for w in warnings if any(f"'{c}'" in w for c in CELL_INDEX_OPS_COLUMNS)]
        assert len(ops_warnings) == len(CELL_INDEX_OPS_COLUMNS)

    def test_ops_has_null_timelapse_columns(self):
        """16. OPS parquet has time-lapse columns as null."""
        df = _make_ops_df()
        for col in (
            CELL_INDEX_TIMELAPSE_COLUMNS
            + CELL_INDEX_BIOLOGY_COLUMNS
            + CELL_INDEX_IMAGING_COLUMNS
            + CELL_INDEX_NORMALIZATION_COLUMNS
        ):
            df[col] = None
        warnings = validate_cell_index(df, strict=True)
        tl_warnings = [w for w in warnings if any(f"'{c}'" in w for c in CELL_INDEX_TIMELAPSE_COLUMNS)]
        assert len(tl_warnings) == len(CELL_INDEX_TIMELAPSE_COLUMNS)

    def test_concat_schema_compatible(self, tmp_path):
        """17. Both can be pd.concat'd (schema-compatible)."""
        tl = _make_timelapse_df()
        for col in CELL_INDEX_OPS_COLUMNS + CELL_INDEX_BIOLOGY_COLUMNS + CELL_INDEX_IMAGING_COLUMNS:
            tl[col] = None

        ops = _make_ops_df()
        ops["cell_id"] = [f"ops_cell_{i}" for i in range(len(ops))]  # avoid id clash
        for col in CELL_INDEX_TIMELAPSE_COLUMNS:
            ops[col] = None

        # Write both with schema enforcement
        tl_path = tmp_path / "tl.parquet"
        ops_path = tmp_path / "ops.parquet"
        write_cell_index(tl, tl_path)
        write_cell_index(ops, ops_path)

        # Read back and concat
        tl_read = read_cell_index(tl_path)
        ops_read = read_cell_index(ops_path)
        combined = pd.concat([tl_read, ops_read], ignore_index=True)

        assert len(combined) == len(tl) + len(ops)
        # Schema columns all present
        for field in CELL_INDEX_SCHEMA:
            assert field.name in combined.columns


# ---------------------------------------------------------------------------
# convert_ops_parquet (tests 18–20)
# ---------------------------------------------------------------------------


def _make_ops_merged_parquet(tmp_path: Path, n: int = 4) -> Path:
    """Write a synthetic OPS merged parquet mimicking the pipeline output."""
    df = pd.DataFrame(
        {
            "store_key": [f"2023_store_{i % 2}" for i in range(n)],
            "well": [f"A/{i}/0" for i in range(n)],
            "bbox": [f"({10 + i}, {20 + i}, {30 + i}, {40 + i})" for i in range(n)],
            "gene_name": ["RPL35", "NTC", "TP53", "RPL35"],
            "reporter": ["Phase", "Phase", "GFP", "Phase"],
            "sgRNA": ["sg1", "sg2", "sg3", "sg1"],
            "channel": ["Phase", "Phase", "GFP", "Phase"],
            "total_index": [100, 101, 102, 103],
        }
    )
    path = tmp_path / "ops_merged.parquet"
    df.to_parquet(path)
    return path


class TestConvertOpsParquet:
    """Tests for convert_ops_parquet()."""

    def test_produces_valid_schema(self, tmp_path):
        """18. Converted parquet passes validate_cell_index."""
        ops_path = _make_ops_merged_parquet(tmp_path)
        output = tmp_path / "ops_cell_index.parquet"
        df = convert_ops_parquet(ops_path, output, store_root="/nonexistent", store_suffix="fake.zarr")
        warnings = validate_cell_index(df)
        assert isinstance(warnings, list)

    def test_fov_and_well_mapping(self, tmp_path):
        """19. OPS 'well' splits into fov (last part) and well (parent)."""
        ops_path = _make_ops_merged_parquet(tmp_path)
        output = tmp_path / "ops_cell_index.parquet"
        df = convert_ops_parquet(ops_path, output, store_root="/nonexistent", store_suffix="fake.zarr")
        # fov should be just the FOV part (e.g. "0")
        assert df["fov"].iloc[0] == "0"
        # well should be the parent (e.g. "A/0")
        assert df["well"].iloc[0] == "A/0"

    def test_cell_id_unique(self, tmp_path):
        """20. cell_id is unique across all rows."""
        ops_path = _make_ops_merged_parquet(tmp_path)
        output = tmp_path / "ops_cell_index.parquet"
        df = convert_ops_parquet(ops_path, output, store_root="/nonexistent", store_suffix="fake.zarr")
        assert not df["cell_id"].duplicated().any()

    def test_gene_name_and_perturbation(self, tmp_path):
        """OPS gene_name populates perturbation column."""
        ops_path = _make_ops_merged_parquet(tmp_path)
        output = tmp_path / "ops_cell_index.parquet"
        df = convert_ops_parquet(ops_path, output, store_root="/nonexistent", store_suffix="fake.zarr")
        assert df["gene_name"].iloc[0] == "RPL35"
        assert df["perturbation"].iloc[0] == "RPL35"

    def test_round_trip_parquet(self, tmp_path):
        """Written parquet can be read back with correct schema."""
        ops_path = _make_ops_merged_parquet(tmp_path)
        output = tmp_path / "ops_cell_index.parquet"
        convert_ops_parquet(ops_path, output, store_root="/nonexistent", store_suffix="fake.zarr")
        result = read_cell_index(output)
        for field in CELL_INDEX_SCHEMA:
            assert field.name in result.columns
