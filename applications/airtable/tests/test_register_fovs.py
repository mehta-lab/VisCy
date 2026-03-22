"""Tests for airtable_utils.registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from airtable_utils.registration import (
    MAX_CHANNELS,
    copy_well_template_fields,
    parse_position_path,
    register_fovs,
    zarr_fields_for_position,
)
from airtable_utils.schemas import DatasetRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_well_template(well_id: str, record_id: str | None = None, **overrides) -> DatasetRecord:
    """Create a well-level DatasetRecord (no fov)."""
    defaults = {
        "dataset": "test_dataset",
        "well_id": well_id,
        "fov": None,
        "cell_type": "A549",
        "cell_state": "Live",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "perturbation": "ZIKV",
        "hours_post_perturbation": 5.0,
        "moi": 5.0,
        "time_interval_min": 30.0,
        "fluorescence_modality": "Light-sheet",
        "channel_0_biology": "brightfield",
        "channel_1_biology": "mitochondria",
        "record_id": record_id,
    }
    defaults.update(overrides)
    return DatasetRecord(**defaults)


def _make_fov_record(well_id: str, fov: str, record_id: str, **overrides) -> DatasetRecord:
    """Create a per-FOV DatasetRecord."""
    defaults = {
        "dataset": "test_dataset",
        "well_id": well_id,
        "fov": fov,
        "cell_type": "A549",
        "marker": "TOMM20",
        "organelle": "mitochondria",
        "record_id": record_id,
    }
    defaults.update(overrides)
    return DatasetRecord(**defaults)


def _make_mock_plate(positions_data: dict[str, tuple[int, ...]], channel_names: list[str] | None = None):
    """Build a mock plate with given positions and shapes.

    Parameters
    ----------
    positions_data : dict[str, tuple[int, ...]]
        Mapping of position name -> array shape.
    channel_names : list[str] or None
        Channel names. Defaults to Phase3D + GFP + mCherry.
    """
    if channel_names is None:
        channel_names = ["Phase3D", "raw GFP EX488 EM525-45", "raw mCherry EX561 EM600-37"]

    plate = MagicMock()
    plate.channel_names = channel_names

    def _getitem(key):
        pos = MagicMock()
        pos.data.shape = positions_data[key]
        return pos

    plate.__getitem__ = MagicMock(side_effect=_getitem)
    plate.__enter__ = MagicMock(return_value=plate)
    plate.__exit__ = MagicMock(return_value=False)
    return plate


# ---------------------------------------------------------------------------
# parse_position_path
# ---------------------------------------------------------------------------


class TestParsePositionPath:
    """Tests for parse_position_path."""

    def test_standard_path(self):
        root, pos = parse_position_path(Path("/data/test_dataset.zarr/A/1/000000"))
        assert root == Path("/data/test_dataset.zarr")
        assert pos == "A/1/000000"

    def test_deep_zarr_path(self):
        root, pos = parse_position_path(Path("/hpc/projects/org/ds.zarr/B/2/001001"))
        assert root == Path("/hpc/projects/org/ds.zarr")
        assert pos == "B/2/001001"

    def test_no_zarr_raises(self):
        with pytest.raises(ValueError, match="No .zarr component"):
            parse_position_path(Path("/data/not_a_zarr/A/1/000000"))


# ---------------------------------------------------------------------------
# register_fovs
# ---------------------------------------------------------------------------


class TestRegisterFovs:
    """Tests for the register_fovs core function."""

    def test_creates_new_fov_records_from_well_templates(self):
        """New FOVs get created with template fields + zarr-derived fields."""
        template_a1 = _make_well_template("A/1", record_id="recWELL1")
        db = MagicMock()
        db.get_dataset_records.return_value = [template_a1]

        positions = {
            "A/1/000000": (10, 3, 1, 512, 512),
            "A/1/000001": (10, 3, 1, 512, 512),
        }
        mock_plate = _make_mock_plate(positions)

        paths = [
            Path("/data/test_dataset.zarr/A/1/000000"),
            Path("/data/test_dataset.zarr/A/1/000001"),
        ]
        with patch("airtable_utils.registration.open_ome_zarr", return_value=mock_plate):
            result = register_fovs(paths, db=db)

        assert result.dataset == "test_dataset"
        assert len(result.created) == 2
        assert len(result.updated) == 0
        assert len(result.unmatched) == 0

        rec0 = result.created[0]["fields"]
        assert rec0["dataset"] == "test_dataset"
        assert rec0["well_id"] == "A/1"
        assert rec0["fov"] == "000000"
        assert rec0["data_path"] == "/data/test_dataset.zarr/A/1/000000"
        assert rec0["channel_0_name"] == "Phase3D"
        assert rec0["channel_1_name"] == "raw GFP EX488 EM525-45"
        assert rec0["channel_2_name"] == "raw mCherry EX561 EM600-37"
        assert rec0["t_shape"] == 10
        assert rec0["c_shape"] == 3
        assert rec0["z_shape"] == 1
        assert rec0["y_shape"] == 512
        assert rec0["x_shape"] == 512
        assert rec0["cell_type"] == "A549"
        assert rec0["marker"] == "TOMM20"
        assert rec0["organelle"] == "mitochondria"
        assert rec0["perturbation"] == "ZIKV"
        assert rec0["moi"] == 5.0
        assert rec0["channel_0_biology"] == "brightfield"
        assert rec0["channel_1_biology"] == "mitochondria"

    def test_updates_existing_fov_records(self):
        """Existing per-FOV records get updated with zarr-derived fields only."""
        existing = _make_fov_record("A/1", "000000", record_id="recFOV1")
        db = MagicMock()
        db.get_dataset_records.return_value = [existing]

        positions = {"A/1/000000": (20, 3, 1, 256, 256)}
        mock_plate = _make_mock_plate(positions)

        paths = [Path("/data/test_dataset.zarr/A/1/000000")]
        with patch("airtable_utils.registration.open_ome_zarr", return_value=mock_plate):
            result = register_fovs(paths, db=db)

        assert len(result.created) == 0
        assert len(result.updated) == 1

        upd = result.updated[0]
        assert upd["id"] == "recFOV1"
        assert upd["fields"]["data_path"] == "/data/test_dataset.zarr/A/1/000000"
        assert upd["fields"]["t_shape"] == 20
        assert upd["fields"]["channel_0_name"] == "Phase3D"
        assert "cell_type" not in upd["fields"]
        assert "marker" not in upd["fields"]

    def test_unmatched_positions(self):
        """Positions without a well template or existing record are unmatched."""
        template_a1 = _make_well_template("A/1")
        db = MagicMock()
        db.get_dataset_records.return_value = [template_a1]

        positions = {
            "A/1/000000": (10, 3, 1, 512, 512),
            "B/2/000000": (10, 3, 1, 512, 512),
        }
        mock_plate = _make_mock_plate(positions)

        paths = [
            Path("/data/test_dataset.zarr/A/1/000000"),
            Path("/data/test_dataset.zarr/B/2/000000"),
        ]
        with patch("airtable_utils.registration.open_ome_zarr", return_value=mock_plate):
            result = register_fovs(paths, db=db)

        assert len(result.created) == 1
        assert len(result.unmatched) == 1
        assert result.unmatched[0] == "B/2/000000"

    def test_mixed_create_and_update(self):
        """Mix of new FOVs (create) and existing FOVs (update)."""
        template_a1 = _make_well_template("A/1")
        existing_a1_fov0 = _make_fov_record("A/1", "000000", record_id="recEXIST")
        db = MagicMock()
        db.get_dataset_records.return_value = [template_a1, existing_a1_fov0]

        positions = {
            "A/1/000000": (10, 3, 1, 512, 512),
            "A/1/000001": (10, 3, 1, 512, 512),
        }
        mock_plate = _make_mock_plate(positions)

        paths = [
            Path("/data/test_dataset.zarr/A/1/000000"),
            Path("/data/test_dataset.zarr/A/1/000001"),
        ]
        with patch("airtable_utils.registration.open_ome_zarr", return_value=mock_plate):
            result = register_fovs(paths, db=db)

        assert len(result.updated) == 1
        assert result.updated[0]["id"] == "recEXIST"
        assert len(result.created) == 1
        assert result.created[0]["fields"]["fov"] == "000001"

    def test_raises_on_no_airtable_records(self):
        """ValueError raised when no Airtable records exist for dataset."""
        db = MagicMock()
        db.get_dataset_records.return_value = []

        paths = [Path("/data/test_dataset.zarr/A/1/000000")]
        with pytest.raises(ValueError, match="No Airtable records"):
            register_fovs(paths, db=db)

    def test_raises_on_empty_paths(self):
        """ValueError raised when no position paths provided."""
        db = MagicMock()
        with pytest.raises(ValueError, match="No position paths"):
            register_fovs([], db=db)

    def test_raises_on_mixed_zarr_stores(self):
        """ValueError raised when paths span multiple zarr stores."""
        db = MagicMock()
        paths = [
            Path("/data/store_a.zarr/A/1/000000"),
            Path("/data/store_b.zarr/A/1/000000"),
        ]
        with pytest.raises(ValueError, match="same zarr store"):
            register_fovs(paths, db=db)

    def test_all_records_already_per_fov_no_templates(self):
        """When all records are per-FOV and no templates exist, only updates happen."""
        existing = _make_fov_record("A/1", "000000", record_id="recFOV1")
        db = MagicMock()
        db.get_dataset_records.return_value = [existing]

        positions = {
            "A/1/000000": (10, 3, 1, 512, 512),
            "A/1/000001": (10, 3, 1, 512, 512),
        }
        mock_plate = _make_mock_plate(positions)

        paths = [
            Path("/data/test_dataset.zarr/A/1/000000"),
            Path("/data/test_dataset.zarr/A/1/000001"),
        ]
        with patch("airtable_utils.registration.open_ome_zarr", return_value=mock_plate):
            result = register_fovs(paths, db=db)

        assert len(result.updated) == 1
        assert len(result.created) == 0
        assert len(result.unmatched) == 1
        assert result.unmatched[0] == "A/1/000001"


# ---------------------------------------------------------------------------
# zarr_fields_for_position
# ---------------------------------------------------------------------------


class TestZarrFieldsForPosition:
    """Tests for zarr_fields_for_position helper."""

    def test_basic_fields(self):
        fields = zarr_fields_for_position(
            zarr_path=Path("/data/ds.zarr"),
            pos_name="A/1/000000",
            channel_names=["Phase3D", "GFP"],
            shape=(10, 2, 1, 256, 256),
        )
        assert fields["data_path"] == "/data/ds.zarr/A/1/000000"
        assert fields["channel_0_name"] == "Phase3D"
        assert fields["channel_1_name"] == "GFP"
        assert "channel_2_name" not in fields
        assert fields["t_shape"] == 10
        assert fields["c_shape"] == 2
        assert fields["z_shape"] == 1
        assert fields["y_shape"] == 256
        assert fields["x_shape"] == 256

    def test_truncates_at_max_channels(self):
        channels = [f"ch_{i}" for i in range(6)]
        fields = zarr_fields_for_position(
            zarr_path=Path("/data/ds.zarr"),
            pos_name="A/1/000000",
            channel_names=channels,
            shape=(1, 6, 1, 64, 64),
        )
        for i in range(MAX_CHANNELS):
            assert fields[f"channel_{i}_name"] == f"ch_{i}"
        assert f"channel_{MAX_CHANNELS}_name" not in fields


# ---------------------------------------------------------------------------
# copy_well_template_fields
# ---------------------------------------------------------------------------


class TestCopyWellTemplateFields:
    """Tests for copy_well_template_fields helper."""

    def test_copies_non_none_fields(self):
        template = _make_well_template("A/1")
        fields = copy_well_template_fields(template)

        assert fields["cell_type"] == "A549"
        assert fields["marker"] == "TOMM20"
        assert fields["organelle"] == "mitochondria"
        assert fields["perturbation"] == "ZIKV"
        assert fields["moi"] == 5.0
        assert fields["time_interval_min"] == 30.0
        assert fields["channel_0_biology"] == "brightfield"
        assert fields["channel_1_biology"] == "mitochondria"

    def test_skips_none_fields(self):
        template = _make_well_template("A/1", seeding_density=None, treatment_concentration_nm=None)
        fields = copy_well_template_fields(template)

        assert "seeding_density" not in fields
        assert "treatment_concentration_nm" not in fields
