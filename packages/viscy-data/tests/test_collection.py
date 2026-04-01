"""Tests for viscy_data.collection: Collection, load/save, build_collection."""

import pytest

from viscy_data.collection import (
    ChannelEntry,
    Collection,
    ExperimentEntry,
    _group_records,
    build_collection,
    load_collection,
    save_collection,
)
from viscy_data.schemas import FOVRecord


def _make_experiment(name="exp1", channel_names=None, interval_minutes=15.0, **kwargs):
    """Create an ExperimentEntry with sensible defaults."""
    ch_names = channel_names or ["Phase", "GFP"]
    defaults = dict(
        name=name,
        data_path=f"/data/{name}.zarr",
        tracks_path=f"/tracks/{name}",
        channels=[ChannelEntry(name=ch, marker=ch) for ch in ch_names],
        channel_names=ch_names,
        perturbation_wells={"mock": ["A/1"], "infected": ["B/1"]},
        interval_minutes=interval_minutes,
    )
    defaults.update(kwargs)
    return ExperimentEntry(**defaults)


def _make_collection(experiments=None, **kwargs):
    """Create a valid Collection with sensible defaults."""
    experiments = experiments or [_make_experiment()]
    return Collection(
        name=kwargs.pop("name", "test_collection"),
        experiments=experiments,
        **kwargs,
    )


class TestCollectionValidation:
    """Test Collection model_validator rules."""

    def test_valid_collection(self):
        """Verify a well-formed collection passes validation."""
        coll = _make_collection()
        assert coll.name == "test_collection"
        assert len(coll.experiments) == 1

    def test_duplicate_experiment_names(self):
        """Raise ValueError when experiment names are not unique."""
        exp = _make_experiment(name="dup")
        with pytest.raises(ValueError, match="Duplicate experiment name"):
            _make_collection(experiments=[exp, exp])

    def test_zero_interval_minutes_allowed(self):
        """Zero interval_minutes is valid (non-timelapse data)."""
        exp = _make_experiment(name="exp1", interval_minutes=0.0)
        _make_collection(experiments=[exp])

    def test_negative_interval_minutes(self):
        """Raise ValueError when interval_minutes is negative."""
        exp = _make_experiment(name="exp1", interval_minutes=-5.0)
        with pytest.raises(ValueError, match="interval_minutes must be non-negative"):
            _make_collection(experiments=[exp])

    def test_perturbation_wells_empty(self):
        """Raise ValueError when perturbation_wells is empty."""
        exp = _make_experiment(name="exp1", perturbation_wells={})
        with pytest.raises(ValueError, match="perturbation_wells must not be empty"):
            _make_collection(experiments=[exp])


class TestCollectionIO:
    """Test round-trip save/load."""

    def test_round_trip(self, tmp_path):
        """Verify save_collection then load_collection produces equal data."""
        original = _make_collection(description="round-trip test")
        yaml_path = tmp_path / "collection.yml"
        save_collection(original, yaml_path)
        loaded = load_collection(yaml_path)
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert len(loaded.experiments) == len(original.experiments)
        assert loaded.experiments[0].name == original.experiments[0].name
        assert loaded.experiments[0].channel_names == original.experiments[0].channel_names
        assert loaded.experiments[0].perturbation_wells == original.experiments[0].perturbation_wells
        assert loaded.experiments[0].interval_minutes == original.experiments[0].interval_minutes
        assert len(loaded.experiments[0].channels) == len(original.experiments[0].channels)
        for orig_ch, load_ch in zip(original.experiments[0].channels, loaded.experiments[0].channels):
            assert orig_ch.name == load_ch.name
            assert orig_ch.marker == load_ch.marker


class TestBuildCollection:
    """Test build_collection grouping logic."""

    def test_groups_by_dataset(self):
        """Verify build_collection groups FOVRecords by dataset into ExperimentEntry."""
        records = [
            FOVRecord(
                dataset="exp_a",
                well_id="A/1",
                data_path="/data/a.zarr",
                tracks_path="/tracks/a",
                channel_names=["Phase", "GFP"],
                time_interval_min=10.0,
                cell_state="mock",
            ),
            FOVRecord(
                dataset="exp_a",
                well_id="B/1",
                data_path="/data/a.zarr",
                tracks_path="/tracks/a",
                channel_names=["Phase", "GFP"],
                time_interval_min=10.0,
                cell_state="infected",
            ),
            FOVRecord(
                dataset="exp_b",
                well_id="C/1",
                data_path="/data/b.zarr",
                tracks_path="/tracks/b",
                channel_names=["Phase", "GFP"],
                time_interval_min=20.0,
                cell_state="mock",
            ),
        ]
        coll = build_collection(records, name="built")

        assert coll.name == "built"
        assert len(coll.experiments) == 2

        exp_names = {e.name for e in coll.experiments}
        assert exp_names == {"exp_a", "exp_b"}

        exp_a = next(e for e in coll.experiments if e.name == "exp_a")
        assert exp_a.interval_minutes == 10.0
        assert "mock" in exp_a.perturbation_wells
        assert "infected" in exp_a.perturbation_wells
        assert "A/1" in exp_a.perturbation_wells["mock"]
        assert "B/1" in exp_a.perturbation_wells["infected"]

        assert len(coll.fov_records) == 3

    def test_splits_multi_marker_dataset(self):
        """When one dataset has multiple markers, split into separate experiments."""
        records = [
            FOVRecord(
                dataset="plate1",
                well_id="A/1",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                cell_state="uninfected",
                marker="TOMM20",
                organelle="mitochondria",
            ),
            FOVRecord(
                dataset="plate1",
                well_id="A/2",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                cell_state="infected",
                marker="TOMM20",
                organelle="mitochondria",
            ),
            FOVRecord(
                dataset="plate1",
                well_id="B/1",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                cell_state="uninfected",
                marker="SEC61B",
                organelle="endoplasmic_reticulum",
            ),
            FOVRecord(
                dataset="plate1",
                well_id="B/2",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                cell_state="infected",
                marker="SEC61B",
                organelle="endoplasmic_reticulum",
            ),
        ]
        grouped = _group_records(records)
        assert len(grouped) == 2
        assert "plate1_TOMM20" in grouped
        assert "plate1_SEC61B" in grouped
        assert len(grouped["plate1_TOMM20"]) == 2
        assert len(grouped["plate1_SEC61B"]) == 2

        coll = build_collection(records, name="multi_marker")
        assert len(coll.experiments) == 2

        tomm = next(e for e in coll.experiments if e.name == "plate1_TOMM20")
        assert tomm.marker == "TOMM20"
        assert tomm.organelle == "mitochondria"
        assert tomm.data_path == "/data/plate1.zarr"
        assert set(tomm.perturbation_wells["uninfected"]) == {"A/1"}
        assert set(tomm.perturbation_wells["infected"]) == {"A/2"}

        sec = next(e for e in coll.experiments if e.name == "plate1_SEC61B")
        assert sec.marker == "SEC61B"
        assert set(sec.perturbation_wells["uninfected"]) == {"B/1"}

    def test_single_marker_dataset_not_split(self):
        """When all records in a dataset share one marker, no split occurs."""
        records = [
            FOVRecord(
                dataset="plate1",
                well_id="A/1",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                marker="TOMM20",
            ),
            FOVRecord(
                dataset="plate1",
                well_id="A/2",
                data_path="/data/plate1.zarr",
                tracks_path="/tracks/plate1",
                channel_names=["Phase", "GFP"],
                time_interval_min=30.0,
                marker="TOMM20",
            ),
        ]
        grouped = _group_records(records)
        assert len(grouped) == 1
        assert "plate1" in grouped
