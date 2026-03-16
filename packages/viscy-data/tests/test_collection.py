"""Tests for viscy_data.collection: Collection, load/save, build_collection."""

import pytest

from viscy_data.collection import (
    Collection,
    ExperimentEntry,
    SourceChannel,
    _group_records,
    build_collection,
    load_collection,
    save_collection,
)
from viscy_data.schemas import FOVRecord


def _make_experiment(name="exp1", channel_names=None, interval_minutes=15.0, **kwargs):
    """Create an ExperimentEntry with sensible defaults."""
    defaults = dict(
        name=name,
        data_path=f"/data/{name}.zarr",
        tracks_path=f"/tracks/{name}",
        channel_names=channel_names or ["Phase", "GFP"],
        condition_wells={"mock": ["A/1"], "infected": ["B/1"]},
        interval_minutes=interval_minutes,
    )
    defaults.update(kwargs)
    return ExperimentEntry(**defaults)


def _make_source_channels(experiment_names, channels=None):
    """Create SourceChannel list mapping each label to all experiments."""
    channels = channels or {"labelfree": "Phase", "reporter": "GFP"}
    return [
        SourceChannel(label=label, per_experiment={n: ch for n in experiment_names}) for label, ch in channels.items()
    ]


def _make_collection(experiments=None, source_channels=None, **kwargs):
    """Create a valid Collection with sensible defaults."""
    experiments = experiments or [_make_experiment()]
    exp_names = [e.name for e in experiments]
    source_channels = source_channels or _make_source_channels(exp_names)
    return Collection(
        name=kwargs.pop("name", "test_collection"),
        source_channels=source_channels,
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

    def test_source_channel_references_unknown_experiment(self):
        """Raise ValueError when per_experiment key is not a valid experiment."""
        exp = _make_experiment(name="real")
        bad_sc = [
            SourceChannel(label="labelfree", per_experiment={"real": "Phase", "ghost": "Phase"}),
        ]
        with pytest.raises(ValueError, match="unknown experiment 'ghost'"):
            _make_collection(experiments=[exp], source_channels=bad_sc)

    def test_source_channel_partial_experiment_coverage(self):
        """Experiments may omit a source channel — partial per_experiment is valid."""
        exp1 = _make_experiment(name="a")
        exp2 = _make_experiment(name="b")
        # exp2 has no reporter channel — this should succeed
        partial_sc = [
            SourceChannel(label="labelfree", per_experiment={"a": "Phase", "b": "Phase"}),
            SourceChannel(label="reporter", per_experiment={"a": "GFP"}),
        ]
        collection = _make_collection(experiments=[exp1, exp2], source_channels=partial_sc)
        assert len(collection.source_channels) == 2

    def test_mapped_channel_not_in_experiment(self):
        """Raise ValueError when mapped channel name does not exist in experiment."""
        exp = _make_experiment(name="exp1", channel_names=["Phase", "GFP"])
        bad_sc = [
            SourceChannel(label="labelfree", per_experiment={"exp1": "MISSING_CHANNEL"}),
        ]
        with pytest.raises(ValueError, match="channel 'MISSING_CHANNEL'"):
            _make_collection(experiments=[exp], source_channels=bad_sc)

    def test_interval_minutes_not_positive(self):
        """Raise ValueError when interval_minutes <= 0."""
        exp = _make_experiment(name="exp1", interval_minutes=0.0)
        with pytest.raises(ValueError, match="interval_minutes must be positive"):
            _make_collection(experiments=[exp])

    def test_negative_interval_minutes(self):
        """Raise ValueError when interval_minutes is negative."""
        exp = _make_experiment(name="exp1", interval_minutes=-5.0)
        with pytest.raises(ValueError, match="interval_minutes must be positive"):
            _make_collection(experiments=[exp])

    def test_condition_wells_empty(self):
        """Raise ValueError when condition_wells is empty."""
        exp = _make_experiment(name="exp1", condition_wells={})
        with pytest.raises(ValueError, match="condition_wells must not be empty"):
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
        assert loaded.experiments[0].condition_wells == original.experiments[0].condition_wells
        assert loaded.experiments[0].interval_minutes == original.experiments[0].interval_minutes
        assert len(loaded.source_channels) == len(original.source_channels)
        for orig_sc, load_sc in zip(original.source_channels, loaded.source_channels):
            assert orig_sc.label == load_sc.label
            assert orig_sc.per_experiment == load_sc.per_experiment


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
        source_channels = _make_source_channels(["exp_a", "exp_b"])
        coll = build_collection(records, source_channels, name="built")

        assert coll.name == "built"
        assert len(coll.experiments) == 2

        exp_names = {e.name for e in coll.experiments}
        assert exp_names == {"exp_a", "exp_b"}

        exp_a = next(e for e in coll.experiments if e.name == "exp_a")
        assert exp_a.interval_minutes == 10.0
        assert "mock" in exp_a.condition_wells
        assert "infected" in exp_a.condition_wells
        assert "A/1" in exp_a.condition_wells["mock"]
        assert "B/1" in exp_a.condition_wells["infected"]

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

        source_channels = _make_source_channels(["plate1_TOMM20", "plate1_SEC61B"])
        coll = build_collection(records, source_channels, name="multi_marker")
        assert len(coll.experiments) == 2

        tomm = next(e for e in coll.experiments if e.name == "plate1_TOMM20")
        assert tomm.marker == "TOMM20"
        assert tomm.organelle == "mitochondria"
        assert tomm.data_path == "/data/plate1.zarr"
        assert set(tomm.condition_wells["uninfected"]) == {"A/1"}
        assert set(tomm.condition_wells["infected"]) == {"A/2"}

        sec = next(e for e in coll.experiments if e.name == "plate1_SEC61B")
        assert sec.marker == "SEC61B"
        assert set(sec.condition_wells["uninfected"]) == {"B/1"}

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
