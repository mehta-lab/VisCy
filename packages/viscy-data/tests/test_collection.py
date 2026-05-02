"""Tests for viscy_data.collection: Collection, load/save, build_collection."""

import pytest
import yaml

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


class TestChannelWells:
    """Test per-well channel validity restriction via ChannelEntry.wells."""

    def _make_viral_sensor_records(self):
        """FOVRecords for a mixed plate where viral sensor is only in B/3 and C/2."""
        common = dict(
            dataset="2025_01_24",
            data_path="/data/2025_01_24.zarr",
            tracks_path="/tracks/2025_01_24",
            channel_names=["Phase3D", "raw mCherry EX561 EM600-37"],
            time_interval_min=15.0,
        )
        # B/1, B/2: no viral sensor (channel_markers has no entry for mCherry)
        no_sensor = [
            FOVRecord(**common, well_id="B/1", cell_state="uninfected", channel_markers={"Phase3D": "Phase3D"}),
            FOVRecord(**common, well_id="B/2", cell_state="uninfected", channel_markers={"Phase3D": "Phase3D"}),
        ]
        # B/3, C/2: viral sensor present
        sensor = [
            FOVRecord(
                **common,
                well_id="B/3",
                cell_state="uninfected",
                channel_markers={"Phase3D": "Phase3D", "raw mCherry EX561 EM600-37": "pAL40"},
            ),
            FOVRecord(
                **common,
                well_id="C/2",
                cell_state="infected",
                channel_markers={"Phase3D": "Phase3D", "raw mCherry EX561 EM600-37": "pAL40"},
            ),
        ]
        return no_sensor + sensor

    def test_wells_auto_populated_for_partial_channel(self):
        """build_collection restricts a channel to wells where it has a marker."""
        records = self._make_viral_sensor_records()
        coll = build_collection(records, name="test")
        exp = coll.experiments[0]

        phase = next(ch for ch in exp.channels if ch.name == "Phase3D")
        mcherry = next(ch for ch in exp.channels if ch.name == "raw mCherry EX561 EM600-37")

        assert phase.wells == [], "Phase3D is valid in all wells — wells must be empty"
        assert sorted(mcherry.wells) == ["B/3", "C/2"], "mCherry only valid in B/3, C/2"

    def test_wells_empty_when_all_wells_have_marker(self):
        """When all wells share a marker, wells stays empty (no restriction needed)."""
        records = [
            FOVRecord(
                dataset="exp",
                well_id="A/1",
                data_path="/d.zarr",
                tracks_path="/t",
                channel_names=["Phase3D"],
                cell_state="uninfected",
                channel_markers={"Phase3D": "Phase3D"},
            ),
            FOVRecord(
                dataset="exp",
                well_id="A/2",
                data_path="/d.zarr",
                tracks_path="/t",
                channel_names=["Phase3D"],
                cell_state="infected",
                channel_markers={"Phase3D": "Phase3D"},
            ),
        ]
        coll = build_collection(records, name="test")
        phase = coll.experiments[0].channels[0]
        assert phase.wells == []

    def test_wells_round_trips_yaml(self, tmp_path):
        """wells field survives save_collection → load_collection."""
        records = self._make_viral_sensor_records()
        coll = build_collection(records, name="test")
        path = tmp_path / "col.yml"
        save_collection(coll, path)
        loaded = load_collection(path)
        mcherry = next(ch for ch in loaded.experiments[0].channels if ch.name == "raw mCherry EX561 EM600-37")
        assert sorted(mcherry.wells) == ["B/3", "C/2"]

    def test_channel_entry_wells_default_empty(self):
        """ChannelEntry.wells defaults to empty list."""
        ch = ChannelEntry(name="Phase3D", marker="Phase3D")
        assert ch.wells == []


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _minimal_experiment(name, data_path, tracks_path):
    return {
        "name": name,
        "data_path": data_path,
        "tracks_path": tracks_path,
        "channels": [{"name": "Phase3D", "marker": "Phase3D"}],
        "perturbation_wells": {"mock": ["A/1"]},
    }


class TestDatasetsRoot:
    """Test ${datasets_root} substitution in load/save round-trip."""

    def test_resolve_datasets_root(self, tmp_path):
        """Paths with ${datasets_root} are fully resolved after load."""
        data = {
            "name": "test",
            "datasets_root": "/hpc/projects/organelle_phenotyping",
            "experiments": [
                _minimal_experiment(
                    "exp1",
                    "${datasets_root}/datasets/exp1/exp1.zarr",
                    "${datasets_root}/datasets/exp1/tracking.zarr",
                )
            ],
        }
        _write_yaml(tmp_path / "col.yml", data)
        coll = load_collection(tmp_path / "col.yml")
        assert coll.experiments[0].data_path == "/hpc/projects/organelle_phenotyping/datasets/exp1/exp1.zarr"
        assert coll.experiments[0].tracks_path == "/hpc/projects/organelle_phenotyping/datasets/exp1/tracking.zarr"
        assert coll.datasets_root == "/hpc/projects/organelle_phenotyping"

    def test_round_trip_preserves_templates(self, tmp_path):
        """save_collection writes ${datasets_root} back; reload resolves again."""
        data = {
            "name": "test",
            "datasets_root": "/hpc/projects/organelle_phenotyping",
            "experiments": [
                _minimal_experiment(
                    "exp1",
                    "${datasets_root}/datasets/exp1/exp1.zarr",
                    "${datasets_root}/datasets/exp1/tracking.zarr",
                )
            ],
        }
        yaml_path = tmp_path / "col.yml"
        _write_yaml(yaml_path, data)
        coll = load_collection(yaml_path)
        out_path = tmp_path / "col_out.yml"
        save_collection(coll, out_path)

        with open(out_path) as f:
            on_disk = yaml.safe_load(f)

        assert "${datasets_root}" in on_disk["experiments"][0]["data_path"]
        assert "${datasets_root}" in on_disk["experiments"][0]["tracks_path"]

        reloaded = load_collection(out_path)
        assert reloaded.experiments[0].data_path == "/hpc/projects/organelle_phenotyping/datasets/exp1/exp1.zarr"

    def test_mixed_paths_non_root_stays_absolute(self, tmp_path):
        """Paths not under datasets_root survive save unchanged."""
        data = {
            "name": "test",
            "datasets_root": "/hpc/projects/organelle_phenotyping",
            "experiments": [
                _minimal_experiment(
                    "exp_vast",
                    "${datasets_root}/datasets/exp1/exp1.zarr",
                    "${datasets_root}/datasets/exp1/tracking.zarr",
                ),
                _minimal_experiment(
                    "exp_nfs",
                    "${datasets_root}/datasets/exp2/exp2.zarr",
                    "/hpc/projects/intracellular_dashboard/viral-sensor/tracking.zarr",
                ),
            ],
        }
        yaml_path = tmp_path / "col.yml"
        _write_yaml(yaml_path, data)
        coll = load_collection(yaml_path)
        assert coll.experiments[1].tracks_path == "/hpc/projects/intracellular_dashboard/viral-sensor/tracking.zarr"

        out_path = tmp_path / "col_out.yml"
        save_collection(coll, out_path)
        with open(out_path) as f:
            on_disk = yaml.safe_load(f)
        nfs_path = "/hpc/projects/intracellular_dashboard/viral-sensor/tracking.zarr"
        assert on_disk["experiments"][1]["tracks_path"] == nfs_path

    def test_no_datasets_root_passthrough(self, tmp_path):
        """Collections without datasets_root load and save unchanged."""
        data = {
            "name": "test",
            "experiments": [
                _minimal_experiment(
                    "exp1",
                    "/absolute/data/exp1.zarr",
                    "/absolute/tracks/exp1",
                )
            ],
        }
        yaml_path = tmp_path / "col.yml"
        _write_yaml(yaml_path, data)
        coll = load_collection(yaml_path)
        assert coll.datasets_root is None
        assert coll.experiments[0].data_path == "/absolute/data/exp1.zarr"

        out_path = tmp_path / "col_out.yml"
        save_collection(coll, out_path)
        with open(out_path) as f:
            on_disk = yaml.safe_load(f)
        assert on_disk["experiments"][0]["data_path"] == "/absolute/data/exp1.zarr"
