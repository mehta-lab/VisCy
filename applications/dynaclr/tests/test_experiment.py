"""Tests for ExperimentRegistry with Collection-based API."""

import logging

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr

from dynaclr.data.experiment import ExperimentRegistry
from viscy_data.collection import Collection, ExperimentEntry, SourceChannel, save_collection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_zarr(tmp_path):
    """Create a minimal HCS OME-Zarr store with channels ['Phase', 'GFP', 'RFP']."""
    zarr_path = tmp_path / "exp_a.zarr"
    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=["Phase", "GFP", "RFP"]) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 3, 1, 64, 64), dtype=np.float32)
    return zarr_path


@pytest.fixture()
def mini_zarr_mito(tmp_path):
    """Create a second zarr with channels ['Phase', 'Mito']."""
    zarr_path = tmp_path / "exp_b.zarr"
    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=["Phase", "Mito"]) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 2, 1, 64, 64), dtype=np.float32)
    return zarr_path


@pytest.fixture()
def exp_entry_a(mini_zarr, tmp_path):
    """ExperimentEntry for experiment A with 3 channels."""
    return ExperimentEntry(
        name="exp_a",
        data_path=str(mini_zarr),
        tracks_path=str(tmp_path / "tracks_a"),
        channel_names=["Phase", "GFP", "RFP"],
        condition_wells={"uninfected": ["A/1"], "infected": ["B/1"]},
        interval_minutes=30.0,
    )


@pytest.fixture()
def exp_entry_b(mini_zarr_mito, tmp_path):
    """ExperimentEntry for experiment B with 2 channels."""
    return ExperimentEntry(
        name="exp_b",
        data_path=str(mini_zarr_mito),
        tracks_path=str(tmp_path / "tracks_b"),
        channel_names=["Phase", "Mito"],
        condition_wells={"control": ["A/1"]},
        interval_minutes=15.0,
    )


def _make_collection_ab(exp_entry_a, exp_entry_b):
    """Create a Collection with two experiments and two source channels."""
    source_channels = [
        SourceChannel(label="labelfree", per_experiment={"exp_a": "Phase", "exp_b": "Phase"}),
        SourceChannel(label="reporter", per_experiment={"exp_a": "RFP", "exp_b": "Mito"}),
    ]
    return Collection(
        name="test",
        source_channels=source_channels,
        experiments=[exp_entry_a, exp_entry_b],
    )


def _make_collection_single(exp_entry, source_channel_names):
    """Create a Collection with a single experiment."""
    source_channels = [
        SourceChannel(label=f"ch{i}", per_experiment={exp_entry.name: ch}) for i, ch in enumerate(source_channel_names)
    ]
    return Collection(
        name="test",
        source_channels=source_channels,
        experiments=[exp_entry],
    )


# ---------------------------------------------------------------------------
# ExperimentRegistry tests
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    def test_registry_channel_maps(self, exp_entry_a):
        """channel_maps correctly maps source_channel position -> zarr index."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        # source_channels: ch0->Phase(idx0), ch1->RFP(idx2) in ["Phase", "GFP", "RFP"]
        assert registry.channel_maps["exp_a"] == {0: 0, 1: 2}

    def test_registry_channel_maps_different_names(self, exp_entry_a, exp_entry_b):
        """Positional alignment: different channel names, same position count."""
        collection = _make_collection_ab(exp_entry_a, exp_entry_b)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        # exp_a: labelfree->Phase(0), reporter->RFP(2) in ["Phase", "GFP", "RFP"]
        assert registry.channel_maps["exp_a"] == {0: 0, 1: 2}
        # exp_b: labelfree->Phase(0), reporter->Mito(1) in ["Phase", "Mito"]
        assert registry.channel_maps["exp_b"] == {0: 0, 1: 1}

    def test_registry_source_channel_not_in_channel_names(self, mini_zarr, tmp_path):
        """ValueError when source_channel references a channel not in channel_names."""
        exp = ExperimentEntry(
            name="bad_source",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        source_channels = [
            SourceChannel(label="labelfree", per_experiment={"bad_source": "Phase"}),
            SourceChannel(label="reporter", per_experiment={"bad_source": "DAPI"}),
        ]
        with pytest.raises(ValueError, match="DAPI"):
            Collection(
                name="test",
                source_channels=source_channels,
                experiments=[exp],
            )

    def test_registry_duplicate_names(self, exp_entry_a):
        """ValueError when two experiments share the same name."""
        dup = ExperimentEntry(
            name="exp_a",
            data_path=exp_entry_a.data_path,
            tracks_path=exp_entry_a.tracks_path,
            channel_names=exp_entry_a.channel_names,
            condition_wells=exp_entry_a.condition_wells,
            interval_minutes=exp_entry_a.interval_minutes,
        )
        source_channels = [
            SourceChannel(label="labelfree", per_experiment={"exp_a": "Phase"}),
        ]
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            Collection(
                name="test",
                source_channels=source_channels,
                experiments=[exp_entry_a, dup],
            )

    def test_registry_empty_experiments(self):
        """ValueError when experiments list is empty."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            ExperimentRegistry(
                collection=Collection(name="test", source_channels=[], experiments=[]),
                z_window=1,
            )

    def test_registry_zarr_validation(self, exp_entry_a):
        """Opens zarr and validates channel_names match metadata."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        assert registry.num_source_channels == 2

    def test_registry_zarr_channel_mismatch(self, mini_zarr, tmp_path):
        """ValueError when channel_names don't match zarr metadata."""
        exp = ExperimentEntry(
            name="mismatch",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "Mito"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        collection = _make_collection_single(exp, ["Phase"])
        with pytest.raises(ValueError, match="channel"):
            ExperimentRegistry(collection=collection, z_window=1)

    def test_registry_data_path_not_exists(self, tmp_path):
        """ValueError when data_path does not exist."""
        exp = ExperimentEntry(
            name="no_path",
            data_path=str(tmp_path / "nonexistent.zarr"),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        collection = _make_collection_single(exp, ["Phase"])
        with pytest.raises(ValueError, match="data_path"):
            ExperimentRegistry(collection=collection, z_window=1)

    def test_from_collection(self, mini_zarr, tmp_path):
        """Round-trip: write collection YAML, load, verify registry."""
        exp = ExperimentEntry(
            name="yaml_exp",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            condition_wells={
                "uninfected": ["A/1"],
                "infected": ["B/1"],
            },
            interval_minutes=30.0,
            start_hpi=3.0,
        )
        source_channels = [
            SourceChannel(label="labelfree", per_experiment={"yaml_exp": "Phase"}),
            SourceChannel(label="reporter", per_experiment={"yaml_exp": "GFP"}),
        ]
        collection = Collection(
            name="test",
            source_channels=source_channels,
            experiments=[exp],
        )
        collection_path = tmp_path / "collection.yml"
        save_collection(collection, collection_path)

        registry = ExperimentRegistry.from_collection(collection_path, z_window=1)
        assert len(registry.experiments) == 1
        assert registry.experiments[0].name == "yaml_exp"
        assert registry.experiments[0].start_hpi == 3.0
        assert registry.channel_maps["yaml_exp"] == {0: 0, 1: 1}

    def test_tau_range_frames_30min(self, exp_entry_a):
        """tau_range_hours=(0.5, 2.0) at 30min -> (1, 4)."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.tau_range_frames("exp_a", (0.5, 2.0))
        assert result == (1, 4)

    def test_tau_range_frames_15min(self, exp_entry_b):
        """tau_range_hours=(0.5, 2.0) at 15min -> (2, 8)."""
        collection = _make_collection_single(exp_entry_b, ["Phase", "Mito"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.tau_range_frames("exp_b", (0.5, 2.0))
        assert result == (2, 8)

    def test_tau_range_frames_warns_few_frames(self, exp_entry_a, caplog):
        """Warns when min_frames >= max_frames."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        with caplog.at_level(logging.WARNING):
            # (0.0, 0.0) at 30min -> (0, 0), min >= max
            registry.tau_range_frames("exp_a", (0.0, 0.0))
        assert any("fewer than 2" in msg.lower() or "few" in msg.lower() for msg in caplog.messages)

    def test_get_experiment(self, exp_entry_a):
        """Lookup by name returns the correct config."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.get_experiment("exp_a")
        assert result.name == "exp_a"
        assert result is exp_entry_a

    def test_get_experiment_not_found(self, exp_entry_a):
        """KeyError when experiment name not found."""
        collection = _make_collection_single(exp_entry_a, ["Phase", "RFP"])
        registry = ExperimentRegistry(collection=collection, z_window=1)
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get_experiment("nonexistent")

    def test_negative_interval_minutes(self, mini_zarr, tmp_path):
        """ValueError when interval_minutes is negative."""
        exp = ExperimentEntry(
            name="neg_interval",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=-5.0,
        )
        source_channels = [
            SourceChannel(label="labelfree", per_experiment={"neg_interval": "Phase"}),
        ]
        with pytest.raises(ValueError, match="interval_minutes"):
            Collection(
                name="test",
                source_channels=source_channels,
                experiments=[exp],
            )

    def test_empty_condition_wells(self, mini_zarr, tmp_path):
        """ValueError when condition_wells is empty."""
        exp = ExperimentEntry(
            name="empty_wells",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            condition_wells={},
            interval_minutes=30.0,
        )
        source_channels = [
            SourceChannel(label="labelfree", per_experiment={"empty_wells": "Phase"}),
        ]
        with pytest.raises(ValueError, match="condition_wells"):
            Collection(
                name="test",
                source_channels=source_channels,
                experiments=[exp],
            )
