"""Tests for ExperimentRegistry with Collection-based API."""

import logging

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr

from dynaclr.data.experiment import ExperimentRegistry
from viscy_data.collection import ChannelEntry, Collection, ExperimentEntry, save_collection

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
        channels=[
            ChannelEntry(name="Phase", marker="Phase"),
            ChannelEntry(name="GFP", marker="GFP"),
            ChannelEntry(name="RFP", marker="RFP"),
        ],
        channel_names=["Phase", "GFP", "RFP"],
        perturbation_wells={"uninfected": ["A/1"], "infected": ["B/1"]},
        interval_minutes=30.0,
    )


@pytest.fixture()
def exp_entry_b(mini_zarr_mito, tmp_path):
    """ExperimentEntry for experiment B with 2 channels."""
    return ExperimentEntry(
        name="exp_b",
        data_path=str(mini_zarr_mito),
        tracks_path=str(tmp_path / "tracks_b"),
        channels=[
            ChannelEntry(name="Phase", marker="Phase"),
            ChannelEntry(name="Mito", marker="Mito"),
        ],
        channel_names=["Phase", "Mito"],
        perturbation_wells={"control": ["A/1"]},
        interval_minutes=15.0,
    )


def _make_collection_ab(exp_entry_a, exp_entry_b):
    """Create a Collection with two experiments."""
    return Collection(
        name="test",
        experiments=[exp_entry_a, exp_entry_b],
    )


def _make_collection_single(exp_entry):
    """Create a Collection with a single experiment."""
    return Collection(
        name="test",
        experiments=[exp_entry],
    )


# ---------------------------------------------------------------------------
# ExperimentRegistry tests
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    def test_registry_source_channel_labels(self, exp_entry_a, exp_entry_b):
        """source_channel_labels returns unique markers from all experiments."""
        collection = _make_collection_ab(exp_entry_a, exp_entry_b)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        labels = registry.source_channel_labels
        assert "Phase" in labels
        assert "RFP" in labels
        assert "Mito" in labels

    def test_registry_duplicate_names(self, exp_entry_a):
        """ValueError when two experiments share the same name."""
        dup = ExperimentEntry(
            name="exp_a",
            data_path=exp_entry_a.data_path,
            tracks_path=exp_entry_a.tracks_path,
            channel_names=exp_entry_a.channel_names,
            perturbation_wells=exp_entry_a.perturbation_wells,
            interval_minutes=exp_entry_a.interval_minutes,
        )
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            Collection(
                name="test",
                experiments=[exp_entry_a, dup],
            )

    def test_registry_empty_experiments(self):
        """ValueError when experiments list is empty."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            ExperimentRegistry(
                collection=Collection(name="test", experiments=[]),
                z_window=1,
            )

    def test_registry_zarr_validation(self, exp_entry_a):
        """Opens zarr and validates channel_names match metadata."""
        collection = _make_collection_single(exp_entry_a)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        assert len(registry.source_channel_labels) == 3

    def test_registry_zarr_channel_mismatch(self, mini_zarr, tmp_path):
        """ValueError when channel_names don't match zarr metadata."""
        exp = ExperimentEntry(
            name="mismatch",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "Mito"],
            perturbation_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        collection = _make_collection_single(exp)
        with pytest.raises(ValueError, match="channel"):
            ExperimentRegistry(collection=collection, z_window=1)

    def test_registry_data_path_not_exists(self, tmp_path):
        """ValueError when data_path does not exist."""
        exp = ExperimentEntry(
            name="no_path",
            data_path=str(tmp_path / "nonexistent.zarr"),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase"],
            perturbation_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        collection = _make_collection_single(exp)
        with pytest.raises(ValueError, match="data_path"):
            ExperimentRegistry(collection=collection, z_window=1)

    def test_from_collection(self, mini_zarr, tmp_path):
        """Round-trip: write collection YAML, load, verify registry."""
        exp = ExperimentEntry(
            name="yaml_exp",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channels=[
                ChannelEntry(name="Phase", marker="Phase"),
                ChannelEntry(name="GFP", marker="GFP"),
                ChannelEntry(name="RFP", marker="RFP"),
            ],
            channel_names=["Phase", "GFP", "RFP"],
            perturbation_wells={
                "uninfected": ["A/1"],
                "infected": ["B/1"],
            },
            interval_minutes=30.0,
            start_hpi=3.0,
        )
        collection = Collection(
            name="test",
            experiments=[exp],
        )
        collection_path = tmp_path / "collection.yml"
        save_collection(collection, collection_path)

        registry = ExperimentRegistry.from_collection(collection_path, z_window=1)
        assert len(registry.experiments) == 1
        assert registry.experiments[0].name == "yaml_exp"
        assert registry.experiments[0].start_hpi == 3.0

    def test_tau_range_frames_30min(self, exp_entry_a):
        """tau_range_hours=(0.5, 2.0) at 30min -> (1, 4)."""
        collection = _make_collection_single(exp_entry_a)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.tau_range_frames("exp_a", (0.5, 2.0))
        assert result == (1, 4)

    def test_tau_range_frames_15min(self, exp_entry_b):
        """tau_range_hours=(0.5, 2.0) at 15min -> (2, 8)."""
        collection = _make_collection_single(exp_entry_b)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.tau_range_frames("exp_b", (0.5, 2.0))
        assert result == (2, 8)

    def test_tau_range_frames_warns_few_frames(self, exp_entry_a, caplog):
        """Warns when min_frames >= max_frames."""
        collection = _make_collection_single(exp_entry_a)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        with caplog.at_level(logging.WARNING):
            # (0.0, 0.0) at 30min -> (0, 0), min >= max
            registry.tau_range_frames("exp_a", (0.0, 0.0))
        assert any("fewer than 2" in msg.lower() or "few" in msg.lower() for msg in caplog.messages)

    def test_get_experiment(self, exp_entry_a):
        """Lookup by name returns the correct config."""
        collection = _make_collection_single(exp_entry_a)
        registry = ExperimentRegistry(collection=collection, z_window=1)
        result = registry.get_experiment("exp_a")
        assert result.name == "exp_a"
        assert result is exp_entry_a

    def test_get_experiment_not_found(self, exp_entry_a):
        """KeyError when experiment name not found."""
        collection = _make_collection_single(exp_entry_a)
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
            perturbation_wells={"ctrl": ["A/1"]},
            interval_minutes=-5.0,
        )
        with pytest.raises(ValueError, match="interval_minutes"):
            Collection(
                name="test",
                experiments=[exp],
            )

    def test_empty_perturbation_wells(self, mini_zarr, tmp_path):
        """ValueError when perturbation_wells is empty."""
        exp = ExperimentEntry(
            name="empty_wells",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            perturbation_wells={},
            interval_minutes=30.0,
        )
        with pytest.raises(ValueError, match="perturbation_wells"):
            Collection(
                name="test",
                experiments=[exp],
            )
