"""Tests for ExperimentConfig and ExperimentRegistry."""

import logging

import numpy as np
import pytest
import yaml
from iohub.ngff import open_ome_zarr

from dynaclr.experiment import ExperimentConfig, ExperimentRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_zarr(tmp_path):
    """Create a minimal HCS OME-Zarr store with channels ['Phase', 'GFP', 'RFP']."""
    zarr_path = tmp_path / "exp_a.zarr"
    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=["Phase", "GFP", "RFP"]
    ) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 3, 1, 64, 64), dtype=np.float32)
    return zarr_path


@pytest.fixture()
def mini_zarr_mito(tmp_path):
    """Create a second zarr with channels ['Phase', 'Mito']."""
    zarr_path = tmp_path / "exp_b.zarr"
    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=["Phase", "Mito"]
    ) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 2, 1, 64, 64), dtype=np.float32)
    return zarr_path


@pytest.fixture()
def exp_config_a(mini_zarr, tmp_path):
    """ExperimentConfig for experiment A with 3 channels, 2 source channels."""
    return ExperimentConfig(
        name="exp_a",
        data_path=str(mini_zarr),
        tracks_path=str(tmp_path / "tracks_a"),
        channel_names=["Phase", "GFP", "RFP"],
        source_channel=["Phase", "RFP"],
        condition_wells={"uninfected": ["A/1"], "infected": ["B/1"]},
        interval_minutes=30.0,
    )


@pytest.fixture()
def exp_config_b(mini_zarr_mito, tmp_path):
    """ExperimentConfig for experiment B with 2 channels, 2 source channels."""
    return ExperimentConfig(
        name="exp_b",
        data_path=str(mini_zarr_mito),
        tracks_path=str(tmp_path / "tracks_b"),
        channel_names=["Phase", "Mito"],
        source_channel=["Phase", "Mito"],
        condition_wells={"control": ["A/1"]},
        interval_minutes=15.0,
    )


# ---------------------------------------------------------------------------
# ExperimentConfig tests
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_experiment_config_creation(self, exp_config_a):
        """All fields are accessible after creation."""
        assert exp_config_a.name == "exp_a"
        assert exp_config_a.data_path == str(exp_config_a.data_path)
        assert exp_config_a.channel_names == ["Phase", "GFP", "RFP"]
        assert exp_config_a.source_channel == ["Phase", "RFP"]
        assert exp_config_a.condition_wells == {
            "uninfected": ["A/1"],
            "infected": ["B/1"],
        }
        assert exp_config_a.interval_minutes == 30.0

    def test_experiment_config_defaults(self, mini_zarr, tmp_path):
        """Default values for optional fields."""
        cfg = ExperimentConfig(
            name="defaults_test",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            source_channel=["Phase"],
            condition_wells={"ctrl": ["A/1"]},
        )
        assert cfg.interval_minutes == 30.0
        assert cfg.start_hpi == 0.0
        assert cfg.organelle == ""
        assert cfg.date == ""
        assert cfg.moi == 0.0


# ---------------------------------------------------------------------------
# ExperimentRegistry tests
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    def test_registry_channel_maps(self, exp_config_a):
        """channel_maps correctly maps source_channel position -> zarr index."""
        registry = ExperimentRegistry(experiments=[exp_config_a])
        # source_channel=["Phase", "RFP"], channel_names=["Phase", "GFP", "RFP"]
        # Position 0 -> index 0 (Phase), Position 1 -> index 2 (RFP)
        assert registry.channel_maps["exp_a"] == {0: 0, 1: 2}

    def test_registry_channel_maps_different_names(
        self, exp_config_a, exp_config_b
    ):
        """Positional alignment: different channel names, same position count."""
        registry = ExperimentRegistry(
            experiments=[exp_config_a, exp_config_b]
        )
        # exp_a: source=["Phase", "RFP"] in ["Phase", "GFP", "RFP"] -> {0:0, 1:2}
        assert registry.channel_maps["exp_a"] == {0: 0, 1: 2}
        # exp_b: source=["Phase", "Mito"] in ["Phase", "Mito"] -> {0:0, 1:1}
        assert registry.channel_maps["exp_b"] == {0: 0, 1: 1}

    def test_registry_source_channel_not_in_channel_names(
        self, mini_zarr, tmp_path
    ):
        """ValueError when source_channel has entry not in channel_names."""
        cfg = ExperimentConfig(
            name="bad_source",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            source_channel=["Phase", "DAPI"],  # DAPI not in channel_names
            condition_wells={"ctrl": ["A/1"]},
        )
        with pytest.raises(ValueError, match="DAPI"):
            ExperimentRegistry(experiments=[cfg])

    def test_registry_mismatched_source_channel_count(
        self, mini_zarr, mini_zarr_mito, tmp_path
    ):
        """ValueError when experiments have different source_channel counts."""
        cfg_a = ExperimentConfig(
            name="exp_a",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks_a"),
            channel_names=["Phase", "GFP", "RFP"],
            source_channel=["Phase", "RFP"],  # 2 channels
            condition_wells={"ctrl": ["A/1"]},
        )
        cfg_b = ExperimentConfig(
            name="exp_b",
            data_path=str(mini_zarr_mito),
            tracks_path=str(tmp_path / "tracks_b"),
            channel_names=["Phase", "Mito"],
            source_channel=["Phase"],  # 1 channel -- mismatch
            condition_wells={"ctrl": ["A/1"]},
        )
        with pytest.raises(ValueError, match="source_channel"):
            ExperimentRegistry(experiments=[cfg_a, cfg_b])

    def test_registry_duplicate_names(self, exp_config_a):
        """ValueError when two experiments share the same name."""
        dup = ExperimentConfig(
            name="exp_a",  # duplicate
            data_path=exp_config_a.data_path,
            tracks_path=exp_config_a.tracks_path,
            channel_names=exp_config_a.channel_names,
            source_channel=exp_config_a.source_channel,
            condition_wells=exp_config_a.condition_wells,
        )
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            ExperimentRegistry(experiments=[exp_config_a, dup])

    def test_registry_empty_experiments(self):
        """ValueError when experiments list is empty."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            ExperimentRegistry(experiments=[])

    def test_registry_zarr_validation(self, exp_config_a):
        """Opens zarr and validates channel_names match metadata."""
        # Should succeed -- channel_names match the zarr store
        registry = ExperimentRegistry(experiments=[exp_config_a])
        assert registry.num_source_channels == 2

    def test_registry_zarr_channel_mismatch(self, mini_zarr, tmp_path):
        """ValueError when channel_names don't match zarr metadata."""
        cfg = ExperimentConfig(
            name="mismatch",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "Mito"],  # Mito != RFP in zarr
            source_channel=["Phase"],
            condition_wells={"ctrl": ["A/1"]},
        )
        with pytest.raises(ValueError, match="channel"):
            ExperimentRegistry(experiments=[cfg])

    def test_registry_data_path_not_exists(self, tmp_path):
        """ValueError when data_path does not exist."""
        cfg = ExperimentConfig(
            name="no_path",
            data_path=str(tmp_path / "nonexistent.zarr"),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase"],
            source_channel=["Phase"],
            condition_wells={"ctrl": ["A/1"]},
        )
        with pytest.raises(ValueError, match="data_path"):
            ExperimentRegistry(experiments=[cfg])

    def test_from_yaml(self, mini_zarr, tmp_path):
        """Round-trip: write YAML, load, verify registry."""
        yaml_data = {
            "experiments": [
                {
                    "name": "yaml_exp",
                    "data_path": str(mini_zarr),
                    "tracks_path": str(tmp_path / "tracks"),
                    "channel_names": ["Phase", "GFP", "RFP"],
                    "source_channel": ["Phase", "GFP"],
                    "condition_wells": {
                        "uninfected": ["A/1"],
                        "infected": ["B/1"],
                    },
                    "interval_minutes": 30.0,
                    "start_hpi": 3.0,
                }
            ]
        }
        yaml_path = tmp_path / "experiments.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        registry = ExperimentRegistry.from_yaml(yaml_path)
        assert len(registry.experiments) == 1
        assert registry.experiments[0].name == "yaml_exp"
        assert registry.experiments[0].start_hpi == 3.0
        assert registry.channel_maps["yaml_exp"] == {0: 0, 1: 1}

    def test_tau_range_frames_30min(self, exp_config_a):
        """tau_range_hours=(0.5, 2.0) at 30min -> (1, 4)."""
        registry = ExperimentRegistry(experiments=[exp_config_a])
        result = registry.tau_range_frames("exp_a", (0.5, 2.0))
        assert result == (1, 4)

    def test_tau_range_frames_15min(self, exp_config_b):
        """tau_range_hours=(0.5, 2.0) at 15min -> (2, 8)."""
        registry = ExperimentRegistry(experiments=[exp_config_b])
        result = registry.tau_range_frames("exp_b", (0.5, 2.0))
        assert result == (2, 8)

    def test_tau_range_frames_warns_few_frames(self, exp_config_a, caplog):
        """Warns when min_frames >= max_frames."""
        registry = ExperimentRegistry(experiments=[exp_config_a])
        with caplog.at_level(logging.WARNING):
            # (0.0, 0.0) at 30min -> (0, 0), min >= max
            registry.tau_range_frames("exp_a", (0.0, 0.0))
        assert any("fewer than 2" in msg.lower() or "few" in msg.lower() for msg in caplog.messages)

    def test_get_experiment(self, exp_config_a):
        """Lookup by name returns the correct config."""
        registry = ExperimentRegistry(experiments=[exp_config_a])
        result = registry.get_experiment("exp_a")
        assert result.name == "exp_a"
        assert result is exp_config_a

    def test_get_experiment_not_found(self, exp_config_a):
        """KeyError when experiment name not found."""
        registry = ExperimentRegistry(experiments=[exp_config_a])
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get_experiment("nonexistent")

    def test_negative_interval_minutes(self, mini_zarr, tmp_path):
        """ValueError when interval_minutes is negative."""
        cfg = ExperimentConfig(
            name="neg_interval",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            source_channel=["Phase"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=-5.0,
        )
        with pytest.raises(ValueError, match="interval_minutes"):
            ExperimentRegistry(experiments=[cfg])

    def test_empty_condition_wells(self, mini_zarr, tmp_path):
        """ValueError when condition_wells is empty."""
        cfg = ExperimentConfig(
            name="empty_wells",
            data_path=str(mini_zarr),
            tracks_path=str(tmp_path / "tracks"),
            channel_names=["Phase", "GFP", "RFP"],
            source_channel=["Phase"],
            condition_wells={},
        )
        with pytest.raises(ValueError, match="condition_wells"):
            ExperimentRegistry(experiments=[cfg])
