"""Tests for MultiExperimentDataModule: experiment-level train/val split,
FlexibleBatchSampler wiring, ChannelDropout integration, and hyperparameter
exposure for Lightning CLI configurability."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from dynaclr.experiment import ExperimentConfig, ExperimentRegistry
from dynaclr.index import MultiExperimentIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = ["Phase", "GFP"]
_IMG_H = 64
_IMG_W = 64
_N_T = 10
_N_Z = 1
_N_TRACKS = 5
_YX_PATCH = (32, 32)
_FINAL_YX_PATCH = (24, 24)
_Z_RANGE = (0, 1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracks_csv(
    path: Path,
    n_tracks: int = _N_TRACKS,
    n_t: int = _N_T,
    *,
    start_t: int = 0,
) -> None:
    """Write a tracking CSV with standard columns."""
    rows = []
    for tid in range(n_tracks):
        for t in range(start_t, start_t + n_t):
            rows.append(
                {
                    "track_id": tid,
                    "t": t,
                    "id": tid * n_t + t,
                    "parent_track_id": float("nan"),
                    "parent_id": float("nan"),
                    "z": 0,
                    "y": 32.0,
                    "x": 32.0,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _create_experiment(
    tmp_path: Path,
    name: str,
    wells: list[tuple[str, str]],
    condition_wells: dict[str, list[str]],
    fovs_per_well: int = 1,
    n_tracks: int = _N_TRACKS,
    n_t: int = _N_T,
) -> ExperimentConfig:
    """Create a mini HCS OME-Zarr store, tracking CSVs, and return an ExperimentConfig."""
    from iohub.ngff import open_ome_zarr

    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(_CHANNEL_NAMES)

    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=_CHANNEL_NAMES
    ) as plate:
        for row, col in wells:
            for fov_idx in range(fovs_per_well):
                pos = plate.create_position(row, col, str(fov_idx))
                arr = pos.create_zeros(
                    "0",
                    shape=(n_t, n_ch, _N_Z, _IMG_H, _IMG_W),
                    dtype=np.float32,
                )
                rng = np.random.default_rng(42)
                arr[:] = rng.standard_normal(arr.shape).astype(np.float32)
                fov_name = f"{row}/{col}/{fov_idx}"
                csv_path = tracks_root / fov_name / "tracks.csv"
                _make_tracks_csv(csv_path, n_tracks=n_tracks, n_t=n_t)

    return ExperimentConfig(
        name=name,
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channel_names=_CHANNEL_NAMES,
        source_channel=["Phase", "GFP"],
        condition_wells=condition_wells,
        interval_minutes=30.0,
    )


def _create_four_experiments(tmp_path: Path) -> list[ExperimentConfig]:
    """Create 4 experiments for train/val split testing."""
    configs = []
    for i, name in enumerate(["exp_a", "exp_b", "exp_c", "exp_d"]):
        row_letter = chr(ord("A") + i)
        configs.append(
            _create_experiment(
                tmp_path,
                name=name,
                wells=[(row_letter, "1")],
                condition_wells={"control": [f"{row_letter}/1"]},
            )
        )
    return configs


def _write_experiments_yaml(
    tmp_path: Path, configs: list[ExperimentConfig]
) -> Path:
    """Write experiments YAML from a list of ExperimentConfig objects."""
    import yaml

    yaml_path = tmp_path / "experiments.yaml"
    data = {
        "experiments": [
            {
                "name": c.name,
                "data_path": c.data_path,
                "tracks_path": c.tracks_path,
                "channel_names": c.channel_names,
                "source_channel": c.source_channel,
                "condition_wells": c.condition_wells,
                "interval_minutes": c.interval_minutes,
            }
            for c in configs
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return yaml_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def four_experiments(tmp_path):
    """Four synthetic experiments with YAML config."""
    configs = _create_four_experiments(tmp_path)
    yaml_path = _write_experiments_yaml(tmp_path, configs)
    return yaml_path, configs


@pytest.fixture()
def two_experiments(tmp_path):
    """Two synthetic experiments for simpler tests."""
    configs = [
        _create_experiment(
            tmp_path,
            name="exp_a",
            wells=[("A", "1")],
            condition_wells={"control": ["A/1"]},
        ),
        _create_experiment(
            tmp_path,
            name="exp_b",
            wells=[("B", "1")],
            condition_wells={"treated": ["B/1"]},
        ),
    ]
    yaml_path = _write_experiments_yaml(tmp_path, configs)
    return yaml_path, configs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitExposesAllHyperparameters:
    """DATA-05: All hyperparameters are exposed as __init__ parameters."""

    def test_init_exposes_all_hyperparameters(self, two_experiments):
        """Instantiate with all hyperparameters explicitly set and verify storage."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            tau_decay_rate=3.0,
            batch_size=64,
            num_workers=2,
            experiment_aware=False,
            condition_balanced=False,
            leaky=0.1,
            temporal_enrichment=True,
            temporal_window_hours=3.0,
            temporal_global_fraction=0.5,
            hcl_beta=0.7,
            channel_dropout_channels=[0, 1],
            channel_dropout_prob=0.8,
            cache_pool_bytes=1024,
            seed=42,
        )

        assert dm.tau_range == (0.5, 2.0)
        assert dm.tau_decay_rate == 3.0
        assert dm.batch_size == 64
        assert dm.num_workers == 2
        assert dm.experiment_aware is False
        assert dm.condition_balanced is False
        assert dm.leaky == 0.1
        assert dm.temporal_enrichment is True
        assert dm.temporal_window_hours == 3.0
        assert dm.temporal_global_fraction == 0.5
        assert dm.hcl_beta == 0.7
        assert dm.channel_dropout_channels == [0, 1]
        assert dm.channel_dropout_prob == 0.8
        assert dm.cache_pool_bytes == 1024
        assert dm.seed == 42


class TestTrainValSplitByExperiment:
    """DATA-04: Train/val split is by whole experiments, not individual FOVs."""

    def test_train_val_split_by_experiment(self, four_experiments):
        """With 4 experiments and val_experiments=[exp_c, exp_d], verify correct split."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = four_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_c", "exp_d"],
            tau_range=(0.5, 2.0),
            batch_size=8,
        )
        dm.setup("fit")

        # Train dataset should only contain exp_a and exp_b
        train_experiments = set(dm.train_dataset.index.tracks["experiment"].unique())
        assert train_experiments == {"exp_a", "exp_b"}, (
            f"Train experiments {train_experiments} should be {{exp_a, exp_b}}"
        )

        # Val dataset should only contain exp_c and exp_d
        val_experiments = set(dm.val_dataset.index.tracks["experiment"].unique())
        assert val_experiments == {"exp_c", "exp_d"}, (
            f"Val experiments {val_experiments} should be {{exp_c, exp_d}}"
        )

        # No overlap: train FOVs should not appear in val
        train_fovs = set(dm.train_dataset.index.tracks["fov_name"].unique())
        val_fovs = set(dm.val_dataset.index.tracks["fov_name"].unique())
        assert train_fovs.isdisjoint(val_fovs), (
            f"FOV overlap between train and val: {train_fovs & val_fovs}"
        )


class TestTrainDataloaderUsesFlexibleBatchSampler:
    """DATA-03: Training uses FlexibleBatchSampler."""

    def test_train_dataloader_uses_flexible_batch_sampler(self, two_experiments):
        """train_dataloader() returns a ThreadDataLoader with FlexibleBatchSampler."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            experiment_aware=True,
            condition_balanced=True,
            temporal_enrichment=False,
        )
        dm.setup("fit")
        train_dl = dm.train_dataloader()

        from monai.data.thread_buffer import ThreadDataLoader
        from viscy_data.sampler import FlexibleBatchSampler

        assert isinstance(train_dl, ThreadDataLoader), (
            f"Expected ThreadDataLoader, got {type(train_dl)}"
        )
        # The batch_sampler should be a FlexibleBatchSampler
        assert isinstance(train_dl.batch_sampler, FlexibleBatchSampler), (
            f"Expected FlexibleBatchSampler, got {type(train_dl.batch_sampler)}"
        )
        # Verify sampler settings match
        sampler = train_dl.batch_sampler
        assert sampler.experiment_aware is True
        assert sampler.condition_balanced is True
        assert sampler.temporal_enrichment is False


class TestValDataloaderNoBatchSampler:
    """Validation should be deterministic without FlexibleBatchSampler."""

    def test_val_dataloader_no_batch_sampler(self, two_experiments):
        """val_dataloader uses simple sequential loading."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
        )
        dm.setup("fit")
        val_dl = dm.val_dataloader()

        from viscy_data.sampler import FlexibleBatchSampler

        # val_dataloader should NOT use FlexibleBatchSampler
        assert not isinstance(val_dl.batch_sampler, FlexibleBatchSampler), (
            "Validation should NOT use FlexibleBatchSampler"
        )


class TestOnAfterBatchTransferAppliesTransforms:
    """Verify on_after_batch_transfer applies transforms and ChannelDropout."""

    def test_on_after_batch_transfer_applies_channel_dropout_and_transforms(
        self, two_experiments
    ):
        """Create a mock batch and verify on_after_batch_transfer processes it."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            channel_dropout_channels=[1],
            channel_dropout_prob=0.0,  # No dropout for this test
        )
        dm.setup("fit")

        # Create a synthetic batch dict
        B, C, Z, Y, X = 4, 2, 1, 32, 32
        batch = {
            "anchor": torch.randn(B, C, Z, Y, X),
            "positive": torch.randn(B, C, Z, Y, X),
            "anchor_norm_meta": [None] * B,
            "positive_norm_meta": [None] * B,
        }

        result = dm.on_after_batch_transfer(batch, 0)

        # Output should have anchor and positive as Tensors
        assert isinstance(result["anchor"], torch.Tensor)
        assert isinstance(result["positive"], torch.Tensor)

        # norm_meta keys should be consumed (removed)
        assert "anchor_norm_meta" not in result
        assert "positive_norm_meta" not in result

        # Final crop should reduce spatial size to final_yx_patch_size
        assert result["anchor"].shape[-2:] == (
            _FINAL_YX_PATCH[0],
            _FINAL_YX_PATCH[1],
        ), f"Expected spatial {_FINAL_YX_PATCH}, got {result['anchor'].shape[-2:]}"


class TestChannelDropoutIntegration:
    """Verify ChannelDropout behavior in train vs eval mode."""

    def test_channel_dropout_integration(self, two_experiments):
        """With p=1.0 on channel 1, training zeros ch1; eval preserves it."""
        from dynaclr.datamodule import MultiExperimentDataModule

        yaml_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            experiments_yaml=str(yaml_path),
            z_range=_Z_RANGE,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            channel_dropout_channels=[1],
            channel_dropout_prob=1.0,  # Always drop channel 1
        )
        dm.setup("fit")

        B, C, Z, Y, X = 4, 2, 1, 32, 32
        batch_train = {
            "anchor": torch.randn(B, C, Z, Y, X).abs() + 0.1,  # all positive
            "positive": torch.randn(B, C, Z, Y, X).abs() + 0.1,
            "anchor_norm_meta": [None] * B,
            "positive_norm_meta": [None] * B,
        }

        # Training mode: channel 1 should be zeroed
        dm.channel_dropout.train()
        result_train = dm.on_after_batch_transfer(batch_train, 0)
        assert torch.all(result_train["anchor"][:, 1] == 0.0), (
            "Training: channel 1 should be all zeros with p=1.0"
        )
        assert torch.all(result_train["positive"][:, 1] == 0.0), (
            "Training: positive channel 1 should be all zeros with p=1.0"
        )

        # Eval mode: channel 1 should be preserved
        dm.channel_dropout.eval()
        batch_eval = {
            "anchor": torch.randn(B, C, Z, Y, X).abs() + 0.1,
            "positive": torch.randn(B, C, Z, Y, X).abs() + 0.1,
            "anchor_norm_meta": [None] * B,
            "positive_norm_meta": [None] * B,
        }
        result_eval = dm.on_after_batch_transfer(batch_eval, 0)
        assert not torch.all(result_eval["anchor"][:, 1] == 0.0), (
            "Eval: channel 1 should NOT be zeroed"
        )
