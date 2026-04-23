"""Tests for MultiExperimentDataModule: experiment-level train/val split,
FlexibleBatchSampler wiring, ChannelDropout integration, and hyperparameter
exposure for Lightning CLI configurability."""

from __future__ import annotations

import pytest
import torch

from viscy_data.cell_index import build_timelapse_cell_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = ["Phase", "GFP"]
_YX_PATCH = (32, 32)
_FINAL_YX_PATCH = (24, 24)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def four_experiments(tmp_path, _create_experiment, _write_collection_yaml):
    """Four synthetic experiments with collection YAML and cell index parquet."""
    entries = []
    for i, name in enumerate(["exp_a", "exp_b", "exp_c", "exp_d"]):
        row_letter = chr(ord("A") + i)
        entries.append(
            _create_experiment(
                tmp_path,
                name=name,
                channel_names=_CHANNEL_NAMES,
                wells=[(row_letter, "1")],
                perturbation_wells={"control": [f"{row_letter}/1"]},
            )
        )
    collection_path = _write_collection_yaml(tmp_path, entries)
    parquet_path = tmp_path / "cell_index.parquet"
    build_timelapse_cell_index(collection_path, parquet_path)
    return parquet_path, entries


@pytest.fixture()
def two_experiments(tmp_path, _create_experiment, _write_collection_yaml):
    """Two synthetic experiments with cell index parquet."""
    entries = [
        _create_experiment(
            tmp_path,
            name="exp_a",
            channel_names=_CHANNEL_NAMES,
            wells=[("A", "1")],
            perturbation_wells={"control": ["A/1"]},
        ),
        _create_experiment(
            tmp_path,
            name="exp_b",
            channel_names=_CHANNEL_NAMES,
            wells=[("B", "1")],
            perturbation_wells={"treated": ["B/1"]},
        ),
    ]
    collection_path = _write_collection_yaml(tmp_path, entries)
    parquet_path = tmp_path / "cell_index.parquet"
    build_timelapse_cell_index(collection_path, parquet_path)
    return parquet_path, entries


@pytest.fixture()
def multi_fov_experiments(tmp_path, _create_experiment, _write_collection_yaml):
    """Two experiments with 5 FOVs each for FOV-level split testing."""
    entries = [
        _create_experiment(
            tmp_path,
            name="exp_a",
            channel_names=_CHANNEL_NAMES,
            wells=[("A", "1")],
            perturbation_wells={"control": ["A/1"]},
            fovs_per_well=5,
        ),
        _create_experiment(
            tmp_path,
            name="exp_b",
            channel_names=_CHANNEL_NAMES,
            wells=[("B", "1")],
            perturbation_wells={"treated": ["B/1"]},
            fovs_per_well=5,
        ),
    ]
    collection_path = _write_collection_yaml(tmp_path, entries)
    parquet_path = tmp_path / "cell_index.parquet"
    build_timelapse_cell_index(collection_path, parquet_path)
    return parquet_path, entries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitExposesAllHyperparameters:
    """DATA-05: All hyperparameters are exposed as __init__ parameters."""

    def test_init_exposes_all_hyperparameters(self, two_experiments):
        """Instantiate with all hyperparameters explicitly set and verify storage."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            split_ratio=0.7,
            tau_range=(0.5, 2.0),
            tau_decay_rate=3.0,
            batch_size=64,
            num_workers=2,
            batch_group_by=None,
            stratify_by=None,
            leaky=0.1,
            temporal_enrichment=True,
            temporal_window_hours=3.0,
            temporal_global_fraction=0.5,
            channel_dropout_channels=[0, 1],
            channel_dropout_prob=0.8,
            cache_pool_bytes=1024,
            seed=42,
        )

        assert dm.split_ratio == 0.7
        assert dm.tau_range == (0.5, 2.0)
        assert dm.tau_decay_rate == 3.0
        assert dm.batch_size == 64
        assert dm.num_workers == 2
        assert dm.batch_group_by is None
        assert dm.stratify_by is None
        assert dm.leaky == 0.1
        assert dm.temporal_enrichment is True
        assert dm.temporal_window_hours == 3.0
        assert dm.temporal_global_fraction == 0.5
        assert dm.channel_dropout_channels == [0, 1]
        assert dm.channel_dropout_prob == 0.8
        assert dm.cache_pool_bytes == 1024
        assert dm.seed == 42


class TestTrainValSplitByExperiment:
    """DATA-04: Train/val split is by whole experiments, not individual FOVs."""

    def test_train_val_split_by_experiment(self, four_experiments):
        """With 4 experiments and val_experiments=[exp_c, exp_d], verify correct split."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = four_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
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
        assert val_experiments == {"exp_c", "exp_d"}, f"Val experiments {val_experiments} should be {{exp_c, exp_d}}"

        # No overlap: train FOVs should not appear in val
        train_fovs = set(dm.train_dataset.index.tracks["fov_name"].unique())
        val_fovs = set(dm.val_dataset.index.tracks["fov_name"].unique())
        assert train_fovs.isdisjoint(val_fovs), f"FOV overlap between train and val: {train_fovs & val_fovs}"


class TestTrainDataloaderUsesFlexibleBatchSampler:
    """DATA-03: Training uses FlexibleBatchSampler."""

    def test_train_dataloader_uses_flexible_batch_sampler(self, two_experiments):
        """train_dataloader() returns a ThreadDataLoader with FlexibleBatchSampler."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            batch_group_by="experiment",
            stratify_by="perturbation",
            temporal_enrichment=False,
        )
        dm.setup("fit")
        train_dl = dm.train_dataloader()

        from monai.data.thread_buffer import ThreadDataLoader

        from viscy_data.sampler import FlexibleBatchSampler

        assert isinstance(train_dl, ThreadDataLoader), f"Expected ThreadDataLoader, got {type(train_dl)}"
        # The batch_sampler should be a FlexibleBatchSampler
        assert isinstance(train_dl.batch_sampler, FlexibleBatchSampler), (
            f"Expected FlexibleBatchSampler, got {type(train_dl.batch_sampler)}"
        )
        # Verify sampler settings match
        sampler = train_dl.batch_sampler
        assert sampler.batch_group_by == ["experiment"]
        assert sampler.stratify_by == ["perturbation"]
        assert sampler.temporal_enrichment is False


class TestTrainDataloaderWiresDDPTopology:
    """train_dataloader must forward Trainer world_size/rank to the sampler."""

    def test_reads_world_size_and_rank_from_trainer(self, two_experiments):
        from types import SimpleNamespace

        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            batch_group_by="experiment",
            stratify_by="perturbation",
            temporal_enrichment=False,
        )
        dm.setup("fit")
        # Bypass LightningDataModule.trainer descriptor state check; we
        # only need world_size/global_rank to flow into the sampler.
        dm.__dict__["trainer"] = SimpleNamespace(world_size=4, global_rank=2)
        sampler = dm.train_dataloader().batch_sampler
        assert (sampler.num_replicas, sampler.rank) == (4, 2)


class TestValDataloaderNoBatchSampler:
    """Validation should be deterministic without FlexibleBatchSampler."""

    def test_val_dataloader_no_batch_sampler(self, two_experiments):
        """val_dataloader uses simple sequential loading."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
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

    def test_on_after_batch_transfer_applies_channel_dropout_and_transforms(self, two_experiments):
        """Create a mock batch and verify on_after_batch_transfer processes it."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
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
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
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
        assert torch.all(result_train["anchor"][:, 1] == 0.0), "Training: channel 1 should be all zeros with p=1.0"
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
        assert not torch.all(result_eval["anchor"][:, 1] == 0.0), "Eval: channel 1 should NOT be zeroed"


class TestFovLevelSplit:
    """FOV-level split when val_experiments is empty."""

    def test_fov_split_no_overlap(self, multi_fov_experiments):
        """With split_ratio=0.6, FOVs are split within each experiment with no overlap."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = multi_fov_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=[],
            split_ratio=0.6,
            tau_range=(0.5, 2.0),
            batch_size=8,
            seed=42,
        )
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        train_fovs = set(dm.train_dataset.index.tracks["fov_name"].unique())
        val_fovs = set(dm.val_dataset.index.tracks["fov_name"].unique())

        # No overlap
        assert train_fovs.isdisjoint(val_fovs), f"FOV overlap: {train_fovs & val_fovs}"

        # Both experiments should be represented in train
        train_exps = set(dm.train_dataset.index.tracks["experiment"].unique())
        assert train_exps == {"exp_a", "exp_b"}

        # Val should also have FOVs from both experiments
        val_exps = set(dm.val_dataset.index.tracks["experiment"].unique())
        assert val_exps == {"exp_a", "exp_b"}

    def test_fov_split_ratio_1_no_val(self, multi_fov_experiments):
        """With split_ratio=1.0, all FOVs go to train and val_dataset is None."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = multi_fov_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=[],
            split_ratio=1.0,
            tau_range=(0.5, 2.0),
            batch_size=8,
        )
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is None

    def test_fov_split_default_val_experiments(self, multi_fov_experiments):
        """Default val_experiments=[] triggers FOV split."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = multi_fov_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            split_ratio=0.8,
            tau_range=(0.5, 2.0),
            batch_size=8,
        )
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        train_fovs = set(dm.train_dataset.index.tracks["fov_name"].unique())
        val_fovs = set(dm.val_dataset.index.tracks["fov_name"].unique())
        assert train_fovs.isdisjoint(val_fovs)


class TestNewPositiveParams:
    """Test new positive_cell_source / positive_match_columns / positive_channel_source params."""

    def test_positive_cell_source_self_stores_on_dm(self, two_experiments):
        """positive_cell_source='self' is stored and passed to datasets."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            positive_cell_source="self",
        )
        assert dm.positive_cell_source == "self"
        dm.setup("fit")
        assert dm.train_dataset.positive_cell_source == "self"

    def test_positive_match_columns_stored_on_dm(self, two_experiments):
        """positive_match_columns is stored on datamodule."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            positive_match_columns=["perturbation"],
        )
        assert dm.positive_match_columns == ["perturbation"]

    def test_positive_channel_source_any_stored(self, two_experiments):
        """positive_channel_source='any' is stored on datamodule and dataset."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            positive_channel_source="any",
        )
        assert dm.positive_channel_source == "any"
        dm.setup("fit")
        assert dm.train_dataset.positive_channel_source == "any"

    def test_self_positive_all_tracks_are_valid_anchors(self, two_experiments):
        """With positive_cell_source='self', all tracks become valid anchors."""
        from dynaclr.data.datamodule import MultiExperimentDataModule

        parquet_path, _ = two_experiments
        dm = MultiExperimentDataModule(
            cell_index_path=str(parquet_path),
            z_window=1,
            yx_patch_size=_YX_PATCH,
            final_yx_patch_size=_FINAL_YX_PATCH,
            val_experiments=["exp_b"],
            tau_range=(0.5, 2.0),
            batch_size=8,
            positive_cell_source="self",
        )
        dm.setup("fit")
        n_unique_cells = dm.train_dataset.index.tracks["cell_id"].nunique()
        n_anchors = len(dm.train_dataset.index.valid_anchors)
        assert n_anchors == n_unique_cells
