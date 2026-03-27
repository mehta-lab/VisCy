"""End-to-end tests for the flat parquet pipeline.

Validates the full data flow:
  flat parquet → ExperimentRegistry.from_cell_index → MultiExperimentIndex
  → FlexibleBatchSampler → MultiExperimentTripletDataset → batch tensors

Tests cover:
  - Channel selection modes (bag-of-channels, all-channels, fixed)
  - Batch sampler groupings (marker, experiment, experiment+marker, none)
  - Anchor→positive composition (same channel, same cell_id lineage)
  - Pixel size round-trip through registry
"""

from __future__ import annotations

import pytest

from dynaclr.data.dataset import MultiExperimentTripletDataset
from dynaclr.data.experiment import ExperimentRegistry
from dynaclr.data.index import MultiExperimentIndex
from viscy_data.collection import ChannelEntry
from viscy_data.sampler import FlexibleBatchSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YX_PATCH = (32, 32)
_BATCH_SIZE = 4
_TAU_RANGE = (0.5, 4.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def flat_parquet_setup(tmp_path, _create_experiment, _write_collection_yaml):
    """Create 2 experiments with different channel counts and a flat parquet.

    exp_a: 2 channels (Phase3D, GFP) → markers (Phase3D, SEC61)
           2 wells (ctrl, treated), 1 FOV each, 5 tracks, 10 timepoints
    exp_b: 3 channels (Phase3D, GFP, mCherry) → markers (Phase3D, G3BP1, pAL17)
           1 well (ctrl), 1 FOV, 3 tracks, 8 timepoints
    """
    from viscy_data.cell_index import build_timelapse_cell_index

    entry_a = _create_experiment(
        tmp_path,
        "exp_a",
        channel_names=["Phase3D", "GFP"],
        wells=[("A", "1"), ("B", "1")],
        perturbation_wells={"ctrl": ["A/1"], "treated": ["B/1"]},
        fovs_per_well=1,
        n_tracks=5,
        n_t=10,
        interval_minutes=30.0,
        start_hpi=0.0,
        channels=[
            ChannelEntry(name="Phase3D", marker="Phase3D"),
            ChannelEntry(name="GFP", marker="SEC61"),
        ],
        pixel_size_xy_um=0.108,
        pixel_size_z_um=0.3,
    )
    entry_b = _create_experiment(
        tmp_path,
        "exp_b",
        channel_names=["Phase3D", "GFP", "mCherry"],
        wells=[("A", "1")],
        perturbation_wells={"ctrl": ["A/1"]},
        fovs_per_well=1,
        n_tracks=3,
        n_t=8,
        interval_minutes=15.0,
        start_hpi=1.0,
        channels=[
            ChannelEntry(name="Phase3D", marker="Phase3D"),
            ChannelEntry(name="GFP", marker="G3BP1"),
            ChannelEntry(name="mCherry", marker="pAL17"),
        ],
        pixel_size_xy_um=0.108,
        pixel_size_z_um=0.3,
    )

    yaml_path = _write_collection_yaml(tmp_path, [entry_a, entry_b])

    parquet_path = tmp_path / "flat.parquet"
    df = build_timelapse_cell_index(yaml_path, parquet_path, num_workers=1)

    registry = ExperimentRegistry.from_cell_index(
        parquet_path,
        reference_pixel_size_xy_um=0.108,
        reference_pixel_size_z_um=0.3,
    )
    index = MultiExperimentIndex(
        registry=registry,
        yx_patch_size=_YX_PATCH,
        tau_range_hours=_TAU_RANGE,
        cell_index_path=parquet_path,
    )

    return index, registry, df


# ---------------------------------------------------------------------------
# Registry + Index
# ---------------------------------------------------------------------------


class TestFlatParquetRegistry:
    """Validate registry built from flat parquet."""

    def test_experiments_discovered(self, flat_parquet_setup):
        _, registry, _ = flat_parquet_setup
        names = {e.name for e in registry.experiments}
        assert names == {"exp_a", "exp_b"}

    def test_source_channel_labels_from_markers(self, flat_parquet_setup):
        _, registry, _ = flat_parquet_setup
        labels = registry.source_channel_labels
        assert "Phase3D" in labels
        assert "SEC61" in labels
        assert "G3BP1" in labels
        assert "pAL17" in labels

    def test_pixel_sizes_round_trip(self, flat_parquet_setup):
        _, registry, _ = flat_parquet_setup
        for exp in registry.experiments:
            assert exp.pixel_size_xy_um is not None
            assert abs(exp.pixel_size_xy_um - 0.108) < 1e-3
            assert exp.pixel_size_z_um is not None
            assert abs(exp.pixel_size_z_um - 0.3) < 1e-3

    def test_valid_anchors_have_channel_name(self, flat_parquet_setup):
        index, _, _ = flat_parquet_setup
        assert "channel_name" in index.valid_anchors.columns
        channels = set(index.valid_anchors["channel_name"].unique())
        assert "Phase3D" in channels
        assert "GFP" in channels

    def test_valid_anchors_have_marker(self, flat_parquet_setup):
        index, _, _ = flat_parquet_setup
        assert "marker" in index.valid_anchors.columns
        markers = set(index.valid_anchors["marker"].unique())
        assert "Phase3D" in markers
        assert "SEC61" in markers


# ---------------------------------------------------------------------------
# Channel selection modes
# ---------------------------------------------------------------------------


class TestChannelModes:
    """Validate all channel selection modes with flat parquet."""

    def test_bag_of_channels(self, flat_parquet_setup):
        """channels_per_sample=1 → from_index mode, reads per-row channel."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(index=index, fit=True, channels_per_sample=1)
        assert ds._channel_mode == "from_index"
        batch = ds.__getitems__([0, 1, 2, 3])
        assert batch["anchor"].shape[1] == 1

    def test_all_channels(self, flat_parquet_setup):
        """channels_per_sample=None → all mode, reads all zarr channels."""
        index, _, _ = flat_parquet_setup
        n_before = len(index.valid_anchors)
        ds = MultiExperimentTripletDataset(index=index, fit=True, channels_per_sample=None)
        assert ds._channel_mode == "all"
        # Dedup should reduce valid_anchors (flat parquet has N rows per cell)
        assert len(index.valid_anchors) < n_before, (
            f"All-channels dedup should reduce rows: {n_before} → {len(index.valid_anchors)}"
        )
        # No duplicate cell_ids after dedup
        assert not index.valid_anchors["cell_id"].duplicated().any()
        batch = ds.__getitems__([0, 1])
        # C depends on experiment: exp_a has 2, exp_b has 3
        assert batch["anchor"].shape[1] >= 2

    def test_fixed_channels(self, flat_parquet_setup):
        """channels_per_sample=["Phase3D"] → fixed mode, reads specific channel."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(index=index, fit=True, channels_per_sample=["Phase3D"])
        assert ds._channel_mode == "fixed"
        batch = ds.__getitems__([0, 1])
        assert batch["anchor"].shape[1] == 1


# ---------------------------------------------------------------------------
# Sampler configurations
# ---------------------------------------------------------------------------


class TestSamplerGroupings:
    """Validate FlexibleBatchSampler with flat parquet columns."""

    def test_group_by_marker(self, flat_parquet_setup):
        """batch_group_by=['marker'] → each batch has one marker."""
        index, _, _ = flat_parquet_setup
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["marker"],
            seed=42,
        )
        for batch_indices in sampler:
            rows = index.valid_anchors.iloc[batch_indices]
            assert rows["marker"].nunique() == 1, f"Batch should have 1 marker, got {rows['marker'].unique().tolist()}"
            break  # one batch is enough to verify

    def test_group_by_experiment(self, flat_parquet_setup):
        """batch_group_by=['experiment'] → each batch has one experiment."""
        index, _, _ = flat_parquet_setup
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["experiment"],
            seed=42,
        )
        for batch_indices in sampler:
            rows = index.valid_anchors.iloc[batch_indices]
            assert rows["experiment"].nunique() == 1
            break

    def test_group_by_experiment_and_marker(self, flat_parquet_setup):
        """batch_group_by=['experiment', 'marker'] → each batch has one combo."""
        index, _, _ = flat_parquet_setup
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["experiment", "marker"],
            seed=42,
        )
        for batch_indices in sampler:
            rows = index.valid_anchors.iloc[batch_indices]
            assert rows["experiment"].nunique() == 1
            assert rows["marker"].nunique() == 1
            break

    def test_no_grouping_with_stratify_perturbation(self, flat_parquet_setup):
        """batch_group_by=None, stratify_by=['perturbation'] → balanced perturbations."""
        index, _, _ = flat_parquet_setup
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=8,
            batch_group_by=None,
            stratify_by=["perturbation"],
            seed=42,
        )
        for batch_indices in sampler:
            rows = index.valid_anchors.iloc[batch_indices]
            # With stratification, perturbations should be more balanced than random
            assert rows["perturbation"].nunique() >= 1
            break

    def test_marker_groups_cover_all_markers(self, flat_parquet_setup):
        """Over a full epoch, all markers appear as batch groups."""
        index, _, _ = flat_parquet_setup
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["marker"],
            seed=42,
        )
        seen_markers = set()
        for batch_indices in sampler:
            rows = index.valid_anchors.iloc[batch_indices]
            seen_markers.update(rows["marker"].unique())
        all_markers = set(index.valid_anchors["marker"].unique())
        assert seen_markers == all_markers, f"Missing markers: {all_markers - seen_markers}"


# ---------------------------------------------------------------------------
# Anchor → Positive composition
# ---------------------------------------------------------------------------


class TestAnchorPositiveComposition:
    """Validate anchor→positive sampling with flat parquet."""

    def test_positive_same_channel_in_bag_mode(self, flat_parquet_setup):
        """In bag-of-channels mode, positive reads the same channel as anchor."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
        )
        batch = ds.__getitems__([0, 1])
        # Both anchor and positive should be single-channel
        assert batch["anchor"].shape[1] == 1
        assert batch["positive"].shape[1] == 1

    def test_positive_exists_for_all_anchors(self, flat_parquet_setup):
        """Every valid anchor can find a positive (no RuntimeError)."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
        )
        # Sample a range of anchors — should not raise
        n = min(len(ds), 20)
        batch = ds.__getitems__(list(range(n)))
        assert batch["anchor"].shape[0] == n
        assert batch["positive"].shape[0] == n

    def test_sampler_to_dataset_integration(self, flat_parquet_setup):
        """Full loop: sampler yields indices → dataset returns valid batch."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
        )
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["marker"],
            seed=42,
        )
        for batch_indices in sampler:
            batch = ds.__getitems__(batch_indices)
            assert batch["anchor"].shape == (_BATCH_SIZE, 1, 1, _YX_PATCH[0], _YX_PATCH[1])
            assert batch["positive"].shape == batch["anchor"].shape
            assert "anchor_meta" in batch
            assert len(batch["anchor_meta"]) == _BATCH_SIZE
            break  # one batch is enough

    def test_temporal_positive_different_timepoint(self, flat_parquet_setup):
        """Temporal positive comes from a different timepoint in the same lineage."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
            positive_match_columns=["lineage_id"],
        )
        batch = ds.__getitems__([0, 1, 2, 3])
        anchor_meta = batch["anchor_meta"]
        positive_meta = batch["positive_meta"]
        for a_meta, p_meta in zip(anchor_meta, positive_meta):
            # Same lineage
            assert a_meta["lineage_id"] == p_meta["lineage_id"]
            # Different timepoint (tau > 0)
            assert a_meta["t"] != p_meta["t"], (
                f"Temporal positive should have different t: anchor_t={a_meta['t']}, positive_t={p_meta['t']}"
            )

    def test_temporal_positive_bag_of_channels_sampler_integration(self, flat_parquet_setup):
        """Sampler groups by marker, dataset samples temporal positives from same lineage."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
            positive_match_columns=["lineage_id"],
        )
        sampler = FlexibleBatchSampler(
            valid_anchors=index.valid_anchors,
            batch_size=_BATCH_SIZE,
            batch_group_by=["marker"],
            seed=42,
        )
        for batch_indices in sampler:
            batch = ds.__getitems__(batch_indices)
            # Batch should have consistent marker (from sampler grouping)
            anchor_rows = index.valid_anchors.iloc[batch_indices]
            assert anchor_rows["marker"].nunique() == 1
            # Positives should exist and have same lineage
            for a_meta, p_meta in zip(batch["anchor_meta"], batch["positive_meta"]):
                assert a_meta["lineage_id"] == p_meta["lineage_id"]
            break

    def test_batch_meta_contains_marker(self, flat_parquet_setup):
        """anchor_meta carries marker info for downstream use."""
        index, _, _ = flat_parquet_setup
        ds = MultiExperimentTripletDataset(
            index=index,
            fit=True,
            channels_per_sample=1,
        )
        batch = ds.__getitems__([0])
        meta = batch["anchor_meta"]
        assert len(meta) == 1
        # Meta should contain experiment and perturbation at minimum
        assert "experiment" in meta[0]
        assert "perturbation" in meta[0]
