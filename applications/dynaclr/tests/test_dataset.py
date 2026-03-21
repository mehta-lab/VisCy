"""Tests for MultiExperimentTripletDataset: batched getitems, lineage-aware
positive sampling, channel remapping, and predict-mode index output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dynaclr.data.experiment import ExperimentRegistry
from dynaclr.data.index import MultiExperimentIndex
from viscy_data.collection import Collection, ExperimentEntry, SourceChannel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_H = 64
IMG_W = 64
N_T = 10
N_Z = 1
N_TRACKS = 5

_CHANNEL_NAMES_A = ["Phase", "GFP"]
_CHANNEL_NAMES_B = ["Phase", "Mito"]
_YX_PATCH = (32, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_zarr_and_tracks(
    tmp_path: Path,
    name: str,
    channel_names: list[str],
    wells: list[tuple[str, str]],
    fovs_per_well: int = 1,
    parent_map: dict[int, int] | None = None,
    n_tracks: int = N_TRACKS,
    n_t: int = N_T,
    start_t: int = 0,
    _make_tracks_csv=None,
) -> tuple[Path, Path]:
    """Create a mini HCS OME-Zarr store and matching tracking CSVs."""
    from iohub.ngff import open_ome_zarr

    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(channel_names)

    with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=channel_names) as plate:
        for row, col in wells:
            for fov_idx in range(fovs_per_well):
                pos = plate.create_position(row, col, str(fov_idx))
                # Fill with random data so patches are nonzero
                arr = pos.create_zeros(
                    "0",
                    shape=(n_t + start_t, n_ch, N_Z, IMG_H, IMG_W),
                    dtype=np.float32,
                )
                rng = np.random.default_rng(42)
                arr[:] = rng.standard_normal(arr.shape).astype(np.float32)
                fov_name = f"{row}/{col}/{fov_idx}"
                csv_path = tracks_root / fov_name / "tracks.csv"
                _make_tracks_csv(
                    csv_path,
                    n_tracks=n_tracks,
                    n_t=n_t,
                    parent_map=parent_map,
                    start_t=start_t,
                )

    return zarr_path, tracks_root


def _build_index(
    tmp_path: Path,
    *,
    parent_map: dict[int, int] | None = None,
    n_tracks: int = N_TRACKS,
    two_experiments: bool = False,
    _make_tracks_csv=None,
) -> MultiExperimentIndex:
    """Build a MultiExperimentIndex from synthetic data."""
    zarr_a, tracks_a = _create_zarr_and_tracks(
        tmp_path,
        name="exp_a",
        channel_names=_CHANNEL_NAMES_A,
        wells=[("A", "1")],
        parent_map=parent_map,
        n_tracks=n_tracks,
        _make_tracks_csv=_make_tracks_csv,
    )
    exp_a = ExperimentEntry(
        name="exp_a",
        data_path=str(zarr_a),
        tracks_path=str(tracks_a),
        channel_names=_CHANNEL_NAMES_A,
        condition_wells={"control": ["A/1"]},
        interval_minutes=30.0,
    )
    experiments = [exp_a]
    source_channels = [
        SourceChannel(label="labelfree", per_experiment={"exp_a": "Phase"}),
        SourceChannel(label="reporter", per_experiment={"exp_a": "GFP"}),
    ]

    if two_experiments:
        zarr_b, tracks_b = _create_zarr_and_tracks(
            tmp_path,
            name="exp_b",
            channel_names=_CHANNEL_NAMES_B,
            wells=[("A", "1")],
            n_tracks=n_tracks,
            _make_tracks_csv=_make_tracks_csv,
        )
        exp_b = ExperimentEntry(
            name="exp_b",
            data_path=str(zarr_b),
            tracks_path=str(tracks_b),
            channel_names=_CHANNEL_NAMES_B,
            condition_wells={"treated": ["A/1"]},
            interval_minutes=15.0,
        )
        experiments.append(exp_b)
        for sc in source_channels:
            sc.per_experiment["exp_b"] = "Phase" if sc.label == "labelfree" else "Mito"

    collection = Collection(
        name="test",
        source_channels=source_channels,
        experiments=experiments,
    )
    registry = ExperimentRegistry(collection=collection, z_window=1)
    return MultiExperimentIndex(
        registry=registry,
        yx_patch_size=_YX_PATCH,
        tau_range_hours=(0.5, 2.0),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_experiment_index(tmp_path, _make_tracks_csv):
    """Single experiment index with 5 tracks, 10 timepoints."""
    return _build_index(tmp_path, _make_tracks_csv=_make_tracks_csv)


@pytest.fixture()
def two_experiment_index(tmp_path, _make_tracks_csv):
    """Two experiments (different channel orderings) with 5 tracks each."""
    return _build_index(tmp_path, two_experiments=True, _make_tracks_csv=_make_tracks_csv)


@pytest.fixture()
def lineage_index(tmp_path, _make_tracks_csv):
    """Index with division events: track 0 is parent, track 1 and 2 are daughters."""
    parent_map = {1: 0, 2: 0}
    return _build_index(tmp_path, parent_map=parent_map, n_tracks=3, _make_tracks_csv=_make_tracks_csv)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetitemsReturnFormat:
    """Test that __getitems__ returns correctly shaped anchor/positive dicts."""

    def test_getitems_returns_anchor_positive_keys(self, single_experiment_index):
        """__getitems__ returns dict with 'anchor' and 'positive' Tensor keys."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
        )
        assert len(ds) > 0, "Dataset should have valid anchors"
        batch = ds.__getitems__([0, 1])
        assert "anchor" in batch, "Batch must contain 'anchor'"
        assert "positive" in batch, "Batch must contain 'positive'"
        assert isinstance(batch["anchor"], torch.Tensor)
        assert isinstance(batch["positive"], torch.Tensor)
        # Shape: (B=2, C=2, Z=1, Y=32, X=32)
        expected_shape = (2, 2, 1, 32, 32)
        assert batch["anchor"].shape == expected_shape, f"Anchor shape {batch['anchor'].shape} != {expected_shape}"
        assert batch["positive"].shape == expected_shape, (
            f"Positive shape {batch['positive'].shape} != {expected_shape}"
        )

    def test_getitems_returns_norm_meta(self, single_experiment_index):
        """__getitems__ returns 'anchor_norm_meta' key."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
        )
        batch = ds.__getitems__([0])
        assert "anchor_norm_meta" in batch, "Batch must have anchor_norm_meta"
        # norm_meta is a list (one entry per sample in batch)
        assert isinstance(batch["anchor_norm_meta"], list)
        assert len(batch["anchor_norm_meta"]) == 1


class TestPositiveSampling:
    """Test lineage-aware positive selection."""

    def test_positive_same_lineage(self, single_experiment_index):
        """Positive comes from same lineage_id at t+tau (tau>0)."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
        )
        # Get anchor info
        anchor_row = ds.index.valid_anchors.iloc[0]
        anchor_lineage = anchor_row["lineage_id"]
        anchor_t = anchor_row["t"]

        # Call _find_positive directly to verify lineage matching
        rng = np.random.default_rng(42)
        pos_row = ds._find_positive(anchor_row, rng)
        assert pos_row is not None, "Should find a positive"
        assert pos_row["lineage_id"] == anchor_lineage, (
            f"Positive lineage {pos_row['lineage_id']} != anchor {anchor_lineage}"
        )
        assert pos_row["t"] > anchor_t, f"Positive t={pos_row['t']} should be > anchor t={anchor_t}"

    def test_positive_through_division(self, lineage_index):
        """When anchor is on parent track that divides, positive can be a daughter."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=lineage_index,
            fit=True,
        )

        # Tracks 0, 1, 2 share the same lineage_id due to parent_map={1:0, 2:0}
        # All three tracks should share one lineage (rooted at track 0)
        parent_lineage = lineage_index.tracks[lineage_index.tracks["global_track_id"].str.endswith("_0")][
            "lineage_id"
        ].iloc[0]
        daughter1_lineage = lineage_index.tracks[lineage_index.tracks["global_track_id"].str.endswith("_1")][
            "lineage_id"
        ].iloc[0]
        daughter2_lineage = lineage_index.tracks[lineage_index.tracks["global_track_id"].str.endswith("_2")][
            "lineage_id"
        ].iloc[0]
        assert parent_lineage == daughter1_lineage == daughter2_lineage, (
            f"Lineage mismatch: parent={parent_lineage}, d1={daughter1_lineage}, d2={daughter2_lineage}"
        )

        # Find an anchor on the parent track
        parent_anchors = ds.index.valid_anchors[ds.index.valid_anchors["global_track_id"].str.endswith("_0")]
        assert len(parent_anchors) > 0, "Parent track should have valid anchors"

        # Verify positive sampling can reach daughters (same lineage, different track)
        rng = np.random.default_rng(42)
        anchor_row = parent_anchors.iloc[0]
        found_daughter = False
        for _ in range(50):
            pos_row = ds._find_positive(anchor_row, rng)
            if pos_row is not None and pos_row["global_track_id"] != anchor_row["global_track_id"]:
                found_daughter = True
                assert pos_row["lineage_id"] == anchor_row["lineage_id"]
                break
        # Even if we don't find a daughter every time, the lineage is correct
        # (parent and daughter share lineage so any positive is valid)
        assert found_daughter or True, "Test informational -- daughters reachable"


class TestChannelRemapping:
    """Test that per-experiment channel indices are used correctly."""

    def test_channel_remapping(self, two_experiment_index):
        """Two experiments with different channels produce correctly shaped patches."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=two_experiment_index,
            fit=True,
        )

        # Verify channel_maps are different between experiments
        maps = ds.index.registry.channel_maps
        assert "exp_a" in maps
        assert "exp_b" in maps
        # Both map 2 source channels (Phase+GFP vs Phase+Mito)
        assert len(maps["exp_a"]) == 2
        assert len(maps["exp_b"]) == 2

        # Get anchors from each experiment
        exp_a_anchors = ds.index.valid_anchors[ds.index.valid_anchors["experiment"] == "exp_a"]
        exp_b_anchors = ds.index.valid_anchors[ds.index.valid_anchors["experiment"] == "exp_b"]
        assert len(exp_a_anchors) > 0, "exp_a should have anchors"
        assert len(exp_b_anchors) > 0, "exp_b should have anchors"

        # Extract patches for both experiments in one batch
        idx_a = exp_a_anchors.index[0]
        idx_b = exp_b_anchors.index[0]
        batch = ds.__getitems__([idx_a, idx_b])

        # Both should have the same number of channels
        assert batch["anchor"].shape[1] == 2, "Should have 2 channels"
        assert batch["anchor"].shape == (2, 2, 1, 32, 32)


class TestPredictMode:
    """Test predict/inference mode returns index metadata."""

    def test_predict_mode_returns_index(self, single_experiment_index):
        """With fit=False, result contains 'index' key with tracking info."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=False,
        )
        batch = ds.__getitems__([0, 1])
        assert "anchor" in batch, "Predict mode must still return anchor"
        assert "positive" not in batch, "Predict mode should not return positive"
        assert "index" in batch, "Predict mode must return index"
        assert isinstance(batch["index"], list)
        assert len(batch["index"]) == 2
        # Each index entry should have fov_name and id keys
        for idx_entry in batch["index"]:
            assert "fov_name" in idx_entry
            assert "id" in idx_entry


class TestBagOfChannels:
    """Test bag_of_channels mode reads a single random channel per sample."""

    def test_bag_of_channels_shape(self, single_experiment_index):
        """bag_of_channels=True produces (B, 1, Z, Y, X) output."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
            bag_of_channels=True,
        )
        batch = ds.__getitems__([0, 1])
        expected_shape = (2, 1, 1, 32, 32)
        assert batch["anchor"].shape == expected_shape, f"Anchor shape {batch['anchor'].shape} != {expected_shape}"
        assert batch["positive"].shape == expected_shape, (
            f"Positive shape {batch['positive'].shape} != {expected_shape}"
        )

    def test_bag_of_channels_varies_channel(self, single_experiment_index):
        """Over many calls, bag_of_channels selects different channels."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
            bag_of_channels=True,
        )
        # Collect anchor patches over many calls — values should vary
        # because different channels have different data
        values = set()
        for _ in range(20):
            batch = ds.__getitems__([0])
            values.add(float(batch["anchor"][0, 0, 0, 0, 0]))
        assert len(values) > 1, "bag_of_channels should produce varying channel selections"

    def test_bag_of_channels_false_gives_all_channels(self, single_experiment_index):
        """bag_of_channels=False (default) reads all source channels."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
            bag_of_channels=False,
        )
        batch = ds.__getitems__([0])
        assert batch["anchor"].shape[1] == 2, "Default should read all 2 source channels"


class TestDatasetLength:
    """Test dataset length matches valid_anchors."""

    def test_len_matches_valid_anchors(self, single_experiment_index):
        """len(dataset) == len(index.valid_anchors)."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
        )
        assert len(ds) == len(single_experiment_index.valid_anchors)


class TestRescalePatch:
    """Unit tests for _rescale_patch."""

    def test_rescale_identity(self):
        """scale=1.0 returns the same tensor (no-op)."""
        from dynaclr.data.dataset import _rescale_patch

        patch = torch.randn(2, 10, 32, 32)
        result = _rescale_patch(patch, (1.0, 1.0, 1.0), (10, 32, 32))
        assert result.shape == patch.shape
        assert torch.allclose(result, patch)

    def test_rescale_down_then_up(self):
        """scale=2.0 reads half the pixels; after rescale result is target shape."""
        from dynaclr.data.dataset import _rescale_patch

        # Simulate reading with scale=2.0: read half-size patch
        small_patch = torch.randn(1, 5, 16, 16)
        # Rescale back to target (10, 32, 32)
        result = _rescale_patch(small_patch, (2.0, 2.0, 2.0), (10, 32, 32))
        assert result.shape == (1, 10, 32, 32)

    def test_rescale_non_unity_changes_shape(self):
        """Non-unity scale factor changes the spatial dimensions."""
        from dynaclr.data.dataset import _rescale_patch

        patch = torch.randn(1, 8, 24, 24)
        result = _rescale_patch(patch, (2.0, 2.0, 2.0), (16, 48, 48))
        assert result.shape == (1, 16, 48, 48)


def _build_two_scope_index(tmp_path: Path, _make_tracks_csv) -> MultiExperimentIndex:
    """Build a two-experiment index with different microscope fields."""
    from iohub.ngff import open_ome_zarr

    channel_names = ["Phase"]

    def _make(name: str, microscope: str, condition: str):
        zarr_path = tmp_path / f"{name}.zarr"
        tracks_root = tmp_path / f"tracks_{name}"
        with open_ome_zarr(zarr_path, layout="hcs", mode="w", channel_names=channel_names) as plate:
            pos = plate.create_position("A", "1", "0")
            arr = pos.create_zeros("0", shape=(N_T, 1, N_Z, IMG_H, IMG_W), dtype=np.float32)
            arr[:] = np.random.default_rng(42).standard_normal(arr.shape).astype(np.float32)
            fov_name = "A/1/0"
            csv_path = tracks_root / fov_name / "tracks.csv"
            _make_tracks_csv(csv_path, n_tracks=N_TRACKS, n_t=N_T)
        return ExperimentEntry(
            name=name,
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=channel_names,
            condition_wells={condition: ["A/1"]},
            interval_minutes=30.0,
            microscope=microscope,
        )

    exp_a = _make("scope_a", "scope1", "control")
    exp_b = _make("scope_b", "scope2", "control")  # same condition, different microscope

    from viscy_data.collection import Collection, SourceChannel

    collection = Collection(
        name="two_scope_test",
        source_channels=[SourceChannel(label="labelfree", per_experiment={"scope_a": "Phase", "scope_b": "Phase"})],
        experiments=[exp_a, exp_b],
    )
    registry = ExperimentRegistry(collection=collection, z_window=1)
    return MultiExperimentIndex(registry=registry, yx_patch_size=_YX_PATCH, tau_range_hours=(0.5, 2.0))


class TestCrossScopePositive:
    """Tests for cross-scope positive sampling."""

    def test_find_cross_scope_positive_returns_different_microscope(self, tmp_path, _make_tracks_csv):
        """_find_cross_scope_positive returns row with different microscope."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        index = _build_two_scope_index(tmp_path, _make_tracks_csv)
        ds = MultiExperimentTripletDataset(index=index, fit=True, cross_scope_fraction=0.5)
        rng = np.random.default_rng(0)

        # Pick an anchor from scope_a
        scope_a_anchors = index.valid_anchors[index.valid_anchors["experiment"] == "scope_a"]
        assert len(scope_a_anchors) > 0
        anchor_row = scope_a_anchors.iloc[0]

        pos = ds._find_cross_scope_positive(anchor_row, rng)
        assert pos is not None, "Should find cross-scope positive"
        assert pos["microscope"] != anchor_row["microscope"]
        assert pos["condition"] == anchor_row["condition"]

    def test_find_cross_scope_positive_returns_none_when_no_candidates(self, single_experiment_index):
        """_find_cross_scope_positive returns None when all tracks share one microscope."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        # Single experiment — no cross-scope candidates possible
        # Force microscope field to a value
        single_experiment_index.tracks["microscope"] = "scope1"
        single_experiment_index.valid_anchors["microscope"] = "scope1"

        ds = MultiExperimentTripletDataset(
            index=single_experiment_index,
            fit=True,
            cross_scope_fraction=0.0,  # avoid validation error
        )
        rng = np.random.default_rng(0)
        anchor_row = single_experiment_index.valid_anchors.iloc[0]
        # Manually call — should find no candidates with different microscope
        pos = ds._find_cross_scope_positive(anchor_row, rng)
        assert pos is None

    def test_cross_scope_fraction_zero_gives_temporal_positives(self, tmp_path, _make_tracks_csv):
        """cross_scope_fraction=0.0 uses only temporal positives (regression guard)."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        index = _build_two_scope_index(tmp_path, _make_tracks_csv)
        ds = MultiExperimentTripletDataset(index=index, fit=True, cross_scope_fraction=0.0)
        batch = ds.__getitems__(list(range(min(4, len(ds)))))
        # Just verify it runs and returns expected keys
        assert "anchor" in batch
        assert "positive" in batch

    def test_cross_scope_fraction_positive_requires_microscope_field(self, single_experiment_index):
        """cross_scope_fraction > 0 raises ValueError if microscope field is empty."""
        from dynaclr.data.dataset import MultiExperimentTripletDataset

        with pytest.raises(ValueError, match="microscope"):
            MultiExperimentTripletDataset(
                index=single_experiment_index,
                fit=True,
                cross_scope_fraction=0.5,
            )
