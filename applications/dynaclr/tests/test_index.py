"""Tests for MultiExperimentIndex: tracks, lineage, border clamping, valid anchors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from iohub.ngff import open_ome_zarr

from dynaclr.experiment import ExperimentConfig, ExperimentRegistry
from dynaclr.index import MultiExperimentIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHANNEL_NAMES_A = ["Phase", "GFP"]
_CHANNEL_NAMES_B = ["Phase", "Mito"]

_IMG_H = 64
_IMG_W = 64
_N_T = 10
_N_Z = 1
_N_TRACKS = 5
_YX_PATCH = (32, 32)


def _make_tracks_csv(
    path: Path,
    n_tracks: int = _N_TRACKS,
    n_t: int = _N_T,
    *,
    parent_map: dict[int, int] | None = None,
    border_cell_track: int | None = None,
    outside_cell_track: int | None = None,
) -> None:
    """Write a tracking CSV with standard columns.

    Parameters
    ----------
    path : Path
        Where to write the CSV file.
    n_tracks : int
        Number of tracks.
    n_t : int
        Number of timepoints per track.
    parent_map : dict[int, int] | None
        Mapping child_track_id -> parent_track_id for lineage testing.
    border_cell_track : int | None
        Track ID to place near the border (y=2, x=2).
    outside_cell_track : int | None
        Track ID to place outside the image boundary (y=-1).
    """
    rows = []
    for tid in range(n_tracks):
        for t in range(n_t):
            y = 32.0  # center by default
            x = 32.0
            if border_cell_track is not None and tid == border_cell_track:
                y = 2.0
                x = 2.0
            if outside_cell_track is not None and tid == outside_cell_track:
                y = -1.0
                x = -1.0
            ptid = float("nan")
            if parent_map and tid in parent_map:
                ptid = parent_map[tid]
            rows.append(
                {
                    "track_id": tid,
                    "t": t,
                    "id": tid * n_t + t,
                    "parent_track_id": ptid,
                    "parent_id": float("nan"),
                    "z": 0,
                    "y": y,
                    "x": x,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _create_zarr_and_tracks(
    tmp_path: Path,
    name: str,
    channel_names: list[str],
    wells: list[tuple[str, str]],
    fovs_per_well: int = 2,
    parent_map: dict[int, int] | None = None,
    border_cell_track: int | None = None,
    outside_cell_track: int | None = None,
) -> tuple[Path, Path]:
    """Create a mini HCS OME-Zarr store and matching tracking CSVs.

    Returns (zarr_path, tracks_root_path).
    """
    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(channel_names)

    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=channel_names
    ) as plate:
        for row, col in wells:
            for fov_idx in range(fovs_per_well):
                pos = plate.create_position(row, col, str(fov_idx))
                pos.create_zeros(
                    "0",
                    shape=(_N_T, n_ch, _N_Z, _IMG_H, _IMG_W),
                    dtype=np.float32,
                )
                fov_name = f"{row}/{col}/{fov_idx}"
                csv_path = tracks_root / fov_name / "tracks.csv"
                _make_tracks_csv(
                    csv_path,
                    parent_map=parent_map,
                    border_cell_track=border_cell_track,
                    outside_cell_track=outside_cell_track,
                )

    return zarr_path, tracks_root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_experiment_setup(tmp_path):
    """Create 2 experiments, 2 wells each, 2 FOVs each, with tracking CSVs."""
    zarr_a, tracks_a = _create_zarr_and_tracks(
        tmp_path,
        name="exp_a",
        channel_names=_CHANNEL_NAMES_A,
        wells=[("A", "1"), ("B", "1")],
        fovs_per_well=2,
    )
    zarr_b, tracks_b = _create_zarr_and_tracks(
        tmp_path,
        name="exp_b",
        channel_names=_CHANNEL_NAMES_B,
        wells=[("A", "1"), ("B", "1")],
        fovs_per_well=2,
    )

    cfg_a = ExperimentConfig(
        name="exp_a",
        data_path=str(zarr_a),
        tracks_path=str(tracks_a),
        channel_names=_CHANNEL_NAMES_A,
        source_channel=["Phase", "GFP"],
        condition_wells={"uninfected": ["A/1"], "infected": ["B/1"]},
        interval_minutes=30.0,
        start_hpi=0.0,
    )
    cfg_b = ExperimentConfig(
        name="exp_b",
        data_path=str(zarr_b),
        tracks_path=str(tracks_b),
        channel_names=_CHANNEL_NAMES_B,
        source_channel=["Phase", "Mito"],
        condition_wells={"control": ["A/1"], "treated": ["B/1"]},
        interval_minutes=15.0,
        start_hpi=2.0,
    )

    registry = ExperimentRegistry(experiments=[cfg_a, cfg_b])
    return registry, cfg_a, cfg_b


@pytest.fixture()
def lineage_setup(tmp_path):
    """Create an experiment with lineage (parent_track_id) relationships.

    Track lineage: track 0 (root) -> track 1 (daughter) -> track 2 (granddaughter)
    Track 3: has parent_track_id=99 (not in data, should fallback)
    Track 4: no parent (independent root)
    """
    parent_map = {1: 0, 2: 1, 3: 99}
    zarr_path, tracks_root = _create_zarr_and_tracks(
        tmp_path,
        name="lineage_exp",
        channel_names=_CHANNEL_NAMES_A,
        wells=[("A", "1")],
        fovs_per_well=1,
        parent_map=parent_map,
    )

    cfg = ExperimentConfig(
        name="lineage_exp",
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channel_names=_CHANNEL_NAMES_A,
        source_channel=["Phase", "GFP"],
        condition_wells={"ctrl": ["A/1"]},
        interval_minutes=30.0,
    )

    registry = ExperimentRegistry(experiments=[cfg])
    return registry


@pytest.fixture()
def border_setup(tmp_path):
    """Create an experiment with border cells and one outside-image cell.

    Track 3: near border (y=2, x=2)
    Track 4: outside image (y=-1)
    """
    zarr_path, tracks_root = _create_zarr_and_tracks(
        tmp_path,
        name="border_exp",
        channel_names=_CHANNEL_NAMES_A,
        wells=[("A", "1")],
        fovs_per_well=1,
        border_cell_track=3,
        outside_cell_track=4,
    )

    cfg = ExperimentConfig(
        name="border_exp",
        data_path=str(zarr_path),
        tracks_path=str(tracks_root),
        channel_names=_CHANNEL_NAMES_A,
        source_channel=["Phase", "GFP"],
        condition_wells={"ctrl": ["A/1"]},
        interval_minutes=30.0,
    )

    registry = ExperimentRegistry(experiments=[cfg])
    return registry


# ---------------------------------------------------------------------------
# CELL-01: Unified tracks DataFrame
# ---------------------------------------------------------------------------


class TestUnifiedTracksDataFrame:
    """Tests for MultiExperimentIndex track building across experiments."""

    def test_all_observations_present(self, two_experiment_setup):
        """2 experiments x 2 wells x 2 FOVs x 5 tracks x 10 timepoints = 400 rows."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # 2 experiments * 2 wells * 2 FOVs * 5 tracks * 10 timepoints = 400
        assert len(index.tracks) == 400

    def test_experiment_column(self, two_experiment_setup):
        """'experiment' column matches exp.name for each row."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        assert set(index.tracks["experiment"].unique()) == {"exp_a", "exp_b"}
        # Each experiment contributes half the rows
        exp_a_rows = index.tracks[index.tracks["experiment"] == "exp_a"]
        exp_b_rows = index.tracks[index.tracks["experiment"] == "exp_b"]
        assert len(exp_a_rows) == 200
        assert len(exp_b_rows) == 200

    def test_condition_column(self, two_experiment_setup):
        """'condition' column correctly maps wells to conditions."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # exp_a: A/1 -> uninfected, B/1 -> infected
        exp_a_well_a = index.tracks[
            (index.tracks["experiment"] == "exp_a")
            & (index.tracks["well_name"] == "A/1")
        ]
        assert (exp_a_well_a["condition"] == "uninfected").all()

        exp_a_well_b = index.tracks[
            (index.tracks["experiment"] == "exp_a")
            & (index.tracks["well_name"] == "B/1")
        ]
        assert (exp_a_well_b["condition"] == "infected").all()

    def test_global_track_id_format(self, two_experiment_setup):
        """global_track_id is '{exp_name}_{fov_name}_{track_id}'."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        sample = index.tracks.iloc[0]
        expected_prefix = f"{sample['experiment']}_{sample['fov_name']}_{sample['track_id']}"
        assert sample["global_track_id"] == expected_prefix

    def test_global_track_id_unique_across_experiments(self, two_experiment_setup):
        """global_track_ids are unique across experiments."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Each track_id+fov combination appears in both experiments
        # but global_track_id should be unique due to experiment prefix
        # 2 exp * 2 wells * 2 FOVs * 5 tracks = 40 unique global_track_ids
        assert index.tracks["global_track_id"].nunique() == 40

    def test_hours_post_infection(self, two_experiment_setup):
        """hours_post_infection = start_hpi + t * interval_minutes / 60."""
        registry, cfg_a, cfg_b = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Check exp_a (start_hpi=0.0, interval=30min)
        row_a = index.tracks[
            (index.tracks["experiment"] == "exp_a") & (index.tracks["t"] == 3)
        ].iloc[0]
        expected_a = 0.0 + 3 * 30.0 / 60.0  # = 1.5
        assert row_a["hours_post_infection"] == pytest.approx(expected_a)

        # Check exp_b (start_hpi=2.0, interval=15min)
        row_b = index.tracks[
            (index.tracks["experiment"] == "exp_b") & (index.tracks["t"] == 4)
        ].iloc[0]
        expected_b = 2.0 + 4 * 15.0 / 60.0  # = 3.0
        assert row_b["hours_post_infection"] == pytest.approx(expected_b)

    def test_fluorescence_channel(self, two_experiment_setup):
        """fluorescence_channel is source_channel[1] when len > 1."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        exp_a_rows = index.tracks[index.tracks["experiment"] == "exp_a"]
        assert (exp_a_rows["fluorescence_channel"] == "GFP").all()

        exp_b_rows = index.tracks[index.tracks["experiment"] == "exp_b"]
        assert (exp_b_rows["fluorescence_channel"] == "Mito").all()

    def test_required_columns_present(self, two_experiment_setup):
        """All required columns exist in the final DataFrame."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        required = {
            "track_id",
            "t",
            "y",
            "x",
            "z",
            "position",
            "fov_name",
            "well_name",
            "experiment",
            "condition",
            "global_track_id",
            "hours_post_infection",
            "fluorescence_channel",
            "lineage_id",
            "y_clamp",
            "x_clamp",
        }
        assert required.issubset(set(index.tracks.columns))

    def test_include_wells_filter(self, two_experiment_setup):
        """include_wells filters to only specified wells."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            include_wells=["A/1"],
        )
        assert set(index.tracks["well_name"].unique()) == {"A/1"}
        # Only A/1 wells: 2 experiments * 1 well * 2 FOVs * 5 tracks * 10 t = 200
        assert len(index.tracks) == 200

    def test_exclude_fovs_filter(self, two_experiment_setup):
        """exclude_fovs removes specified FOVs."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            exclude_fovs=["A/1/0"],
        )
        assert "A/1/0" not in index.tracks["fov_name"].to_numpy()
        # Removed 1 FOV from each experiment: 2 * (4 - 1) * 5 * 10 = 300
        assert len(index.tracks) == 300

    def test_positions_stored(self, two_experiment_setup):
        """Position objects are stored in self.positions."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # 2 experiments * 2 wells * 2 FOVs = 8 positions
        assert len(index.positions) == 8

    def test_position_column_is_position_object(self, two_experiment_setup):
        """'position' column contains iohub Position objects."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        from iohub.ngff import Position

        sample_pos = index.tracks.iloc[0]["position"]
        assert isinstance(sample_pos, Position)


# ---------------------------------------------------------------------------
# CELL-02: Lineage reconstruction
# ---------------------------------------------------------------------------


class TestLineageReconstruction:
    """Tests for lineage_id reconstruction from parent_track_id."""

    def test_root_track_lineage(self, lineage_setup):
        """Track without parent -> lineage_id = own global_track_id."""
        index = MultiExperimentIndex(
            registry=lineage_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Track 0 is root (no parent)
        track0 = index.tracks[index.tracks["track_id"] == 0].iloc[0]
        assert track0["lineage_id"] == track0["global_track_id"]

    def test_daughter_track_lineage(self, lineage_setup):
        """Track with parent -> lineage_id = parent's lineage_id."""
        index = MultiExperimentIndex(
            registry=lineage_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Track 1 is daughter of track 0
        track0 = index.tracks[index.tracks["track_id"] == 0].iloc[0]
        track1 = index.tracks[index.tracks["track_id"] == 1].iloc[0]
        assert track1["lineage_id"] == track0["global_track_id"]

    def test_granddaughter_lineage_chain(self, lineage_setup):
        """Chain: track 0 -> track 1 -> track 2, all share track 0's lineage_id."""
        index = MultiExperimentIndex(
            registry=lineage_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        track0 = index.tracks[index.tracks["track_id"] == 0].iloc[0]
        track2 = index.tracks[index.tracks["track_id"] == 2].iloc[0]
        # Granddaughter should have root's lineage_id
        assert track2["lineage_id"] == track0["global_track_id"]

    def test_missing_parent_fallback(self, lineage_setup):
        """parent_track_id=99 (not in data) -> lineage_id = own global_track_id."""
        index = MultiExperimentIndex(
            registry=lineage_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        track3 = index.tracks[index.tracks["track_id"] == 3].iloc[0]
        assert track3["lineage_id"] == track3["global_track_id"]

    def test_independent_track_lineage(self, lineage_setup):
        """Track 4: no parent -> lineage_id = own global_track_id."""
        index = MultiExperimentIndex(
            registry=lineage_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        track4 = index.tracks[index.tracks["track_id"] == 4].iloc[0]
        assert track4["lineage_id"] == track4["global_track_id"]


# ---------------------------------------------------------------------------
# CELL-03: Border clamping
# ---------------------------------------------------------------------------


class TestBorderClamping:
    """Tests for border clamping of cell centroids."""

    def test_center_cell_unchanged(self, border_setup):
        """Cell at center (y=32, x=32) in 64x64 with 32x32 patch -> unchanged."""
        index = MultiExperimentIndex(
            registry=border_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Track 0 is at center (y=32, x=32)
        center_cell = index.tracks[index.tracks["track_id"] == 0].iloc[0]
        assert center_cell["y_clamp"] == 32.0
        assert center_cell["x_clamp"] == 32.0

    def test_border_cell_clamped(self, border_setup):
        """Cell near border (y=2, x=2) -> clamped to (16, 16) for 32x32 patch."""
        index = MultiExperimentIndex(
            registry=border_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Track 3 is at y=2, x=2 (border)
        border_cell = index.tracks[index.tracks["track_id"] == 3].iloc[0]
        # y_half = 16, x_half = 16 -> clamp to (16, 16)
        assert border_cell["y_clamp"] == 16.0
        assert border_cell["x_clamp"] == 16.0

    def test_border_cell_original_preserved(self, border_setup):
        """Original y, x are preserved even after clamping."""
        index = MultiExperimentIndex(
            registry=border_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        border_cell = index.tracks[index.tracks["track_id"] == 3].iloc[0]
        assert border_cell["y"] == 2.0
        assert border_cell["x"] == 2.0

    def test_outside_cell_excluded(self, border_setup):
        """Cell completely outside image (y=-1) is excluded."""
        index = MultiExperimentIndex(
            registry=border_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        # Track 4 had y=-1 -> should be excluded
        track4_rows = index.tracks[index.tracks["track_id"] == 4]
        assert len(track4_rows) == 0

    def test_border_cells_retained_count(self, border_setup):
        """Border cells are retained (not excluded like old approach).

        5 tracks total, 1 outside (track 4) -> 4 tracks remain.
        4 tracks * 10 timepoints = 40 rows.
        """
        index = MultiExperimentIndex(
            registry=border_setup, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )
        assert len(index.tracks) == 40

    def test_edge_cell_clamped(self, tmp_path):
        """Cell at exact edge (y=0, x=0) -> clamped to (y_half, x_half)."""
        # Create a special setup with cell at y=0, x=0
        zarr_path = tmp_path / "edge.zarr"
        tracks_root = tmp_path / "tracks_edge"

        with open_ome_zarr(
            zarr_path, layout="hcs", mode="w", channel_names=_CHANNEL_NAMES_A
        ) as plate:
            pos = plate.create_position("A", "1", "0")
            pos.create_zeros(
                "0", shape=(1, 2, 1, _IMG_H, _IMG_W), dtype=np.float32
            )

        # Create CSV with cell at exact edge
        csv_path = tracks_root / "A" / "1" / "0" / "tracks.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "track_id": 0,
                    "t": 0,
                    "id": 0,
                    "parent_track_id": float("nan"),
                    "parent_id": float("nan"),
                    "z": 0,
                    "y": 0.0,
                    "x": 0.0,
                }
            ]
        )
        df.to_csv(csv_path, index=False)

        cfg = ExperimentConfig(
            name="edge_exp",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry, z_range=slice(0, 1), yx_patch_size=_YX_PATCH
        )

        cell = index.tracks.iloc[0]
        assert cell["y_clamp"] == 16.0  # y_half
        assert cell["x_clamp"] == 16.0  # x_half


# ---------------------------------------------------------------------------
# Helpers for anchor / property tests
# ---------------------------------------------------------------------------


def _make_tracks_csv_custom(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a tracking CSV from explicit row dicts.

    Each dict should contain at least: track_id, t, y, x, z.
    Missing columns (id, parent_track_id, parent_id) get defaults.
    """
    for r in rows:
        r.setdefault("id", r["track_id"] * 1000 + r["t"])
        r.setdefault("parent_track_id", float("nan"))
        r.setdefault("parent_id", float("nan"))
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _create_zarr_and_custom_tracks(
    tmp_path: Path,
    name: str,
    channel_names: list[str],
    well: tuple[str, str],
    track_rows: list[dict],
    n_t: int = _N_T,
) -> tuple[Path, Path]:
    """Create a mini HCS OME-Zarr with one FOV and custom tracking rows."""
    zarr_path = tmp_path / f"{name}.zarr"
    tracks_root = tmp_path / f"tracks_{name}"
    n_ch = len(channel_names)

    with open_ome_zarr(
        zarr_path, layout="hcs", mode="w", channel_names=channel_names
    ) as plate:
        pos = plate.create_position(well[0], well[1], "0")
        pos.create_zeros(
            "0",
            shape=(n_t, n_ch, _N_Z, _IMG_H, _IMG_W),
            dtype=np.float32,
        )

    fov_name = f"{well[0]}/{well[1]}/0"
    csv_path = tracks_root / fov_name / "tracks.csv"
    _make_tracks_csv_custom(csv_path, track_rows)

    return zarr_path, tracks_root


# ---------------------------------------------------------------------------
# CELL-04: Valid anchors with variable tau range and lineage continuity
# ---------------------------------------------------------------------------


class TestValidAnchors:
    """Tests for valid_anchors computation."""

    def test_basic_anchor_validity(self, two_experiment_setup):
        """Tracks with enough future timepoints are valid anchors.

        With tau_range_hours=(0.5, 1.5) and exp_a interval=30min:
        tau_range_frames = (1, 3).
        Track at t=0 with observations at t=1,2,...,9 -> valid (t+1 exists).
        """
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        assert hasattr(index, "valid_anchors")
        # valid_anchors is a subset of tracks
        assert len(index.valid_anchors) <= len(index.tracks)
        assert len(index.valid_anchors) > 0

    def test_anchor_is_subset_of_tracks(self, two_experiment_setup):
        """valid_anchors rows are all present in tracks."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        # Every global_track_id+t pair in valid_anchors should exist in tracks
        anchor_keys = set(
            zip(
                index.valid_anchors["global_track_id"],
                index.valid_anchors["t"],
            )
        )
        track_keys = set(
            zip(index.tracks["global_track_id"], index.tracks["t"])
        )
        assert anchor_keys.issubset(track_keys)

    def test_end_of_track_not_valid(self, tmp_path):
        """Observations near the end of a track with no future positives are excluded.

        Single track with t=0..9, tau_range_frames=(1,3).
        t=9: needs t=10,11,12 -> none exist -> NOT valid.
        t=8: needs t=9 -> exists -> valid.
        t=7: needs t=8 -> exists -> valid.
        """
        track_rows = [
            {"track_id": 0, "t": t, "z": 0, "y": 32.0, "x": 32.0}
            for t in range(10)
        ]
        zarr_path, tracks_root = _create_zarr_and_custom_tracks(
            tmp_path,
            name="end_test",
            channel_names=_CHANNEL_NAMES_A,
            well=("A", "1"),
            track_rows=track_rows,
        )
        cfg = ExperimentConfig(
            name="end_test",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        # tau_range_frames at 30min = (1, 3)
        # t=9 should NOT be valid (no t=10,11,12)
        anchors_t = set(index.valid_anchors["t"].to_numpy())
        assert 9 not in anchors_t
        # t=7 and t=8 should be valid
        assert 7 in anchors_t
        assert 8 in anchors_t

    def test_lineage_continuity_across_tracks(self, tmp_path):
        """Parent anchor is valid because daughter track provides future positive.

        Parent track (tid=0): t=0..4
        Daughter track (tid=1, parent=0): t=5..9
        They share lineage_id. So parent at t=3 with tau=2 -> t=5 is
        in daughter track (same lineage) -> valid.
        """
        track_rows = []
        # Parent track: t=0..4
        for t in range(5):
            track_rows.append(
                {"track_id": 0, "t": t, "z": 0, "y": 32.0, "x": 32.0}
            )
        # Daughter track: t=5..9, parent=0
        for t in range(5, 10):
            track_rows.append(
                {
                    "track_id": 1,
                    "t": t,
                    "z": 0,
                    "y": 32.0,
                    "x": 32.0,
                    "parent_track_id": 0,
                }
            )
        zarr_path, tracks_root = _create_zarr_and_custom_tracks(
            tmp_path,
            name="lineage_anchor",
            channel_names=_CHANNEL_NAMES_A,
            well=("A", "1"),
            track_rows=track_rows,
        )
        cfg = ExperimentConfig(
            name="lineage_anchor",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        # tau_range_frames = (1, 3) at 30min
        # Parent t=3: check t=4 (parent has it, same lineage) -> valid
        # Parent t=4: check t=5 (daughter has it, same lineage) -> valid
        parent_anchors = index.valid_anchors[
            index.valid_anchors["track_id"] == 0
        ]
        parent_anchor_times = set(parent_anchors["t"].to_numpy())
        assert 3 in parent_anchor_times
        assert 4 in parent_anchor_times

    def test_different_tau_for_different_intervals(self, tmp_path):
        """Different experiments with different intervals yield different tau_range_frames.

        exp_fast: interval=15min, tau_range_hours=(0.5,1.5) -> tau_range_frames=(2,6)
        exp_slow: interval=30min, tau_range_hours=(0.5,1.5) -> tau_range_frames=(1,3)
        """
        # Fast experiment: 15min interval
        fast_rows = [
            {"track_id": 0, "t": t, "z": 0, "y": 32.0, "x": 32.0}
            for t in range(10)
        ]
        zarr_fast, tracks_fast = _create_zarr_and_custom_tracks(
            tmp_path,
            name="fast_exp",
            channel_names=_CHANNEL_NAMES_A,
            well=("A", "1"),
            track_rows=fast_rows,
        )
        cfg_fast = ExperimentConfig(
            name="fast_exp",
            data_path=str(zarr_fast),
            tracks_path=str(tracks_fast),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=15.0,
        )

        # Slow experiment: 30min interval
        slow_rows = [
            {"track_id": 0, "t": t, "z": 0, "y": 32.0, "x": 32.0}
            for t in range(10)
        ]
        zarr_slow, tracks_slow = _create_zarr_and_custom_tracks(
            tmp_path,
            name="slow_exp",
            channel_names=_CHANNEL_NAMES_B,
            well=("A", "1"),
            track_rows=slow_rows,
        )
        cfg_slow = ExperimentConfig(
            name="slow_exp",
            data_path=str(zarr_slow),
            tracks_path=str(tracks_slow),
            channel_names=_CHANNEL_NAMES_B,
            source_channel=["Phase", "Mito"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )

        registry = ExperimentRegistry(experiments=[cfg_fast, cfg_slow])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )

        # fast_exp: tau_range_frames=(2,6), so t=9 needs t=11..15 -> invalid
        # t=4 needs t=6..10 -> t=6,7,8,9 exist -> valid
        fast_anchors = index.valid_anchors[
            index.valid_anchors["experiment"] == "fast_exp"
        ]
        fast_anchor_times = set(fast_anchors["t"].to_numpy())
        assert 9 not in fast_anchor_times  # no future at tau=2..6
        assert 4 in fast_anchor_times  # t=6 exists

        # slow_exp: tau_range_frames=(1,3), so t=9 needs t=10..12 -> invalid
        # t=7 needs t=8..10 -> t=8 exists -> valid
        slow_anchors = index.valid_anchors[
            index.valid_anchors["experiment"] == "slow_exp"
        ]
        slow_anchor_times = set(slow_anchors["t"].to_numpy())
        assert 9 not in slow_anchor_times
        assert 7 in slow_anchor_times

    def test_empty_tracks_empty_anchors(self, tmp_path):
        """When tracks is empty, valid_anchors is also empty."""
        zarr_path = tmp_path / "empty.zarr"
        tracks_root = tmp_path / "tracks_empty"

        with open_ome_zarr(
            zarr_path, layout="hcs", mode="w", channel_names=_CHANNEL_NAMES_A
        ) as plate:
            pos = plate.create_position("A", "1", "0")
            pos.create_zeros(
                "0", shape=(1, 2, 1, _IMG_H, _IMG_W), dtype=np.float32
            )

        # No CSV -> no tracks loaded -> skip with warning
        cfg = ExperimentConfig(
            name="empty_exp",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        assert len(index.valid_anchors) == 0

    def test_track_with_gap_still_valid(self, tmp_path):
        """Track with missing timepoint -> anchor at t=2 still valid if t=4 exists.

        Track: t=0,1,2,4,5 (missing t=3).
        tau_range_frames=(1,3): t=2 checks t=3(missing),t=4(exists!),t=5 -> valid.
        """
        track_rows = [
            {"track_id": 0, "t": t, "z": 0, "y": 32.0, "x": 32.0}
            for t in [0, 1, 2, 4, 5]
        ]
        zarr_path, tracks_root = _create_zarr_and_custom_tracks(
            tmp_path,
            name="gap_test",
            channel_names=_CHANNEL_NAMES_A,
            well=("A", "1"),
            track_rows=track_rows,
            n_t=6,
        )
        cfg = ExperimentConfig(
            name="gap_test",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        anchor_times = set(index.valid_anchors["t"].to_numpy())
        # t=2: check t=3(missing), t=4(exists!) -> valid
        assert 2 in anchor_times

    def test_anchor_self_not_positive(self, tmp_path):
        """An anchor cannot be its own positive (tau=0 is skipped).

        Single observation at t=5 with tau_range including 0 frames:
        tau_range_hours=(0.0, 0.5) with interval=30min -> tau_range_frames=(0, 1).
        t=5 checks tau=1 -> t=6 doesn't exist -> NOT valid.
        """
        track_rows = [
            {"track_id": 0, "t": 5, "z": 0, "y": 32.0, "x": 32.0}
        ]
        zarr_path, tracks_root = _create_zarr_and_custom_tracks(
            tmp_path,
            name="self_test",
            channel_names=_CHANNEL_NAMES_A,
            well=("A", "1"),
            track_rows=track_rows,
            n_t=10,
        )
        cfg = ExperimentConfig(
            name="self_test",
            data_path=str(zarr_path),
            tracks_path=str(tracks_root),
            channel_names=_CHANNEL_NAMES_A,
            source_channel=["Phase", "GFP"],
            condition_wells={"ctrl": ["A/1"]},
            interval_minutes=30.0,
        )
        registry = ExperimentRegistry(experiments=[cfg])
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.0, 0.5),
        )
        # Only 1 observation, tau=0 skipped, tau=1 -> t=6 missing -> not valid
        assert len(index.valid_anchors) == 0


# ---------------------------------------------------------------------------
# CELL-04: Properties and summary
# ---------------------------------------------------------------------------


class TestMultiExperimentIndexProperties:
    """Tests for experiment_groups, condition_groups, and summary()."""

    def test_experiment_groups_keys(self, two_experiment_setup):
        """experiment_groups returns dict with experiment names as keys."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        groups = index.experiment_groups
        assert set(groups.keys()) == {"exp_a", "exp_b"}

    def test_experiment_groups_values_are_index_arrays(self, two_experiment_setup):
        """experiment_groups values are numpy arrays of row indices."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        groups = index.experiment_groups
        for name, indices in groups.items():
            assert isinstance(indices, np.ndarray)
            # Verify these indices actually correspond to the experiment
            for idx in indices:
                assert index.tracks.loc[idx, "experiment"] == name

    def test_experiment_groups_cover_all_rows(self, two_experiment_setup):
        """All row indices are covered by experiment_groups."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        groups = index.experiment_groups
        all_indices = np.concatenate(list(groups.values()))
        assert len(all_indices) == len(index.tracks)

    def test_condition_groups_keys(self, two_experiment_setup):
        """condition_groups returns dict with condition labels as keys."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        groups = index.condition_groups
        # exp_a: uninfected, infected; exp_b: control, treated
        assert set(groups.keys()) == {
            "uninfected",
            "infected",
            "control",
            "treated",
        }

    def test_condition_groups_values_correct(self, two_experiment_setup):
        """condition_groups values point to rows with matching condition."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        groups = index.condition_groups
        for cond, indices in groups.items():
            assert isinstance(indices, np.ndarray)
            for idx in indices:
                assert index.tracks.loc[idx, "condition"] == cond

    def test_summary_returns_string(self, two_experiment_setup):
        """summary() returns a non-empty string."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        result = index.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_contains_experiment_count(self, two_experiment_setup):
        """summary() mentions the number of experiments."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        result = index.summary()
        assert "2 experiments" in result

    def test_summary_contains_per_experiment_lines(self, two_experiment_setup):
        """summary() has a line per experiment with observation and anchor counts."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        result = index.summary()
        assert "exp_a:" in result
        assert "exp_b:" in result
        assert "observations" in result
        assert "anchors" in result
        assert "conditions:" in result

    def test_summary_contains_total_counts(self, two_experiment_setup):
        """summary() header line contains total observations and valid anchors."""
        registry, _, _ = two_experiment_setup
        index = MultiExperimentIndex(
            registry=registry,
            z_range=slice(0, 1),
            yx_patch_size=_YX_PATCH,
            tau_range_hours=(0.5, 1.5),
        )
        result = index.summary()
        assert f"{len(index.tracks)} total observations" in result
        assert f"{len(index.valid_anchors)} valid anchors" in result
