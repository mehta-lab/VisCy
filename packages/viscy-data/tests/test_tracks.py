"""Tests for viscy_data.tracks — per-FOV GEFF / CSV track readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from iohub import open_ome_zarr

from viscy_data.cell_index import build_timelapse_cell_index
from viscy_data.collection import ExperimentEntry
from viscy_data.tracks import read_fov_tracks, read_geff_tracks

geff = pytest.importorskip("geff")
from geff.core_io import write_arrays  # noqa: E402
from geff_spec import Axis  # noqa: E402

FOV = "A/1/0"


def _expected_tracks(with_z: bool) -> pd.DataFrame:
    """Ultrack ``to_tracks_layer`` CSV for a small forest with one division.

    Track 1 (nodes 10, 11) divides at t=2 into tracks 2 (nodes 20, 21) and
    3 (node 30); track 4 (nodes 40, 41) is an unrelated root track.
    """
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 3, 4, 4],
            "t": [0, 1, 2, 3, 2, 1, 2],
            "z": [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
            "y": [10.5, 11.0, 12.0, 13.25, 20.0, 40.0, 41.0],
            "x": [5.0, 6.0, 7.5, 8.0, 15.0, 30.0, 31.0],
            "id": [10, 11, 20, 21, 30, 40, 41],
            "parent_track_id": [-1, -1, 1, 1, 1, -1, -1],
            "parent_id": [-1, 10, 11, 20, 11, -1, 40],
        }
    )
    return df if with_z else df.drop(columns="z")


def _write_geff(path: Path, tracks: pd.DataFrame, store_parent_track_id: bool = False) -> None:
    """Write ``tracks`` as a GEFF the way ultrack's ``to_geff`` lays it out."""
    # Shuffle node order: the reader must not rely on the on-disk order.
    tracks = tracks.sample(frac=1.0, random_state=0)
    spatial = [c for c in ("z", "y", "x") if c in tracks.columns]
    props = {name: tracks[name].to_numpy() for name in ("t", *spatial, "track_id")}
    props["lineage_id"] = np.where(tracks["track_id"] == 4, 4, 1)
    props["seg_id"] = tracks["track_id"].to_numpy()
    if store_parent_track_id:
        props["parent_track_id"] = tracks["parent_track_id"].to_numpy()
    edges = tracks.loc[tracks["parent_id"] >= 0, ["parent_id", "id"]].to_numpy(dtype=np.uint64)
    metadata = geff.GeffMetadata(
        directed=True,
        axes=[Axis(name="t", type="time"), *(Axis(name=c, type="space") for c in spatial)],
        node_props_metadata={},
        edge_props_metadata={},
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
    )
    write_arrays(
        path,
        node_ids=tracks["id"].to_numpy(dtype=np.uint64),
        node_props={name: {"values": values, "missing": None} for name, values in props.items()},
        edge_ids=edges,
        edge_props={},
        metadata=metadata,
        zarr_format=3,
    )


@pytest.mark.parametrize("with_z", [False, True], ids=["2d", "3d"])
@pytest.mark.parametrize("store_parent_track_id", [False, True], ids=["derived", "stored"])
def test_geff_matches_csv(tmp_path, with_z, store_parent_track_id):
    """GEFF with a division reads back as the equivalent ultrack CSV."""
    expected = _expected_tracks(with_z)
    _write_geff(tmp_path / "tracks.geff", expected, store_parent_track_id)
    csv_path = tmp_path / "tracks.csv"
    expected.to_csv(csv_path, index=False)

    result = read_geff_tracks(tmp_path / "tracks.geff")

    pd.testing.assert_frame_equal(result, pd.read_csv(csv_path), check_dtype=False)
    assert list(result.columns) == list(expected.columns)


def test_geff_missing_track_id_raises(tmp_path):
    tracks = _expected_tracks(with_z=False)
    write_arrays(
        tmp_path / "tracks.geff",
        node_ids=tracks["id"].to_numpy(dtype=np.uint64),
        node_props={c: {"values": tracks[c].to_numpy(), "missing": None} for c in ("t", "y", "x")},
        edge_ids=np.empty((0, 2), dtype=np.uint64),
        edge_props={},
        metadata=geff.GeffMetadata(directed=True, node_props_metadata={}, edge_props_metadata={}),
        zarr_format=3,
    )
    with pytest.raises(ValueError, match="track_id"):
        read_geff_tracks(tmp_path / "tracks.geff")


class TestReadFovTracks:
    def test_prefers_geff_over_csv(self, tmp_path):
        fov_dir = tmp_path / FOV
        fov_dir.mkdir(parents=True)
        expected = _expected_tracks(with_z=False)
        _write_geff(fov_dir / "tracks.geff", expected)
        # A stale CSV next to the GEFF must not be read.
        expected.iloc[:2].to_csv(fov_dir / "tracks_A_1_0.csv", index=False)

        result = read_fov_tracks(tmp_path, FOV)

        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_csv_fallback(self, tmp_path):
        fov_dir = tmp_path / FOV
        fov_dir.mkdir(parents=True)
        expected = _expected_tracks(with_z=False)
        expected.to_csv(fov_dir / "tracks.csv", index=False)

        pd.testing.assert_frame_equal(read_fov_tracks(str(tmp_path), FOV), expected, check_dtype=False)

    def test_no_tracks_raises(self, tmp_path):
        (tmp_path / FOV).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No tracking CSV"):
            read_fov_tracks(tmp_path, FOV)

    def test_multiple_csv_raises(self, tmp_path):
        fov_dir = tmp_path / FOV
        fov_dir.mkdir(parents=True)
        for name in ("a.csv", "b.csv"):
            _expected_tracks(with_z=False).to_csv(fov_dir / name, index=False)
        with pytest.raises(ValueError, match="exactly one tracking CSV"):
            read_fov_tracks(tmp_path, FOV)


class TestTracksPathDefault:
    def test_defaults_to_data_path(self):
        entry = ExperimentEntry(name="exp", data_path="/data/plate.zarr", perturbation_wells={"ctrl": ["A/1"]})
        assert entry.tracks_path == "/data/plate.zarr"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_defaults_to_data_path(self, value):
        entry = ExperimentEntry(name="exp", data_path="/data/plate.zarr", tracks_path=value)
        assert entry.tracks_path == "/data/plate.zarr"

    def test_explicit_tracks_path_kept(self):
        entry = ExperimentEntry(name="exp", data_path="/data/plate.zarr", tracks_path="/data/tracking.zarr")
        assert entry.tracks_path == "/data/tracking.zarr"

    def test_build_cell_index_from_in_plate_geff(self, tmp_path):
        """Collection without tracks_path reads <plate>/<fov>/tracks.geff."""
        plate_path = tmp_path / "plate.zarr"
        with open_ome_zarr(plate_path, layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
            pos = plate.create_position("A", "1", "0")
            pos.create_image("0", np.zeros((4, 1, 1, 64, 64), dtype=np.float32))
        _write_geff(plate_path / FOV / "tracks.geff", _expected_tracks(with_z=False))

        yaml_path = tmp_path / "collection.yml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "geff_collection",
                    "experiments": [
                        {
                            "name": "geff_exp",
                            "data_path": str(plate_path),
                            "channels": [{"name": "Phase3D", "marker": "Phase3D"}],
                            "perturbation_wells": {"ctrl": ["A/1"]},
                        }
                    ],
                }
            )
        )

        df = build_timelapse_cell_index(yaml_path, tmp_path / "index.parquet")

        assert len(df) == 7
        assert set(df["tracks_path"]) == {str(plate_path)}
        daughters = df[df["track_id"].isin([2, 3])]
        assert (daughters["lineage_id"] == "geff_exp_A/1/0_1").all()
