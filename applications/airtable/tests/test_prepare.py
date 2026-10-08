"""Tests for the tracks-copy step of airtable_utils.prepare."""

from __future__ import annotations

from pathlib import Path

from airtable_utils.prepare import generate_concatenate_script, has_in_plate_tracks


def _script(tmp_path: Path, **kwargs) -> str:
    defaults = {
        "crop_concat_path": tmp_path / "crop_concat.yml",
        "vast_zarr_path": tmp_path / "vast" / "ds.zarr",
        "nfs_tracking_path": tmp_path / "nfs" / "tracking.zarr",
        "vast_tracking_path": tmp_path / "vast" / "tracking.zarr",
        "conda_env": "biahub",
    }
    return generate_concatenate_script(**{**defaults, **kwargs})


def test_has_in_plate_tracks(tmp_path):
    plate = tmp_path / "plate.zarr"
    (plate / "A" / "1" / "0").mkdir(parents=True)
    assert not has_in_plate_tracks(plate)
    (plate / "A" / "1" / "0" / "tracks.geff").mkdir()
    assert has_in_plate_tracks(plate)


def test_legacy_tracking_zarr_rsync(tmp_path):
    script = _script(tmp_path)
    assert "biahub concatenate" in script
    assert "Copy tracking zarr" in script
    assert str(tmp_path / "nfs" / "tracking.zarr") in script
    assert "tracks.geff" not in script


def test_in_plate_tracks_copy(tmp_path):
    nfs_zarr = tmp_path / "nfs" / "ds.zarr"
    script = _script(tmp_path, nfs_tracking_path=None, vast_tracking_path=None, nfs_zarr_path=nfs_zarr)
    assert "Copy in-plate tracks" in script
    assert f'src_root="{nfs_zarr}"' in script
    assert "--include='tracks.geff/***'" in script
    assert "tracking.zarr" not in script


def test_no_tracks_step(tmp_path):
    script = _script(tmp_path, nfs_tracking_path=None, vast_tracking_path=None)
    assert "Step 2" not in script
