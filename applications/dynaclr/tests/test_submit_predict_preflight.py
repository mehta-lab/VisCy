"""Tests for the AI-ready preflight in submit_predict (zattrs detection + flagging)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from iohub.ngff import open_ome_zarr

from dynaclr.evaluation.orchestration import predict_batch as submit_predict

CHANNELS = ["Phase3D"]


def _make_zarr(path: Path, *, normalization: bool, focus_slice: bool) -> None:
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=CHANNELS) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_zeros("0", shape=(1, 1, 4, 8, 8), dtype=np.float32)
        if normalization:
            pos.zattrs["normalization"] = {"Phase3D": {"fov_statistics": {"mean": 0.0, "std": 1.0}}}
        if focus_slice:
            pos.zattrs["focus_slice"] = {"Phase3D": {"fov_statistics": {"z_focus_mean": 2}}}


def _collection(tmp_path: Path, data_path: Path, name: str = "ds") -> Path:
    coll = tmp_path / "collection.yml"
    coll.write_text(
        yaml.safe_dump(
            {
                "name": "c",
                "experiments": [
                    {
                        "name": name,
                        "data_path": str(data_path),
                        "tracks_path": str(tmp_path / "tracks"),
                        "channels": [{"name": "Phase3D", "marker": "Phase3D"}],
                        "perturbation_wells": {"uninfected": ["A/1"]},
                    }
                ],
            }
        )
    )
    return coll


def test_check_ai_ready_all_present(tmp_path):
    z = tmp_path / "ok.zarr"
    _make_zarr(z, normalization=True, focus_slice=True)
    report = submit_predict.check_ai_ready(_collection(tmp_path, z))
    assert report[0]["missing_normalization"] is False
    assert report[0]["missing_focus_slice"] is False


def test_check_ai_ready_flags_missing(tmp_path):
    z = tmp_path / "raw.zarr"
    _make_zarr(z, normalization=False, focus_slice=False)
    report = submit_predict.check_ai_ready(_collection(tmp_path, z))
    assert report[0]["missing_normalization"] is True
    assert report[0]["missing_focus_slice"] is True


def test_preflight_raises_on_missing_focus(tmp_path):
    """focus_slice is never auto-run — a missing one must raise with QC guidance."""
    z = tmp_path / "nofocus.zarr"
    _make_zarr(z, normalization=True, focus_slice=False)
    with pytest.raises(ValueError, match="focus_slice"):
        submit_predict.preflight(_collection(tmp_path, z), "/ws", auto_normalize=True)


def test_preflight_raises_on_missing_norm_without_auto(tmp_path):
    z = tmp_path / "nonorm.zarr"
    _make_zarr(z, normalization=False, focus_slice=True)
    with pytest.raises(ValueError, match="normalization"):
        submit_predict.preflight(_collection(tmp_path, z), "/ws", auto_normalize=False)


def test_preflight_passes_when_ai_ready(tmp_path):
    z = tmp_path / "ready.zarr"
    _make_zarr(z, normalization=True, focus_slice=True)
    # Should not raise.
    submit_predict.preflight(_collection(tmp_path, z), "/ws", auto_normalize=False)
