"""Integration test for dynacell.evaluation.precompute_cli."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

pytest.importorskip("zarr")

from dynacell.evaluation.cache import cache_paths, load_manifest, read_mask  # noqa: E402


def _write_tiny_hcs(path: Path, positions: list[tuple[str, str, str]], channel: str = "target") -> None:
    """Create a minimal HCS OME-Zarr plate with deterministic content."""
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel], version="0.5") as plate:
        for row, col, fov in positions:
            pos = plate.create_position(row, col, fov)
            # Shape: (T=1, C=1, Z=2, Y=4, X=4)
            data = np.full((1, 1, 2, 4, 4), 0.5, dtype=np.float32)
            pos.create_image("0", data)


def _seg_fn(img, target_name, seg_model=None):
    """Deterministic segmentation stub: everything > 0 is foreground."""
    del target_name, seg_model
    return (np.asarray(img) > 0).astype(bool)


def test_precompute_gt_masks_only_writes_mask_plate(tmp_path: Path, monkeypatch) -> None:
    """build.masks=true (only) writes organelle_masks/{target}.zarr and updates the manifest."""
    gt_path = tmp_path / "gt.zarr"
    cache_dir = tmp_path / "cache"
    _write_tiny_hcs(gt_path, [("A", "1", "0"), ("A", "1", "1")])

    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn)
    # Avoid loading real segmentation checkpoints.
    monkeypatch.setattr(segmentation, "prepare_segmentation_model", lambda cfg: None)

    config = OmegaConf.create(
        {
            "target_name": "er",
            "io": {
                "gt_path": str(gt_path),
                "pred_path": "/unused",
                "cell_segmentation_path": None,
                "gt_cache_dir": str(cache_dir),
                "pred_channel_name": "prediction",
                "gt_channel_name": "target",
                "require_complete_cache": False,
            },
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
            "feature_metrics": {"patch_size": 4},
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "final_metrics": False,
            },
            "build": {"masks": True, "cp": False, "dinov3": False, "dynaclr": False},
            "compute_microssim": False,
            "compute_feature_metrics": False,
            "limit_positions": None,
        }
    )

    from dynacell.evaluation.precompute_cli import precompute_gt_artifacts

    precompute_gt_artifacts(config)

    paths = cache_paths(cache_dir)
    assert paths.mask_plate("er").exists()
    mask_a = read_mask(paths, "er", "A/1/0")
    mask_b = read_mask(paths, "er", "A/1/1")
    assert mask_a is not None and mask_a.shape == (1, 2, 4, 4)
    assert mask_b is not None and mask_b.shape == (1, 2, 4, 4)
    assert mask_a.dtype == bool
    assert mask_a.all()  # seg_fn returns all-True for positive input

    manifest = load_manifest(paths)
    assert manifest["gt"]["plate_path"] == str(gt_path)
    assert manifest["gt"]["channel_name"] == "target"
    er_entry = manifest["artifacts"]["organelle_masks"]["er"]
    assert sorted(er_entry["positions"]) == ["A/1/0", "A/1/1"]


def test_precompute_gt_requires_cache_dir(tmp_path: Path) -> None:
    """Missing io.gt_cache_dir raises with a clear message."""
    gt_path = tmp_path / "gt.zarr"
    _write_tiny_hcs(gt_path, [("A", "1", "0")])

    config = OmegaConf.create(
        {
            "target_name": "er",
            "io": {
                "gt_path": str(gt_path),
                "pred_path": "/unused",
                "cell_segmentation_path": None,
                "gt_cache_dir": None,
                "pred_channel_name": "prediction",
                "gt_channel_name": "target",
                "require_complete_cache": False,
            },
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
            "feature_metrics": {"patch_size": 4},
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "final_metrics": False,
            },
            "build": {"masks": True, "cp": False, "dinov3": False, "dynaclr": False},
            "compute_microssim": False,
            "compute_feature_metrics": False,
            "limit_positions": None,
        }
    )

    from dynacell.evaluation.precompute_cli import precompute_gt_artifacts

    with pytest.raises(ValueError, match="io.gt_cache_dir is required"):
        precompute_gt_artifacts(config)


def test_precompute_features_require_cell_segmentation(tmp_path: Path) -> None:
    """build.cp=true without io.cell_segmentation_path raises."""
    gt_path = tmp_path / "gt.zarr"
    _write_tiny_hcs(gt_path, [("A", "1", "0")])

    config = OmegaConf.create(
        {
            "target_name": "er",
            "io": {
                "gt_path": str(gt_path),
                "pred_path": "/unused",
                "cell_segmentation_path": None,
                "gt_cache_dir": str(tmp_path / "cache"),
                "pred_channel_name": "prediction",
                "gt_channel_name": "target",
                "require_complete_cache": False,
            },
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
            "feature_metrics": {"patch_size": 4},
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "final_metrics": False,
            },
            "build": {"masks": False, "cp": True, "dinov3": False, "dynaclr": False},
            "compute_microssim": False,
            "compute_feature_metrics": False,
            "limit_positions": None,
        }
    )

    from dynacell.evaluation.precompute_cli import precompute_gt_artifacts

    with pytest.raises(ValueError, match="cell_segmentation_path is required"):
        precompute_gt_artifacts(config)


def test_precompute_respects_limit_positions(tmp_path: Path, monkeypatch) -> None:
    """limit_positions trims the FOV iteration."""
    gt_path = tmp_path / "gt.zarr"
    cache_dir = tmp_path / "cache"
    _write_tiny_hcs(gt_path, [("A", "1", "0"), ("A", "1", "1"), ("A", "1", "2")])

    import dynacell.evaluation.segmentation as segmentation

    monkeypatch.setattr(segmentation, "segment", _seg_fn)
    monkeypatch.setattr(segmentation, "prepare_segmentation_model", lambda cfg: None)

    config = OmegaConf.create(
        {
            "target_name": "er",
            "io": {
                "gt_path": str(gt_path),
                "pred_path": "/unused",
                "cell_segmentation_path": None,
                "gt_cache_dir": str(cache_dir),
                "pred_channel_name": "prediction",
                "gt_channel_name": "target",
                "require_complete_cache": False,
            },
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
            "feature_metrics": {"patch_size": 4},
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "final_metrics": False,
            },
            "build": {"masks": True, "cp": False, "dinov3": False, "dynaclr": False},
            "compute_microssim": False,
            "compute_feature_metrics": False,
            "limit_positions": 2,
        }
    )

    from dynacell.evaluation.precompute_cli import precompute_gt_artifacts

    precompute_gt_artifacts(config)

    paths = cache_paths(cache_dir)
    manifest = load_manifest(paths)
    positions = manifest["artifacts"]["organelle_masks"]["er"]["positions"]
    assert sorted(positions) == ["A/1/0", "A/1/1"]  # third position skipped
