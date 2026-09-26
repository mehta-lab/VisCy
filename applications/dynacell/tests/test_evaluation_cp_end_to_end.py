"""End-to-end CP (GLCM+) scoring through the real ``evaluate_predictions``.

Tiny HCS plates (3 positions x 2 timepoints) carry a GT volume, a cell
segmentation with 16 textured cells plus one single-voxel cell (whose CP row is
non-finite), and a prediction derived from the GT. CP regionprops, the GT/pred CP
caches, the CP reference, pixel and mask metrics all run for real; only the deep
extractors are stubbed (``fov_deep_features`` returns seeded random rows, one per
cell), since they are not what these tests are about.

The reference is fit by the production builder path on the GT CP cache the first
eval step writes, so the eval's content gate compares two independently collected
views of the same GT cells.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.cache import cache_paths, open_features_group, write_features_to_group
from dynacell.evaluation.model_loader import EvalModels

from ._eval_fixtures import N_POSITIONS, D, H, T, W, build_eval_config, live_pipeline_module, make_mask_cache

_DATASET = "a549-mantis-sec61b-mock"
_POSITIONS = [f"A/1/{i}" for i in range(N_POSITIONS)]


def _segmentation() -> np.ndarray:
    """``(T, D, H, W)`` labels: a 4 x 4 grid of 6 x 6 x 6 cells plus one single-voxel cell."""
    seg = np.zeros((T, D, H, W), dtype=np.int32)
    label = 1
    for row in range(4):
        for col in range(4):
            y, x = 1 + 8 * row, 1 + 8 * col
            seg[:, 2:8, y : y + 6, x : x + 6] = label
            label += 1
    seg[:, 9, 0, 0] = label  # one voxel: skewness/kurtosis are NaN -> a non-finite GT row
    return seg


def _gt_volume(seed: int) -> np.ndarray:
    """``(T, D, H, W)`` GT: uniform background, cells with per-cell texture inside the background range."""
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.0, 1.0, size=(T, D, H, W))
    seg = _segmentation()
    for label in range(1, 17):
        inside = seg == label
        gt[inside] = rng.uniform(0.25, 0.75) + 0.2 * rng.standard_normal(inside.sum())
    return gt.astype(np.float32)


def _write_plate(path: Path, channel: str, volumes: list[np.ndarray], dtype=np.float32) -> None:
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel], version="0.5") as plate:
        for i, vol in enumerate(volumes):
            plate.create_position("A", "1", str(i)).create_image("0", vol[:, None].astype(dtype))


def _deep_stub(ctx, pos_name, image, cell_segmentation, extractor, kind, z_slabs=None):
    """One seeded random 8-d row per cell and timepoint, identical for GT and pred."""
    rng = np.random.default_rng(zlib.crc32(f"{pos_name}/{kind}".encode()))
    return [rng.standard_normal((len(np.unique(cell_segmentation[t])) - 1, 8)) for t in range(image.shape[0])]


def _stub_models() -> EvalModels:
    return EvalModels(
        seg_model=None,
        dinov3=object(),
        dynaclr=object(),
        celldino=None,
        morphem=None,
        dinov3_model_name=None,
        dynaclr_ckpt_path=None,
        dynaclr_encoder_cfg=None,
        celldino_weights_path=None,
        morphem_model_name=None,
    )


class Harness:
    """GT/seg plates, mask caches, a CP reference fit on the GT CP cache, and a runner."""

    def __init__(self, root: Path, monkeypatch):
        self.root = root
        self.pipeline = live_pipeline_module()
        from dynacell.evaluation import cp_reference, pipeline_cache

        self.cp_reference = cp_reference
        self.gt = [_gt_volume(seed=i) for i in range(N_POSITIONS)]
        self.seg = _segmentation()
        _write_plate(root / "gt.zarr", "target", self.gt)
        _write_plate(root / "seg.zarr", "cell_segmentation", [self.seg] * N_POSITIONS, dtype=np.int32)
        make_mask_cache(root / "gt_cache", root / "gt.zarr", "target", side="gt")
        monkeypatch.setattr(self.pipeline, "fov_deep_features", _deep_stub)
        monkeypatch.setattr(self.pipeline, "precompute_deep_features", lambda *a, **k: None)

        # GT CP cache, written by the pipeline's own cache writer, then the reference from it.
        config = self.config(root / "unused_pred.zarr", root / "unused_out")
        ctx = pipeline_cache.init_cache_context(config, side="gt")
        for name, vol in zip(_POSITIONS, self.gt, strict=True):
            pipeline_cache.fov_cp_features(ctx, name, vol, self.seg)
        pipeline_cache.flush_manifest(ctx)
        identity, names = cp_reference.cp_space(config)
        cells, record = cp_reference.read_gt_cp_cells(ctx, root / "gt.zarr", len(names))
        fit = cp_reference.DatasetFit(dataset=_DATASET, cells=cells, record=record, in_mask_fit=True)
        payload = cp_reference.fit_cp_reference([fit], target_name="er", feature_names=names, cp_identity=identity)
        cp_reference.write_cp_reference(payload, root / "er.json")

    def config(self, pred_path: Path, save_dir: Path):
        config = build_eval_config(
            pred_path,
            self.root / "gt.zarr",
            self.root / "gt_cache",
            save_dir.parent / f"{save_dir.name}_pred_cache",
            save_dir,
            executor="serial",
            fov_workers=1,
        )
        config.io.require_complete_cache = False
        config.io.cell_segmentation_path = str(self.root / "seg.zarr")
        config.compute_feature_metrics = True
        config.feature_metrics.compute_fid = False
        config.feature_metrics.compute_prc = False
        config.feature_metrics.compute_mind = False
        config.feature_metrics.cp = {
            "norm": {"p_lo": 1.0, "p_hi": 99.0},
            "glcm": {"enabled": False},
            "reference_path": str(self.root / "er.json"),
        }
        config.benchmark = {"dataset_ref": {"dataset": _DATASET, "target": "sec61b"}}
        return config

    def run(self, name: str, pred: list[np.ndarray], **overrides) -> tuple[dict, object]:
        """Evaluate ``pred`` against the GT; return the dataset-level feature row and the config."""
        pred_path = self.root / f"{name}.zarr"
        _write_plate(pred_path, "prediction", pred)
        config = self.config(pred_path, self.root / name)
        make_mask_cache(Path(config.io.pred_cache_dir), pred_path, "prediction", side="pred")
        for key, value in overrides.items():
            config.force_recompute[key] = value
        _, _, feature_rows = self.pipeline.evaluate_predictions(
            config, models=_stub_models(), cp_space=self.pipeline.eval_cp_space(config)
        )
        return feature_rows[0], config


@pytest.fixture
def harness(tmp_path: Path, monkeypatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_identical_gt_recache_passes_the_content_gate(harness: Harness) -> None:
    """``force_recompute.gt_cp`` rewrites the GT CP cache (new built_at) with identical cells: scoring proceeds."""
    row, _ = harness.run("recache", [g.copy() for g in harness.gt], gt_cp=True)
    assert np.isfinite(row["Dataset_CP_KID"])


def test_same_count_gt_value_change_is_refused(harness: Harness) -> None:
    """A GT CP cache whose values moved (same cell count) since the fit fails before any metric is written."""
    with open_features_group(cache_paths(harness.root / "gt_cache"), "cp", mode="a") as group:
        feats = np.asarray(group["A/1/1/t1"])
        feats[0, 0] += 0.5
        write_features_to_group(group, "A/1/1", 1, feats)
    save_dir = harness.root / "moved"
    with pytest.raises(Exception, match="differ from the CP reference fit") as err:
        harness.run("moved", [g.copy() for g in harness.gt])
    assert type(err.value).__name__ == "StaleCacheError"
    assert not list(save_dir.glob("*_metrics.*")) and not (save_dir / "cp_selected_feature_mask.json").exists()
