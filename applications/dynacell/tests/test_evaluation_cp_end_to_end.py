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

import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr

from dynacell.evaluation.cache import (
    cache_paths,
    load_manifest,
    open_features_group,
    save_manifest,
    write_features_to_group,
)
from dynacell.evaluation.model_loader import EvalModels

from ._eval_fixtures import N_POSITIONS, D, H, T, W, build_eval_config, live_pipeline_module, make_mask_cache

_DATASET = "a549-mantis-sec61b-mock"
_POSITIONS = [f"A/1/{i}" for i in range(N_POSITIONS)]


def _segmentation(single_voxel: bool = True) -> np.ndarray:
    """``(T, D, H, W)`` labels: a 4 x 4 grid of 6 x 6 x 6 cells, plus (optionally) one single-voxel cell.

    The GPU variants leave the single voxel out: cuCIM's GPU ``regionprops_table``
    raises a TypeError on a one-voxel region (a cuCIM limitation, not this code's).
    """
    seg = np.zeros((T, D, H, W), dtype=np.int32)
    label = 1
    for row in range(4):
        for col in range(4):
            y, x = 1 + 8 * row, 1 + 8 * col
            seg[:, 2:8, y : y + 6, x : x + 6] = label
            label += 1
    if single_voxel:
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

    def __init__(self, root: Path, monkeypatch, single_voxel: bool = True):
        self.root = root
        self.pipeline = live_pipeline_module()
        from dynacell.evaluation import cp_reference, pipeline_cache

        self.cp_reference = cp_reference
        self.gt = [_gt_volume(seed=i) for i in range(N_POSITIONS)]
        self.seg = _segmentation(single_voxel)
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

    def run(self, name: str, pred: list[np.ndarray], use_gpu: bool = False, **overrides) -> tuple[dict, object]:
        """Evaluate ``pred`` against the GT; return the dataset-level feature row and the config."""
        pred_path = self.root / f"{name}.zarr"
        _write_plate(pred_path, "prediction", pred)
        config = self.config(pred_path, self.root / name)
        config.use_gpu = use_gpu
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


def test_gpu_level_jitter_in_every_gt_cell_passes(harness: Harness) -> None:
    """Every cached GT CP cell scaled by (1 + 1e-12) -- far above GPU regionprops jitter -- still scores."""
    with open_features_group(cache_paths(harness.root / "gt_cache"), "cp", mode="a") as group:
        for pos in _POSITIONS:
            for t in range(T):
                write_features_to_group(group, pos, t, np.asarray(group[f"{pos}/t{t}"]) * (1 + 1e-12))
    row, _ = harness.run("jitter", [g.copy() for g in harness.gt])
    assert np.isfinite(row["Dataset_CP_KID"])


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@_CUDA
def test_gpu_recompute_of_the_gt_passes_the_content_gate(tmp_path: Path, monkeypatch) -> None:
    """``force_recompute.gt_cp`` with ``use_gpu=True`` against the CPU-built reference scores (no exact-bit gate)."""
    harness = Harness(tmp_path, monkeypatch, single_voxel=False)
    row, _ = harness.run("gpu_recache", [g.copy() for g in harness.gt], use_gpu=True, gt_cp=True)
    assert np.isfinite(row["Dataset_CP_KID"])


@_CUDA
def test_two_gpu_recomputes_of_one_block_pass_the_gate() -> None:
    """One real-shaped block recomputed twice on the GPU: whatever the bit-level jitter, the gate passes."""
    from dynacell.evaluation.cp_reference import DatasetFit, fit_cp_reference, write_cp_reference
    from dynacell.evaluation.cp_reference import load_cp_reference as load
    from dynacell.evaluation.metrics import active_cp_feature_names, cp_regionprops

    image, seg = _gt_volume(seed=3)[0], _segmentation(single_voxel=False)[0]
    runs = [
        cp_regionprops(
            image, seg, [1.0, 1.0, 1.0], norm={"p_lo": 1.0, "p_hi": 99.0}, glcm_cfg={"enabled": True}, use_gpu=True
        )
        for _ in range(2)
    ]
    finite = [r[np.isfinite(r).all(axis=1)] for r in runs]
    names = active_cp_feature_names(True)
    fit = DatasetFit(dataset="gpu", cells=finite[0], record={"positions": ["A/1/0"]}, in_mask_fit=True)
    payload = fit_cp_reference([fit], target_name="er", feature_names=names, cp_identity={})
    path = Path(tempfile.mkdtemp()) / "er.json"
    write_cp_reference(payload, path)
    load(path, target_name="er").for_dataset("gpu").check_gt_cells({("A/1/0", 0): finite[1]})
    print(
        f"GPU runs bit-identical: {np.array_equal(runs[0], runs[1], equal_nan=True)}, "
        f"max |diff| {np.nanmax(np.abs(runs[0] - runs[1])):.3g}"
    )


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


def test_gt_cache_with_reordered_columns_is_refused(harness: Harness) -> None:
    """A GT CP cache whose recorded column names are reordered fails before the FOV loop."""
    paths = cache_paths(harness.root / "gt_cache")
    manifest = load_manifest(paths)
    names = manifest["artifacts"]["cp_features"]["cp_feature_names"]
    names[0], names[1] = names[1], names[0]
    save_manifest(paths, manifest)
    with pytest.raises(Exception, match="Masking by position would misalign them") as err:
        harness.run("reordered", [g.copy() for g in harness.gt])
    assert type(err.value).__name__ == "StaleCacheError"


def _inside_cells(harness: Harness, fn) -> list[np.ndarray]:
    """A prediction equal to the GT outside the cells and ``fn(GT)`` inside them."""
    inside = (harness.seg > 0) & (harness.seg <= 16)
    return [np.where(inside, fn(g), g).astype(np.float32) for g in harness.gt]


def _old_per_side_kid(save_dir: Path, cp_space) -> float:
    """The retired transform on the same run's cells: reference mask, then each side by its own stats."""
    from dynacell.evaluation.feature_metrics import compute_feature_similarity

    def zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(0)) / (x.std(0) + 1e-8)

    emb = {
        side: np.load(save_dir / "embeddings" / f"{side}_cp_single_cell_embeddings.npz")["embeddings"]
        for side in ("pred", "gt")
    }
    pred, gt = cp_space.select(emb["pred"]), cp_space.select(emb["gt"])
    return compute_feature_similarity(
        zscore(pred), zscore(gt), "CP", compute_fid=False, compute_prc=False, compute_mind=False
    )["CP_KID"]


@pytest.mark.parametrize(
    ("name", "fn"),
    [("offset", lambda g: g + 0.15), ("contracted", lambda g: 0.5 * g)],
    ids=["pred=GT+offset", "pred=GTx0.5"],
)
def test_cp_kid_registers_offset_and_contraction_end_to_end(harness: Harness, name: str, fn) -> None:
    """Through the real pipeline, an offset or contracted prediction scores clearly above the pred==GT floor.

    Contrast: the retired per-side z-score, applied to the same run's CP cells,
    scores both near the floor, because it standardizes each side by its own
    statistics and so absorbs a per-feature offset or scale.
    """
    floor_row, _ = harness.run("identity", [g.copy() for g in harness.gt])
    row, config = harness.run(name, _inside_cells(harness, fn))
    floor, kid = floor_row["Dataset_CP_KID"], row["Dataset_CP_KID"]
    old = _old_per_side_kid(Path(config.save.save_dir), harness.pipeline.eval_cp_space(config))
    assert kid > floor + 1.0, (kid, floor)
    assert abs(old - floor) < 0.1 * (kid - floor), (old, floor, kid)
