"""``segmentation.semantic_focus_halfwidth`` through the eval pipeline: focus-plane ER masks.

Runs ``evaluate_predictions`` end to end with the real ER segmenter (cubic's
``workflow_sec61b`` on CPU) on synthetic tubular volumes, and checks the mask rows
against segmenting each focus slab by hand, for both plane anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

from dynacell.evaluation.cache import prediction_sources, prediction_sources_sha256_12

from ._eval_fixtures import build_eval_config, live_pipeline_module

pytest.importorskip("cubic.segmentation")

T, D, H, W = 2, 11, 48, 48
POSITIONS = ("A/1/0", "A/1/1")
#: Designed nucleus-area planes per position and timepoint (ellipsoid centers).
NUCLEUS_PLANES = {"A/1/0": (4, 6), "A/1/1": (5, 3)}
#: Phase-midband planes written as ``focus_slice`` zattrs; 0 and D - 1 exercise the clipped slab.
PHASE_PLANES = {"A/1/0": (0, 5), "A/1/1": (D - 1, 7)}


def _tubes(rng: np.random.Generator) -> np.ndarray:
    """A ``(D, H, W)`` volume of blurred line segments on a noisy background (ER-like)."""
    vol = np.zeros((D, H, W), np.float32)
    for _ in range(14):
        z = rng.integers(2, D - 2)
        y0, x0 = rng.integers(0, H, 2)
        angle = rng.uniform(0, np.pi)
        for s in np.linspace(-20, 20, 120):
            y, x = int(y0 + s * np.sin(angle)), int(x0 + s * np.cos(angle))
            if 0 <= y < H and 0 <= x < W:
                vol[z, y, x] = 1.0
    vol = gaussian_filter(vol, (1.0, 0.8, 0.8)) * 50 + rng.normal(0, 0.02, vol.shape) + 0.1
    return vol.astype(np.float32)


def _ellipsoid(center_z: int) -> np.ndarray:
    """A ``(D, H, W)`` solid nucleus whose widest cross-section is at ``center_z``."""
    z, y, x = np.ogrid[:D, :H, :W]
    inside = ((z - center_z) / 3.0) ** 2 + ((y - H / 2) / 12.0) ** 2 + ((x - W / 2) / 12.0) ** 2 <= 1.0
    return inside.astype(np.float32)


def _build_stores(root: Path) -> tuple[Path, Path]:
    """GT store (``target`` + ``Nuclei``, ``focus_slice`` zattrs) and a prediction store sharing its tubes."""
    rng = np.random.default_rng(0)
    gt_path, pred_path = root / "gt.zarr", root / "pred.zarr"
    with (
        open_ome_zarr(gt_path, mode="w", layout="hcs", channel_names=["target", "Nuclei"], version="0.5") as gt,
        open_ome_zarr(pred_path, mode="w", layout="hcs", channel_names=["prediction"], version="0.5") as pred,
    ):
        for name in POSITIONS:
            row, col, fov = name.split("/")
            target = np.stack([_tubes(rng) for _ in range(T)])
            # The prediction keeps the tubes, adds noise and some of its own, so Dice < 1.
            predict = 0.7 * target + 0.3 * np.stack([_tubes(rng) for _ in range(T)])
            nuclei = np.stack([_ellipsoid(z) for z in NUCLEUS_PLANES[name]])
            gt_pos = gt.create_position(row, col, fov)
            gt_pos.create_image("0", np.stack([target, nuclei], axis=1))
            gt_pos.zattrs["focus_slice"] = {
                "Phase3D": {"per_timepoint": {str(t): z for t, z in enumerate(PHASE_PLANES[name])}}
            }
            pred.create_position(row, col, fov).create_image("0", predict[:, None].astype(np.float32))
    return gt_path, pred_path


@pytest.fixture
def stores(tmp_path: Path):
    """Store paths plus a ``config(save_name, **segmentation)`` factory with empty caches."""
    gt_path, pred_path = _build_stores(tmp_path)

    def config(save_name: str, **segmentation):
        cfg = build_eval_config(
            pred_path,
            gt_path,
            tmp_path / "gt_cache",
            tmp_path / "pred_cache",
            tmp_path / save_name,
            executor="serial",
            fov_workers=1,
        )
        cfg.io.require_complete_cache = False
        OmegaConf.update(cfg, "segmentation", segmentation, merge=True)
        return cfg

    return gt_path, pred_path, config


def _expected_rows(gt_path: Path, pred_path: Path, planes: dict[str, tuple[int, ...]]) -> dict[tuple[str, int], dict]:
    """Segment each focus slab by hand with the production segmenter and score its focus plane."""
    from dynacell.evaluation.metrics import evaluate_segmentations
    from dynacell.evaluation.segmentation import segment

    expected = {}
    with open_ome_zarr(gt_path, mode="r") as gt, open_ome_zarr(pred_path, mode="r") as pred:
        for name in POSITIONS:
            target = np.asarray(gt[name].data[:, 0])
            predict = np.asarray(pred[name].data[:, 0])
            for t, z in enumerate(planes[name]):
                lo, hi = max(0, z - 1), min(D, z + 2)

                def at_focus(vol: np.ndarray, lo: int = lo, hi: int = hi, z: int = z) -> np.ndarray:
                    return np.asarray(segment(vol[lo:hi], "er", use_gpu=False))[z - lo].astype(bool)

                expected[(name, t)] = evaluate_segmentations(at_focus(predict[t]), at_focus(target[t]))
    return expected


def _rows_by_key(mask_rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["FOV"], row["Timepoint"]): row for row in mask_rows}


@pytest.mark.parametrize(
    ("anchor", "planes"),
    [("nucleus_area", NUCLEUS_PLANES), ("phase_midband", PHASE_PLANES)],
    ids=["nucleus_area", "phase_midband"],
)
def test_focus_masks_score_the_slab_center_plane(stores, anchor, planes):
    """Each mask row is the focus plane of a segmented 3-plane slab, on the anchor's plane."""
    gt_path, pred_path, config = stores
    pipeline = live_pipeline_module()
    cfg = config("focus", semantic_focus_halfwidth=1, focus_anchor=anchor, nuclei_channel_name="Nuclei")
    _, mask_rows, _ = pipeline.evaluate_predictions(cfg)

    rows = _rows_by_key(mask_rows)
    expected = _expected_rows(gt_path, pred_path, planes)
    assert rows.keys() == expected.keys()
    for key, want in expected.items():
        assert {k: rows[key][k] for k in want} == want, key
    # Non-degenerate: real overlap that is not perfect.
    assert all(0.0 < row["Dice"] < 1.0 for row in rows.values())


def test_focus_masks_differ_from_whole_volume_masks(stores):
    """The same data scored over the whole volume gives different rows, so the mode is observable."""
    _, _, config = stores
    pipeline = live_pipeline_module()
    _, volume_rows, _ = pipeline.evaluate_predictions(config("volume"))
    _, focus_rows, _ = pipeline.evaluate_predictions(
        config("focus", semantic_focus_halfwidth=1, focus_anchor="nucleus_area", nuclei_channel_name="Nuclei")
    )
    volume, focus = _rows_by_key(volume_rows), _rows_by_key(focus_rows)
    assert volume.keys() == focus.keys()
    assert any(volume[key]["Dice"] != focus[key]["Dice"] for key in volume)


def test_stamp_gates_the_final_metrics_cache_both_ways(stores, tmp_path: Path):
    """The recipe is stamped; a whole-volume run refuses a focus cache, and the reverse."""
    _, pred_path, config = stores
    pipeline = live_pipeline_module()
    digest = prediction_sources_sha256_12(prediction_sources(pred_path, "prediction"))
    focus_kwargs = {"semantic_focus_halfwidth": 1, "focus_anchor": "phase_midband"}

    def run(cfg):
        Path(cfg.save.save_dir).mkdir(parents=True, exist_ok=True)
        pixel, mask, feature = pipeline.evaluate_predictions(cfg)
        pipeline.save_metrics(cfg, pixel, mask, feature, cp_space=None, prediction_digest=digest)

    focus_cfg = config("focus", **focus_kwargs)
    run(focus_cfg)
    stamp = json.loads((tmp_path / "focus" / "metrics_provenance.json").read_text())
    assert stamp["semantic_focus"] == {
        "halfwidth": 1,
        "focus_anchor": "phase_midband",
        "focus_channel_name": "Phase3D",
        "na_det": 1.35,
        "lambda_ill": 0.45,
        "pixel_size": 1.0,
    }
    assert pipeline._final_metrics_cache_valid(focus_cfg) is True
    assert pipeline._final_metrics_cache_valid(config("focus")) is False
    assert (
        pipeline._final_metrics_cache_valid(config("focus", **{**focus_kwargs, "semantic_focus_halfwidth": 2})) is False
    )

    run(config("volume"))
    assert "semantic_focus" not in json.loads((tmp_path / "volume" / "metrics_provenance.json").read_text())
    assert pipeline._final_metrics_cache_valid(config("volume")) is True
    assert pipeline._final_metrics_cache_valid(config("volume", **focus_kwargs)) is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"semantic_focus_halfwidth": -1}, ">= 0"),
        ({"semantic_focus_halfwidth": 1, "focus_anchor": "nucleus_area"}, "nuclei_channel_name"),
        ({"semantic_focus_halfwidth": 1, "focus_anchor": "sharpest"}, "focus_anchor"),
    ],
)
def test_invalid_recipe_fails_at_the_cache_gate(stores, overrides, match):
    """A bad recipe raises before any model load, even when the final metrics are forced."""
    _, _, config = stores
    pipeline = live_pipeline_module()
    cfg = config("bad", **overrides)
    cfg.force_recompute.final_metrics = True
    with pytest.raises(ValueError, match=match):
        pipeline._final_metrics_cache_valid(cfg)


def test_instance_targets_are_refused(stores):
    """Instance targets already score one plane through ``slice_selection``."""
    _, _, config = stores
    cfg = config("instance", semantic_focus_halfwidth=1, focus_anchor="phase_midband")
    cfg.compute_instance_ap = True
    with pytest.raises(ValueError, match="slice_selection"):
        live_pipeline_module()._semantic_focus_settings(cfg)
