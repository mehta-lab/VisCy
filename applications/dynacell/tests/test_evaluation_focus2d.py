"""``focus2d.halfwidth`` (the 2D benchmark) through the eval pipeline.

Runs ``evaluate_predictions`` end to end with the real ER segmenter (cubic's
``workflow_sec61b`` on CPU) and the real pixel metrics on synthetic tubular volumes,
and checks every metric family against the same computation done by hand on the
focus plane (or its slab), for both plane anchors. Also covers a plane-restricted
prediction store (``HCSDataModule.predict_z_planes``), the plane file
``precompute-gt`` writes for it, the MicroMS3IM calibration pool, the cache
isolation and the provenance stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

from dynacell.evaluation.cache import prediction_sources, prediction_sources_sha256_12
from viscy_data import HCSDataModule
from viscy_utils.callbacks.prediction_writer import HCSPredictionWriter

from ._eval_fixtures import build_eval_config, live_pipeline_module

pytest.importorskip("cubic.segmentation")

T, D, H, W = 2, 11, 48, 48
POSITIONS = ("A/1/0", "A/1/1")
#: Designed nucleus-area planes per position and timepoint (ellipsoid centers).
NUCLEUS_PLANES = {"A/1/0": (4, 6), "A/1/1": (5, 3)}
#: Phase-midband planes written as ``focus_slice`` zattrs; 0 and D - 1 exercise the clipped slab.
PHASE_PLANES = {"A/1/0": (0, 5), "A/1/1": (D - 1, 7)}
ANCHORS = [("nucleus_area", NUCLEUS_PLANES), ("phase_midband", PHASE_PLANES)]
_EVAL_YAML = Path(__file__).resolve().parents[1] / "src/dynacell/evaluation/_configs/eval.yaml"
_PRECOMPUTE_YAML = _EVAL_YAML.with_name("precompute.yaml")


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


def _slab(z: int) -> list[int]:
    return list(range(max(0, z - 1), min(D, z + 2)))


@pytest.fixture
def stores(tmp_path: Path):
    """Store paths plus a ``config(save_name, pred_path=None, halfwidth=None, **segmentation)`` factory."""
    gt_path, pred_path = _build_stores(tmp_path)

    def config(save_name: str, *, pred=None, halfwidth=None, **segmentation):
        cfg = build_eval_config(
            pred or pred_path,
            gt_path,
            tmp_path / "gt_cache",
            tmp_path / "pred_cache",
            tmp_path / save_name,
            executor="serial",
            fov_workers=1,
        )
        cfg.io.require_complete_cache = False
        cfg.focus2d = {"halfwidth": halfwidth}
        OmegaConf.update(cfg, "segmentation", segmentation, merge=True)
        return cfg

    return gt_path, pred_path, config


def _focus_kwargs(anchor: str) -> dict:
    return {"halfwidth": 1, "focus_anchor": anchor, "nuclei_channel_name": "Nuclei"}


def _volumes(gt_path: Path, pred_path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with open_ome_zarr(gt_path, mode="r") as gt, open_ome_zarr(pred_path, mode="r") as pred:
        return np.asarray(gt[name].data[:, 0]), np.asarray(pred[name].data[:, 0])


def _rows_by_key(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["FOV"], row["Timepoint"]): row for row in rows}


@pytest.mark.parametrize(("anchor", "planes"), ANCHORS, ids=[a for a, _ in ANCHORS])
def test_masks_and_pixel_metrics_score_the_focus_plane(stores, anchor, planes):
    """Mask rows are the focus plane of a segmented slab; pixel rows are 2D metrics on that plane."""
    from dynacell.evaluation.metrics import compute_pixel_metrics, evaluate_segmentations
    from dynacell.evaluation.segmentation import segment

    gt_path, pred_path, config = stores
    pipeline = live_pipeline_module()
    cfg = config("focus", **_focus_kwargs(anchor))
    pixel_rows, mask_rows, _ = pipeline.evaluate_predictions(cfg)
    pixel, mask = _rows_by_key(pixel_rows), _rows_by_key(mask_rows)

    for name in POSITIONS:
        target, predict = _volumes(gt_path, pred_path, name)
        for t, z in enumerate(planes[name]):
            lo = _slab(z)[0]

            def at_focus(vol: np.ndarray, lo: int = lo, z: int = z) -> np.ndarray:
                return np.asarray(segment(vol[lo : lo + len(_slab(z))], "er", use_gpu=False))[z - lo].astype(bool)

            want_mask = evaluate_segmentations(at_focus(predict[t]), at_focus(target[t]))
            assert {k: mask[(name, t)][k] for k in want_mask} == want_mask
            want_pixel = compute_pixel_metrics(
                predict[t, z], target[t, z], spacing=[1.0, 1.0, 1.0], fsc_kwargs=None, spectral_pcc_kwargs=None
            )
            got_pixel = {k: float(pixel[(name, t)][k]) for k in want_pixel}
            assert got_pixel == pytest.approx({k: float(v) for k, v in want_pixel.items()}, nan_ok=True)
            assert not np.isnan(pixel[(name, t)]["SI_SSIM"])
    assert all(0.0 < row["Dice"] < 1.0 for row in mask.values())

    # Discriminates: the same data scored over the full volume gives other pixel and mask rows.
    pixel_3d, mask_3d, _ = live_pipeline_module().evaluate_predictions(config("volume"))
    pixel_3d, mask_3d = _rows_by_key(pixel_3d), _rows_by_key(mask_3d)
    assert all(pixel_3d[key]["SI_SSIM"] != pixel[key]["SI_SSIM"] for key in pixel)
    assert any(mask_3d[key]["Dice"] != mask[key]["Dice"] for key in mask)


def test_cp_and_deep_features_read_the_plane_and_its_slab(stores):
    """CP regionprops run on the focus plane alone; deep-feature crops project its slab."""
    from dynacell.evaluation.metrics import build_crops, cp_regionprops
    from dynacell.evaluation.pipeline_cache import init_cache_context

    gt_path, pred_path, config = stores
    pipeline = live_pipeline_module()
    _, predict = _volumes(gt_path, pred_path, "A/1/0")
    # Labels whose cells change with z, so the plane choice shows in the CP rows.
    cells = np.zeros((T, D, H, W), np.uint16)
    for z in range(D):
        cells[:, z, 4 + z : 20 + z, 6:22] = 1
        cells[:, z, 26:42, 8 + z : 24 + z] = 2
    planes = list(NUCLEUS_PLANES["A/1/0"])
    slabs = [slice(z - 1, z + 2) for z in planes]

    class _MeanExtractor:
        def extract_features_batch(self, crops):
            return torch.stack([torch.as_tensor(np.asarray(c)).float().mean().reshape(1) for c in crops])

    cfg = config("feat", **_focus_kwargs("nucleus_area"))
    cfg.io.pred_cache_dir = None
    ctx = init_cache_context(cfg, side="pred")
    assert not ctx.enabled
    got = pipeline._fov_pred_features_per_t(
        ctx, "A/1/0", predict, cells, _MeanExtractor(), _MeanExtractor(), None, None, 16, [1.0, 1.0, 1.0],
        z_slabs=slabs, cp_planes=planes,
    )  # fmt: skip
    for t, z in enumerate(planes):
        want_cp = cp_regionprops(
            predict[t, z : z + 1], cells[t, z : z + 1], [1.0, 1.0, 1.0], norm=ctx.cp_norm, glcm_cfg=ctx.cp_glcm,
            use_gpu=False,
        )  # fmt: skip
        np.testing.assert_array_equal(got["cp"][t], want_cp)
        full_cp = cp_regionprops(
            predict[t], cells[t], [1.0, 1.0, 1.0], norm=ctx.cp_norm, glcm_cfg=ctx.cp_glcm, use_gpu=False
        )
        assert not np.array_equal(got["cp"][t], full_cp)
        crops = build_crops(predict[t], cells[t], 16, z_slab=slabs[t])
        np.testing.assert_array_equal(got["dinov3"][t], _MeanExtractor().extract_features_batch(crops).numpy())


def test_microssim_calibration_pools_only_the_focus_planes(tmp_path):
    """The fit ignores the zero planes a plane-restricted store holds; without the planes it does not."""
    from dynacell.evaluation.metrics import fit_microssim

    pipeline = live_pipeline_module()
    rng = np.random.default_rng(3)
    size, depth = 192, 5
    target = gaussian_filter(rng.random((T, depth, size, size)), (0, 0, 3, 3)).astype(np.float32)
    predict = (0.8 * target + 0.05 * rng.random(target.shape)).astype(np.float32)
    planes = [2, 3]
    cut = predict.copy()
    for t, z in enumerate(planes):
        cut[t, [p for p in range(depth) if p != z]] = 0
    stores = {}
    for label, pred in (("full", predict), ("cut", cut)):
        with (
            open_ome_zarr(tmp_path / f"{label}.zarr", mode="w", layout="hcs", channel_names=["prediction"]) as p,
            open_ome_zarr(tmp_path / f"gt_{label}.zarr", mode="w", layout="hcs", channel_names=["target"]) as g,
        ):
            p.create_position("A", "1", "0").create_image("0", pred[:, None])
            g.create_position("A", "1", "0").create_image("0", target[:, None])
        stores[label] = (tmp_path / f"{label}.zarr", tmp_path / f"gt_{label}.zarr")

    io = OmegaConf.create({"pred_channel_name": "prediction", "gt_channel_name": "target"})

    def factor(label: str, focus: bool) -> float:
        with open_ome_zarr(stores[label][0], mode="r") as p, open_ome_zarr(stores[label][1], mode="r") as g:
            sim, _ = pipeline._calibrate_microssim(
                list(p.positions()), list(g.positions()), io, use_gpu=False, max_pairs=12, seed=0,
                cache_reads=False, focus_planes=(lambda *_: planes) if focus else None,
            )  # fmt: skip
        return float(sim._ri_factor)

    want = fit_microssim(
        np.stack([target[t, z] for t, z in enumerate(planes)]),
        np.stack([predict[t, z] for t, z in enumerate(planes)]),
        use_gpu=False,
    )._ri_factor
    assert factor("full", True) == factor("cut", True) == pytest.approx(want)
    assert factor("cut", False) != pytest.approx(factor("full", False))


def test_focus2d_caches_live_apart_and_refuse_another_recipe(stores, tmp_path):
    """focus2d caches sit in their own subdir, which records the plane recipe and refuses another."""
    # Imported here: live_pipeline_module() in other tests reloads dynacell.evaluation.*,
    # so a module-level StaleCacheError could be another class than the one raised.
    from dynacell.evaluation.cache import StaleCacheError, save_manifest
    from dynacell.evaluation.pipeline_cache import init_cache_context

    _, _, config = stores
    volume_ctx = init_cache_context(config("v"), side="gt")
    focus_ctx = init_cache_context(config("f", **_focus_kwargs("nucleus_area")), side="gt")
    assert volume_ctx.paths.root == tmp_path / "gt_cache"
    assert focus_ctx.paths.root == tmp_path / "gt_cache" / "focus2d_h1"
    assert focus_ctx.manifest["focus2d"]["focus_anchor"] == "nucleus_area"
    save_manifest(focus_ctx.paths, focus_ctx.manifest)
    with pytest.raises(StaleCacheError, match="another plane recipe"):
        init_cache_context(config("g", **_focus_kwargs("phase_midband")), side="gt")


def test_stamp_gates_the_final_metrics_cache_both_ways(stores, tmp_path: Path):
    """The recipe is stamped; a full-volume run refuses a focus2d cache, and the reverse."""
    _, pred_path, config = stores
    pipeline = live_pipeline_module()
    digest = prediction_sources_sha256_12(prediction_sources(pred_path, "prediction"))
    focus_kwargs = {"halfwidth": 1, "focus_anchor": "phase_midband"}

    def run(cfg):
        Path(cfg.save.save_dir).mkdir(parents=True, exist_ok=True)
        pixel, mask, feature = pipeline.evaluate_predictions(cfg)
        pipeline.save_metrics(cfg, pixel, mask, feature, cp_space=None, prediction_digest=digest)

    focus_cfg = config("focus", **focus_kwargs)
    run(focus_cfg)
    stamp = json.loads((tmp_path / "focus" / "metrics_provenance.json").read_text())
    assert stamp["focus2d"] == {
        "halfwidth": 1,
        "focus_anchor": "phase_midband",
        "focus_channel_name": "Phase3D",
        "na_det": 1.35,
        "lambda_ill": 0.45,
        "pixel_size": 1.0,
    }
    assert pipeline._final_metrics_cache_valid(focus_cfg) is True
    assert pipeline._final_metrics_cache_valid(config("focus")) is False
    assert pipeline._final_metrics_cache_valid(config("focus", **{**focus_kwargs, "halfwidth": 2})) is False

    run(config("volume"))
    assert "focus2d" not in json.loads((tmp_path / "volume" / "metrics_provenance.json").read_text())
    assert pipeline._final_metrics_cache_valid(config("volume")) is True
    assert pipeline._final_metrics_cache_valid(config("volume", **focus_kwargs)) is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"halfwidth": -1}, ">= 0"),
        ({"halfwidth": 1, "focus_anchor": "nucleus_area"}, "nuclei_channel_name"),
        ({"halfwidth": 1, "focus_anchor": "sharpest"}, "focus_anchor"),
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


def test_instance_targets_must_use_the_same_slab(stores):
    """Instance targets score the focus2d slab only through the matching 2D instance geometry."""
    from dynacell.evaluation.focus import read_focus2d_config

    _, _, config = stores
    cfg = config("instance", halfwidth=1, focus_anchor="phase_midband", slice_selection="focus")
    cfg.compute_instance_ap = True
    cfg.segmentation.focus_slab_halfwidth = 2
    with pytest.raises(ValueError, match="focus_slab_halfwidth=1"):
        read_focus2d_config(cfg)
    cfg.segmentation.focus_slab_halfwidth = 1
    assert read_focus2d_config(cfg)["halfwidth"] == 1


class _SourceModule(LightningModule):
    """Predict the source itself, so a written plane names the source plane it came from."""

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return batch["source"].clone()


def test_precompute_plane_file_drives_a_predict_the_eval_accepts(stores, tmp_path):
    """precompute-gt writes the eval's slabs; a predict restricted to them scores like the full store, and only so."""
    from dynacell.evaluation.precompute_cli import precompute_gt_artifacts

    gt_path, _, config = stores
    planes_file = tmp_path / "planes.json"
    pre = OmegaConf.merge(OmegaConf.load(_EVAL_YAML), {k: v for k, v in OmegaConf.load(_PRECOMPUTE_YAML).items()})
    del pre["defaults"]
    pre = OmegaConf.merge(
        pre,
        {
            "target_name": "er",
            "use_gpu": False,
            "io": {"gt_path": str(gt_path), "gt_channel_name": "target", "gt_cache_dir": str(tmp_path / "pre_cache")},
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
            "build": {k: False for k in ("masks", "cp", "dinov3", "dynaclr", "celldino", "morphem")},
            "focus2d": {"halfwidth": 1},
            "segmentation": {"focus_anchor": "nucleus_area", "nuclei_channel_name": "Nuclei"},
        },
    )
    pre.build.focus_planes = str(planes_file)
    precompute_gt_artifacts(pre)
    payload = json.loads(planes_file.read_text())
    assert payload["positions"] == {name: [_slab(z) for z in NUCLEUS_PLANES[name]] for name in POSITIONS}

    def predict(out: Path, planes) -> Path:
        dm = HCSDataModule(
            data_path=str(gt_path), source_channel=["target"], target_channel=["Nuclei"], z_window_size=1,
            batch_size=4, num_workers=0, yx_patch_size=[H, W], normalizations=[], augmentations=[],
            predict_z_planes=planes,
        )  # fmt: skip
        Trainer(
            accelerator="cpu", logger=False, enable_progress_bar=False, callbacks=[HCSPredictionWriter(str(out))]
        ).predict(_SourceModule(), datamodule=dm, return_predictions=False)
        return out

    full = predict(tmp_path / "full_pred.zarr", None)
    cut = predict(tmp_path / "cut_pred.zarr", planes_file)

    def eval_config(name: str, store: Path, **overrides):
        cfg = config(name, pred=store, **overrides)
        cfg.io.pred_channel_name = "Nuclei_prediction"
        return cfg

    rows = [
        live_pipeline_module().evaluate_predictions(
            eval_config(f"eval_{name}", store, **_focus_kwargs("nucleus_area"))
        )[:2]
        for name, store in (("full", full), ("cut", cut))
    ]
    for full_rows, cut_rows in zip(*rows, strict=True):
        assert _rows_by_key(full_rows) == _rows_by_key(cut_rows)
    # The prediction is the target itself, so the plane metrics are perfect.
    assert all(row["Dice"] == 1.0 for row in rows[1][1])

    with pytest.raises(ValueError, match="plane-restricted predict"):
        live_pipeline_module().evaluate_predictions(eval_config("cut3d", cut))
    # The phase anchor scores other planes than the store holds: refused, not scored on zeros.
    with pytest.raises(ValueError, match="resolve the focus plane differently"):
        live_pipeline_module().evaluate_predictions(eval_config("anchor", cut, **_focus_kwargs("phase_midband")))
