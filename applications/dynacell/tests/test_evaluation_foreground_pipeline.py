"""``pixel_metrics.foreground`` through the eval pipeline: default-off identity, columns, stamp, cache gate.

Runs the cache-only fixture end to end (``evaluate_predictions`` + ``save_metrics``)
with the real metrics module, so the ``FG_*`` columns come from the production code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dynacell.evaluation.cache import prediction_sources, prediction_sources_sha256_12

from ._eval_fixtures import build_eval_config, build_fixture, live_pipeline_module

pytest.importorskip("cubic.metrics")

_FG_COLUMNS = ("FG_PCC", "FG_SI_SSIM", "FG_SI_NRMSE", "FG_SI_PSNR", "FG_frac")


def _run(pipeline, config, pred_path: Path) -> list[dict]:
    """Score the fixture and save it the way ``evaluate_model`` does; return the pixel rows."""
    Path(config.save.save_dir).mkdir(parents=True, exist_ok=True)
    pixel, mask, feature = pipeline.evaluate_predictions(config)
    digest = prediction_sources_sha256_12(prediction_sources(pred_path, "prediction"))
    pipeline.save_metrics(config, pixel, mask, feature, cp_space=None, prediction_digest=digest)
    return pixel


@pytest.fixture
def scored(tmp_path: Path):
    """Fixture stores plus a ``config(save_name, foreground=None)`` factory."""
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = build_fixture(tmp_path / "fixture")

    def config(save_name: str, foreground: dict | None = None):
        cfg = build_eval_config(
            pred_path, gt_path, gt_cache_dir, pred_cache_dir, tmp_path / save_name, executor="serial", fov_workers=1
        )
        if foreground is not None:
            OmegaConf.update(cfg, "pixel_metrics.foreground", foreground, merge=True)
        return cfg

    return pred_path, config


def test_foreground_settings_resolution():
    """Absent or disabled -> None; null sigmas take the target default; explicit values win."""
    pipeline = live_pipeline_module()
    resolve = pipeline._foreground_settings
    base = {"target_name": "er", "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]}}
    assert resolve(OmegaConf.create(base)) is None
    assert resolve(OmegaConf.create({**base, "pixel_metrics": {"foreground": {"enabled": False}}})) is None

    enabled = {"enabled": True, "source": "smooth_otsu", "smooth_sigma_um": None, "feather_sigma_um": None}
    cfg = OmegaConf.create({**base, "pixel_metrics": {"foreground": enabled}})
    assert resolve(cfg) == {"source": "smooth_otsu", "smooth_sigma_um": 1.0, "feather_sigma_um": 0.5}
    cfg.target_name = "nucleus"
    assert resolve(cfg)["smooth_sigma_um"] == 0.5
    cfg.pixel_metrics.foreground.smooth_sigma_um = 2
    assert resolve(cfg) == {"source": "smooth_otsu", "smooth_sigma_um": 2.0, "feather_sigma_um": 0.5}
    cfg.target_name = "lysosomes"
    with pytest.raises(ValueError, match="feather_sigma_um is null"):
        resolve(cfg)


def test_otsu_source_resolves_no_smoothing_sigma():
    """``otsu`` ignores ``smooth_sigma_um``: it is neither resolved, stamped, nor required to have a default."""
    resolve = live_pipeline_module()._foreground_settings
    fg = {"enabled": True, "source": "otsu", "smooth_sigma_um": None, "feather_sigma_um": None}
    cfg = OmegaConf.create({"target_name": "er", "pixel_metrics": {"spacing": [1.0, 1.0, 1.0], "foreground": fg}})
    assert resolve(cfg) == {"source": "otsu", "feather_sigma_um": 0.5}
    cfg.pixel_metrics.foreground.smooth_sigma_um = 3.0
    assert resolve(cfg) == {"source": "otsu", "feather_sigma_um": 0.5}
    cfg.target_name = "lysosomes"
    cfg.pixel_metrics.foreground.feather_sigma_um = 0.25
    assert resolve(cfg) == {"source": "otsu", "feather_sigma_um": 0.25}


def test_default_off_is_byte_identical_and_unstamped(scored, tmp_path: Path):
    """No block and ``enabled: false`` write the same CSV, no FG_* column and no stamp key."""
    pred_path, config = scored
    pipeline = live_pipeline_module()
    rows = _run(pipeline, config("absent"), pred_path)
    _run(pipeline, config("disabled", {"enabled": False, "smooth_sigma_um": 3.0}), pred_path)
    assert not any(key.startswith("FG_") for key in rows[0])
    for name in ("pixel_metrics.csv", "metrics_provenance.json"):
        assert (tmp_path / "absent" / name).read_bytes() == (tmp_path / "disabled" / name).read_bytes(), name
    assert "pixel_foreground" not in json.loads((tmp_path / "absent" / "metrics_provenance.json").read_text())


def test_foreground_on_adds_columns_and_gates_the_cache(scored, tmp_path: Path):
    """On: FG_* columns appended (base columns untouched), recipe stamped, and a recipe change recomputes."""
    pred_path, config = scored
    pipeline = live_pipeline_module()
    off_rows = _run(pipeline, config("off"), pred_path)
    on_cfg = config("on", {"enabled": True, "source": "smooth_otsu", "smooth_sigma_um": None, "feather_sigma_um": 0.5})
    on_rows = _run(pipeline, on_cfg, pred_path)

    for off, on in zip(off_rows, on_rows, strict=True):
        assert list(on) == [*off, *_FG_COLUMNS]
        assert all(on[k] == off[k] for k in off)
        assert 0.0 < on["FG_frac"] < 1.0 and np.isfinite(on["FG_SI_SSIM"])

    stamp = json.loads((tmp_path / "on" / "metrics_provenance.json").read_text())
    assert stamp["pixel_foreground"] == {"source": "smooth_otsu", "smooth_sigma_um": 1.0, "feather_sigma_um": 0.5}

    assert pipeline._final_metrics_cache_valid(on_cfg) is True
    on_cfg.pixel_metrics.foreground.feather_sigma_um = 1.0
    assert pipeline._final_metrics_cache_valid(on_cfg) is False  # another recipe -> recompute
    on_cfg.pixel_metrics.foreground.feather_sigma_um = 0.5
    on_cfg.pixel_metrics.foreground.enabled = False
    assert pipeline._final_metrics_cache_valid(on_cfg) is False  # FG rows, run without FG -> recompute

    # A cache scored without FG must not satisfy a run that asks for it.
    off_cfg = config("off", {"enabled": True, "smooth_sigma_um": 1.0, "feather_sigma_um": 0.5})
    assert pipeline._final_metrics_cache_valid(off_cfg) is False
    off_cfg.pixel_metrics.foreground.enabled = False
    assert pipeline._final_metrics_cache_valid(off_cfg) is True


def test_stamped_recipe_without_columns_is_recomputed(scored, tmp_path: Path):
    """A stamp naming a recipe whose rows lack the FG_* columns is not reusable."""
    pred_path, config = scored
    pipeline = live_pipeline_module()
    cfg = config("partial", {"enabled": True, "smooth_sigma_um": 1.0, "feather_sigma_um": 0.5})
    rows = _run(pipeline, cfg, pred_path)
    assert pipeline._final_metrics_cache_valid(cfg) is True
    stripped = [{k: v for k, v in row.items() if not k.startswith("FG_")} for row in rows]
    np.save(tmp_path / "partial" / "pixel_metrics.npy", stripped)
    assert pipeline._final_metrics_cache_valid(cfg) is False
