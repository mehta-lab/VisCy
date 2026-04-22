"""Hydra-side compose integration tests for ``apply_dataset_ref``.

Layer 1
    Parametrized checks that composing a real eval leaf (target /
    predict_set / leaf groups selected from the repo's ``_internal/``
    tree) and then calling :func:`apply_dataset_ref` produces the
    manifest-derived ``io.*`` and ``pixel_metrics.spacing`` values.

Layer 2
    End-to-end wiring checks that the real ``@hydra.main`` entry points
    ``evaluate_model`` and ``precompute_gt`` call the hook before the
    heavy work.

The tests replicate the external-searchpath injection that
``dynacell.__main__._inject_external_configs`` performs in production
CLI calls by passing ``hydra.searchpath`` overrides to ``compose``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from dynacell.data.manifests import load_manifest
from dynacell.evaluation._ref_hook import apply_dataset_ref

_DYNACELL_ROOT = Path(__file__).resolve().parents[1]
_INTERNAL = _DYNACELL_ROOT / "configs" / "benchmarks" / "virtual_staining" / "_internal"
_SHARED_EVAL = _INTERNAL / "shared" / "eval"
_LEAF_ROOT = _INTERNAL / "leaf"

_EXPECTED_SPACING = [0.29, 0.108, 0.108]

# (organelle, manifest-target-slug, gt_channel, gt_store_suffix, seg_store_suffix, cache_suffix)
_ORGANELLE_EXPECTATIONS: dict[str, tuple[str, str, str, str, str]] = {
    "er": ("er_sec61b", "Structure", "test_cropped/SEC61B.zarr", "SEC61B_segmented_cleaned.zarr", "eval_cache/SEC61B"),
    "mito": (
        "mito_tomm20",
        "Structure",
        "test_cropped/TOMM20.zarr",
        "TOMM20_segmented_cleaned.zarr",
        "eval_cache/TOMM20",
    ),
    "nucleus": (
        "nucleus",
        "Nuclei",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/nucleus",
    ),
    "membrane": (
        "membrane",
        "Membrane",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/membrane",
    ),
}

_LEAF_MATRIX = [(organelle, model) for organelle in _ORGANELLE_EXPECTATIONS for model in ("celldiff", "unetvit3d")]


@pytest.fixture(autouse=True)
def _clear_manifest_cache() -> Iterator[None]:
    """Clear the ``load_manifest`` LRU cache so tests don't leak state."""
    load_manifest.cache_clear()
    yield
    load_manifest.cache_clear()


@pytest.fixture(autouse=True)
def _clear_global_hydra() -> Iterator[None]:
    """Ensure Hydra's global state is pristine before and after each test."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _searchpath_override() -> str:
    """Build a ``hydra.searchpath=[...]`` override pointing at the external tree.

    Mirrors ``dynacell.__main__._inject_external_configs`` so test compose
    calls see the same ``target/``, ``leaf/``, and
    ``feature_extractor/dynaclr/`` groups that production CLI calls do.
    """
    return f"hydra.searchpath=[file://{_INTERNAL},file://{_SHARED_EVAL}]"


def _compose_eval_cfg(overrides: list[str], config_name: str = "eval") -> DictConfig:
    """Compose an eval or precompute config with the external searchpath injected."""
    with initialize_config_module(config_module="dynacell.evaluation._configs", version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=[*overrides, _searchpath_override()])
    return cfg


# -- Layer 1: compose + hook produces correct resolved values ---------------


@pytest.mark.parametrize("organelle,model", _LEAF_MATRIX)
def test_eval_leaf_composes_and_splices(organelle: str, model: str) -> None:
    """Compose a real eval leaf + call apply_dataset_ref; check manifest splicing."""
    leaf_selector = f"{organelle}/{model}/ipsc_confocal/eval__ipsc_confocal"
    leaf_symlink = _LEAF_ROOT / organelle / model / "ipsc_confocal" / "eval__ipsc_confocal.yaml"
    if not leaf_symlink.exists():
        pytest.skip(f"leaf symlink missing for {organelle}/{model}: {leaf_symlink}")

    target_slug, gt_channel, gt_suffix, seg_suffix, cache_suffix = _ORGANELLE_EXPECTATIONS[organelle]

    cfg = _compose_eval_cfg(
        [
            f"target={target_slug}",
            "predict_set=ipsc_confocal",
            f"leaf={leaf_selector}",
        ]
    )

    # The hook is *not* auto-invoked by compose alone.
    apply_dataset_ref(cfg)

    assert str(cfg.io.gt_path).endswith(gt_suffix)
    assert str(cfg.io.cell_segmentation_path).endswith(seg_suffix)
    assert str(cfg.io.gt_cache_dir).endswith(cache_suffix)
    assert cfg.io.gt_channel_name == gt_channel
    assert cfg.io.pred_channel_name == f"{gt_channel}_prediction"
    assert list(cfg.pixel_metrics.spacing) == _EXPECTED_SPACING


def test_nucleus_vs_membrane_share_store_but_differ_elsewhere() -> None:
    """Nucleus and membrane share cell.zarr but split on channel + cache_dir."""
    nuc = _compose_eval_cfg(
        [
            "target=nucleus",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            "save.save_dir=/tmp/out",
        ]
    )
    mem = _compose_eval_cfg(
        [
            "target=membrane",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            "save.save_dir=/tmp/out",
        ]
    )
    apply_dataset_ref(nuc)
    apply_dataset_ref(mem)

    assert str(nuc.io.gt_path) == str(mem.io.gt_path)
    assert str(nuc.io.gt_path).endswith("test_cropped/cell.zarr")

    assert str(nuc.io.gt_cache_dir) != str(mem.io.gt_cache_dir)
    assert str(nuc.io.gt_cache_dir).endswith("eval_cache/nucleus")
    assert str(mem.io.gt_cache_dir).endswith("eval_cache/membrane")

    assert nuc.io.gt_channel_name == "Nuclei"
    assert mem.io.gt_channel_name == "Membrane"


def test_collision_raises_with_both_paths_in_message() -> None:
    """Full ref + conflicting explicit io.gt_path raises with both paths in message."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"gt_path": "/other/path.zarr"},
        }
    )
    with pytest.raises(ValueError) as exc:
        apply_dataset_ref(cfg)
    msg = str(exc.value)
    assert "/other/path.zarr" in msg
    assert "SEC61B.zarr" in msg


# -- Layer 2: real entry points invoke the hook ----------------------------


def test_evaluate_model_wires_hook(monkeypatch, tmp_path) -> None:
    """``evaluate_model`` runs ``apply_dataset_ref`` before ``evaluate_predictions``."""
    captured: list[DictConfig] = []

    def _fake_evaluate_predictions(cfg: DictConfig):
        captured.append(cfg)
        return ([], [], [])

    def _fake_save_metrics(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("dynacell.evaluation.pipeline.evaluate_predictions", _fake_evaluate_predictions)
    monkeypatch.setattr("dynacell.evaluation.pipeline.save_metrics", _fake_save_metrics)

    cfg = _compose_eval_cfg(
        [
            "target=er_sec61b",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            f"save.save_dir={tmp_path}",
        ]
    )

    from dynacell.evaluation.pipeline import evaluate_model

    evaluate_model.__wrapped__(cfg)

    assert len(captured) == 1
    spliced = captured[0]
    assert str(spliced.io.gt_path).endswith("test_cropped/SEC61B.zarr")
    assert str(spliced.io.cell_segmentation_path).endswith("SEC61B_segmented_cleaned.zarr")
    assert spliced.io.gt_channel_name == "Structure"
    assert spliced.io.pred_channel_name == "Structure_prediction"
    assert list(spliced.pixel_metrics.spacing) == _EXPECTED_SPACING


def test_precompute_gt_wires_hook(monkeypatch, tmp_path) -> None:
    """``precompute_gt`` runs ``apply_dataset_ref`` before ``precompute_gt_artifacts``."""
    captured: list[DictConfig] = []

    def _fake_precompute_gt_artifacts(cfg: DictConfig) -> None:
        captured.append(cfg)

    monkeypatch.setattr(
        "dynacell.evaluation.precompute_cli.precompute_gt_artifacts",
        _fake_precompute_gt_artifacts,
    )

    cfg = _compose_eval_cfg(
        [
            "target=er_sec61b",
            "predict_set=ipsc_confocal",
            "io.pred_path=/tmp/fake",
            f"save.save_dir={tmp_path}",
        ]
    )

    from dynacell.evaluation.precompute_cli import precompute_gt

    precompute_gt.__wrapped__(cfg)

    assert len(captured) == 1
    spliced = captured[0]
    assert str(spliced.io.gt_path).endswith("test_cropped/SEC61B.zarr")
    assert spliced.io.gt_channel_name == "Structure"
    assert spliced.io.pred_channel_name == "Structure_prediction"
    assert list(spliced.pixel_metrics.spacing) == _EXPECTED_SPACING
