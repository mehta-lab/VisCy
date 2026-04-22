"""Unit tests for ``dynacell.evaluation._ref_hook.apply_dataset_ref``."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dynacell.evaluation._ref_hook import apply_dataset_ref

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manifests"

_EXPECTED_SPACING = [0.29, 0.108, 0.108]

_TARGET_EXPECTATIONS = [
    (
        "sec61b",
        "Structure",
        "Structure_prediction",
        "test_cropped/SEC61B.zarr",
        "SEC61B_segmented_cleaned.zarr",
        "eval_cache/SEC61B",
    ),
    (
        "tomm20",
        "Structure",
        "Structure_prediction",
        "test_cropped/TOMM20.zarr",
        "TOMM20_segmented_cleaned.zarr",
        "eval_cache/TOMM20",
    ),
    (
        "nucleus",
        "Nuclei",
        "Nuclei_prediction",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/nucleus",
    ),
    (
        "membrane",
        "Membrane",
        "Membrane_prediction",
        "test_cropped/cell.zarr",
        "cell_segmented_cleaned.zarr",
        "eval_cache/membrane",
    ),
]


@pytest.mark.parametrize(
    "target,target_channel,pred_channel,gt_suffix,seg_suffix,cache_suffix",
    _TARGET_EXPECTATIONS,
)
def test_full_ref_happy_path_all_targets(
    target: str,
    target_channel: str,
    pred_channel: str,
    gt_suffix: str,
    seg_suffix: str,
    cache_suffix: str,
) -> None:
    """Full ref splices io.* and pixel_metrics.spacing for every fixture target."""
    cfg = OmegaConf.create({"benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": target}}})
    apply_dataset_ref(cfg)
    assert str(cfg.io.gt_path).endswith(gt_suffix)
    assert str(cfg.io.cell_segmentation_path).endswith(seg_suffix)
    assert str(cfg.io.gt_cache_dir).endswith(cache_suffix)
    assert cfg.io.gt_channel_name == target_channel
    assert cfg.io.pred_channel_name == pred_channel
    assert list(cfg.pixel_metrics.spacing) == _EXPECTED_SPACING


def test_partial_ref_dataset_only_is_noop() -> None:
    """Ref with only ``dataset`` key is a no-op."""
    cfg = OmegaConf.create({"benchmark": {"dataset_ref": {"dataset": "aics-hipsc"}}})
    before = OmegaConf.to_yaml(cfg)
    apply_dataset_ref(cfg)
    assert OmegaConf.to_yaml(cfg) == before


def test_partial_ref_target_only_is_noop() -> None:
    """Ref with only ``target`` key is a no-op."""
    cfg = OmegaConf.create({"benchmark": {"dataset_ref": {"target": "sec61b"}}})
    before = OmegaConf.to_yaml(cfg)
    apply_dataset_ref(cfg)
    assert OmegaConf.to_yaml(cfg) == before


def test_no_benchmark_key_is_noop() -> None:
    """Config without a ``benchmark`` key is a no-op."""
    cfg = OmegaConf.create({"something": "else"})
    before = OmegaConf.to_yaml(cfg)
    apply_dataset_ref(cfg)
    assert OmegaConf.to_yaml(cfg) == before


def test_null_benchmark_is_noop() -> None:
    """``benchmark: null`` placeholder is a no-op."""
    cfg = OmegaConf.create({"benchmark": None})
    before = OmegaConf.to_yaml(cfg)
    apply_dataset_ref(cfg)
    assert OmegaConf.to_yaml(cfg) == before


def test_dataset_present_but_null_raises() -> None:
    """Both keys present but ``dataset: null`` raises ValueError via pydantic."""
    cfg = OmegaConf.create({"benchmark": {"dataset_ref": {"dataset": None, "target": "sec61b"}}})
    with pytest.raises(ValueError, match="Invalid benchmark.dataset_ref"):
        apply_dataset_ref(cfg)


def test_collision_gt_path_differs() -> None:
    """Explicit ``io.gt_path`` disagreeing with manifest raises ValueError."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"gt_path": "/other/path.zarr"},
        }
    )
    with pytest.raises(ValueError, match="conflicts with explicit fields"):
        apply_dataset_ref(cfg)


def test_collision_spacing_differs() -> None:
    """Explicit ``pixel_metrics.spacing`` disagreeing with manifest raises."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0]},
        }
    )
    with pytest.raises(ValueError, match="conflicts with explicit fields"):
        apply_dataset_ref(cfg)


def test_agreement_spacing_matches_manifest() -> None:
    """Explicit spacing matching the manifest is not a collision."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "pixel_metrics": {"spacing": list(_EXPECTED_SPACING)},
        }
    )
    apply_dataset_ref(cfg)
    assert list(cfg.pixel_metrics.spacing) == _EXPECTED_SPACING
    assert str(cfg.io.gt_path).endswith("test_cropped/SEC61B.zarr")


def test_missing_field_treated_as_unset() -> None:
    """OmegaConf ``???`` in io.gt_path is treated as unset (no collision)."""
    cfg = OmegaConf.create(
        """
benchmark:
  dataset_ref:
    dataset: aics-hipsc
    target: sec61b
io:
  gt_path: ???
"""
    )
    apply_dataset_ref(cfg)
    assert str(cfg.io.gt_path).endswith("test_cropped/SEC61B.zarr")


def test_explicit_pred_channel_agrees() -> None:
    """Explicit ``io.pred_channel_name`` matching the derived value is fine."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"pred_channel_name": "Structure_prediction"},
        }
    )
    apply_dataset_ref(cfg)
    assert cfg.io.pred_channel_name == "Structure_prediction"


def test_explicit_pred_channel_disagrees() -> None:
    """Explicit ``io.pred_channel_name`` disagreeing with derived value raises."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"pred_channel_name": "Something_else"},
        }
    )
    with pytest.raises(ValueError, match="conflicts with explicit fields"):
        apply_dataset_ref(cfg)


def test_explicit_null_is_not_collision() -> None:
    """Explicit ``io.cell_segmentation_path: null`` is treated as unset."""
    cfg = OmegaConf.create(
        {
            "benchmark": {"dataset_ref": {"dataset": "aics-hipsc", "target": "sec61b"}},
            "io": {"cell_segmentation_path": None},
        }
    )
    apply_dataset_ref(cfg)
    assert str(cfg.io.cell_segmentation_path).endswith("SEC61B_segmented_cleaned.zarr")
