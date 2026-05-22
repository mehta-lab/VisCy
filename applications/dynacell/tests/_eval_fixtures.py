"""Shared cache-only fixture helpers for dynacell eval integration tests.

Underscore prefix keeps pytest from collecting this file as tests. Imported
by ``test_evaluation_pipeline_parallel_cpu.py`` and
``test_evaluation_grouped.py`` (and any future eval test that wants the
same scaffolding) — see CLAUDE.md "no ``sys.path`` edits" rule.

Builds tiny iohub HCS plates + pre-populated mask caches so the
cache-only eval path (``target_name=er`` + ``require_complete_cache=true``
+ ``compute_feature_metrics=false``) runs end-to-end without loading any
segmenter / extractor / cubic — small enough to fit a single-CPU CI in
under ~30 s.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf

N_POSITIONS = 3
T = 2
D = 4
H = 32
W = 32


def live_pipeline_module():
    """Force-import a fresh ``dynacell.evaluation.pipeline`` module.

    Other tests (e.g. ``test_evaluation_pipeline.py``) stub
    ``dynacell.evaluation.metrics`` via ``monkeypatch.setitem(sys.modules,
    ...)``. After teardown ``sys.modules`` may still hold the stubbed
    modules AND the pipeline module's bound ``compute_pixel_metrics``
    reference. Clear every ``dynacell.evaluation.*`` cache entry before
    re-import so we get fresh modules with real implementations.
    """
    for name in list(sys.modules):
        if name.startswith("dynacell.evaluation"):
            sys.modules.pop(name, None)
    return importlib.import_module("dynacell.evaluation.pipeline")


def make_hcs_plate(path: Path, channel_name: str, seed: int) -> None:
    """Create a tiny HCS plate with ``N_POSITIONS`` positions of shape ``(T, 1, D, H, W)``.

    Positions are named ``A/1/{i}`` for ``i in range(N_POSITIONS)``. Pixel
    values are deterministic from ``seed`` so pred and GT differ
    predictably.
    """
    rng = np.random.default_rng(seed)
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel_name], version="0.5") as plate:
        for i in range(N_POSITIONS):
            pos = plate.create_position("A", "1", str(i))
            data = rng.uniform(0.0, 1.0, size=(T, 1, D, H, W)).astype(np.float32)
            pos.create_image("0", data)


def make_mask_cache(
    cache_dir: Path,
    plate_path: Path,
    channel_name: str,
    side: str,
    target_name: str = "er",
) -> None:
    """Prebuild a mask cache (zarr + manifest) for ``N_POSITIONS`` positions.

    Mask values are deterministic per (side, position, t): GT masks fill
    the upper half of the (H, W) plane; pred masks fill the upper
    quadrant. This makes DICE / IoU deterministic and non-trivially
    different between sides. ``side`` is ``"gt"`` or ``"pred"``.
    """
    from dynacell.evaluation.cache import (
        CACHE_SCHEMA_VERSION,
        cache_paths,
        save_manifest,
        write_mask,
    )

    paths = cache_paths(cache_dir)
    mask_channel = "target_seg" if side == "gt" else "prediction_seg"

    pos_names = [f"A/1/{i}" for i in range(N_POSITIONS)]
    for pos_name in pos_names:
        masks = np.zeros((T, D, H, W), dtype=bool)
        if side == "gt":
            masks[:, :, : H // 2, :] = True
        else:
            masks[:, :, : H // 2, : W // 2] = True
        write_mask(paths, target_name, pos_name, masks, channel_name=mask_channel)

    source_tag: dict[str, str] = {} if side == "gt" else {"source": "prediction"}
    now = datetime.now(UTC).isoformat()
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "gt" if side == "gt" else "pred": {
            "plate_path": str(plate_path),
            "channel_name": channel_name,
        },
        "artifacts": {
            "organelle_masks": {
                target_name: {
                    "target_name": target_name,
                    "path": f"organelle_masks/{target_name}.zarr",
                    "built_at": now,
                    "positions": pos_names,
                    **source_tag,
                }
            },
        },
    }
    save_manifest(paths, manifest)


def build_eval_config(
    pred_path: Path,
    gt_path: Path,
    gt_cache_dir: Path,
    pred_cache_dir: Path,
    save_dir: Path,
    *,
    executor: str,
    fov_workers: int,
):
    """Compose a minimal cache-only eval config."""
    return OmegaConf.create(
        {
            "target_name": "er",
            "io": {
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "pred_channel_name": "prediction",
                "gt_channel_name": "target",
                "cell_segmentation_path": None,
                "gt_cache_dir": str(gt_cache_dir),
                "pred_cache_dir": str(pred_cache_dir),
                "require_complete_cache": True,
            },
            "pixel_metrics": {
                "spacing": [1.0, 1.0, 1.0],
                "fsc": None,
                "spectral_pcc": None,
            },
            "feature_metrics": {
                "patch_size": 64,
                "deep_feature_batch_threshold": 256,
            },
            "use_gpu": False,
            "compute_microssim": False,
            "compute_feature_metrics": False,
            "limit_positions": None,
            "force_recompute": {
                "all": False,
                "gt_masks": False,
                "gt_cp": False,
                "gt_dinov3": False,
                "gt_dynaclr": False,
                "gt_celldino": False,
                "pred_masks": False,
                "pred_cp": False,
                "pred_dinov3": False,
                "pred_dynaclr": False,
                "pred_celldino": False,
                "final_metrics": False,
            },
            "save": {
                "save_dir": str(save_dir),
                "pixel_csv_filename": "pixel_metrics.csv",
                "pixel_metrics_filename": "pixel_metrics.npy",
                "mask_csv_filename": "mask_metrics.csv",
                "mask_metrics_filename": "mask_metrics.npy",
                "feature_csv_filename": "feature_metrics.csv",
                "feature_metrics_filename": "feature_metrics.npy",
            },
            "runtime": {
                "fov_workers": fov_workers,
                "threads_per_worker": "auto",
                "executor": executor,
                "cuda_empty_cache_every_n_timepoints": 0,
                "gc_collect_every_n_fovs": 0,
            },
        }
    )


def build_fixture(root: Path):
    """Build matched pred + GT plates and mask caches under ``root``.

    Returns ``(pred_path, gt_path, gt_cache_dir, pred_cache_dir)``.
    """
    pred_path = root / "pred.zarr"
    gt_path = root / "gt.zarr"
    gt_cache_dir = root / "gt_cache"
    pred_cache_dir = root / "pred_cache"
    make_hcs_plate(pred_path, "prediction", seed=42)
    make_hcs_plate(gt_path, "target", seed=7)
    make_mask_cache(gt_cache_dir, gt_path, "target", side="gt")
    make_mask_cache(pred_cache_dir, pred_path, "prediction", side="pred")
    return pred_path, gt_path, gt_cache_dir, pred_cache_dir


def read_position_arrays(plate_path: Path) -> dict[str, np.ndarray]:
    """Return ``{pos_name: data}`` for every position in an HCS plate."""
    out: dict[str, np.ndarray] = {}
    with open_ome_zarr(plate_path, mode="r") as plate:
        for name, pos in plate.positions():
            out[name] = np.asarray(pos.data[:])
    return out
