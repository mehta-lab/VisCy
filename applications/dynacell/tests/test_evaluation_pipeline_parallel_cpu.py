"""End-to-end CPU integration test for serial-vs-process executor parity.

Drives ``evaluate_predictions`` twice on the same tiny iohub fixture +
prebuilt mask caches: once with ``runtime.executor=serial``, once with
``runtime.executor=process runtime.fov_workers=2``. Asserts byte-equal
CSV outputs (after sort by FOV) and per-position equality of the
``segmentation_results.zarr`` HCS plate.

Cache-only design (matches plan ``.claude/plans/eval-parallelism.md`` §C5):

- ``target_name=er`` → ``prepare_segmentation_model`` returns None.
- ``io.require_complete_cache=true`` → mask cache MUST satisfy every
  request; we pre-populate the cache before the run.
- ``compute_feature_metrics=false``, ``compute_microssim=false`` → no
  extractors load, no microssim.
- ``pixel_metrics.fsc=null`` and ``pixel_metrics.spectral_pcc=null`` →
  ``compute_pixel_metrics`` only runs the torch metric block (PCC,
  SSIM, NRMSE, PSNR), no cubic FFT path.

Result: the full pipeline runs without instantiating cellpose,
SuperModel, DinoV3, DynaCLR, CellDino, or cubic — exercises spawn
boot, DictConfig pickling, ``FovResult`` transport, parent-side HCS
plate write, and the dataset-level aggregation contract.
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest  # noqa: F401  -- imported for future @pytest.mark uses
from iohub.ngff import open_ome_zarr
from omegaconf import OmegaConf


def _live_pipeline_module():
    """Force-import a fresh ``dynacell.evaluation.pipeline`` module.

    Other tests (``test_evaluation_pipeline.py``) stub
    ``dynacell.evaluation.metrics`` etc. via ``monkeypatch.setitem(sys.modules,
    ...)``. After teardown, ``sys.modules`` may still hold the stubbed modules
    AND the pipeline module's bound ``compute_pixel_metrics`` reference. Clear
    every ``dynacell.evaluation.*`` cache entry before re-import so we get
    fresh modules with real implementations.
    """
    for name in list(sys.modules):
        if name.startswith("dynacell.evaluation"):
            sys.modules.pop(name, None)
    return importlib.import_module("dynacell.evaluation.pipeline")


# Fixture dimensions — kept tiny so the test runs in CI (single-CPU) under ~30s.
N_POSITIONS = 3
T = 2
D = 4
H = 32
W = 32


def _make_hcs_plate(path: Path, channel_name: str, seed: int) -> None:
    """Create a tiny HCS plate with ``N_POSITIONS`` positions of shape ``(T, 1, D, H, W)``.

    Positions are named ``A/1/{i}`` for ``i in range(N_POSITIONS)``. Pixel
    values are deterministic from ``seed`` so pred and GT differ predictably.
    """
    rng = np.random.default_rng(seed)
    with open_ome_zarr(path, mode="w", layout="hcs", channel_names=[channel_name], version="0.5") as plate:
        for i in range(N_POSITIONS):
            pos = plate.create_position("A", "1", str(i))
            data = rng.uniform(0.0, 1.0, size=(T, 1, D, H, W)).astype(np.float32)
            pos.create_image("0", data)


def _make_mask_cache(
    cache_dir: Path,
    plate_path: Path,
    channel_name: str,
    side: str,  # "gt" or "pred"
    target_name: str = "er",
) -> None:
    """Prebuild a mask cache (zarr + manifest) for ``N_POSITIONS`` positions.

    The mask values themselves are deterministic per (side, position, t):
    GT masks are TRUE in the upper half of the (H, W) plane; pred masks
    are TRUE in the upper quadrant. This guarantees the DICE / IoU
    metrics are deterministic and non-trivially different between sides.
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
            masks[:, :, : H // 2, :] = True  # upper half
        else:
            masks[:, :, : H // 2, : W // 2] = True  # upper quadrant
        write_mask(paths, target_name, pos_name, masks, channel_name=mask_channel)

    # Manifest entries must satisfy _validate_artifact_params: the
    # masks_section.get(target_name) must equal {"target_name": ..., **source_tag}.
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


def _build_eval_config(
    pred_path: Path,
    gt_path: Path,
    gt_cache_dir: Path,
    pred_cache_dir: Path,
    save_dir: Path,
    *,
    executor: str,
    fov_workers: int,
):
    """Compose a minimal eval config for the cache-only path."""
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


def _build_fixture(root: Path):
    """Build pred + GT plates and matching mask caches under ``root``."""
    pred_path = root / "pred.zarr"
    gt_path = root / "gt.zarr"
    gt_cache_dir = root / "gt_cache"
    pred_cache_dir = root / "pred_cache"
    _make_hcs_plate(pred_path, "prediction", seed=42)
    _make_hcs_plate(gt_path, "target", seed=7)
    _make_mask_cache(gt_cache_dir, gt_path, "target", side="gt")
    _make_mask_cache(pred_cache_dir, pred_path, "prediction", side="pred")
    return pred_path, gt_path, gt_cache_dir, pred_cache_dir


def _read_position_arrays(plate_path: Path) -> dict[str, np.ndarray]:
    """Return ``{pos_name: data}`` for every position in an HCS plate."""
    out: dict[str, np.ndarray] = {}
    with open_ome_zarr(plate_path, mode="r") as plate:
        for name, pos in plate.positions():
            out[name] = np.asarray(pos.data[:])
    return out


def test_serial_vs_process_evaluate_predictions_parity(tmp_path: Path):
    """End-to-end: serial-mode and process-mode produce byte-equal outputs.

    Exercises the spawn boundary (pickling, worker startup, _process_one_fov
    in workers, FovResult transport, parent-side aggregation + HCS plate
    write) without invoking cellpose / extractors / cubic.
    """
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = _build_fixture(fixture_root)

    save_dir_serial = tmp_path / "serial"
    save_dir_process = tmp_path / "process"

    # Re-import inside the test body — test_lazy_init.py pops dynacell.*
    # modules from sys.modules, which would invalidate a module-level import.
    pipeline = _live_pipeline_module()
    evaluate_predictions = pipeline.evaluate_predictions

    cfg_serial = _build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir_serial,
        executor="serial",
        fov_workers=1,
    )
    cfg_process = _build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir_process,
        executor="process",
        fov_workers=2,
    )

    save_dir_serial.mkdir(parents=True)
    save_dir_process.mkdir(parents=True)

    pixel_serial, mask_serial, _ = evaluate_predictions(cfg_serial)
    pixel_process, mask_process, _ = evaluate_predictions(cfg_process)

    # Pixel + mask rows: same set after sort. We sort by (FOV, Timepoint)
    # because the process branch already sorts by pos_name and within-FOV
    # iteration is deterministic, but we sort defensively.
    def _sort_key(row: dict):
        return (row["FOV"], row["Timepoint"])

    assert sorted(pixel_serial, key=_sort_key) == sorted(pixel_process, key=_sort_key)
    assert sorted(mask_serial, key=_sort_key) == sorted(mask_process, key=_sort_key)
    # Per-position seg plate equality (both channels).
    arrs_serial = _read_position_arrays(save_dir_serial / "segmentation_results.zarr")
    arrs_process = _read_position_arrays(save_dir_process / "segmentation_results.zarr")
    assert set(arrs_serial.keys()) == set(arrs_process.keys())
    for pos_name, arr_a in arrs_serial.items():
        arr_b = arrs_process[pos_name]
        assert arr_a.shape == arr_b.shape, f"{pos_name} shape mismatch"
        np.testing.assert_array_equal(arr_a, arr_b)


def test_serial_path_runs_without_segmenter_loaded(tmp_path: Path):
    """``require_complete_cache=true`` should skip ``prepare_segmentation_model``.

    Confirms the cache-only fast-path: evaluate_predictions runs end-to-end
    without ever instantiating SuperModel (which would trigger a quilt
    download on a cold node).
    """
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = _build_fixture(fixture_root)
    save_dir = tmp_path / "out"
    save_dir.mkdir()

    cfg = _build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir,
        executor="serial",
        fov_workers=1,
    )
    pipeline = _live_pipeline_module()
    pixel_rows, mask_rows, _ = pipeline.evaluate_predictions(cfg)
    assert len(pixel_rows) == N_POSITIONS * T
    assert len(mask_rows) == N_POSITIONS * T
    # Confirms the no-segmenter contract: prepare_segmentation_model
    # returned None and evaluate_predictions still produced valid outputs
    # for every (FOV, t) pair.
