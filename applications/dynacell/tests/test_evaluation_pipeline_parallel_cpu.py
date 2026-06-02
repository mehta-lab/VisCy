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

Fixture-building helpers live in ``_eval_fixtures.py`` (shared with
``test_evaluation_grouped.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ._eval_fixtures import (
    N_POSITIONS,
    T,
    build_eval_config,
    build_fixture,
    live_pipeline_module,
    make_hcs_plate,
    make_mask_cache,
    read_position_arrays,
)


def test_serial_vs_process_evaluate_predictions_parity(tmp_path: Path):
    """End-to-end: serial-mode and process-mode produce byte-equal outputs.

    Exercises the spawn boundary (pickling, worker startup, _process_one_fov
    in workers, FovResult transport, parent-side aggregation + HCS plate
    write) without invoking cellpose / extractors / cubic.
    """
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = build_fixture(fixture_root)

    save_dir_serial = tmp_path / "serial"
    save_dir_process = tmp_path / "process"

    # Re-import inside the test body — test_lazy_init.py pops dynacell.*
    # modules from sys.modules, which would invalidate a module-level import.
    pipeline = live_pipeline_module()
    evaluate_predictions = pipeline.evaluate_predictions

    cfg_serial = build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir_serial,
        executor="serial",
        fov_workers=1,
    )
    cfg_process = build_eval_config(
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

    def _sort_key(row: dict):
        return (row["FOV"], row["Timepoint"])

    assert sorted(pixel_serial, key=_sort_key) == sorted(pixel_process, key=_sort_key)
    assert sorted(mask_serial, key=_sort_key) == sorted(mask_process, key=_sort_key)
    arrs_serial = read_position_arrays(save_dir_serial / "segmentation_results.zarr")
    arrs_process = read_position_arrays(save_dir_process / "segmentation_results.zarr")
    assert set(arrs_serial.keys()) == set(arrs_process.keys())
    for pos_name, arr_a in arrs_serial.items():
        arr_b = arrs_process[pos_name]
        assert arr_a.shape == arr_b.shape, f"{pos_name} shape mismatch"
        np.testing.assert_array_equal(arr_a, arr_b)


def _build_partial_pred_fixture(root: Path, n_pred: int):
    """Build a fixture where pred has only the first ``n_pred`` positions of GT.

    Returns ``(pred_path, gt_path, gt_cache_dir, pred_cache_dir)``. GT has
    the full ``N_POSITIONS`` positions + cache; pred has ``n_pred``
    positions + a matching pred cache. Mirrors the ``--fast_dev_run``
    predict scenario where the predict writer only flushed the first N
    FOVs before exiting.
    """
    pred_path = root / "pred.zarr"
    gt_path = root / "gt.zarr"
    gt_cache_dir = root / "gt_cache"
    pred_cache_dir = root / "pred_cache"
    make_hcs_plate(pred_path, "prediction", seed=42, n_positions=n_pred)
    make_hcs_plate(gt_path, "target", seed=7, n_positions=N_POSITIONS)
    make_mask_cache(gt_cache_dir, gt_path, "target", side="gt", n_positions=N_POSITIONS)
    make_mask_cache(pred_cache_dir, pred_path, "prediction", side="pred", n_positions=n_pred)
    return pred_path, gt_path, gt_cache_dir, pred_cache_dir


def test_limit_positions_with_partial_pred_zarr(tmp_path: Path):
    """``limit_positions=N`` should accept a pred zarr with only N positions.

    The eval loop computes metrics over the pred positions paired against
    the matching-by-name positions in GT (which has more). Without the
    subset-alignment fix in evaluate_predictions, the unconditional
    ``pred=N != gt=M`` count check would raise before limit_positions had
    a chance to truncate.
    """
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = _build_partial_pred_fixture(fixture_root, n_pred=1)
    save_dir = tmp_path / "out"
    save_dir.mkdir()

    cfg = build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir,
        executor="serial",
        fov_workers=1,
    )
    cfg.limit_positions = 1
    pipeline = live_pipeline_module()
    pixel_rows, mask_rows, _ = pipeline.evaluate_predictions(cfg)
    assert len(pixel_rows) == 1 * T
    assert len(mask_rows) == 1 * T


def test_limit_positions_rejects_pred_with_unknown_position(tmp_path: Path):
    """``limit_positions`` only relaxes count strict equality when pred ⊂ gt.

    If pred carries a position name absent from gt, the relaxed pairing
    would silently drop it; raise loudly instead so a misaligned smoke
    cannot fail open.
    """
    import numpy as np
    from iohub.ngff import open_ome_zarr

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()

    # Build a pred plate with one in-GT position (A/1/0) plus an orphan
    # position (A/1/999) that doesn't exist on the GT plate, then a GT
    # plate with the standard N_POSITIONS positions. Mask caches are built
    # against the orphan-containing pred and the standard GT so cache
    # manifest plate_paths match.
    pred_path = fixture_root / "pred.zarr"
    gt_path = fixture_root / "gt.zarr"
    gt_cache_dir = fixture_root / "gt_cache"
    pred_cache_dir = fixture_root / "pred_cache"

    make_hcs_plate(pred_path, "prediction", seed=42, n_positions=1)
    with open_ome_zarr(pred_path, mode="r+") as plate:
        pos = plate.create_position("A", "1", "999")
        pos.create_image("0", np.zeros((T, 1, 4, 32, 32), dtype=np.float32))
    make_hcs_plate(gt_path, "target", seed=7, n_positions=N_POSITIONS)
    make_mask_cache(gt_cache_dir, gt_path, "target", side="gt", n_positions=N_POSITIONS)
    # Pred cache covers only the in-GT position (orphan never gets to per-FOV).
    make_mask_cache(pred_cache_dir, pred_path, "prediction", side="pred", n_positions=1)

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    cfg = build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir,
        executor="serial",
        fov_workers=1,
    )
    cfg.limit_positions = 2
    pipeline = live_pipeline_module()
    with pytest.raises(ValueError, match="pred contains positions not present in gt"):
        pipeline.evaluate_predictions(cfg)


def test_serial_path_runs_without_segmenter_loaded(tmp_path: Path):
    """``require_complete_cache=true`` should skip ``prepare_segmentation_model``.

    Confirms the cache-only fast-path: evaluate_predictions runs end-to-end
    without ever instantiating SuperModel (which would trigger a quilt
    download on a cold node).
    """
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    pred_path, gt_path, gt_cache_dir, pred_cache_dir = build_fixture(fixture_root)
    save_dir = tmp_path / "out"
    save_dir.mkdir()

    cfg = build_eval_config(
        pred_path,
        gt_path,
        gt_cache_dir,
        pred_cache_dir,
        save_dir,
        executor="serial",
        fov_workers=1,
    )
    pipeline = live_pipeline_module()
    pixel_rows, mask_rows, _ = pipeline.evaluate_predictions(cfg)
    assert len(pixel_rows) == N_POSITIONS * T
    assert len(mask_rows) == N_POSITIONS * T
