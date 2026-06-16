"""Cross-store GT-nuclei seeds for the cellpose_watershed whole-cell backend.

Drives ``evaluate_predictions`` end-to-end on three tiny single-channel HCS
plates — GT membrane, a *separate* GT-nuclei store, and a membrane prediction —
with the seed/watershed segmenters monkeypatched (no cellpose / cubic seg). This
is the A549 layout (membrane in ``CAAX_*.ozx``, nuclei in ``H2B_*.ozx``,
positions matched 1:1 by name). Asserts:

- the watershed seed generator receives the **nuclei-store** pixels (not the GT
  membrane pixels), proving ``io.nuclei_gt_path`` is honored, and
- a nuclei store missing a pred position raises rather than silently misaligning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from ._eval_fixtures import T, live_pipeline_module, make_hcs_plate


def _wc_config(pred, gt, nuclei_gt_path, save_dir):
    """Minimal cellpose_watershed instance-AP config (caches disabled → compute)."""
    return OmegaConf.create(
        {
            "target_name": "membrane",
            "io": {
                "pred_path": str(pred),
                "gt_path": str(gt),
                "nuclei_gt_path": (str(nuclei_gt_path) if nuclei_gt_path is not None else None),
                "pred_channel_name": "Membrane_prediction",
                "gt_channel_name": "Membrane",
                "cell_segmentation_path": None,
                "gt_cache_dir": None,
                "pred_cache_dir": None,
                "require_complete_cache": False,
            },
            "segmentation": {
                "backend": "cellpose_watershed",
                "dimension": "2d",
                "slice_selection": "frac",
                "slice_fraction": 0.5,
                "nuclei_channel_name": "Nuclei",
                "cellpose": {
                    "target_voxel_um": 0.58,
                    "cellprob_threshold": 0.0,
                    "flow_threshold": 0.4,
                    "min_obj_size": 30,
                },
                "watershed": {
                    "cell_voxel_um": 0.3,
                    "close_um": 2.5,
                    "wall_sigma_um": 0.35,
                    "wall_min_um": 1.0,
                    "hole_um": 3.0,
                    "min_cell_um": 15.0,
                    "memb_clahe": True,
                    "subtract_nuclei": True,
                },
            },
            "compute_instance_ap": True,
            "instance_metrics": {"iou_thresholds": [0.5, 0.75]},
            "pixel_metrics": {"spacing": [1.0, 1.0, 1.0], "fsc": None, "spectral_pcc": None},
            "feature_metrics": {"patch_size": 16, "deep_feature_batch_threshold": 256},
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
                "gt_instances": False,
                "pred_masks": False,
                "pred_cp": False,
                "pred_dinov3": False,
                "pred_dynaclr": False,
                "pred_celldino": False,
                "pred_instances": False,
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
                "fov_workers": 1,
                "threads_per_worker": "auto",
                "executor": "serial",
                "cuda_empty_cache_every_n_timepoints": 0,
                "gc_collect_every_n_fovs": 0,
            },
        }
    )


def _patch_segmenters(monkeypatch, pipeline, captured: dict):
    """Stub the GPU seg entry points; capture each seed-generator input array."""
    from dynacell.evaluation import segmentation, segmentation_cellpose, segmentation_whole_cell

    def fake_seeds(img, spacing, model, do_3d=False, **kw):
        captured.setdefault("seed_inputs", []).append(np.asarray(img).copy())
        lab = np.zeros(np.asarray(img).shape[-2:], dtype=np.uint16)
        lab[:8, :8] = 1
        return lab

    monkeypatch.setattr(segmentation_cellpose, "segment_nucleus_instances", fake_seeds)
    monkeypatch.setattr(
        segmentation_whole_cell,
        "segment_whole_cell",
        lambda memb, nuc, seed, spacing, **kw: np.asarray(seed, dtype=np.uint16),
    )
    # Let the real load_eval_models build EvalModels (extractors are None — no
    # feature metrics); only the cellpose seg-model load needs stubbing.
    monkeypatch.setattr(segmentation, "prepare_segmentation_model", lambda config: object())


def test_watershed_seeds_read_from_nuclei_store(tmp_path: Path, monkeypatch) -> None:
    """Seeds are segmented from the nuclei-store channel, not the GT membrane."""
    from dynacell.evaluation.segmentation_whole_cell import slice_index

    pipeline = live_pipeline_module()
    memb_gt = tmp_path / "memb_gt.zarr"
    nuclei_gt = tmp_path / "nuclei_gt.zarr"
    pred = tmp_path / "memb_pred.zarr"
    make_hcs_plate(memb_gt, "Membrane", seed=1, n_positions=2)
    make_hcs_plate(nuclei_gt, "Nuclei", seed=2, n_positions=2)
    make_hcs_plate(pred, "Membrane_prediction", seed=3, n_positions=2)

    captured: dict = {}
    _patch_segmenters(monkeypatch, pipeline, captured)

    cfg = _wc_config(pred, memb_gt, nuclei_gt, tmp_path / "out")
    pixel_rows, mask_rows, _ = pipeline.evaluate_predictions(cfg)

    assert mask_rows, "instance-AP mask rows should be produced"
    assert any("mAP" in r and "instance_dice" in r for r in mask_rows)

    # Seeds are computed once per (FOV, t) on the GT-nuclei slice. The 2D slice z is
    # picked from the GT membrane (same z applied to nuclei), so reconstruct it and
    # confirm each captured seed input equals the nuclei-store slice, not membrane.
    from iohub.ngff import open_ome_zarr

    with (
        open_ome_zarr(memb_gt, mode="r") as mplate,
        open_ome_zarr(nuclei_gt, mode="r") as nplate,
    ):
        memb = {name: np.asarray(pos.data[:, 0]) for name, pos in mplate.positions()}  # (T, D, H, W)
        nuc = {name: np.asarray(pos.data[:, 0]) for name, pos in nplate.positions()}

    expected_nuclei = []
    expected_membrane = []
    for name in sorted(memb):
        for t in range(T):
            z = slice_index(memb[name][t], selection="frac", fraction=0.5)
            expected_nuclei.append(nuc[name][t, z])
            expected_membrane.append(memb[name][t, z])

    seed_inputs = captured["seed_inputs"]
    assert len(seed_inputs) == len(expected_nuclei)
    # Order is FOV-major then per-t; sets of arrays must match the nuclei store and
    # differ from the membrane store (distinct rng seeds make them non-equal).
    for got, want_nuc, want_memb in zip(seed_inputs, expected_nuclei, expected_membrane):
        assert np.array_equal(got, want_nuc)
        assert not np.array_equal(got, want_memb)


def test_missing_nuclei_position_raises(tmp_path: Path, monkeypatch) -> None:
    """A nuclei store missing a pred position raises (no silent misalignment)."""
    pipeline = live_pipeline_module()
    memb_gt = tmp_path / "memb_gt.zarr"
    nuclei_gt = tmp_path / "nuclei_gt.zarr"
    pred = tmp_path / "memb_pred.zarr"
    make_hcs_plate(memb_gt, "Membrane", seed=1, n_positions=2)
    make_hcs_plate(nuclei_gt, "Nuclei", seed=2, n_positions=1)  # short one position
    make_hcs_plate(pred, "Membrane_prediction", seed=3, n_positions=2)

    _patch_segmenters(monkeypatch, pipeline, {})

    cfg = _wc_config(pred, memb_gt, nuclei_gt, tmp_path / "out")
    with pytest.raises(ValueError, match="nuclei_gt_path store is missing positions"):
        pipeline.evaluate_predictions(cfg)
