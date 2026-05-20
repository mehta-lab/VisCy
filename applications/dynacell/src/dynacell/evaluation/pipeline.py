"""Batch orchestration: load, segment, evaluate, save."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.feature_metrics import (
    compute_feature_similarity,
    compute_feature_similarity_pairwise,
)
from dynacell.evaluation.feature_select import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_FREQ_CUT,
    DEFAULT_UNIQUE_CUT,
    select_features,
)
from dynacell.evaluation.linear_probe import indistinguishability, paired_auroc
from dynacell.evaluation.metrics import (
    build_crops,
    calculate_microssim,
    compute_pixel_metrics,
    cp_regionprops,
    drop_paired_nonfinite_rows,
    evaluate_segmentations,
    features_from_crops,
)
from dynacell.evaluation.model_loader import EvalModels, init_cache_contexts, load_eval_models
from dynacell.evaluation.pipeline_cache import (
    flush_manifest,
    fov_cp_features,
    fov_deep_features,
    fov_masks,
    precompute_deep_features,
)
from dynacell.evaluation.utils import plot_metrics


def _zscore_per_side(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-side z-score: separate (mean, std) computed for each matrix."""
    pred_z = (pred - pred.mean(axis=0)) / (pred.std(axis=0) + 1e-8)
    target_z = (target - target.mean(axis=0)) / (target.std(axis=0) + 1e-8)
    return pred_z, target_z


def _cp_dropzero_zscore(pred_raw: np.ndarray, target_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-(FOV, timepoint) CP cleanup: drop target-zero columns then z-score.

    Returns ``(np.empty(...), np.empty(...))`` when all columns drop, so
    the caller can short-circuit and emit a NaN row.
    """
    non_zero_cols = ~np.all(target_raw == 0, axis=0)
    pred_mat = pred_raw[:, non_zero_cols]
    target_mat = target_raw[:, non_zero_cols]
    if pred_mat.size == 0:
        return pred_mat, target_mat
    return _zscore_per_side(pred_mat, target_mat)


def _real_vs_pred_probe(
    pred_arr: np.ndarray,
    target_arr: np.ndarray,
    pred_fovs: np.ndarray,
    target_fovs: np.ndarray,
    prefix: str,
    rng_seed: int = 2020,
) -> dict[str, float]:
    """Probe A: linear classifier on ``[gt; pred]`` with FOV-stratified CV."""
    result = paired_auroc(target_arr, pred_arr, target_fovs, pred_fovs, rng_seed=rng_seed)
    auroc = float(result["auroc_mean"])
    return {
        f"{prefix}_RealVsPred_AUROC": auroc,
        f"{prefix}_RealVsPred_AUROC_std": float(result["auroc_std"]),
        f"{prefix}_Indistinguishability": indistinguishability(auroc),
    }


def _fov_pred_features_per_t(
    pred_cache_ctx,
    pos_name: str,
    predict: np.ndarray,
    cell_segmentation: np.ndarray | None,
    dinov3_feature_extractor,
    dynaclr_feature_extractor,
    celldino_feature_extractor,
    patch_size: int,
    spacing,
) -> dict[str, list[np.ndarray] | None]:
    """Return per-backbone per-t prediction features.

    Uses the pred cache when configured; otherwise computes fresh
    per-timepoint, sharing one set of 2-D crops across all deep backbones
    to avoid redundant max-projection + crop construction per backbone.
    """
    if pred_cache_ctx.enabled:
        return {
            "cp": fov_cp_features(pred_cache_ctx, pos_name, predict, cell_segmentation),
            "dinov3": fov_deep_features(
                pred_cache_ctx, pos_name, predict, cell_segmentation, dinov3_feature_extractor, "dinov3"
            ),
            "dynaclr": fov_deep_features(
                pred_cache_ctx, pos_name, predict, cell_segmentation, dynaclr_feature_extractor, "dynaclr"
            ),
            "celldino": (
                fov_deep_features(
                    pred_cache_ctx, pos_name, predict, cell_segmentation, celldino_feature_extractor, "celldino"
                )
                if celldino_feature_extractor is not None
                else None
            ),
        }
    t_count = predict.shape[0]
    # Single per-t pass: build crops once per timepoint, fan out to every
    # deep backbone, then drop them before moving to the next t — avoids
    # holding ``t_count`` crop tensors in memory at once.
    cp: list[np.ndarray] = []
    dinov3: list[np.ndarray] = []
    dynaclr: list[np.ndarray] = []
    celldino: list[np.ndarray] | None = [] if celldino_feature_extractor is not None else None
    for t in range(t_count):
        cp.append(cp_regionprops(predict[t], cell_segmentation[t], spacing))
        crops_t = build_crops(predict[t], cell_segmentation[t], patch_size)
        dinov3.append(features_from_crops(crops_t, dinov3_feature_extractor))
        dynaclr.append(features_from_crops(crops_t, dynaclr_feature_extractor))
        if celldino is not None:
            celldino.append(features_from_crops(crops_t, celldino_feature_extractor))
    return {"cp": cp, "dinov3": dinov3, "dynaclr": dynaclr, "celldino": celldino}


def _save_embeddings(save_dir: Path, groups: dict[str, tuple[list, list, list]]) -> None:
    """Save concatenated single-cell embeddings with FOV and Timepoint metadata."""
    embed_dir = save_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    for name, (feats, fovs, ts) in groups.items():
        if not feats:
            continue
        out_path = embed_dir / f"{name}_single_cell_embeddings.npz"
        np.savez(
            out_path,
            embeddings=np.concatenate(feats, axis=0),
            fov=np.concatenate(fovs, axis=0),
            timepoint=np.concatenate(ts, axis=0),
        )
        print(f"Saved embeddings → {out_path}")


@dataclass
class _BackboneLists:
    """Per-FOV per-backbone single-cell feature lists.

    Each list holds one entry per timepoint with non-empty cell features.
    ``pred_*`` and ``gt_*`` indices align: ``pred_feats[i]`` and
    ``gt_feats[i]`` come from the same (pos_name, t) pair, with
    ``pred_fovs[i] == gt_fovs[i]`` and ``pred_ts[i] == gt_ts[i]``.
    """

    pred_feats: list[np.ndarray] = field(default_factory=list)
    gt_feats: list[np.ndarray] = field(default_factory=list)
    pred_fovs: list[np.ndarray] = field(default_factory=list)
    gt_fovs: list[np.ndarray] = field(default_factory=list)
    pred_ts: list[np.ndarray] = field(default_factory=list)
    gt_ts: list[np.ndarray] = field(default_factory=list)


@dataclass
class FovResult:
    """Output of ``_process_one_fov``: everything one FOV contributes to the run.

    Designed for cross-process transport via pickle: all fields are picklable
    types (str, list[dict], list[np.ndarray], np.ndarray) — no iohub handles,
    no torch modules.

    Microssim scores are merged into ``per_t_pixel_rows`` before return
    (matching the existing serial path at ``pipeline.py:478-481``); no
    separate microssim field.
    """

    pos_name: str
    row: str
    col: str
    fov: str
    per_t_pixel_rows: list[dict]
    per_t_mask_rows: list[dict]
    per_t_feature_rows: list[dict]
    seg_array: np.ndarray  # (T, 2, D, H, W) bool: channel 0 = pred, channel 1 = GT
    cp: _BackboneLists = field(default_factory=_BackboneLists)
    dinov3: _BackboneLists = field(default_factory=_BackboneLists)
    dynaclr: _BackboneLists = field(default_factory=_BackboneLists)
    celldino: _BackboneLists = field(default_factory=_BackboneLists)
    timings: list[tuple[str, int | None, str, float]] = field(default_factory=list)


def _process_one_fov(
    config: DictConfig,
    cuda_empty_cache_every_n_timepoints: int,
    pos_name_pred: str,
    pos_pred,
    pos_gt,
    pos_seg,
    io_config,
    cache_ctx,
    pred_cache_ctx,
    seg_model,
    dinov3_feature_extractor,
    dynaclr_feature_extractor,
    celldino_feature_extractor,
) -> FovResult:
    """Compute everything one FOV contributes to the eval and return a FovResult.

    No side effects on shared parent state (no segmentation_results plate
    writes, no manifest flush). The parent aggregator handles those — see
    ``_aggregate_fov_result``. Used by both the serial and process FOV-loop
    paths in ``evaluate_predictions``.
    """
    from dynacell.evaluation.runtime import (
        get_timings,
        gpu_serialization_lock,
        is_worker,
        maybe_empty_cuda_cache,
        region_timer,
    )
    from dynacell.evaluation.segmentation import segment

    timings_start = len(get_timings())
    # Inner per-T tqdm is noise when N workers each emit it to the shared
    # parent stderr — outer per-FOV tqdm in the parent stays visible either way.
    suppress_inner_tqdm = is_worker()
    # GPU serialization lock is a no-op when use_gpu=false: under that
    # setting compute_pixel_metrics, cellpose, and feature extractors all
    # run CPU-only, so cross-worker fcntl serialization would just add
    # latency for nothing.
    use_gpu = bool(getattr(config, "use_gpu", True))

    pred_channel_index = pos_pred.get_channel_index(io_config.pred_channel_name)
    gt_channel_index = pos_gt.get_channel_index(io_config.gt_channel_name)

    predict = np.asarray(pos_pred.data[:, pred_channel_index])  # shape: (T, D, H, W)
    target = np.asarray(pos_gt.data[:, gt_channel_index])
    cell_segmentation = np.asarray(pos_seg.data[:, 0]) if pos_seg is not None else None

    T = predict.shape[0]

    with region_timer("mask_gt", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
        gt_mask_stack = fov_masks(cache_ctx, pos_name_pred, target, seg_model)
    if pred_cache_ctx.enabled:
        with region_timer("mask_pred", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
            pred_mask_stack = fov_masks(pred_cache_ctx, pos_name_pred, predict, seg_model)
    else:
        pred_mask_stack = None

    gt_cp_per_t = None
    gt_dinov3_per_t = None
    gt_dynaclr_per_t = None
    gt_celldino_per_t = None
    pred_per_t = None
    if config.compute_feature_metrics:
        with region_timer("cp_gt", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
            gt_cp_per_t = fov_cp_features(cache_ctx, pos_name_pred, target, cell_segmentation)
        with region_timer("deep_gt_dinov3", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
            gt_dinov3_per_t = fov_deep_features(
                cache_ctx, pos_name_pred, target, cell_segmentation, dinov3_feature_extractor, "dinov3"
            )
        with region_timer("deep_gt_dynaclr", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
            gt_dynaclr_per_t = fov_deep_features(
                cache_ctx, pos_name_pred, target, cell_segmentation, dynaclr_feature_extractor, "dynaclr"
            )
        if celldino_feature_extractor is not None:
            with region_timer("deep_gt_celldino", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                gt_celldino_per_t = fov_deep_features(
                    cache_ctx,
                    pos_name_pred,
                    target,
                    cell_segmentation,
                    celldino_feature_extractor,
                    "celldino",
                )
        with region_timer("features_pred_per_t", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
            pred_per_t = _fov_pred_features_per_t(
                pred_cache_ctx,
                pos_name_pred,
                predict,
                cell_segmentation,
                dinov3_feature_extractor,
                dynaclr_feature_extractor,
                celldino_feature_extractor,
                config.feature_metrics.patch_size,
                config.pixel_metrics.spacing,
            )

    microssim_data: list[dict] = []
    fov_pixel_metrics: list[dict] = []
    fov_mask_metrics: list[dict] = []
    fov_feature_metrics: list[dict] = []
    segmentations: list[np.ndarray] = []
    cp = _BackboneLists()
    dinov3 = _BackboneLists()
    dynaclr = _BackboneLists()
    celldino = _BackboneLists()

    for t in tqdm(range(T), desc="Processing timepoints", leave=False, disable=suppress_inner_tqdm):
        data_info = {"FOV": pos_name_pred, "Timepoint": t}

        with region_timer("pixel_metrics", pos_name_pred, t), gpu_serialization_lock(gate=use_gpu):
            pixel_metrics = compute_pixel_metrics(
                predict[t],
                target[t],
                spacing=config.pixel_metrics.spacing,
                fsc_kwargs=config.pixel_metrics.fsc,
                spectral_pcc_kwargs=config.pixel_metrics.spectral_pcc,
                use_gpu=config.use_gpu,
            )
        if config.compute_microssim:
            microssim_data.append({"target": target[t], "predict": predict[t]})
        fov_pixel_metrics.append({**data_info, **pixel_metrics})

        with region_timer("mask_metrics", pos_name_pred, t):
            segmented_target = gt_mask_stack[t]
            if pred_mask_stack is not None:
                segmented_predict = pred_mask_stack[t]
            else:
                with gpu_serialization_lock(gate=use_gpu):
                    segmented_predict = np.asarray(segment(predict[t], config.target_name, seg_model=seg_model)).astype(
                        bool
                    )
            fov_mask_metrics.append({**data_info, **evaluate_segmentations(segmented_predict, segmented_target)})
            segmentations.append(np.stack([segmented_predict, segmented_target], axis=0))

        if config.compute_feature_metrics:
            with region_timer("feature_pairwise", pos_name_pred, t):
                pred_cp = pred_per_t["cp"][t]
                pred_dinov3 = pred_per_t["dinov3"][t]
                pred_dynaclr = pred_per_t["dynaclr"][t]
                pred_celldino = pred_per_t["celldino"][t] if pred_per_t["celldino"] is not None else None
                pred_cp, gt_cp_t = drop_paired_nonfinite_rows(pred_cp, gt_cp_per_t[t])
                if pred_cp.size and gt_cp_t.size:
                    pred_cp_z, gt_cp_z = _cp_dropzero_zscore(pred_cp, gt_cp_t)
                else:
                    pred_cp_z, gt_cp_z = pred_cp, gt_cp_t
                pairwise_metrics = {
                    **compute_feature_similarity_pairwise(pred_cp_z, gt_cp_z, "CP"),
                    **compute_feature_similarity_pairwise(pred_dinov3, gt_dinov3_per_t[t], "DINOv3"),
                    **compute_feature_similarity_pairwise(pred_dynaclr, gt_dynaclr_per_t[t], "DynaCLR"),
                }
                if pred_celldino is not None:
                    pairwise_metrics.update(
                        compute_feature_similarity_pairwise(pred_celldino, gt_celldino_per_t[t], "CellDINO")
                    )
                fov_feature_metrics.append({**data_info, **pairwise_metrics})
                if pred_cp.size > 0:
                    cp.pred_feats.append(pred_cp)
                    cp.gt_feats.append(gt_cp_t)
                    fov_arr = np.full(len(pred_cp), pos_name_pred)
                    t_arr = np.full(len(pred_cp), t, dtype=np.int32)
                    cp.pred_fovs.append(fov_arr)
                    cp.gt_fovs.append(fov_arr)
                    cp.pred_ts.append(t_arr)
                    cp.gt_ts.append(t_arr)
                if pred_dinov3.size > 0:
                    dinov3.pred_feats.append(pred_dinov3)
                    dinov3.gt_feats.append(gt_dinov3_per_t[t])
                    fov_arr = np.full(len(pred_dinov3), pos_name_pred)
                    t_arr = np.full(len(pred_dinov3), t, dtype=np.int32)
                    dinov3.pred_fovs.append(fov_arr)
                    dinov3.gt_fovs.append(fov_arr)
                    dinov3.pred_ts.append(t_arr)
                    dinov3.gt_ts.append(t_arr)
                if pred_dynaclr.size > 0:
                    dynaclr.pred_feats.append(pred_dynaclr)
                    dynaclr.gt_feats.append(gt_dynaclr_per_t[t])
                    fov_arr = np.full(len(pred_dynaclr), pos_name_pred)
                    t_arr = np.full(len(pred_dynaclr), t, dtype=np.int32)
                    dynaclr.pred_fovs.append(fov_arr)
                    dynaclr.gt_fovs.append(fov_arr)
                    dynaclr.pred_ts.append(t_arr)
                    dynaclr.gt_ts.append(t_arr)
                if pred_celldino is not None and pred_celldino.size > 0:
                    celldino.pred_feats.append(pred_celldino)
                    celldino.gt_feats.append(gt_celldino_per_t[t])
                    fov_arr = np.full(len(pred_celldino), pos_name_pred)
                    t_arr = np.full(len(pred_celldino), t, dtype=np.int32)
                    celldino.pred_fovs.append(fov_arr)
                    celldino.gt_fovs.append(fov_arr)
                    celldino.pred_ts.append(t_arr)
                    celldino.gt_ts.append(t_arr)

        maybe_empty_cuda_cache(t, cuda_empty_cache_every_n_timepoints)

    seg_array = np.stack(segmentations, axis=0)  # shape: (T, 2, D, H, W)

    if config.compute_microssim:
        with region_timer("microssim", pos_name_pred):
            microssim_scores = calculate_microssim(microssim_data)
            for i in range(T):
                fov_pixel_metrics[i]["MicroMS3IM"] = float(microssim_scores[i]["MicroMS3IM"])

    row, col, fov = pos_name_pred.split("/")
    return FovResult(
        pos_name=pos_name_pred,
        row=row,
        col=col,
        fov=fov,
        per_t_pixel_rows=fov_pixel_metrics,
        per_t_mask_rows=fov_mask_metrics,
        per_t_feature_rows=fov_feature_metrics,
        seg_array=seg_array.astype(bool),
        cp=cp,
        dinov3=dinov3,
        dynaclr=dynaclr,
        celldino=celldino,
        timings=get_timings()[timings_start:],
    )


def _aggregate_fov_result(
    result: FovResult,
    segmentation_results,
    all_pixel_metrics: list[dict],
    all_mask_metrics: list[dict],
    all_feature_metrics: list[dict],
    pred_cp_feats: list[np.ndarray],
    pred_cp_fovs: list[np.ndarray],
    pred_cp_ts: list[np.ndarray],
    gt_cp_feats: list[np.ndarray],
    gt_cp_fovs: list[np.ndarray],
    gt_cp_ts: list[np.ndarray],
    pred_dinov3_feats: list[np.ndarray],
    pred_dinov3_fovs: list[np.ndarray],
    pred_dinov3_ts: list[np.ndarray],
    gt_dinov3_feats: list[np.ndarray],
    gt_dinov3_fovs: list[np.ndarray],
    gt_dinov3_ts: list[np.ndarray],
    pred_dynaclr_feats: list[np.ndarray],
    pred_dynaclr_fovs: list[np.ndarray],
    pred_dynaclr_ts: list[np.ndarray],
    gt_dynaclr_feats: list[np.ndarray],
    gt_dynaclr_fovs: list[np.ndarray],
    gt_dynaclr_ts: list[np.ndarray],
    pred_celldino_feats: list[np.ndarray],
    pred_celldino_fovs: list[np.ndarray],
    pred_celldino_ts: list[np.ndarray],
    gt_celldino_feats: list[np.ndarray],
    gt_celldino_fovs: list[np.ndarray],
    gt_celldino_ts: list[np.ndarray],
    *,
    extend_worker_timings: bool,
) -> None:
    """Apply one FOV's contributions to the parent-side run state.

    Writes the segmentation array to the HCS plate, extends the per-T row
    lists, and extends the 24 dataset-level accumulator lists.

    ``extend_worker_timings`` toggles whether to append ``result.timings``
    to the parent's global ``_TIMINGS`` collector. Set ``False`` in serial
    mode (workers and parent share one collector, so the timings are
    already there); set ``True`` in process mode (workers have separate
    per-process collectors, so the parent must aggregate).
    """
    from dynacell.evaluation.runtime import extend_timings, region_timer

    if extend_worker_timings:
        extend_timings(result.timings)

    with region_timer("seg_write", result.pos_name):
        seg_pos = segmentation_results.create_position(result.row, result.col, result.fov)
        seg_pos.create_image("0", result.seg_array)

    all_pixel_metrics.extend(result.per_t_pixel_rows)
    all_mask_metrics.extend(result.per_t_mask_rows)
    all_feature_metrics.extend(result.per_t_feature_rows)

    pred_cp_feats.extend(result.cp.pred_feats)
    pred_cp_fovs.extend(result.cp.pred_fovs)
    pred_cp_ts.extend(result.cp.pred_ts)
    gt_cp_feats.extend(result.cp.gt_feats)
    gt_cp_fovs.extend(result.cp.gt_fovs)
    gt_cp_ts.extend(result.cp.gt_ts)
    pred_dinov3_feats.extend(result.dinov3.pred_feats)
    pred_dinov3_fovs.extend(result.dinov3.pred_fovs)
    pred_dinov3_ts.extend(result.dinov3.pred_ts)
    gt_dinov3_feats.extend(result.dinov3.gt_feats)
    gt_dinov3_fovs.extend(result.dinov3.gt_fovs)
    gt_dinov3_ts.extend(result.dinov3.gt_ts)
    pred_dynaclr_feats.extend(result.dynaclr.pred_feats)
    pred_dynaclr_fovs.extend(result.dynaclr.pred_fovs)
    pred_dynaclr_ts.extend(result.dynaclr.pred_ts)
    gt_dynaclr_feats.extend(result.dynaclr.gt_feats)
    gt_dynaclr_fovs.extend(result.dynaclr.gt_fovs)
    gt_dynaclr_ts.extend(result.dynaclr.gt_ts)
    pred_celldino_feats.extend(result.celldino.pred_feats)
    pred_celldino_fovs.extend(result.celldino.pred_fovs)
    pred_celldino_ts.extend(result.celldino.pred_ts)
    gt_celldino_feats.extend(result.celldino.gt_feats)
    gt_celldino_fovs.extend(result.celldino.gt_fovs)
    gt_celldino_ts.extend(result.celldino.gt_ts)


# Worker-side state for ``executor=process``. A spawn-context child Python
# interpreter calls ``_worker_run_fov`` once per submitted FOV; the first
# call triggers ``_worker_setup`` which lazy-loads seg model + extractors +
# cache contexts under the GPU serialization lock and caches them here.
# Subsequent FOVs in the same worker reuse the cached state.
#
# Plate handles are intentionally *not* cached here — they're context-managed
# per-FOV inside ``_worker_run_fov`` so iohub file descriptors close when the
# worker is between FOVs, and so the worker has nothing to clean up on
# ProcessPoolExecutor shutdown.
_WORKER_STATE: dict[str, Any] = {}


def _worker_setup(config: DictConfig) -> None:
    """First-FOV-per-worker initialization: load models + init cache contexts.

    GPU touch sites (model + extractor loads) run under
    ``gpu_serialization_lock(gate=use_gpu)`` so under ``use_gpu=true`` N workers don't initialize CUDA + load
    weights concurrently. Cache contexts (no GPU touch) are built outside
    the lock. Plate handles are opened lazily per-FOV in ``_worker_run_fov``.
    """
    if _WORKER_STATE.get("initialized"):
        return
    from dynacell.evaluation.runtime import gpu_serialization_lock

    use_gpu = bool(getattr(config, "use_gpu", True))
    with gpu_serialization_lock(gate=use_gpu):
        models = load_eval_models(config)

    cache_ctx, pred_cache_ctx = init_cache_contexts(config, models)

    _WORKER_STATE.update(
        {
            "initialized": True,
            "seg_model": models.seg_model,
            "dinov3": models.dinov3,
            "dynaclr": models.dynaclr,
            "celldino": models.celldino,
            "cache_ctx": cache_ctx,
            "pred_cache_ctx": pred_cache_ctx,
        }
    )


def _find_position(plate, pos_name: str):
    """Return the iohub Position with name ``pos_name`` from ``plate``.

    Scans ``plate.positions()`` since iohub doesn't index by name. Cheap
    for the ≤ tens of positions per eval.
    """
    for name, pos in plate.positions():
        if name == pos_name:
            return pos
    raise KeyError(f"position {pos_name!r} not found in plate")


def _worker_run_fov(config: DictConfig, pos_name: str, cuda_empty_every_n: int) -> FovResult:
    """Worker entry point: process one FOV by name and return FovResult.

    Submitted via ``pool.submit`` in ``executor=process`` mode. Plates
    are opened with a context manager scoped to this single call — iohub
    file descriptors close before the worker accepts its next FOV. Models
    + cache contexts stay cached in ``_WORKER_STATE`` across FOVs.
    """
    from dynacell.evaluation.pipeline_cache import flush_manifest

    _worker_setup(config)
    state = _WORKER_STATE

    seg_path = config.io.cell_segmentation_path
    with (
        open_ome_zarr(Path(config.io.pred_path), mode="r") as pred_plate,
        open_ome_zarr(Path(config.io.gt_path), mode="r") as gt_plate,
    ):
        pos_pred = _find_position(pred_plate, pos_name)
        pos_gt = _find_position(gt_plate, pos_name)

        if seg_path is not None:
            with open_ome_zarr(Path(seg_path), mode="r") as seg_plate:
                pos_seg = _find_position(seg_plate, pos_name)
                result = _process_one_fov(
                    config,
                    cuda_empty_every_n,
                    pos_name,
                    pos_pred,
                    pos_gt,
                    pos_seg,
                    config.io,
                    state["cache_ctx"],
                    state["pred_cache_ctx"],
                    state["seg_model"],
                    state["dinov3"],
                    state["dynaclr"],
                    state["celldino"],
                )
        else:
            result = _process_one_fov(
                config,
                cuda_empty_every_n,
                pos_name,
                pos_pred,
                pos_gt,
                None,
                config.io,
                state["cache_ctx"],
                state["pred_cache_ctx"],
                state["seg_model"],
                state["dinov3"],
                state["dynaclr"],
                state["celldino"],
            )

    # Worker-side manifest flush so interrupted runs preserve progress even
    # when the parent dies. The global manifest lock at
    # _pos_write_lock(ctx, "manifest", "global") serializes flushes across
    # workers.
    flush_manifest(state["cache_ctx"])
    flush_manifest(state["pred_cache_ctx"])
    return result


def evaluate_predictions(config: DictConfig, *, models: EvalModels | None = None):
    """Evaluate predictions on all test images.

    Parameters
    ----------
    config : DictConfig
        Resolved eval config.
    models : EvalModels | None, optional
        Pre-loaded segmenter + feature extractors. When provided, the
        inline ``load_eval_models(config)`` call is skipped — used by the
        grouped multi-condition driver to amortize model loads across
        conditions sharing the same target/extractors. Default ``None``
        preserves the historical single-condition behavior. Note: under
        ``runtime.executor=process``, workers still load their own model
        copies; this kwarg saves only the parent-side load.
    """
    from dynacell.evaluation.runtime import (
        apply_thread_budget,
        dump_timings_csv,
        make_fov_executor,
        maybe_gc_collect,
        reset_timings,
        resolve_runtime,
    )

    # Phase 1 runtime resolution: lock in executor + thread caps before any
    # heavy work. fov_workers may be provisional when "auto"; re-resolved in
    # Phase 2 once the position list is known (C4). threads_per_worker stays
    # frozen across phases for parent/worker BLAS-cap consistency.
    runtime = resolve_runtime(config)
    apply_thread_budget(runtime.threads_per_worker)
    # Resolve ${...} interpolations in-place so spawn workers don't lazy-resolve
    # refs in their child interpreters. ??? MISSING fields stay unresolved
    # (OmegaConf 2.3 _resolve walks only interpolation nodes, not value nodes),
    # so this is safe under feature-metrics-disabled runs.
    OmegaConf.resolve(config)
    reset_timings()

    all_pixel_metrics = []
    all_mask_metrics = []
    all_feature_metrics = []

    io_config = config.io
    pred_path = Path(io_config.pred_path)
    gt_path = Path(io_config.gt_path)
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if config.compute_feature_metrics and io_config.cell_segmentation_path is None:
        raise ValueError("io.cell_segmentation_path is required when compute_feature_metrics=true")

    if models is None:
        models = load_eval_models(config)
    seg_model = models.seg_model
    dinov3_feature_extractor = models.dinov3
    dynaclr_feature_extractor = models.dynaclr
    celldino_feature_extractor = models.celldino

    cache_ctx, pred_cache_ctx = init_cache_contexts(config, models)

    seg_path = Path(io_config.cell_segmentation_path) if io_config.cell_segmentation_path is not None else None

    pred_cp_feats: list[np.ndarray] = []
    pred_cp_fovs: list[np.ndarray] = []
    pred_cp_ts: list[np.ndarray] = []
    gt_cp_feats: list[np.ndarray] = []
    gt_cp_fovs: list[np.ndarray] = []
    gt_cp_ts: list[np.ndarray] = []
    pred_dinov3_feats: list[np.ndarray] = []
    pred_dinov3_fovs: list[np.ndarray] = []
    pred_dinov3_ts: list[np.ndarray] = []
    gt_dinov3_feats: list[np.ndarray] = []
    gt_dinov3_fovs: list[np.ndarray] = []
    gt_dinov3_ts: list[np.ndarray] = []
    pred_dynaclr_feats: list[np.ndarray] = []
    pred_dynaclr_fovs: list[np.ndarray] = []
    pred_dynaclr_ts: list[np.ndarray] = []
    gt_dynaclr_feats: list[np.ndarray] = []
    gt_dynaclr_fovs: list[np.ndarray] = []
    gt_dynaclr_ts: list[np.ndarray] = []
    pred_celldino_feats: list[np.ndarray] = []
    pred_celldino_fovs: list[np.ndarray] = []
    pred_celldino_ts: list[np.ndarray] = []
    gt_celldino_feats: list[np.ndarray] = []
    gt_celldino_fovs: list[np.ndarray] = []
    gt_celldino_ts: list[np.ndarray] = []

    channel_names = ["prediction_seg", "target_seg"]
    with (
        open_ome_zarr(
            save_dir / "segmentation_results.zarr",
            mode="w",
            layout="hcs",
            channel_names=channel_names,
            version="0.5",
        ) as segmentation_results,
        open_ome_zarr(pred_path, mode="r") as pred_plate,
        open_ome_zarr(gt_path, mode="r") as gt_plate,
    ):
        pred_positions = list(pred_plate.positions())
        gt_positions = list(gt_plate.positions())
        if len(pred_positions) != len(gt_positions):
            raise ValueError(f"Position count mismatch: pred={len(pred_positions)}, gt={len(gt_positions)}")
        if seg_path is not None:
            seg_plate = open_ome_zarr(seg_path, mode="r")
            seg_positions = list(seg_plate.positions())
            if len(seg_positions) != len(pred_positions):
                seg_plate.close()
                raise ValueError(f"Position count mismatch: pred={len(pred_positions)}, seg={len(seg_positions)}")
        else:
            seg_plate = None
            seg_positions = [(name, None) for name, _ in pred_positions]

        limit = getattr(config, "limit_positions", None)
        if limit is not None:
            pred_positions = pred_positions[:limit]
            gt_positions = gt_positions[:limit]
            seg_positions = seg_positions[:limit]
        try:
            # Hoist paired-name validation so precompute (which runs before
            # the per-FOV loop) cannot write to mismatched cache slots.
            for (pos_name_pred, _), (pos_name_gt, _), (pos_name_seg, _) in zip(
                pred_positions, gt_positions, seg_positions
            ):
                if pos_name_pred != pos_name_gt:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, gt={pos_name_gt!r}")
                if seg_plate is not None and pos_name_seg != pos_name_pred:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, seg={pos_name_seg!r}")

            if config.compute_feature_metrics:
                deep_extractors = {
                    "dinov3": dinov3_feature_extractor,
                    "dynaclr": dynaclr_feature_extractor,
                }
                if celldino_feature_extractor is not None:
                    deep_extractors["celldino"] = celldino_feature_extractor
                flush_threshold = int(
                    OmegaConf.select(config, "feature_metrics.deep_feature_batch_threshold", default=256)
                )
                # Skip precompute on any side under require_complete_cache;
                # the per-FOV path will fail-loud on misses for that side.
                sides_for_precompute: dict = {}
                side_positions: dict = {}
                side_channels: dict = {}
                if cache_ctx.enabled and not cache_ctx.require_complete:
                    sides_for_precompute["gt"] = cache_ctx
                    side_positions["gt"] = gt_positions
                    side_channels["gt"] = io_config.gt_channel_name
                if pred_cache_ctx.enabled and not pred_cache_ctx.require_complete:
                    sides_for_precompute["pred"] = pred_cache_ctx
                    side_positions["pred"] = pred_positions
                    side_channels["pred"] = io_config.pred_channel_name
                if sides_for_precompute:
                    precompute_deep_features(
                        sides_for_precompute,
                        side_positions,
                        side_channels,
                        seg_positions,
                        deep_extractors,
                        flush_threshold=flush_threshold,
                    )
                    for ctx in sides_for_precompute.values():
                        flush_manifest(ctx)

            # Phase 2 runtime resolution: clamp fov_workers to n_positions
            # now that we know it. threads_per_worker is frozen at Phase 1
            # so the parent's BLAS cap matches what workers see.
            runtime = resolve_runtime(
                config,
                n_positions=len(pred_positions),
                freeze_threads_per_worker=runtime.threads_per_worker,
            )

            # Under executor=process, workers lazy-load their own seg_model
            # copies (under the GPU lock). The parent's copy is held only for
            # the checkpoint-cache-warm side effect (segmenter_model_zoo's
            # validate_model has no file lock around its quilt download, so
            # pre-warming in the parent prevents N workers from racing on a
            # cold cache). After Phase 2 we know the final executor — if it's
            # process, drop the parent copy so we don't keep two seg model
            # copies resident on the GPU.
            if runtime.executor == "process" and seg_model is not None:
                del seg_model
                seg_model = None
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Whether to fold result.timings into the parent's _TIMINGS:
            # only when results come from spawn workers (separate per-process
            # collector). In serial mode, _process_one_fov already wrote to
            # the parent's collector via region_timer; re-extending here
            # would double-count every row.
            extend_worker_timings = runtime.executor == "process"

            def _aggregate(result: FovResult) -> None:
                _aggregate_fov_result(
                    result,
                    segmentation_results,
                    all_pixel_metrics,
                    all_mask_metrics,
                    all_feature_metrics,
                    pred_cp_feats,
                    pred_cp_fovs,
                    pred_cp_ts,
                    gt_cp_feats,
                    gt_cp_fovs,
                    gt_cp_ts,
                    pred_dinov3_feats,
                    pred_dinov3_fovs,
                    pred_dinov3_ts,
                    gt_dinov3_feats,
                    gt_dinov3_fovs,
                    gt_dinov3_ts,
                    pred_dynaclr_feats,
                    pred_dynaclr_fovs,
                    pred_dynaclr_ts,
                    gt_dynaclr_feats,
                    gt_dynaclr_fovs,
                    gt_dynaclr_ts,
                    pred_celldino_feats,
                    pred_celldino_fovs,
                    pred_celldino_ts,
                    gt_celldino_feats,
                    gt_celldino_fovs,
                    gt_celldino_ts,
                    extend_worker_timings=extend_worker_timings,
                )

            if runtime.executor == "serial":
                for fov_idx, (p1, p2, p3) in enumerate(
                    tqdm(
                        zip(pred_positions, gt_positions, seg_positions),
                        total=len(pred_positions),
                        desc="Processing positions",
                    )
                ):
                    pos_name_pred, pos_pred = p1
                    _, pos_gt = p2
                    _, pos_seg = p3

                    result = _process_one_fov(
                        config,
                        runtime.cuda_empty_cache_every_n_timepoints,
                        pos_name_pred,
                        pos_pred,
                        pos_gt,
                        pos_seg,
                        io_config,
                        cache_ctx,
                        pred_cache_ctx,
                        seg_model,
                        dinov3_feature_extractor,
                        dynaclr_feature_extractor,
                        celldino_feature_extractor,
                    )
                    _aggregate(result)

                    # Flush manifest after each position so interrupted runs preserve progress.
                    flush_manifest(cache_ctx)
                    flush_manifest(pred_cache_ctx)

                    maybe_gc_collect(fov_idx, runtime.gc_collect_every_n_fovs)
            else:
                # executor == "process": spawn-context ProcessPoolExecutor.
                # Workers lazy-load their own seg_model + extractors under the
                # GPU lock; parent's copy was discarded after Phase-2 resolve.
                #
                # Streaming aggregation: futures arrive in completion order
                # but we want deterministic per-FOV CSV / embedding NPZ row
                # order, so the next-expected pos_name index advances
                # opportunistically as results land. Only out-of-order
                # FovResults stay buffered — once the expected position
                # completes, we drain the buffer in order. Bounds peak
                # parent-side memory to the number of out-of-order FOVs
                # rather than the full N×seg_array (~N × 440 MB).
                from concurrent.futures import as_completed

                pos_names_in_order = [p[0] for p, _, _ in zip(pred_positions, gt_positions, seg_positions)]
                next_idx = 0
                buffer: dict[str, FovResult] = {}
                with make_fov_executor(runtime) as pool:
                    if pool is None:
                        raise RuntimeError("make_fov_executor returned None for executor='process'")
                    futures = {
                        pool.submit(
                            _worker_run_fov, config, pos_name, runtime.cuda_empty_cache_every_n_timepoints
                        ): pos_name
                        for pos_name in pos_names_in_order
                    }
                    with tqdm(total=len(futures), desc="Processing positions") as pbar:
                        for fut in as_completed(futures):
                            pos_name = futures[fut]
                            buffer[pos_name] = fut.result()
                            # Drain in order while the next-expected position
                            # is available, releasing seg_array refs as soon
                            # as the aggregator's plate write completes.
                            while next_idx < len(pos_names_in_order) and (pos_names_in_order[next_idx] in buffer):
                                expected = pos_names_in_order[next_idx]
                                _aggregate(buffer.pop(expected))
                                maybe_gc_collect(next_idx, runtime.gc_collect_every_n_fovs)
                                next_idx += 1
                                pbar.update(1)
        finally:
            if seg_plate is not None:
                seg_plate.close()

    if config.compute_feature_metrics and all_feature_metrics:
        dataset_row: dict[str, float] = {}

        # Stage per-prefix inputs: (pred_for_metric, target_for_metric,
        # pred_for_probe, target_for_probe, pred_fovs, target_fovs).
        # CP gets pruning + z-score; the pre-prune CP arrays feed the
        # linear probe so MADScaler can normalize per-fold.
        prefix_inputs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        if pred_cp_feats:
            pred_cp_raw = np.concatenate(pred_cp_feats, axis=0)
            target_cp_raw = np.concatenate(gt_cp_feats, axis=0)
            target_cp_filtered, pred_cp_filtered, cp_keep_mask = select_features(target_cp_raw, pred_cp_raw)
            mask_payload = {
                "keep_mask": [bool(b) for b in cp_keep_mask],
                "n_kept": int(cp_keep_mask.sum()),
                "n_total": int(cp_keep_mask.size),
                "criteria": {
                    "freq_cut": DEFAULT_FREQ_CUT,
                    "unique_cut": DEFAULT_UNIQUE_CUT,
                    "corr_threshold": DEFAULT_CORR_THRESHOLD,
                },
            }
            (save_dir / "cp_selected_feature_mask.json").write_text(json.dumps(mask_payload, indent=2))
            if pred_cp_filtered.size and target_cp_filtered.size:
                pred_cp_z, target_cp_z = _zscore_per_side(pred_cp_filtered, target_cp_filtered)
            else:
                pred_cp_z, target_cp_z = pred_cp_filtered, target_cp_filtered
            prefix_inputs.append(
                (
                    "CP",
                    pred_cp_z,
                    target_cp_z,
                    pred_cp_filtered,
                    target_cp_filtered,
                    np.concatenate(pred_cp_fovs, axis=0),
                    np.concatenate(gt_cp_fovs, axis=0),
                )
            )

        deep_tracks = [
            ("DINOv3", pred_dinov3_feats, gt_dinov3_feats, pred_dinov3_fovs, gt_dinov3_fovs),
            ("DynaCLR", pred_dynaclr_feats, gt_dynaclr_feats, pred_dynaclr_fovs, gt_dynaclr_fovs),
        ]
        if celldino_feature_extractor is not None:
            deep_tracks.append(
                ("CellDINO", pred_celldino_feats, gt_celldino_feats, pred_celldino_fovs, gt_celldino_fovs)
            )
        for name, pred_feats, gt_feats, pred_fovs, gt_fovs in deep_tracks:
            if pred_feats:
                pred_arr = np.concatenate(pred_feats, axis=0)
                target_arr = np.concatenate(gt_feats, axis=0)
                prefix_inputs.append(
                    (
                        name,
                        pred_arr,
                        target_arr,
                        pred_arr,
                        target_arr,
                        np.concatenate(pred_fovs, axis=0),
                        np.concatenate(gt_fovs, axis=0),
                    )
                )

        # Prefix with "Dataset_" so dataset-level FID/KID/cosine don't clobber
        # per-FOV columns of the same name when merged into per-FOV rows.
        def _compute_one(args):
            name, p_metric, t_metric, p_probe, t_probe, fov_p, fov_t = args
            raw = {
                **compute_feature_similarity(p_metric, t_metric, name),
                **_real_vs_pred_probe(p_probe, t_probe, fov_p, fov_t, name),
            }
            return {f"Dataset_{k}": v for k, v in raw.items()}

        if prefix_inputs:
            # Threads suffice: torch-fidelity, sklearn LBFGS, and numpy BLAS
            # all release the GIL inside their hot loops. Cap inner BLAS to
            # 1 thread so the outer threads don't oversubscribe cores.
            with (
                threadpool_limits(limits=1),
                ThreadPoolExecutor(max_workers=min(4, len(prefix_inputs))) as pool,
            ):
                for result in pool.map(_compute_one, prefix_inputs):
                    dataset_row.update(result)

        # NaN-fill any prefix that had no cells (parallel pool would
        # otherwise skip it). Cheap; runs on empty arrays.
        expected_prefixes = ["CP", "DINOv3", "DynaCLR"]
        if celldino_feature_extractor is not None:
            expected_prefixes.append("CellDINO")
        for name in expected_prefixes:
            if f"Dataset_{name}_FID" not in dataset_row:
                raw = {
                    **compute_feature_similarity(np.empty((0, 0)), np.empty((0, 0)), name),
                    **_real_vs_pred_probe(np.empty((0, 0)), np.empty((0, 0)), np.empty(0), np.empty(0), name),
                }
                dataset_row.update({f"Dataset_{k}": v for k, v in raw.items()})

        for row in all_feature_metrics:
            row.update(dataset_row)
        embedding_groups = {
            "pred_cp": (pred_cp_feats, pred_cp_fovs, pred_cp_ts),
            "gt_cp": (gt_cp_feats, gt_cp_fovs, gt_cp_ts),
            "pred_dinov3": (pred_dinov3_feats, pred_dinov3_fovs, pred_dinov3_ts),
            "gt_dinov3": (gt_dinov3_feats, gt_dinov3_fovs, gt_dinov3_ts),
            "pred_dynaclr": (pred_dynaclr_feats, pred_dynaclr_fovs, pred_dynaclr_ts),
            "gt_dynaclr": (gt_dynaclr_feats, gt_dynaclr_fovs, gt_dynaclr_ts),
        }
        if celldino_feature_extractor is not None:
            embedding_groups["pred_celldino"] = (pred_celldino_feats, pred_celldino_fovs, pred_celldino_ts)
            embedding_groups["gt_celldino"] = (gt_celldino_feats, gt_celldino_fovs, gt_celldino_ts)
        _save_embeddings(save_dir, embedding_groups)

    dump_timings_csv(save_dir)

    return all_pixel_metrics, all_mask_metrics, all_feature_metrics


def save_metrics(config: DictConfig, pixel_metrics=None, mask_metrics=None, feature_metrics=None):
    """Save metrics to files."""
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for metrics, csv_name, npy_name, plot_dir in (
        (mask_metrics, config.save.mask_csv_filename, config.save.mask_metrics_filename, "mask_metrics"),
        (pixel_metrics, config.save.pixel_csv_filename, config.save.pixel_metrics_filename, "pixel_metrics"),
        (feature_metrics, config.save.feature_csv_filename, config.save.feature_metrics_filename, "feature_metrics"),
    ):
        if not metrics:
            continue
        df = pd.DataFrame(metrics)
        df.to_csv(save_dir / csv_name, index=False)
        np.save(save_dir / npy_name, metrics)
        print(f"Saved {plot_dir} to {save_dir / csv_name} and {save_dir / npy_name}")
        if not df.empty:
            plot_metrics(df, save_dir, plot_dir)
            print(f"Saved {plot_dir} plots to {save_dir / plot_dir}")


def _final_metrics_cache_valid(config: DictConfig) -> bool:
    """Return True when the saved CSV/NPY caches can be reused."""
    force = config.force_recompute
    if force.all or force.final_metrics:
        return False
    save_dir = Path(config.save.save_dir)
    pixel_ok = (save_dir / config.save.pixel_metrics_filename).exists()
    mask_ok = (save_dir / config.save.mask_metrics_filename).exists()
    feature_ok = (save_dir / config.save.feature_metrics_filename).exists() if config.compute_feature_metrics else True
    return pixel_ok and mask_ok and feature_ok


def _load_cached_final_metrics(config: DictConfig) -> tuple[list, list, list]:
    """Reload the (pixel, mask, feature) NPY caches saved by ``save_metrics``."""
    save_dir = Path(config.save.save_dir)
    pixel = np.load(save_dir / config.save.pixel_metrics_filename, allow_pickle=True).tolist()
    mask = np.load(save_dir / config.save.mask_metrics_filename, allow_pickle=True).tolist()
    if config.compute_feature_metrics:
        feature = np.load(save_dir / config.save.feature_metrics_filename, allow_pickle=True).tolist()
    else:
        feature = []
    return pixel, mask, feature


_MODEL_LOADING_FIELDS: tuple[str, ...] = (
    "target_name",
    "feature_extractor",
    "compute_feature_metrics",
    "use_gpu",
)


def _snapshot_field(cfg: DictConfig, cfg_field: str):
    """Resolve a model-loading field to a comparable plain Python value."""
    node = OmegaConf.select(cfg, cfg_field, default=None)
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=False)
    return node


def _merge_condition(base: DictConfig, overrides: DictConfig | dict) -> DictConfig:
    """Return a fresh DictConfig with ``overrides`` deep-merged into ``base``.

    ``OmegaConf.merge`` already returns a config independent of its inputs,
    so no upfront deep-copy is needed. The ``conditions`` field (the list
    we're iterating over) and the per-overlay ``name`` label are stripped
    from the result so each per-condition run sees a normal
    single-condition config shape.
    """
    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
    for key in ("conditions", "name"):
        if key in merged:
            del merged[key]
    return merged  # type: ignore[return-value]


def _check_grouped_field_invariants(base_snapshot: dict[str, object], merged: DictConfig, condition_name: str) -> None:
    """Raise if a per-condition merged config disagrees with the baseline on model-loading fields.

    ``base_snapshot`` is computed once per grouped run from the
    conditions-stripped base, so condition 0 is validated symmetrically
    with all later conditions (a condition 0 overlay that sneaks in e.g.
    ``target_name`` would be rejected, not silently adopted as the new
    "base").
    """
    for cfg_field in _MODEL_LOADING_FIELDS:
        merged_val = _snapshot_field(merged, cfg_field)
        if base_snapshot[cfg_field] != merged_val:
            raise ValueError(
                f"Condition {condition_name!r}: overrides changed model-loading field "
                f"{cfg_field!r}. Move it to the base config or run this condition separately."
            )


def evaluate_predictions_grouped(config: DictConfig) -> list[tuple[str, tuple]]:
    """Run ``evaluate_predictions`` over a list of conditions sharing one model load.

    Reads ``config.conditions`` (list of per-condition overrides). For each
    condition, merges its overrides into a copy of the base config and
    calls :func:`evaluate_predictions` with the shared ``EvalModels``.
    Models are loaded lazily on the first condition that misses its
    ``_final_metrics_cache_valid`` short-circuit, so restarts where every
    condition is cache-hit pay no cold-start.

    Conditions may freely override ``io.*``, ``save.*``, ``runtime.*``,
    ``limit_positions``, and ``force_recompute.*``. They must NOT change
    ``target_name``, ``feature_extractor.*``, ``compute_feature_metrics``,
    or ``use_gpu`` — those gate which models get loaded.

    Parameters
    ----------
    config : DictConfig
        Eval config with an extra top-level ``conditions: [...]`` list.
        Each entry is a dict-like overlay applied to the base.

    Returns
    -------
    list[tuple[str, tuple]]
        ``[(condition_name, (pixel_rows, mask_rows, feature_rows)), ...]``
        in input order. ``condition_name`` is taken from the entry's
        ``name`` field, falling back to its index as a string.

    Notes
    -----
    Under ``runtime.executor=process``, workers still load their own
    model copies per condition (the pool is rebuilt inside each
    ``evaluate_predictions`` call). Only the parent-side load is shared.
    Use ``executor=serial`` to maximize the amortization benefit.
    """
    conditions = OmegaConf.select(config, "conditions", default=None)
    if not conditions:
        raise ValueError("evaluate_predictions_grouped requires a non-empty top-level 'conditions' list")

    executor = OmegaConf.select(config, "runtime.executor", default="serial")
    require_complete = bool(OmegaConf.select(config, "io.require_complete_cache", default=False))
    n_conditions = len(conditions)
    if executor == "process" and n_conditions > 1:
        if require_complete:
            # Cache-only path: workers skip prepare_segmentation_model (returns
            # None) and don't instantiate extractors. Pool re-spawn is the only
            # overhead — mild informational note, not a warning.
            print(
                "[grouped] note: runtime.executor=process with require_complete_cache=true — "
                f"workers re-init per condition but no models actually load. "
                f"{n_conditions} pool spawns total; consider executor=serial to skip "
                "spawn overhead entirely."
            )
        else:
            # Worst-case: each condition's worker pool independently loads
            # SuperModel + DINOv3 + DynaCLR + CELL-DINO. Total wasted
            # cold-start ≈ load_time × N_workers × N_conditions.
            print(
                "\n"
                "[grouped] !!! WARNING: runtime.executor=process + "
                f"{n_conditions} conditions + require_complete_cache=false !!!\n"
                "  Each condition's worker pool independently loads SuperModel + "
                "DINOv3 + DynaCLR + CELL-DINO.\n"
                "  Expected waste: ~30-90 s × N_workers × "
                f"{n_conditions} conditions of redundant cold-start.\n"
                "  Fix: set runtime.executor=serial to share the parent's pre-loaded "
                "models across conditions.\n"
                "  Use process mode only when you need per-FOV parallelism within a "
                "single condition.\n"
            )

    # Canonical baseline for invariant checks: the input config with the
    # ``conditions`` list dropped. Snapshot once; conditions are validated
    # against this, including condition 0.
    models_base = OmegaConf.merge(config, OmegaConf.create({}))
    if "conditions" in models_base:
        del models_base["conditions"]
    base_snapshot = {field: _snapshot_field(models_base, field) for field in _MODEL_LOADING_FIELDS}

    models: EvalModels | None = None

    def get_models() -> EvalModels:
        nonlocal models
        if models is None:
            print(f"[grouped] loading shared models for {len(conditions)} conditions ...")
            models = load_eval_models(models_base)
        return models

    results: list[tuple[str, tuple]] = []
    for idx, cond in enumerate(conditions):
        name = (
            cond.get("name", str(idx)) if isinstance(cond, dict) else OmegaConf.select(cond, "name", default=str(idx))
        )
        merged = _merge_condition(config, cond)
        apply_dataset_ref(merged)
        _check_grouped_field_invariants(base_snapshot, merged, name)
        print(f"[grouped] ({idx + 1}/{len(conditions)}) evaluating {name!r} → {merged.save.save_dir}")

        if _final_metrics_cache_valid(merged):
            print(f"[grouped] ({idx + 1}/{len(conditions)}) {name!r}: reusing cached final metrics")
            pixel_metrics, mask_metrics, feature_metrics = _load_cached_final_metrics(merged)
        else:
            pixel_metrics, mask_metrics, feature_metrics = evaluate_predictions(merged, models=get_models())
            save_metrics(
                merged,
                pixel_metrics=pixel_metrics,
                mask_metrics=mask_metrics,
                feature_metrics=feature_metrics,
            )
        results.append((name, (pixel_metrics, mask_metrics, feature_metrics)))

    return results


@hydra.main(version_base="1.2", config_path="_configs", config_name="eval")
def evaluate_model(config: DictConfig):
    """Evaluate model on test images."""
    apply_dataset_ref(config)
    if _final_metrics_cache_valid(config):
        print("Found existing metrics.")
        pixel_metrics, mask_metrics, feature_metrics = _load_cached_final_metrics(config)
    else:
        pixel_metrics, mask_metrics, feature_metrics = evaluate_predictions(config)
        save_metrics(
            config,
            pixel_metrics=pixel_metrics,
            mask_metrics=mask_metrics,
            feature_metrics=feature_metrics,
        )
    return pixel_metrics, mask_metrics, feature_metrics


@hydra.main(version_base="1.2", config_path="_configs", config_name="eval_grouped")
def evaluate_model_grouped(config: DictConfig):
    """Run grouped multi-condition eval, amortizing model loads across conditions."""
    return evaluate_predictions_grouped(config)


if __name__ == "__main__":
    evaluate_model()
