"""Batch orchestration: load, segment, evaluate, save."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args

import hydra
import numpy as np
import pandas as pd
import torch
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.cache import FeatureKind
from dynacell.evaluation.cross_condition_probe import run_for_group as _cross_condition_run_for_group
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
    active_cp_feature_names,
    ascupy,
    build_crops,
    compute_pixel_metrics,
    cp_regionprops,
    drop_paired_nonfinite_rows,
    evaluate_segmentations,
    features_from_crops,
    fit_microssim,
    per_cell_similarity,
    score_microssim,
)
from dynacell.evaluation.model_loader import EvalModels, init_cache_contexts, load_eval_models
from dynacell.evaluation.pipeline_cache import (
    flush_manifest,
    fov_cp_features,
    fov_deep_features,
    fov_masks,
    fov_nucleus_instances,
    fov_whole_cell_instances,
    instance_cache_hit,
    precompute_deep_features,
)
from dynacell.evaluation.runtime import (
    apply_thread_budget,
    dump_timings_csv,
    extend_timings,
    get_timings,
    gpu_serialization_lock,
    is_worker,
    make_fov_executor,
    maybe_empty_cuda_cache,
    maybe_gc_collect,
    region_timer,
    reset_timings,
    resolve_runtime,
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
    use_gpu = pred_cache_ctx.use_gpu
    for t in range(t_count):
        cp.append(
            cp_regionprops(
                predict[t],
                cell_segmentation[t],
                spacing,
                norm=pred_cache_ctx.cp_norm,
                glcm_cfg=pred_cache_ctx.cp_glcm,
                use_gpu=use_gpu,
            )
        )
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


# Single source of truth for backbone names is cache.FeatureKind.
# Ordering matters: the post-loop CP block runs before the generic
# deep-track loop, so "cp" must be first in FeatureKind.
_BACKBONE_KEYS: tuple[FeatureKind, ...] = get_args(FeatureKind)
_BB_FIELDS = fields(_BackboneLists)


def _extend_backbone(
    bb: _BackboneLists,
    pred: np.ndarray | None,
    gt: np.ndarray | None,
    pos_name: str,
    t: int,
) -> None:
    """Append one (pred, gt) feature pair for ``(pos_name, t)`` into ``bb``.

    No-op when ``pred`` is None (CellDINO disabled) or zero rows (no cells).
    All six lists in ``_BackboneLists`` (feats, fovs, ts for pred and gt) are
    extended with arrays of length ``len(pred)`` to stay lockstep.

    Raises
    ------
    ValueError
        If ``pred`` has rows but ``gt`` is None, or if ``len(pred) != len(gt)``.
        ``_BackboneLists.gt_feats`` is typed ``list[np.ndarray]`` and the
        post-loop block calls ``np.concatenate`` over it — a stray ``None``
        or row-count mismatch would surface as a confusing crash deep in
        the metrics block; fail loudly at the append site instead.
    """
    if pred is None or pred.size == 0:
        return
    if gt is None:
        raise ValueError(f"_extend_backbone: pred has {len(pred)} rows at ({pos_name!r}, t={t}) but gt is None")
    if len(gt) != len(pred):
        raise ValueError(
            f"_extend_backbone: row count mismatch at ({pos_name!r}, t={t}): pred has {len(pred)}, gt has {len(gt)}"
        )
    bb.pred_feats.append(pred)
    bb.gt_feats.append(gt)
    fov_arr = np.full(len(pred), pos_name)
    t_arr = np.full(len(pred), t, dtype=np.int32)
    bb.pred_fovs.append(fov_arr)
    bb.gt_fovs.append(fov_arr)
    bb.pred_ts.append(t_arr)
    bb.gt_ts.append(t_arr)


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


def _calibrate_microssim(
    pred_positions,
    gt_positions,
    io_config,
    *,
    use_gpu: bool,
    max_pairs: int,
    seed: int,
    cache_reads: bool,
) -> tuple[Any, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit MicroMS3IM once per leaf on a random subsample of (FOV, t) volumes.

    The microSSIM paper (Ashesh & Jug, 2024) is explicit that α is a
    single scalar fit over the whole dataset, not refit per image pair.
    Per-pair fitting inflates scores and prevents cross-FOV / cross-leaf
    comparison. α is a population statistic, so a representative
    subsample is sufficient — fitting on the full pool was OOMing on
    80 GB GPUs (~78 GB allocated for 100 FOVs × D=40 × 512² × 2 sides
    plus cubic's SSIM working buffers).

    Strategy: build the joint pool of `(pos_idx, t_idx)` pairs across all
    positions and their actual T, draw ``max_pairs`` of them with a
    seeded RNG, then load the full ``(D, H, W)`` volume for each sampled
    pair. Concatenating along the D-axis gives a `(sum_D, H, W)` stack
    that's fed to :func:`fit_microssim` exactly once.

    Volumes stay whole — only the (FOV, t) axis is sampled. The in-focus
    z-plane shifts across FOVs/timepoints in live-cell data, so taking
    full z-stacks preserves whatever axial distribution biology has.

    Parameters
    ----------
    max_pairs : int
        Cap on the number of (FOV, t) volumes used for fitting. Default
        from the caller is 12 (small enough to keep ~960 MB headroom on
        GPU even with feature extractors + segmenter resident).
    seed : int
        RNG seed; deterministic across re-runs.
    cache_reads : bool
        When ``True``, the returned ``read_cache`` dict maps each sampled
        position's name to ``(predict_TDHW, target_TDHW)`` arrays so the
        serial per-FOV loop can skip re-reading those positions from zarr.
        Worth ~1 GB host RAM per 12 cached FOVs. Process-mode workers
        can't see the parent's cache and silently re-read; pass
        ``cache_reads=True`` only when ``runtime.executor=serial``.

    Returns
    -------
    (sim, read_cache)
        ``sim`` is the fitted MicroMS3IM (or raises on degenerate input).
        ``read_cache`` is the per-position array dict; empty when
        ``cache_reads=False``.

    Raises
    ------
    ValueError, RuntimeError
        Propagated from ``fit_microssim``; the caller wraps this call in
        ``try/except`` so a degenerate leaf falls back to MicroMS3IM=NaN
        instead of aborting the rest of the eval.
    """
    n_positions = len(pred_positions)
    if n_positions == 0:
        return None, {}

    pairs: list[tuple[int, int]] = []
    for pos_idx, (_, pos_pred) in enumerate(pred_positions):
        T_pos = int(pos_pred.data.shape[0])
        pairs.extend((pos_idx, t) for t in range(T_pos))

    if not pairs:
        return None, {}

    n_sample = min(max_pairs, len(pairs))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(pairs), size=n_sample, replace=False)
    sampled_by_pos: dict[int, list[int]] = {}
    for s in sample_idx:
        pos_idx, t_idx = pairs[int(s)]
        sampled_by_pos.setdefault(pos_idx, []).append(t_idx)

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    read_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for pos_idx, t_indices in sampled_by_pos.items():
        pos_name, pos_pred = pred_positions[pos_idx]
        _, pos_gt = gt_positions[pos_idx]
        pred_ci = pos_pred.get_channel_index(io_config.pred_channel_name)
        gt_ci = pos_gt.get_channel_index(io_config.gt_channel_name)
        predict = np.asarray(pos_pred.data[:, pred_ci])  # (T, D, H, W)
        target = np.asarray(pos_gt.data[:, gt_ci])
        for t_idx in t_indices:
            all_predictions.append(predict[t_idx])  # (D, H, W)
            all_targets.append(target[t_idx])
        if cache_reads:
            read_cache[pos_name] = (predict, target)

    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    sim = fit_microssim(targets, predictions, use_gpu=use_gpu)
    return sim, read_cache


def _validate_instance_ap_config(config: DictConfig) -> None:
    """Validate the instance-AP backend / target / toggle combination.

    Bidirectional guard (raises ``ValueError`` on any invalid pairing):

    - ``backend='cellpose_watershed'`` requires ``target_name='membrane'``, a
      non-null ``segmentation.nuclei_channel_name`` (the GT watershed seeds), and
      ``compute_instance_ap=true`` (whole-cell AP is the backend's sole purpose).
    - ``compute_instance_ap=true`` requires an instance-producing pair:
      ``(cellpose_watershed, membrane)`` or ``(cellpose, nucleus)``.
    """
    backend = OmegaConf.select(config, "segmentation.backend", default="supermodel")
    compute_instance_ap = bool(getattr(config, "compute_instance_ap", False))
    target_name = config.target_name
    nuclei_channel = OmegaConf.select(config, "segmentation.nuclei_channel_name", default=None)

    if backend == "cellpose_watershed":
        if target_name != "membrane":
            raise ValueError("segmentation.backend='cellpose_watershed' requires target_name='membrane'")
        if nuclei_channel is None:
            raise ValueError(
                "segmentation.backend='cellpose_watershed' requires segmentation.nuclei_channel_name "
                "(the GT-plate channel for the watershed seeds)"
            )
        if not compute_instance_ap:
            raise ValueError("segmentation.backend='cellpose_watershed' requires compute_instance_ap=true")

    if compute_instance_ap:
        valid = (backend == "cellpose_watershed" and target_name == "membrane") or (
            backend == "cellpose" and target_name == "nucleus"
        )
        if not valid:
            raise ValueError(
                "compute_instance_ap=true requires an instance-producing backend/target: "
                "(backend='cellpose_watershed', target_name='membrane') or "
                "(backend='cellpose', target_name='nucleus'); "
                f"got backend={backend!r}, target_name={target_name!r}"
            )


def _process_one_fov(
    config: DictConfig,
    cuda_empty_cache_every_n_timepoints: int,
    pos_name_pred: str,
    pos_pred,
    pos_gt,
    pos_seg,
    pos_nuclei,
    io_config,
    cache_ctx,
    pred_cache_ctx,
    seg_model,
    dinov3_feature_extractor,
    dynaclr_feature_extractor,
    celldino_feature_extractor,
    microssim_sim,
    predict_cached=None,
    target_cached=None,
) -> FovResult:
    """Compute everything one FOV contributes to the eval and return a FovResult.

    No side effects on shared parent state (no segmentation_results plate
    writes, no manifest flush). The parent aggregator handles those — see
    ``_aggregate_fov_result``. Used by both the serial and process FOV-loop
    paths in ``evaluate_predictions``.
    """
    from dynacell.evaluation.instance_metrics import instance_average_precision
    from dynacell.evaluation.segmentation import segment
    from dynacell.evaluation.segmentation_cellpose import segment_nucleus_instances
    from dynacell.evaluation.segmentation_whole_cell import slice_index

    timings_start = len(get_timings())
    # Inner per-T tqdm is noise when N workers each emit it to the shared
    # parent stderr — outer per-FOV tqdm in the parent stays visible either way.
    suppress_inner_tqdm = is_worker()
    # GPU serialization lock is a no-op when use_gpu=false: under that
    # setting compute_pixel_metrics, cellpose, and feature extractors all
    # run CPU-only, so cross-worker fcntl serialization would just add
    # latency for nothing.
    use_gpu = bool(getattr(config, "use_gpu", True))
    compute_cell_similarity = bool(OmegaConf.select(config, "compute_cell_similarity", default=False))
    cell_sim_metrics = tuple(OmegaConf.select(config, "cell_similarity.metrics", default=["pcc"]))
    cell_sim_reduce = tuple(OmegaConf.select(config, "cell_similarity.reduce", default=["mean", "median"]))

    if predict_cached is not None and target_cached is not None:
        # Reuse arrays already read by _calibrate_microssim (serial mode opt-in
        # via microssim.calibration.cache=true). Avoids the double zarr read
        # for the small subset of FOVs that contributed to the leaf-level fit.
        predict = predict_cached
        target = target_cached
    else:
        pred_channel_index = pos_pred.get_channel_index(io_config.pred_channel_name)
        gt_channel_index = pos_gt.get_channel_index(io_config.gt_channel_name)
        predict = np.asarray(pos_pred.data[:, pred_channel_index])  # shape: (T, D, H, W)
        target = np.asarray(pos_gt.data[:, gt_channel_index])
    cell_segmentation = np.asarray(pos_seg.data[:, 0]) if pos_seg is not None else None

    T = predict.shape[0]

    # Mask path: when compute_instance_ap is set, the binary fov_masks path is
    # replaced by an instance-label path (nucleus or whole-cell). gt_cells /
    # pred_cells are (T, D, H, W) uint16 instance labels; the binary mask metrics
    # are derived from labels>0 in the per-T loop. Otherwise the legacy binary
    # path runs byte-identically.
    instance_mode = bool(getattr(config, "compute_instance_ap", False))
    backend = OmegaConf.select(config, "segmentation.backend", default="supermodel")
    gt_cells = pred_cells = None
    gt_mask_stack = pred_mask_stack = None
    if instance_mode:
        is3d = OmegaConf.select(config, "segmentation.dimension", default="2d") == "3d"
        # Separate working arrays — never rebind predict/target (the 3D pixel
        # metrics + MicroMS3IM below keep using the full native arrays).
        if is3d:
            target_cells, predict_cells = target, predict
        else:
            sel = OmegaConf.select(config, "segmentation.slice_selection", default="frac")
            frac = float(OmegaConf.select(config, "segmentation.slice_fraction", default=0.30))
            z_idx = [slice_index(target[t], selection=sel, fraction=frac) for t in range(T)]
            target_cells = np.stack([target[t, z_idx[t]] for t in range(T)])  # (T, Y, X)
            predict_cells = np.stack([predict[t, z_idx[t]] for t in range(T)])
        if backend == "cellpose_watershed":
            nuclei_channel = OmegaConf.select(config, "segmentation.nuclei_channel_name", default=None)
            # GT nuclei seeds come from a separate store (pos_nuclei) when the GT
            # nuclei live apart from the GT membrane (A549: membrane in CAAX_*.ozx,
            # nuclei in H2B_*.ozx); pos_nuclei is the same-named position from that
            # store. When None (iPSC cell.zarr), read nuclei from the GT membrane
            # plate, byte-identical to the single-store path.
            nuclei_source = pos_nuclei if pos_nuclei is not None else pos_gt
            nuclei = np.asarray(nuclei_source.data[:, nuclei_source.get_channel_index(nuclei_channel)])  # (T, Z, Y, X)
            nuclei_cells = nuclei if is3d else np.stack([nuclei[t, z_idx[t]] for t in range(T)])
            # Seed preflight: compute GT-nuclei watershed seeds only when at least
            # one side will actually run segment_whole_cell (a disabled cache or a
            # manifest-invalidated/forced slot counts as a miss).
            gt_will_compute = not cache_ctx.enabled or not instance_cache_hit(cache_ctx, pos_name_pred)
            pred_will_compute = not pred_cache_ctx.enabled or not instance_cache_hit(pred_cache_ctx, pos_name_pred)
            seed_stack = None
            if gt_will_compute or pred_will_compute:
                seg_spacing = tuple(config.pixel_metrics.spacing) if is3d else tuple(config.pixel_metrics.spacing[-2:])
                with region_timer("nucleus_seeds", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                    seed_stack = np.stack(
                        [
                            segment_nucleus_instances(
                                nuclei_cells[t], seg_spacing, seg_model, do_3d=is3d, **cache_ctx.cellpose_params
                            )
                            for t in range(T)
                        ]
                    )
            with region_timer("mask_gt", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                gt_cells = fov_whole_cell_instances(cache_ctx, pos_name_pred, target_cells, nuclei_cells, seed_stack)
            with region_timer("mask_pred", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                pred_cells = fov_whole_cell_instances(
                    pred_cache_ctx, pos_name_pred, predict_cells, nuclei_cells, seed_stack
                )
        else:  # backend == "cellpose": independent per-side nucleus instances
            with region_timer("mask_gt", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                gt_cells = fov_nucleus_instances(cache_ctx, pos_name_pred, target_cells, seg_model)
            with region_timer("mask_pred", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                pred_cells = fov_nucleus_instances(pred_cache_ctx, pos_name_pred, predict_cells, seg_model)
    else:
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

    # Bulk-upload predict/target once per FOV so compute_pixel_metrics' per-T
    # ascupy is a no-op via cupy-on-cupy. mask / feature / microssim paths
    # still consume the numpy arrays in parallel — neither SuperModel nor
    # build_crops nor the microssim aggregator accept cupy directly. The lock
    # serializes the cupy allocation across workers under executor=process so
    # N FOVs aren't all trying to allocate (T,D,H,W) fp32 at once.
    if use_gpu and ascupy is not None and torch.cuda.is_available():
        with gpu_serialization_lock(gate=use_gpu):
            predict_xp = ascupy(predict)
            target_xp = ascupy(target)
    else:
        predict_xp = predict
        target_xp = target

    for t in tqdm(range(T), desc="Processing timepoints", leave=False, disable=suppress_inner_tqdm):
        data_info = {"FOV": pos_name_pred, "Timepoint": t}

        with region_timer("pixel_metrics", pos_name_pred, t), gpu_serialization_lock(gate=use_gpu):
            pixel_metrics = compute_pixel_metrics(
                predict_xp[t],
                target_xp[t],
                spacing=config.pixel_metrics.spacing,
                fsc_kwargs=config.pixel_metrics.fsc,
                spectral_pcc_kwargs=config.pixel_metrics.spectral_pcc,
                use_gpu=use_gpu,
            )
        pixel_row = {**data_info, **pixel_metrics}
        if compute_cell_similarity and cell_segmentation is not None:
            with region_timer("cell_similarity", pos_name_pred, t), gpu_serialization_lock(gate=use_gpu):
                pixel_row.update(
                    per_cell_similarity(
                        predict[t],
                        target[t],
                        cell_segmentation[t],
                        metrics=cell_sim_metrics,
                        reduce=cell_sim_reduce,
                        use_gpu=use_gpu,
                    )
                )
        if config.compute_microssim:
            microssim_data.append({"target": target[t], "predict": predict[t]})
        fov_pixel_metrics.append(pixel_row)

        with region_timer("mask_metrics", pos_name_pred, t):
            if instance_mode:
                gt_lab = gt_cells[t]
                pred_lab = pred_cells[t]
                segmented_target = gt_lab > 0
                segmented_predict = pred_lab > 0
                with region_timer("instance_ap", pos_name_pred, t):
                    ap_metrics = instance_average_precision(pred_lab, gt_lab, cache_ctx.iou_thresholds)
                fov_mask_metrics.append(
                    {**data_info, **evaluate_segmentations(segmented_predict, segmented_target), **ap_metrics}
                )
            else:
                segmented_target = gt_mask_stack[t]
                if pred_mask_stack is not None:
                    segmented_predict = pred_mask_stack[t]
                else:
                    with gpu_serialization_lock(gate=use_gpu):
                        segmented_predict = np.asarray(
                            segment(
                                predict[t],
                                config.target_name,
                                seg_model=seg_model,
                                backend=OmegaConf.select(config, "segmentation.backend", default="supermodel"),
                                spacing_zyx=tuple(config.pixel_metrics.spacing),
                            )
                        ).astype(bool)
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
                _extend_backbone(cp, pred_cp, gt_cp_t, pos_name_pred, t)
                _extend_backbone(dinov3, pred_dinov3, gt_dinov3_per_t[t], pos_name_pred, t)
                _extend_backbone(dynaclr, pred_dynaclr, gt_dynaclr_per_t[t], pos_name_pred, t)
                gt_celldino_t = gt_celldino_per_t[t] if pred_celldino is not None else None
                _extend_backbone(celldino, pred_celldino, gt_celldino_t, pos_name_pred, t)

        maybe_empty_cuda_cache(t, cuda_empty_cache_every_n_timepoints)

    seg_array = np.stack(segmentations, axis=0)  # shape: (T, 2, D, H, W)

    if config.compute_microssim:
        if microssim_sim is None:
            # Leaf-level calibration failed (degenerate slice / OOM / cubic
            # bracket failure) — the parent already logged the cause. Emit
            # NaN per timepoint so the column exists and the rest of the
            # pixel / mask / feature metrics still get computed.
            for i in range(T):
                fov_pixel_metrics[i]["MicroMS3IM"] = float("nan")
        else:
            with region_timer("microssim", pos_name_pred), gpu_serialization_lock(gate=use_gpu):
                microssim_scores = score_microssim(microssim_data, microssim_sim, use_gpu=use_gpu)
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
    parent_lists: dict[str, _BackboneLists],
    *,
    extend_worker_timings: bool,
) -> None:
    """Apply one FOV's contributions to the parent-side run state.

    Writes the segmentation array to the HCS plate, extends the per-T row
    lists, and extends each backbone's six lists in ``parent_lists`` from
    the matching ``_BackboneLists`` slot on ``result``.

    ``extend_worker_timings`` toggles whether to append ``result.timings``
    to the parent's global ``_TIMINGS`` collector. Set ``False`` in serial
    mode (workers and parent share one collector, so the timings are
    already there); set ``True`` in process mode (workers have separate
    per-process collectors, so the parent must aggregate).
    """
    if extend_worker_timings:
        extend_timings(result.timings)

    with region_timer("seg_write", result.pos_name):
        seg_pos = segmentation_results.create_position(result.row, result.col, result.fov)
        seg_pos.create_image("0", result.seg_array)

    all_pixel_metrics.extend(result.per_t_pixel_rows)
    all_mask_metrics.extend(result.per_t_mask_rows)
    all_feature_metrics.extend(result.per_t_feature_rows)

    # Canary: if a non-list field is added to _BackboneLists, the .extend(...)
    # below will raise AttributeError on the new field.
    for name in _BACKBONE_KEYS:
        worker_bb = getattr(result, name)
        parent_bb = parent_lists[name]
        for bb_field in _BB_FIELDS:
            getattr(parent_bb, bb_field.name).extend(getattr(worker_bb, bb_field.name))


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


def _separate_nuclei_path(config: DictConfig) -> str | None:
    """GT-nuclei store path when it is a *separate* store from the GT membrane plate.

    Returns ``io.nuclei_gt_path`` when set and distinct from ``io.gt_path`` (the A549
    cross-store case — membrane in ``CAAX_*.ozx``, nuclei in ``H2B_*.ozx``), else
    ``None`` (iPSC single-store ``cell.zarr`` — nuclei read from the GT plate).

    Only the ``cellpose_watershed`` instance-AP path consumes a separate GT-nuclei
    store; for any other backend ``nuclei_gt_path`` is an inert ``io`` field, so the
    helper returns ``None`` rather than opening + position-validating a store that
    will never be read.
    """
    backend = OmegaConf.select(config, "segmentation.backend", default="supermodel")
    compute_instance_ap = bool(getattr(config, "compute_instance_ap", False))
    if not (compute_instance_ap and backend == "cellpose_watershed"):
        return None
    nuclei_path = OmegaConf.select(config, "io.nuclei_gt_path", default=None)
    if nuclei_path is None or str(nuclei_path) == str(config.io.gt_path):
        return None
    return str(nuclei_path)


def _find_position(plate, pos_name: str):
    """Return the iohub Position with name ``pos_name`` from ``plate``.

    Scans ``plate.positions()`` since iohub doesn't index by name. Cheap
    for the ≤ tens of positions per eval.
    """
    for name, pos in plate.positions():
        if name == pos_name:
            return pos
    raise KeyError(f"position {pos_name!r} not found in plate")


def _worker_run_fov(
    config: DictConfig,
    pos_name: str,
    cuda_empty_every_n: int,
    microssim_sim,
) -> FovResult:
    """Worker entry point: process one FOV by name and return FovResult.

    Submitted via ``pool.submit`` in ``executor=process`` mode. Plates
    are opened with a context manager scoped to this single call — iohub
    file descriptors close before the worker accepts its next FOV. Models
    + cache contexts stay cached in ``_WORKER_STATE`` across FOVs.

    ``microssim_sim`` is the leaf-level fitted MicroMS3IM (or ``None`` when
    ``compute_microssim=false``); shipped per submission rather than via
    worker state because the parent fits it after the position list is
    finalized and before any worker pool spawns.
    """
    _worker_setup(config)
    state = _WORKER_STATE

    seg_path = config.io.cell_segmentation_path
    # GT nuclei may live in a separate store (cellpose_watershed cross-store seeds).
    nuclei_path = _separate_nuclei_path(config)
    # ExitStack so the two optional auxiliary stores (cell_segmentation, nuclei)
    # don't nest combinatorially; iohub fds close before the next FOV.
    with ExitStack() as stack:
        pred_plate = stack.enter_context(open_ome_zarr(Path(config.io.pred_path), mode="r"))
        gt_plate = stack.enter_context(open_ome_zarr(Path(config.io.gt_path), mode="r"))
        pos_pred = _find_position(pred_plate, pos_name)
        pos_gt = _find_position(gt_plate, pos_name)

        pos_seg = None
        if seg_path is not None:
            seg_plate = stack.enter_context(open_ome_zarr(Path(seg_path), mode="r"))
            pos_seg = _find_position(seg_plate, pos_name)

        pos_nuclei = None
        if nuclei_path is not None:
            nuclei_plate = stack.enter_context(open_ome_zarr(Path(nuclei_path), mode="r"))
            pos_nuclei = _find_position(nuclei_plate, pos_name)

        result = _process_one_fov(
            config,
            cuda_empty_every_n,
            pos_name,
            pos_pred,
            pos_gt,
            pos_seg,
            pos_nuclei,
            config.io,
            state["cache_ctx"],
            state["pred_cache_ctx"],
            state["seg_model"],
            state["dinov3"],
            state["dynaclr"],
            state["celldino"],
            microssim_sim,
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
    _validate_instance_ap_config(config)

    use_gpu = bool(getattr(config, "use_gpu", True))

    all_pixel_metrics = []
    all_mask_metrics = []
    all_feature_metrics = []

    io_config = config.io
    pred_path = Path(io_config.pred_path)
    gt_path = Path(io_config.gt_path)
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    needs_segmentation = config.compute_feature_metrics or bool(
        OmegaConf.select(config, "compute_cell_similarity", default=False)
    )
    if needs_segmentation and io_config.cell_segmentation_path is None:
        raise ValueError(
            "io.cell_segmentation_path is required when compute_feature_metrics=true or compute_cell_similarity=true"
        )

    with region_timer("parent_load_models", "<parent>"):
        if models is None:
            models = load_eval_models(config)
        seg_model = models.seg_model
        dinov3_feature_extractor = models.dinov3
        dynaclr_feature_extractor = models.dynaclr
        celldino_feature_extractor = models.celldino

        cache_ctx, pred_cache_ctx = init_cache_contexts(config, models)

    seg_path = Path(io_config.cell_segmentation_path) if io_config.cell_segmentation_path is not None else None

    parent_lists: dict[str, _BackboneLists] = {name: _BackboneLists() for name in _BACKBONE_KEYS}

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
        # Auxiliary stores (cell_segmentation + a separate GT-nuclei store for the
        # cross-store cellpose_watershed seeds) are opened under an ExitStack
        # *inside* the try, so a raise mid-open can never leak an already-opened
        # plate; the finally closes whatever was entered (matches _worker_run_fov).
        aux_stack = ExitStack()
        seg_plate = None
        nuclei_plate = None
        nuclei_by_name = None
        try:
            if seg_path is not None:
                seg_plate = aux_stack.enter_context(open_ome_zarr(seg_path, mode="r"))
                seg_positions = list(seg_plate.positions())
            else:
                seg_positions = [(name, None) for name, _ in pred_positions]

            # Optional separate GT-nuclei store (cellpose_watershed cross-store seeds).
            # Positions are matched to pred/gt by name (verified 1:1 for A549 caax/h2b),
            # so a name→position dict suffices; workers (process mode) reopen by name.
            nuclei_path = _separate_nuclei_path(config)
            if nuclei_path is not None:
                nuclei_plate = aux_stack.enter_context(open_ome_zarr(Path(nuclei_path), mode="r"))
                nuclei_by_name = dict(nuclei_plate.positions())
            # Position-count alignment.
            #
            # When ``limit_positions`` is unset (production), require strict
            # equality so partial pred zarrs cannot silently eval against the
            # wrong gt — historical behavior.
            #
            # When ``limit_positions=N`` is set (smoke / iteration), allow pred
            # to be a strict subset of gt by name: reorder gt + seg to align
            # with pred, then truncate. Catches the ``--fast_dev_run`` predict
            # → ``limit_positions=N`` eval case where pred has only the first
            # N FOVs but gt has all of them.
            limit = getattr(config, "limit_positions", None)
            if limit is None:
                if len(pred_positions) != len(gt_positions):
                    raise ValueError(f"Position count mismatch: pred={len(pred_positions)}, gt={len(gt_positions)}")
                if seg_plate is not None and len(seg_positions) != len(pred_positions):
                    raise ValueError(f"Position count mismatch: pred={len(pred_positions)}, seg={len(seg_positions)}")
            else:
                pred_name_set = {name for name, _ in pred_positions}
                extra_in_pred = pred_name_set - {name for name, _ in gt_positions}
                if extra_in_pred:
                    raise ValueError(
                        "limit_positions: pred contains positions not present in gt: "
                        f"{sorted(extra_in_pred)!r}. The relaxed alignment only works "
                        "when pred is a subset of gt by position name."
                    )
                gt_by_name = dict(gt_positions)
                gt_positions = [(name, gt_by_name[name]) for name, _ in pred_positions]
                if seg_plate is not None:
                    seg_by_name = dict(seg_positions)
                    missing_seg = pred_name_set - set(seg_by_name)
                    if missing_seg:
                        raise ValueError(
                            f"limit_positions: pred contains positions not present in seg: {sorted(missing_seg)!r}."
                        )
                    seg_positions = [(name, seg_by_name[name]) for name, _ in pred_positions]
                pred_positions = pred_positions[:limit]
                gt_positions = gt_positions[:limit]
                seg_positions = seg_positions[:limit]
            # Hoist paired-name validation so precompute (which runs before
            # the per-FOV loop) cannot write to mismatched cache slots.
            for (pos_name_pred, _), (pos_name_gt, _), (pos_name_seg, _) in zip(
                pred_positions, gt_positions, seg_positions
            ):
                if pos_name_pred != pos_name_gt:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, gt={pos_name_gt!r}")
                if seg_plate is not None and pos_name_seg != pos_name_pred:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, seg={pos_name_seg!r}")
            if nuclei_by_name is not None:
                missing_nuclei = {n for n, _ in pred_positions} - set(nuclei_by_name)
                if missing_nuclei:
                    raise ValueError(
                        f"nuclei_gt_path store is missing positions {sorted(missing_nuclei)!r} "
                        "(cellpose_watershed reads GT-nuclei seeds there, matched by position name)"
                    )

            # Leaf-level MicroMS3IM calibration: fit α once on a random
            # subsample of (FOV, t) volumes and reuse the fitted sim for
            # scoring every FOV. Per the microSSIM paper (Ashesh & Jug, 2024
            # §3.3) α is a single scalar fit over the dataset — per-pair
            # fitting inflates the metric and breaks cross-FOV comparability.
            # Default ``max_pairs=12`` keeps the GPU fit pool ~960 MB even
            # with extractors + segmenter resident; the full pool (100+ FOVs)
            # was OOMing at ~78 GB during the 2026-05-26 CellDINO regen.
            # Cache is opt-in (serial-only, ~1 GB host RAM held longer to
            # skip re-reading the subsampled FOVs in the per-FOV pass).
            microssim_sim = None
            microssim_read_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if config.compute_microssim:
                max_pairs = int(OmegaConf.select(config, "microssim.calibration.max_pairs", default=12))
                seed = int(OmegaConf.select(config, "microssim.calibration.seed", default=42))
                cache_reads = bool(OmegaConf.select(config, "microssim.calibration.cache", default=False))
                try:
                    with region_timer("microssim_calibrate", "(leaf)"), gpu_serialization_lock(gate=use_gpu):
                        microssim_sim, microssim_read_cache = _calibrate_microssim(
                            pred_positions,
                            gt_positions,
                            io_config,
                            use_gpu=use_gpu,
                            max_pairs=max_pairs,
                            seed=seed,
                            cache_reads=cache_reads,
                        )
                except (ValueError, RuntimeError, MemoryError) as exc:
                    print(
                        f"[microssim] leaf-level calibration failed "
                        f"({type(exc).__name__}: {exc}); MicroMS3IM will be NaN for all FOVs."
                    )
                    microssim_sim = None
                    microssim_read_cache = {}

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
                    with region_timer("precompute_all", "<parent>"):
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
                    parent_lists,
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
                    pos_nuclei = nuclei_by_name[pos_name_pred] if nuclei_by_name is not None else None

                    cached_pair = microssim_read_cache.get(pos_name_pred, (None, None))
                    result = _process_one_fov(
                        config,
                        runtime.cuda_empty_cache_every_n_timepoints,
                        pos_name_pred,
                        pos_pred,
                        pos_gt,
                        pos_seg,
                        pos_nuclei,
                        io_config,
                        cache_ctx,
                        pred_cache_ctx,
                        seg_model,
                        dinov3_feature_extractor,
                        dynaclr_feature_extractor,
                        celldino_feature_extractor,
                        microssim_sim,
                        predict_cached=cached_pair[0],
                        target_cached=cached_pair[1],
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
                            _worker_run_fov,
                            config,
                            pos_name,
                            runtime.cuda_empty_cache_every_n_timepoints,
                            microssim_sim,
                        ): pos_name
                        for pos_name in pos_names_in_order
                    }
                    with tqdm(total=len(futures), desc="Processing positions") as pbar:
                        for fut in as_completed(futures):
                            pos_name = futures[fut]
                            buffer[pos_name] = fut.result()
                            # Advance the bar on every worker completion so the
                            # operator sees real progress even when FOVs arrive
                            # out of order. The in-order drain below releases
                            # seg_array refs as soon as the aggregator's plate
                            # write completes; that's an internal step, not a
                            # user-visible unit of work.
                            pbar.update(1)
                            while next_idx < len(pos_names_in_order) and (pos_names_in_order[next_idx] in buffer):
                                expected = pos_names_in_order[next_idx]
                                _aggregate(buffer.pop(expected))
                                maybe_gc_collect(next_idx, runtime.gc_collect_every_n_fovs)
                                next_idx += 1
        finally:
            aux_stack.close()

    if config.compute_feature_metrics and all_feature_metrics:
        with region_timer("dataset_metrics", "<parent>"):
            dataset_row: dict[str, float] = {}

            # Stage per-prefix inputs: (pred_for_metric, target_for_metric,
            # pred_for_probe, target_for_probe, pred_fovs, target_fovs).
            # CP gets pruning + z-score; the pre-prune CP arrays feed the
            # linear probe so MADScaler can normalize per-fold.
            prefix_inputs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

            cp = parent_lists["cp"]
            if cp.pred_feats:
                pred_cp_raw = np.concatenate(cp.pred_feats, axis=0)
                target_cp_raw = np.concatenate(cp.gt_feats, axis=0)
                target_cp_filtered, pred_cp_filtered, cp_keep_mask = select_features(target_cp_raw, pred_cp_raw)
                cp_glcm_enabled = bool(OmegaConf.select(config, "feature_metrics.cp.glcm.enabled", default=False))
                mask_payload = {
                    "feature_names": list(active_cp_feature_names(cp_glcm_enabled)),
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
                        np.concatenate(cp.pred_fovs, axis=0),
                        np.concatenate(cp.gt_fovs, axis=0),
                    )
                )

            deep_tracks = [("DINOv3", "dinov3"), ("DynaCLR", "dynaclr")]
            if celldino_feature_extractor is not None:
                deep_tracks.append(("CellDINO", "celldino"))
            for display_name, key in deep_tracks:
                bb = parent_lists[key]
                if bb.pred_feats:
                    pred_arr = np.concatenate(bb.pred_feats, axis=0)
                    target_arr = np.concatenate(bb.gt_feats, axis=0)
                    prefix_inputs.append(
                        (
                            display_name,
                            pred_arr,
                            target_arr,
                            pred_arr,
                            target_arr,
                            np.concatenate(bb.pred_fovs, axis=0),
                            np.concatenate(bb.gt_fovs, axis=0),
                        )
                    )

            # Prefix with "Dataset_" so dataset-level FID/KID/cosine don't clobber
            # per-FOV columns of the same name when merged into per-FOV rows.
            def _compute_one(args):
                # MIND stays on CPU here even when use_gpu=True: 4 parallel threads
                # racing on the same CUDA context would either serialize via the
                # allocator (no speedup) or contend for memory with mid-eval FOV
                # work in process executors. CPU MIND in a 4-thread BLAS-capped
                # pool is competitive with serialized GPU MIND and bit-stable
                # across runs that toggle use_gpu (torch's CPU vs CUDA RNG
                # produce different streams for the same seed, breaking cross-
                # leaf comparability of the MIND column).
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
            embedding_groups: dict[str, tuple] = {}
            for key in _BACKBONE_KEYS:
                if key == "celldino" and celldino_feature_extractor is None:
                    continue
                bb = parent_lists[key]
                embedding_groups[f"pred_{key}"] = (bb.pred_feats, bb.pred_fovs, bb.pred_ts)
                embedding_groups[f"gt_{key}"] = (bb.gt_feats, bb.gt_fovs, bb.gt_ts)
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
    mask_path = save_dir / config.save.mask_metrics_filename
    mask_ok = mask_path.exists()
    feature_ok = (save_dir / config.save.feature_metrics_filename).exists() if config.compute_feature_metrics else True
    # A prior instance-AP-less (or pre-Dice) run's mask cache must not suppress
    # newly-requested instance metrics: require both the AP and instance-Dice
    # columns to be present in the cached rows, else recompute.
    if mask_ok and bool(getattr(config, "compute_instance_ap", False)):
        rows = np.load(mask_path, allow_pickle=True).tolist()
        if not rows or "mAP" not in rows[0] or "instance_dice" not in rows[0]:
            return False
    # Same guard for per-cell similarity, keyed to the exact requested columns:
    # a prior run with a different metrics/reduce set (e.g. PCC-only) must not
    # let its partial PerCell_ pixel cache suppress newly-requested columns
    # (e.g. an added ssim). Require every expected PerCell_{METRIC}_{REDUCE}.
    if pixel_ok and bool(OmegaConf.select(config, "compute_cell_similarity", default=False)):
        metrics = OmegaConf.select(config, "cell_similarity.metrics", default=["pcc"])
        reduce = OmegaConf.select(config, "cell_similarity.reduce", default=["mean", "median"])
        expected = {f"PerCell_{str(m).upper()}_{r}" for m in metrics for r in reduce}
        pixel_rows = np.load(save_dir / config.save.pixel_metrics_filename, allow_pickle=True).tolist()
        if not pixel_rows or not expected.issubset(pixel_rows[0]):
            return False
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
    # segmentation.backend selects SuperModel vs Cellpose-SAM in
    # ``prepare_segmentation_model`` and the mask cache path — a per-condition
    # override would load the wrong segmenter for the shared bundle and mix
    # backends across the grouped run's mask caches.
    "segmentation.backend",
    # require_complete_cache flips ``prepare_segmentation_model`` between
    # "load real SuperModel" and "return None" — letting a condition
    # override it would mean the shared seg_model bundle is wrong for that
    # condition (None when cache-miss expects a real model, or loaded but
    # never used). Treat as a grouped invariant.
    "io.require_complete_cache",
    # The instance-AP toggles gate the segmentation behavior and the shared
    # instance-label cache identity. A per-condition override would change the
    # seg backend, the sliced plane, the seed params, or the AP thresholds for
    # one condition while sharing the bundle + caches with the others.
    "compute_instance_ap",
    "segmentation.dimension",
    "segmentation.slice_selection",
    "segmentation.slice_fraction",
    "segmentation.nuclei_channel_name",
    "segmentation.cellpose",
    "segmentation.watershed",
    "instance_metrics.iou_thresholds",
)


def _snapshot_field(cfg: DictConfig, cfg_field: str):
    """Resolve a model-loading field to a comparable plain Python value."""
    node = OmegaConf.select(cfg, cfg_field, default=None)
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=False)
    return node


def _seg_model_required(cfg: DictConfig) -> bool:
    """Mirror ``prepare_segmentation_model``'s load decision without instantiating.

    Returns ``True`` iff a real ``SuperModel`` would be loaded for *cfg*.
    Used as a derived grouped-eval invariant so a per-condition override of
    ``io.pred_cache_dir`` cannot silently flip whether the shared seg_model
    is needed (per-condition pred caches are otherwise free to differ).

    Mirrors ``segmentation.prepare_segmentation_model``:

    - Organelle targets (``er``, ``mitochondria``, ``nucleoli``,
      ``lysosomes``) use ``aicssegmentation`` workflows that don't take a
      ``seg_model`` arg → ``False``.
    - Nucleus/membrane with ``require_complete_cache=false`` → ``True``
      (cellpose loads to compute fresh masks on cache miss).
    - Nucleus/membrane with ``require_complete_cache=true`` AND
      ``io.pred_cache_dir`` set → ``False`` (pred masks served from cache;
      per-T loop never falls back to ``segment(predict[t], seg_model=...)``).
    - Nucleus/membrane with ``require_complete_cache=true`` AND
      ``io.pred_cache_dir is None`` → ``True`` (per-T loop falls back to
      ``segment(predict[t], seg_model=...)``; the seg_model is required).
    """
    target_name = OmegaConf.select(cfg, "target_name", default=None)
    if target_name not in ("nucleus", "membrane"):
        return False
    require_complete = bool(OmegaConf.select(cfg, "io.require_complete_cache", default=False))
    if not require_complete:
        return True
    pred_cache_dir = OmegaConf.select(cfg, "io.pred_cache_dir", default=None)
    return pred_cache_dir is None


def _merge_condition(base: DictConfig, overrides: DictConfig | dict) -> DictConfig:
    """Return a fresh DictConfig with ``overrides`` deep-merged into ``base``.

    Hydra composition always returns struct-mode configs. ``OmegaConf.merge``
    propagates struct mode from its first argument, so merging an overlay
    that carries fields outside the base's schema (notably the per-condition
    ``name`` label) raises ``ConfigKeyError``, and deleting the
    ``conditions`` / ``name`` keys raises ``ConfigTypeError`` even when the
    merge succeeds. The to_container round-trip below escapes struct mode
    while still producing a config that's independent of ``base``.
    """
    base_copy = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    merged = OmegaConf.merge(base_copy, OmegaConf.create(overrides))
    for key in ("conditions", "name"):
        if key in merged:
            del merged[key]
    return merged  # type: ignore[return-value]


def _check_grouped_field_invariants(
    base_snapshot: dict[str, object],
    base_seg_required: bool,
    merged: DictConfig,
    condition_name: str,
) -> None:
    """Raise if a per-condition merged config disagrees with the baseline on model-loading fields.

    ``base_snapshot`` is computed once per grouped run from the
    conditions-stripped base, so condition 0 is validated symmetrically
    with all later conditions (a condition 0 overlay that sneaks in e.g.
    ``target_name`` would be rejected, not silently adopted as the new
    "base").

    ``base_seg_required`` is the baseline value of :func:`_seg_model_required`.
    ``io.pred_cache_dir`` is intentionally NOT in :data:`_MODEL_LOADING_FIELDS`
    (per-condition pred caches are the canonical grouped-run use case), but a
    pred_cache_dir flip *can* indirectly change whether SuperModel is needed
    for nucleus/membrane targets under ``require_complete_cache=true``. Catch
    that specific case here so the shared seg_model bundle is never wrong.
    """
    for cfg_field in _MODEL_LOADING_FIELDS:
        merged_val = _snapshot_field(merged, cfg_field)
        if base_snapshot[cfg_field] != merged_val:
            raise ValueError(
                f"Condition {condition_name!r}: overrides changed model-loading field "
                f"{cfg_field!r}. Move it to the base config or run this condition separately."
            )
    if _seg_model_required(merged) and not base_seg_required:
        raise ValueError(
            f"Condition {condition_name!r}: io.pred_cache_dir override flips "
            f"prepare_segmentation_model's load decision in the dangerous direction. "
            f"The base config does NOT load SuperModel "
            f"(target_name={base_snapshot['target_name']!r}, "
            f"require_complete_cache={base_snapshot['io.require_complete_cache']!r}, "
            f"base io.pred_cache_dir is set), but this condition sets "
            f"io.pred_cache_dir=None which makes the per-T loop fall back to "
            f"segment(predict[t], seg_model=...) — that call would receive None "
            f"and raise. Set io.pred_cache_dir on this condition, or run it separately."
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
            # Cache-only path: workers still re-init per condition (pool spawn +
            # thread budget + ``load_eval_models`` call). Under
            # ``require_complete_cache=true`` the segmenter usually short-circuits
            # to ``None``, but deep extractors still load when
            # ``compute_feature_metrics=true``. Mild informational note.
            print(
                "[grouped] note: runtime.executor=process with require_complete_cache=true — "
                f"workers re-init per condition (segmenter usually skipped; deep extractors "
                f"still load when compute_feature_metrics=true). {n_conditions} pool spawns "
                f"total; use executor=serial to amortize the per-condition load."
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
    # against this, including condition 0. Round-trip through to_container
    # to escape Hydra's struct-mode flag — ``del`` on a struct DictConfig
    # raises ``ConfigTypeError``.
    models_base = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    if "conditions" in models_base:
        del models_base["conditions"]
    base_snapshot = {field: _snapshot_field(models_base, field) for field in _MODEL_LOADING_FIELDS}
    base_seg_required = _seg_model_required(models_base)

    models: EvalModels | None = None

    def get_models() -> EvalModels:
        nonlocal models
        if models is None:
            print(f"[grouped] loading shared models for {len(conditions)} conditions ...")
            models = load_eval_models(models_base)
        return models

    results: list[tuple[str, tuple]] = []
    condition_save_dirs: list[Path] = []
    for idx, cond in enumerate(conditions):
        name = (
            cond.get("name", str(idx)) if isinstance(cond, dict) else OmegaConf.select(cond, "name", default=str(idx))
        )
        merged = _merge_condition(config, cond)
        apply_dataset_ref(merged)
        _check_grouped_field_invariants(base_snapshot, base_seg_required, merged, name)
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
        condition_save_dirs.append(Path(merged.save.save_dir))

    # Cross-condition infection probe: once every condition in the group has
    # its single-cell embeddings on disk, classify each infected condition vs
    # mock (FOV-stratified logistic probe, per feature space, GT and pred).
    # Default-on whenever feature metrics were computed (the probe consumes the
    # embeddings those produce) and a mock + >=1 infected condition are present.
    # Gated by ``cross_condition_probe.enabled``; never fails the eval.
    probe_enabled = bool(
        OmegaConf.select(
            config,
            "cross_condition_probe.enabled",
            default=bool(OmegaConf.select(config, "compute_feature_metrics", default=True)),
        )
    )
    if probe_enabled:
        n_splits = int(OmegaConf.select(config, "cross_condition_probe.n_splits", default=5))
        rng_seed = int(OmegaConf.select(config, "cross_condition_probe.rng_seed", default=2020))
        try:
            written = _cross_condition_run_for_group(condition_save_dirs, n_splits=n_splits, rng_seed=rng_seed)
            if written:
                print(f"[grouped] cross-condition probe wrote {len(written)} CSV(s): {[str(p) for p in written]}")
        except Exception as e:  # noqa: BLE001 -- diagnostic add-on must not fail the eval
            print(f"[grouped] cross-condition probe skipped: {type(e).__name__}: {e}")

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
        with region_timer("save_metrics_csvs", "<parent>"):
            save_metrics(
                config,
                pixel_metrics=pixel_metrics,
                mask_metrics=mask_metrics,
                feature_metrics=feature_metrics,
            )
        # Re-dump so save_metrics_csvs lands in eval_timing.csv. evaluate_predictions
        # dumps once before save_metrics runs; this second dump overwrites with the
        # full set (FOV-loop regions + save_metrics_csvs).
        dump_timings_csv(Path(config.save.save_dir))
    return pixel_metrics, mask_metrics, feature_metrics


@hydra.main(version_base="1.2", config_path="_configs", config_name="eval_grouped")
def evaluate_model_grouped(config: DictConfig):
    """Run grouped multi-condition eval, amortizing model loads across conditions."""
    return evaluate_predictions_grouped(config)


if __name__ == "__main__":
    evaluate_model()
