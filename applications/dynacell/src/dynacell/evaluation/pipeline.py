"""Batch orchestration: load, segment, evaluate, save."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig
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
    build_pred_crops,
    calculate_microssim,
    compute_pixel_metrics,
    cp_drop_invalid_cells,
    cp_pred_regionprops,
    evaluate_segmentations,
    features_from_crops,
)
from dynacell.evaluation.pipeline_cache import (
    flush_manifest,
    fov_gt_cp_features,
    fov_gt_deep_features,
    fov_gt_masks,
    init_cache_context,
    resolve_dynaclr_encoder_cfg,
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


def evaluate_predictions(config: DictConfig):
    """Evaluate predictions on all test images."""
    from dynacell.evaluation.segmentation import prepare_segmentation_model, segment
    from dynacell.evaluation.utils import CellDinoFeatureExtractor, DinoV3FeatureExtractor, DynaCLRFeatureExtractor

    all_pixel_metrics = []
    all_mask_metrics = []
    all_feature_metrics = []

    io_config = config.io
    pred_path = Path(io_config.pred_path)
    gt_path = Path(io_config.gt_path)
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seg_model = prepare_segmentation_model(config)

    dinov3_model_name = None
    dynaclr_ckpt_path = None
    dynaclr_encoder_cfg = None
    celldino_weights_path = None
    dinov3_feature_extractor = None
    dynaclr_feature_extractor = None
    celldino_feature_extractor = None

    if config.compute_feature_metrics:
        if io_config.cell_segmentation_path is None:
            raise ValueError("io.cell_segmentation_path is required when compute_feature_metrics=true")
        dinov3_model_name = config.feature_extractor.dinov3.pretrained_model_name
        dinov3_feature_extractor = DinoV3FeatureExtractor(dinov3_model_name)
        dynaclr_config = config.feature_extractor.dynaclr
        dynaclr_ckpt_path = str(dynaclr_config.checkpoint)
        dynaclr_encoder_cfg = resolve_dynaclr_encoder_cfg(config)
        dynaclr_feature_extractor = DynaCLRFeatureExtractor(
            checkpoint=dynaclr_config.checkpoint,
            encoder_config=dynaclr_encoder_cfg,
        )
        celldino_cfg = config.feature_extractor.celldino
        if celldino_cfg.weights_path is not None:
            celldino_weights_path = str(celldino_cfg.weights_path)
            celldino_feature_extractor = CellDinoFeatureExtractor(
                weights_path=celldino_weights_path,
                img_size=int(celldino_cfg.img_size),
                patch_size=int(celldino_cfg.patch_size),
            )

    cache_ctx = init_cache_context(
        config,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_path=dynaclr_ckpt_path,
        dynaclr_encoder_cfg=dynaclr_encoder_cfg,
        celldino_weights_path=celldino_weights_path,
    )

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
            for p1, p2, p3 in tqdm(
                zip(pred_positions, gt_positions, seg_positions),
                total=len(pred_positions),
                desc="Processing positions",
            ):
                pos_name_pred, pos_pred = p1
                pos_name_gt, pos_gt = p2
                pos_name_seg, pos_seg = p3
                if pos_name_pred != pos_name_gt:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, gt={pos_name_gt!r}")
                if seg_plate is not None and pos_name_seg != pos_name_pred:
                    raise ValueError(f"Position name mismatch: pred={pos_name_pred!r}, seg={pos_name_seg!r}")

                pred_channel_index = pos_pred.get_channel_index(io_config.pred_channel_name)
                gt_channel_index = pos_gt.get_channel_index(io_config.gt_channel_name)

                predict = np.asarray(pos_pred.data[:, pred_channel_index])  # shape: (T, D, H, W)
                target = np.asarray(pos_gt.data[:, gt_channel_index])  # shape: (T, D, H, W)
                cell_segmentation = np.asarray(pos_seg.data[:, 0]) if pos_seg is not None else None

                T = predict.shape[0]

                gt_mask_stack = fov_gt_masks(cache_ctx, pos_name_pred, target, seg_model)

                if config.compute_feature_metrics:
                    gt_cp_per_t = fov_gt_cp_features(cache_ctx, pos_name_pred, target, cell_segmentation)
                    gt_dinov3_per_t = fov_gt_deep_features(
                        cache_ctx, pos_name_pred, target, cell_segmentation, dinov3_feature_extractor, "dinov3"
                    )
                    gt_dynaclr_per_t = fov_gt_deep_features(
                        cache_ctx, pos_name_pred, target, cell_segmentation, dynaclr_feature_extractor, "dynaclr"
                    )
                    gt_celldino_per_t = (
                        fov_gt_deep_features(
                            cache_ctx, pos_name_pred, target, cell_segmentation, celldino_feature_extractor, "celldino"
                        )
                        if celldino_feature_extractor is not None
                        else None
                    )

                microssim_data = []
                fov_pixel_metrics = []
                segmentations = []

                for t in tqdm(range(T), desc="Processing timepoints"):
                    data_info = {"FOV": pos_name_pred, "Timepoint": t}

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

                    # Mask: target side from cache/precompute; predict side always fresh.
                    segmented_target = gt_mask_stack[t]
                    segmented_predict = np.asarray(segment(predict[t], config.target_name, seg_model=seg_model)).astype(
                        bool
                    )
                    all_mask_metrics.append(
                        {**data_info, **evaluate_segmentations(segmented_predict, segmented_target)}
                    )
                    segmentations.append(np.stack([segmented_predict, segmented_target], axis=0))

                    if config.compute_feature_metrics:
                        pred_cp = cp_pred_regionprops(predict[t], cell_segmentation[t], config.pixel_metrics.spacing)
                        # Drop cells with non-finite regionprops on either side
                        # before any downstream use. Degenerate cells (1-voxel
                        # regions etc.) yield NaN intensity_std / moments and
                        # crash FID covariance via np.linalg.eigvals.
                        pred_cp, gt_cp_t = cp_drop_invalid_cells(pred_cp, gt_cp_per_t[t])
                        # Build the per-cell 2-D crops once per timepoint and
                        # reuse them across all 3-4 deep backbones (max-z
                        # projection + cell-iteration + crop construction
                        # are otherwise redundant per backbone).
                        pred_crops_2d = build_pred_crops(
                            predict[t], cell_segmentation[t], config.feature_metrics.patch_size
                        )
                        pred_dinov3 = features_from_crops(pred_crops_2d, dinov3_feature_extractor)
                        pred_dynaclr = features_from_crops(pred_crops_2d, dynaclr_feature_extractor)
                        pred_celldino = (
                            features_from_crops(pred_crops_2d, celldino_feature_extractor)
                            if celldino_feature_extractor is not None
                            else None
                        )
                        # Per-timepoint CP: drop target-zero columns + per-side z-score.
                        # Deep features stay untouched.
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
                        all_feature_metrics.append({**data_info, **pairwise_metrics})
                        if pred_cp.size > 0:
                            pred_cp_feats.append(pred_cp)
                            gt_cp_feats.append(gt_cp_t)
                            n = len(pred_cp)
                            pred_cp_fovs.append(np.full(n, pos_name_pred))
                            pred_cp_ts.append(np.full(n, t, dtype=np.int32))
                            gt_cp_fovs.append(np.full(n, pos_name_pred))
                            gt_cp_ts.append(np.full(n, t, dtype=np.int32))
                        if pred_dinov3.size > 0:
                            pred_dinov3_feats.append(pred_dinov3)
                            gt_dinov3_feats.append(gt_dinov3_per_t[t])
                            n = len(pred_dinov3)
                            pred_dinov3_fovs.append(np.full(n, pos_name_pred))
                            pred_dinov3_ts.append(np.full(n, t, dtype=np.int32))
                            gt_dinov3_fovs.append(np.full(n, pos_name_pred))
                            gt_dinov3_ts.append(np.full(n, t, dtype=np.int32))
                        if pred_dynaclr.size > 0:
                            pred_dynaclr_feats.append(pred_dynaclr)
                            gt_dynaclr_feats.append(gt_dynaclr_per_t[t])
                            n = len(pred_dynaclr)
                            pred_dynaclr_fovs.append(np.full(n, pos_name_pred))
                            pred_dynaclr_ts.append(np.full(n, t, dtype=np.int32))
                            gt_dynaclr_fovs.append(np.full(n, pos_name_pred))
                            gt_dynaclr_ts.append(np.full(n, t, dtype=np.int32))
                        if pred_celldino is not None and pred_celldino.size > 0:
                            pred_celldino_feats.append(pred_celldino)
                            gt_celldino_feats.append(gt_celldino_per_t[t])
                            n = len(pred_celldino)
                            pred_celldino_fovs.append(np.full(n, pos_name_pred))
                            pred_celldino_ts.append(np.full(n, t, dtype=np.int32))
                            gt_celldino_fovs.append(np.full(n, pos_name_pred))
                            gt_celldino_ts.append(np.full(n, t, dtype=np.int32))

                seg = np.stack(segmentations, axis=0)  # shape: (T, 2, D, H, W)
                row, col, fov = pos_name_pred.split("/")
                seg_pos = segmentation_results.create_position(row, col, fov)
                seg_pos.create_image("0", seg.astype(bool))

                if config.compute_microssim:
                    microssim_scores = calculate_microssim(microssim_data)
                    for i in range(T):
                        fov_pixel_metrics[i]["MicroMS3IM"] = float(microssim_scores[i]["MicroMS3IM"])

                all_pixel_metrics.extend(fov_pixel_metrics)

                # Flush manifest after each position so interrupted runs preserve progress.
                flush_manifest(cache_ctx)
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

        # Dataset-level columns are renamed with a "Dataset_" prefix before
        # merging into per-(FOV, timepoint) rows. Without this rename, four
        # keys (FID, KID, KID_std, Median_Cosine_Similarity) collide with the
        # per-FOV values and dict.update would silently clobber them.
        # Per-FOV cohorts are tiny (~3–28 cells) so FID is statistically
        # broken there regardless, but KID and cosine carry real per-FOV
        # signal worth preserving; interpretation belongs to downstream code.
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


@hydra.main(version_base="1.2", config_path="_configs", config_name="eval")
def evaluate_model(config: DictConfig):
    """Evaluate model on test images."""
    apply_dataset_ref(config)
    save_dir = Path(config.save.save_dir)
    pixel_metrics_path = save_dir / config.save.pixel_metrics_filename
    mask_metrics_path = save_dir / config.save.mask_metrics_filename
    feature_metrics_path = save_dir / config.save.feature_metrics_filename
    if _final_metrics_cache_valid(config):
        print("Found existing metrics.")
        pixel_metrics = np.load(pixel_metrics_path, allow_pickle=True).tolist()
        mask_metrics = np.load(mask_metrics_path, allow_pickle=True).tolist()
        if config.compute_feature_metrics:
            feature_metrics = np.load(feature_metrics_path, allow_pickle=True).tolist()
        else:
            feature_metrics = []
    else:
        pixel_metrics, mask_metrics, feature_metrics = evaluate_predictions(config)
        save_metrics(
            config,
            pixel_metrics=pixel_metrics,
            mask_metrics=mask_metrics,
            feature_metrics=feature_metrics,
        )
    return pixel_metrics, mask_metrics, feature_metrics


if __name__ == "__main__":
    evaluate_model()
