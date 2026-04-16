"""Batch orchestration: load, segment, evaluate, save."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig
from tqdm import tqdm

from dynacell.evaluation.metrics import (
    calculate_microssim,
    compute_pixel_metrics,
    cp_pairwise,
    cp_pred_regionprops,
    deep_pairwise,
    deep_pred_features,
    evaluate_segmentations,
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


def _pair_feature_metrics(
    predict_t: np.ndarray,
    cell_segmentation_t: np.ndarray,
    gt_cp_t: np.ndarray,
    gt_dinov3_t: np.ndarray,
    gt_dynaclr_t: np.ndarray,
    dinov3_extractor,
    dynaclr_extractor,
    spacing,
    patch_size: int,
) -> dict[str, float]:
    """Compute prediction-side features and pair them with precomputed GT features."""
    pred_cp = cp_pred_regionprops(predict_t, cell_segmentation_t, spacing)
    pred_dinov3 = deep_pred_features(predict_t, cell_segmentation_t, dinov3_extractor, patch_size)
    pred_dynaclr = deep_pred_features(predict_t, cell_segmentation_t, dynaclr_extractor, patch_size)
    return {
        **cp_pairwise(pred_cp, gt_cp_t),
        **deep_pairwise(pred_dinov3, gt_dinov3_t, "DINOv3"),
        **deep_pairwise(pred_dynaclr, gt_dynaclr_t, "DynaCLR"),
    }


def evaluate_predictions(config: DictConfig):
    """Evaluate predictions on all test images."""
    from dynacell.evaluation.segmentation import prepare_segmentation_model, segment
    from dynacell.evaluation.utils import DinoV3FeatureExtractor, DynaCLRFeatureExtractor

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
    dinov3_feature_extractor = None
    dynaclr_feature_extractor = None

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

    cache_ctx = init_cache_context(
        config,
        dinov3_model_name=dinov3_model_name,
        dynaclr_ckpt_path=dynaclr_ckpt_path,
        dynaclr_encoder_cfg=dynaclr_encoder_cfg,
    )

    seg_path = Path(io_config.cell_segmentation_path) if io_config.cell_segmentation_path is not None else None

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

                # Pre-fetch GT-side artifacts for this FOV (from cache or compute+write).
                gt_mask_stack = fov_gt_masks(cache_ctx, pos_name_pred, target, seg_model)

                if config.compute_feature_metrics:
                    gt_cp_per_t = fov_gt_cp_features(cache_ctx, pos_name_pred, target, cell_segmentation)
                    gt_dinov3_per_t = fov_gt_deep_features(
                        cache_ctx, pos_name_pred, target, cell_segmentation, dinov3_feature_extractor, "dinov3"
                    )
                    gt_dynaclr_per_t = fov_gt_deep_features(
                        cache_ctx, pos_name_pred, target, cell_segmentation, dynaclr_feature_extractor, "dynaclr"
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
                        feature_metrics = _pair_feature_metrics(
                            predict[t],
                            cell_segmentation[t],
                            gt_cp_per_t[t],
                            gt_dinov3_per_t[t],
                            gt_dynaclr_per_t[t],
                            dinov3_feature_extractor,
                            dynaclr_feature_extractor,
                            config.pixel_metrics.spacing,
                            config.feature_metrics.patch_size,
                        )
                        all_feature_metrics.append({**data_info, **feature_metrics})

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
        if metrics is None:
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
