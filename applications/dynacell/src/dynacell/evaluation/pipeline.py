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
    compute_feature_metrics,
    compute_pixel_metrics,
    evaluate_segmentations,
)
from dynacell.evaluation.utils import plot_metrics


def evaluate_segmentation_metrics(
    target,
    predict,
    config: DictConfig,
    seg_model=None,
):
    """Segment both prediction and target, return binary mask metrics and masks."""
    from dynacell.evaluation.segmentation import segment

    segmented_predict = segment(predict, config.target_name, seg_model=seg_model)
    segmented_target = segment(target, config.target_name, seg_model=seg_model)

    mask_metrics = evaluate_segmentations(segmented_predict, segmented_target)

    return mask_metrics, segmented_predict, segmented_target


def evaluate_predictions(config: DictConfig):
    """Evaluate predictions on all test images."""
    from dynacell.evaluation.segmentation import prepare_segmentation_model
    from dynacell.evaluation.utils import DinoV3FeatureExtractor, DynaCLRFeatureExtractor

    all_pixel_metrics = []
    all_mask_metrics = []
    all_feature_metrics = []

    io_config = config.io
    pred_path = Path(io_config.pred_path)
    gt_path = Path(io_config.gt_path)
    seg_path = Path(io_config.cell_segmentation_path)
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seg_model = prepare_segmentation_model(config)

    if config.compute_feature_metrics:
        from omegaconf import OmegaConf

        dinov3_feature_extractor = DinoV3FeatureExtractor(config.feature_extractor.dinov3.pretrained_model_name)
        dynaclr_config = config.feature_extractor.dynaclr
        dynaclr_feature_extractor = DynaCLRFeatureExtractor(
            checkpoint=dynaclr_config.checkpoint,
            encoder_config=OmegaConf.to_container(dynaclr_config.encoder, resolve=True),
        )
    else:
        dinov3_feature_extractor = None
        dynaclr_feature_extractor = None

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
        open_ome_zarr(seg_path, mode="r") as seg_plate,
    ):
        pred_positions = list(pred_plate.positions())
        gt_positions = list(gt_plate.positions())
        seg_positions = list(seg_plate.positions())
        if not (len(pred_positions) == len(gt_positions) == len(seg_positions)):
            raise ValueError(
                f"Position count mismatch: pred={len(pred_positions)}, gt={len(gt_positions)}, seg={len(seg_positions)}"
            )
        for p1, p2, p3 in tqdm(
            zip(pred_positions, gt_positions, seg_positions),
            total=len(pred_positions),
            desc="Processing positions",
        ):
            pos_name_pred, pos_pred = p1
            pos_name_gt, pos_gt = p2
            pos_name_seg, pos_seg = p3
            assert pos_name_pred == pos_name_gt == pos_name_seg, (
                "Prediction, GT, and segmentation position names do not match."
            )

            pred_channel_index = pos_pred.get_channel_index(io_config.pred_channel_name)
            gt_channel_index = pos_gt.get_channel_index(io_config.gt_channel_name)

            predict = np.asarray(pos_pred.data[:, pred_channel_index])  # shape: (T, D, H, W)
            target = np.asarray(pos_gt.data[:, gt_channel_index])  # shape: (T, D, H, W)
            cell_segmentation = np.asarray(pos_seg.data[:, 0])  # shape: (T, D, H, W)

            T = predict.shape[0]

            microssim_data = []
            fov_pixel_metrics = []

            segmentations = []

            for t in tqdm(range(T), desc="Processing timepoints"):
                data_info = {
                    "FOV": pos_name_pred,
                    "Timepoint": t,
                }

                pixel_metrics = compute_pixel_metrics(
                    predict[t],
                    target[t],
                    spacing=config.pixel_metrics.spacing,
                    fsc_kwargs=config.pixel_metrics.fsc,
                    spectral_pcc_kwargs=config.pixel_metrics.spectral_pcc,
                )

                if config.compute_microssim:
                    microssim_data.append(
                        {
                            "target": target[t],
                            "predict": predict[t],
                        }
                    )

                fov_pixel_metrics.append({**data_info, **pixel_metrics})

                # compute segmentation metrics for this timepoint
                mask_metrics, segmented_predict, segmented_target = evaluate_segmentation_metrics(
                    target[t],
                    predict[t],
                    config,
                    seg_model=seg_model,
                )

                all_mask_metrics.append({**data_info, **mask_metrics})
                segmentations.append(np.stack([segmented_predict, segmented_target], axis=0))  # shape: (2, D, H, W)

                if config.compute_feature_metrics:
                    feature_metrics = compute_feature_metrics(
                        predict[t],
                        target[t],
                        cell_segmentation[t],
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

    return all_pixel_metrics, all_mask_metrics, all_feature_metrics


def save_metrics(config: DictConfig, pixel_metrics=None, mask_metrics=None, feature_metrics=None):
    """Save metrics to files."""
    save_dir = Path(config.save.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mask_metrics:
        mask_metrics_df = pd.DataFrame(mask_metrics)
        mask_metrics_df.to_csv(save_dir / config.save.mask_csv_filename, index=False)
        np.save(save_dir / config.save.mask_metrics_filename, mask_metrics)
        print(
            f"Saved mask metrics to {save_dir / config.save.mask_csv_filename} "
            f"and {save_dir / config.save.mask_metrics_filename}"
        )
        plot_metrics(mask_metrics_df, save_dir, "mask_metrics")
        print(f"Saved mask metric plots to {save_dir / 'mask_metrics'}")

    if pixel_metrics:
        pixel_metrics_df = pd.DataFrame(pixel_metrics)
        pixel_metrics_df.to_csv(save_dir / config.save.pixel_csv_filename, index=False)
        np.save(save_dir / config.save.pixel_metrics_filename, pixel_metrics)
        print(
            f"Saved pixel metrics to {save_dir / config.save.pixel_csv_filename} "
            f"and {save_dir / config.save.pixel_metrics_filename}"
        )
        plot_metrics(pixel_metrics_df, save_dir, "pixel_metrics")
        print(f"Saved pixel metric plots to {save_dir / 'pixel_metrics'}")

    if feature_metrics:
        feature_metrics_df = pd.DataFrame(feature_metrics)
        feature_metrics_df.to_csv(save_dir / config.save.feature_csv_filename, index=False)
        np.save(save_dir / config.save.feature_metrics_filename, feature_metrics)
        print(
            f"Saved feature metrics to {save_dir / config.save.feature_csv_filename} "
            f"and {save_dir / config.save.feature_metrics_filename}"
        )
        plot_metrics(feature_metrics_df, save_dir, "feature_metrics")
        print(f"Saved feature metric plots to {save_dir / 'feature_metrics'}")


_EVAL_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "evaluate")


@hydra.main(version_base="1.2", config_path=_EVAL_CONFIG_DIR, config_name="eval")
def evaluate_model(config: DictConfig):
    """Evaluate model on test images."""
    save_dir = Path(config.save.save_dir)
    pixel_metrics_path = save_dir / config.save.pixel_metrics_filename
    mask_metrics_path = save_dir / config.save.mask_metrics_filename
    feature_metrics_path = save_dir / config.save.feature_metrics_filename
    feature_metrics_cached = feature_metrics_path.exists() if config.compute_feature_metrics else True
    if (
        pixel_metrics_path.exists()
        and mask_metrics_path.exists()
        and feature_metrics_cached
        and not config.recalculate_metrics
    ):
        print("Found existing metrics.")
        pixel_metrics = np.load(pixel_metrics_path, allow_pickle=True)
        mask_metrics = np.load(mask_metrics_path, allow_pickle=True)
        if config.compute_feature_metrics:
            feature_metrics = np.load(feature_metrics_path, allow_pickle=True)
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
