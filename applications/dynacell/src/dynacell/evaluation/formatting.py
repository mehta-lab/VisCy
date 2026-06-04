"""DataFrame formatting utilities for evaluation metrics."""

import numpy as np
import pandas as pd


def aps_to_df(metrics, models, segmenters, thresholds, metric="ap_to_gt"):
    """Convert AP metrics to a DataFrame."""
    results = []
    for segmenter in segmenters:
        segmenter_metrics = metrics[segmenter]
        for image_aps in segmenter_metrics:
            for model_ix in range(len(image_aps[metric])):
                image_model_ap = np.asarray(image_aps[metric][model_ix])
                for iou_ix in range(len(image_model_ap.T)):
                    tp_fn = image_model_ap[1][iou_ix] + image_model_ap[3][iou_ix]
                    recall = float(image_model_ap[1][iou_ix] / tp_fn) if tp_fn != 0 else 0.0
                    results.append(
                        {
                            "Segmenter": segmenter,
                            "Model": models[model_ix],
                            "IoU threshold": thresholds[iou_ix],
                            "AP": image_model_ap[0][iou_ix],
                            "Recall": recall,
                        }
                    )

    return pd.DataFrame(results)


def cosine_to_df(metrics, models, segmenters, thresholds, metric="cosine_to_gt"):
    """Convert cosine similarity metrics to a DataFrame."""
    results = []
    for segmenter in segmenters:
        segmenter_metrics = metrics[segmenter]
        for image_aps in segmenter_metrics:
            for model_ix in range(len(image_aps[metric])):
                image_model_ap = image_aps[metric][model_ix]
                for iou_ix in image_model_ap.keys():
                    if iou_ix in thresholds:
                        results.append(
                            {
                                "Segmenter": segmenter,
                                "Model": models[model_ix],
                                "IoU threshold": iou_ix,
                                "Distance": image_model_ap[iou_ix],
                            }
                        )

    return pd.DataFrame(results)


def pixel_metrics_to_df(metrics, models):
    """Convert pixel metrics to a melted DataFrame."""
    pixel_metrics_list = []
    for _, img_metrics in enumerate(metrics):
        for model_idx, model_metrics in enumerate(img_metrics):
            for region, region_metrics in model_metrics.items():
                pixel_metrics_list.append(
                    {
                        "Model": models[model_idx],
                        "Region": region,
                    }
                    | region_metrics
                )
    pixel_metrics_list = pd.DataFrame(pixel_metrics_list)
    return pixel_metrics_list.melt(id_vars=["Model", "Region"], var_name="Metric", value_name="Value")
