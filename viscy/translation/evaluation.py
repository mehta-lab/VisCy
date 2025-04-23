"""Test stage lightning modules for comparing segmentation based on  virtual staining and fluorescence ground truth"""

import logging

import numpy as np
from lightning.pytorch import LightningModule
from panoptica import InputType, Panoptica_Evaluator
from panoptica.metrics import Metric
from torchmetrics.functional import accuracy, jaccard_index
from torchmetrics.functional.segmentation import dice_score

from viscy.data.typing import SegmentationSample
from viscy.translation.evaluation_metrics import mean_average_precision

_logger = logging.getLogger("lightning.pytorch")


class SegmentationMetrics(LightningModule):
    """Test runner for segmentation that handles both 2D and 3D data.

    Parameters
    ----------
    mode : str, optional
        Mode of operation, can be "auto", "2D", or "3D", by default "auto".
        In "auto" mode, dimensionality is determined from input data.
    aggregate_epoch : bool, optional
        Whether to aggregate results over the entire epoch, by default False
    """

    def __init__(self, mode: str = "auto", aggregate_epoch: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.aggregate_epoch = aggregate_epoch
        self._validate_mode()

    def _validate_mode(self):
        """Validate the mode parameter."""
        valid_modes = ["auto", "2D", "3D"]
        if self.mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {self.mode}")

    def test_step(self, batch: SegmentationSample, batch_idx: int) -> None:
        pred = batch["pred"]
        target = batch["target"]

        # Determine dimensionality from input data if in auto mode
        if self.mode == "auto":
            if pred.shape[0] == 1 and target.shape[0] == 1:
                current_mode = "2D"
            elif pred.shape[0] > 1 and target.shape[0] > 1:
                current_mode = "3D"
            else:
                raise ValueError(
                    f"Cannot determine dimensionality from shapes {pred.shape} and {target.shape}"
                )
        else:
            current_mode = self.mode

            # Validate input shapes against selected mode
            if current_mode == "2D" and not (
                pred.shape[0] == 1 and target.shape[0] == 1
            ):
                raise ValueError(
                    f"Expected 2D segmentation, got {pred.shape[0]} and {target.shape[0]}"
                )
            elif current_mode == "3D" and not (
                pred.shape[0] > 1 and target.shape[0] > 1
            ):
                raise ValueError(
                    f"Expected 3D segmentation, got {pred.shape[0]} and {target.shape[0]}"
                )

        pred = pred[0]
        target = target[0]

        # Common preprocessing for both modes
        pred_binary = pred > 0
        target_binary = target > 0

        if current_mode == "2D":
            self._compute_2d_metrics(pred, target, pred_binary, target_binary, batch)
        else:  # 3D mode
            self._compute_3d_metrics(pred, target, pred_binary, target_binary, batch)

    def _compute_2d_metrics(self, pred, target, pred_binary, target_binary, batch):
        """Compute and log metrics for 2D segmentation."""
        coco_metrics = mean_average_precision(pred, target)
        _logger.debug(coco_metrics)
        self.logger.log_metrics(
            {
                "position": batch["position_idx"][0],
                "time": batch["time_idx"][0],
                "accuracy": (accuracy(pred_binary, target_binary, task="binary")),
                "dice": (
                    dice_score(
                        pred_binary.long()[None],
                        target_binary.long()[None],
                        num_classes=2,
                        input_format="index",
                    )
                ),
                "jaccard": (jaccard_index(pred_binary, target_binary, task="binary")),
                "mAP": coco_metrics["map"],
                "mAP_50": coco_metrics["map_50"],
                "mAP_75": coco_metrics["map_75"],
                "mAR_100": coco_metrics["mar_100"],
            }
        )

    def _compute_3d_metrics(self, pred, target, pred_binary, target_binary, batch):
        """Compute and log metrics for 3D segmentation."""
        unique_instances_target = np.unique(target)
        unique_instances_pred = np.unique(pred)

        _logger.debug(
            f"Unique instances: {unique_instances_target} and {unique_instances_pred}"
        )

        ## Measuring Panoptic Quality
        evaluator = Panoptica_Evaluator(
            expected_input=InputType.UnmatchedInstancePair,
            instance_metrics=[Metric.DSC, Metric.IoU],
            decision_metric=Metric.DSC,
            decision_threshold=0.5,
            log_times=True,
        )
        result = evaluator.evaluate(pred, target, verbose=False)
        result = result.to_dict()
        _logger.debug(result)

        self.logger.log_metrics(
            {
                "position": batch["position_idx"][0],
                "time": batch["time_idx"][0],
                "target_instances": unique_instances_target,
                "pred_instances": unique_instances_pred,
                **result,
            }
        )


class BiologicalMetrics(LightningModule):
    """Test runner for biological metrics."""

    def __init__(self, aggregate_epoch: bool = False) -> None:
        super().__init__()
        self.aggregate_epoch = aggregate_epoch

    def test_step(self, batch: SegmentationSample, batch_idx: int) -> None:
        # TODO: Implement biological metrics (i.e regionprops logic)
        NotImplementedError("Biological metrics not implemented")


class IntensityMetrics(LightningModule):
    """Test runner for intensity metrics.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute, by default ["mae", "mse", "ssim", "pearson"]
    aggregate_epoch : bool, optional
        Whether to aggregate results over the entire epoch, by default False
    """

    def __init__(
        self,
        metrics: list[str] = ["mae", "mse", "ssim", "pearson"],
        aggregate_epoch: bool = False,
    ) -> None:
        super().__init__()
        self.metrics = metrics
        self.aggregate_epoch = aggregate_epoch
        self._validate_metrics()

    def _validate_metrics(self):
        """Validate the metrics parameter."""
        valid_metrics = ["mae", "mse", "ssim", "ms_ssim", "pearson", "cosine"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Metric '{metric}' not in {valid_metrics}")

    def test_step(self, batch, batch_idx: int) -> None:
        """Compute intensity metrics between prediction and target."""
        from torchmetrics.functional import (
            cosine_similarity,
            mean_absolute_error,
            mean_squared_error,
            pearson_corrcoef,
            structural_similarity_index_measure,
        )

        from viscy.translation.evaluation_metrics import ms_ssim_25d

        pred = batch["pred"]
        target = batch["target"]

        # Dictionary to store computed metrics
        metrics_dict = {
            "position": batch["position_idx"][0] if "position_idx" in batch else -1,
            "time": batch["time_idx"][0] if "time_idx" in batch else -1,
        }

        # Compute metrics
        for metric in self.metrics:
            if metric == "mae":
                metrics_dict["mae"] = mean_absolute_error(pred, target)
            elif metric == "mse":
                metrics_dict["mse"] = mean_squared_error(pred, target)
            elif metric == "ssim":
                # Handle different dimensionality cases
                if pred.shape[0] > 1:  # 3D/2.5D case
                    metrics_dict["ssim"] = structural_similarity_index_measure(
                        (
                            pred.squeeze(2)
                            if pred.shape[2] == 1
                            else pred[:, :, pred.shape[2] // 2]
                        ),
                        (
                            target.squeeze(2)
                            if target.shape[2] == 1
                            else target[:, :, target.shape[2] // 2]
                        ),
                    )
                else:  # 2D case
                    metrics_dict["ssim"] = structural_similarity_index_measure(
                        pred, target
                    )
            elif metric == "ms_ssim":
                if pred.ndim > 1:
                    metrics_dict["ms_ssim"] = ms_ssim_25d(pred, target)
            elif metric == "pearson":
                metrics_dict["pearson"] = pearson_corrcoef(
                    pred.flatten(), target.flatten()
                )
            elif metric == "cosine":
                metrics_dict["cosine"] = cosine_similarity(
                    pred.flatten(), target.flatten()
                )

        # Log computed metrics
        self.logger.log_metrics(metrics_dict)
