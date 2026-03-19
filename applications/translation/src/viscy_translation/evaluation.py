"""Test stage lightning module for comparing virtual staining and segmentations."""

import logging

from lightning.pytorch import LightningModule
from torchmetrics.functional import accuracy, jaccard_index
from torchmetrics.functional.segmentation import dice_score

from viscy_data import SegmentationSample
from viscy_utils.evaluation.metrics import mean_average_precision

_logger = logging.getLogger("lightning.pytorch")


class SegmentationMetrics2D(LightningModule):
    """Test runner for 2D segmentation.

    Parameters
    ----------
    aggregate_epoch : bool
        Whether to aggregate metrics over the epoch, defaults to False.
    """

    def __init__(self, aggregate_epoch: bool = False) -> None:
        super().__init__()
        self.aggregate_epoch = aggregate_epoch

    def test_step(self, batch: SegmentationSample, batch_idx: int) -> None:
        """Execute a single test step for segmentation evaluation.

        Parameters
        ----------
        batch : SegmentationSample
            Input batch with pred and target segmentations.
        batch_idx : int
            Batch index.
        """
        pred = batch["pred"]
        target = batch["target"]
        if not (pred.shape[0] == 1 and target.shape[0] == 1):
            raise ValueError(f"Expected 2D segmentation, got {pred.shape[0]} and {target.shape[0]}")
        pred = pred[0]
        target = target[0]
        pred_binary = pred > 0
        target_binary = target > 0
        coco_metrics = mean_average_precision(pred, target)
        _logger.debug(coco_metrics)
        self.log_dict(
            {
                "test_metrics/accuracy": accuracy(pred_binary, target_binary, task="binary"),
                "test_metrics/dice": dice_score(
                    pred_binary.long()[None],
                    target_binary.long()[None],
                    num_classes=2,
                    input_format="index",
                ),
                "test_metrics/jaccard": jaccard_index(pred_binary, target_binary, task="binary"),
                "test_metrics/mAP": coco_metrics["map"],
                "test_metrics/mAP_50": coco_metrics["map_50"],
                "test_metrics/mAP_75": coco_metrics["map_75"],
                "test_metrics/mAR_100": coco_metrics["mar_100"],
            },
            on_step=True,
            on_epoch=False,
        )
