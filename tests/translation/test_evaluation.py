import numpy as np
import pandas as pd
import pytest
from lightning.pytorch.loggers import CSVLogger
from numpy.testing import assert_array_equal

from viscy.data.segmentation import SegmentationDataModule
from viscy.trainer import Trainer
from viscy.translation.evaluation import SegmentationMetrics2D


@pytest.mark.parametrize("pred_channel", ["DAPI", "GFP"])
def test_segmentation_metrics_2d(pred_channel, labels_hcs_dataset, tmp_path) -> None:
    dm = SegmentationDataModule(
        pred_dataset=labels_hcs_dataset,
        target_dataset=labels_hcs_dataset,
        target_channel="DAPI",
        pred_channel=pred_channel,
        pred_z_slice=0,
        target_z_slice=0,
        batch_size=1,
        num_workers=0,
    )
    lm = SegmentationMetrics2D()
    trainer = Trainer(logger=CSVLogger(tmp_path, name="", version=""))
    trainer.test(lm, datamodule=dm)
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) > 0
    accuracy = metrics["accuracy"].to_numpy()
    if pred_channel == "DAPI":
        assert_array_equal(accuracy, np.ones_like(accuracy))
    else:
        assert 0 < accuracy.mean() < 1
