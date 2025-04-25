"""
This script is a demo script for the DynaCell application.
It looad the ome-zarr 0.4v format, calculate segmentation metrics and saves the results as csv file

"""

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from lightning.pytorch.loggers import CSVLogger

from viscy.data.dynacell import DynaCellDataModule
from viscy.trainer import Trainer
from viscy.translation.evaluation import SegmentationMetrics

csv_database_path=Path("~/gdrive/publications/dynacell/summary_table/dynacell_summary_table.csv").expanduser()

def main(method: Literal["segmentation2D", "segmentation3D"] = "segmentation2D"):

    tmp_path = Path(tempfile.mkdtemp())

    if method == "segmentation2D":
        dm = DynaCellDataModule(
            csv_database_path=csv_database_path,
            pred_channel="nuclei_prediction",
            target_channel="Phase3D",
            pred_z_slice=16,
            target_z_slice=16,
            batch_size=1,
            num_workers=0,
            cell_types=["A549"],
            organelles=["TOMM20"],
            infection_conditions=["Mock"],
        )
        dm.setup(stage="test")
        
        sample = next(iter(dm.test_dataloader()))


        lm = SegmentationMetrics()
    elif method == "segmentation3D":
        pass
        # dm = Segmentation3DDataModule()
        # lm = SegmentationMetrics3D()
    else:
        raise ValueError(f"Invalid method: {method}")

    trainer = Trainer(logger=CSVLogger(tmp_path, name="", version=""))
    trainer.test(lm, datamodule=dm)
    metrics = pd.read_csv(tmp_path / f"{method}_metrics.csv")
    return metrics


# %%
if __name__ == "__main__":
    main()
