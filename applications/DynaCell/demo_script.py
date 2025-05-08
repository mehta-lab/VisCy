"""
This script is a demo script for the DynaCell application.
It loads the ome-zarr 0.4v format, calculates metrics and saves the results as csv files
"""

import datetime
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from lightning.pytorch.loggers import CSVLogger

from viscy.data.dynacell import DynaCellDataBase, DynaCellDataModule
from viscy.trainer import Trainer
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

csv_database_path = Path(
    "/home/eduardo.hirata/repos/viscy/applications/DynaCell/dynacell_summary_table.csv"
).expanduser()
tmp_path = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/demo_metrics")
tmp_path.mkdir(parents=True, exist_ok=True)


def main(
    method: Literal["segmentation2D", "segmentation3D", "intensity"] = "intensity",
):

    # Generate timestamp for unique versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create target database
    target_db = DynaCellDataBase(
        database_path=csv_database_path,
        cell_types=["HEK293T"],
        organelles=["HIST2H2BE"],
        infection_conditions=["Mock"],
        channel_name="Organelle",
        z_slice=16,
    )

    if method == "segmentation2D":
        # For segmentation, use same channel for pred and target (self-comparison)
        pred_db = DynaCellDataBase(
            database_path=csv_database_path,
            cell_types=["HEK293T"],
            organelles=["HIST2H2BE"],
            infection_conditions=["Mock"],
            channel_name="Organelle",
            z_slice=16,
        )

        # Create data module with both databases
        dm = DynaCellDataModule(
            target_database=target_db,
            pred_database=pred_db,
            batch_size=1,
            num_workers=0,
        )
        dm.setup(stage="test")

        # Print a sample to verify metadata
        sample = next(iter(dm.test_dataloader()))
        print(f"Sample keys: {sample.keys()}")
        print(f"Cell type: {sample['cell_type']}")
        print(f"Organelle: {sample['organelle']}")
        print(f"Infection condition: {sample['infection_condition']}")

        # Run segmentation metrics
        lm = SegmentationMetrics()
        # Use the method name and timestamp for unique identification
        name = f"segmentation_{timestamp}"
        # Set up output directory
        output_dir = tmp_path / "segmentation"
        output_dir.mkdir(exist_ok=True)

        # Use the CSVLogger without version (we'll use our own naming)
        logger = CSVLogger(save_dir=output_dir, name=name)
        trainer = Trainer(logger=logger)
        trainer.test(lm, datamodule=dm)

        # Find the metrics file - this should be in output_dir/name/metrics.csv
        metrics_file = list(output_dir.glob(f"{name}/metrics.csv"))[0]
        metrics = pd.read_csv(metrics_file)
        print(f"Segmentation metrics saved to: {metrics_file}")
        print(f"Segmentation metrics columns: {metrics.columns.tolist()}")

        return metrics

    elif method == "segmentation3D":
        raise NotImplementedError("Segmentation3D is not implemented yet")

    elif method == "intensity":
        # For intensity comparison, use the same channel to compare to itself
        pred_db = DynaCellDataBase(
            database_path=csv_database_path,
            cell_types=["HEK293T"],
            organelles=["HIST2H2BE"],
            infection_conditions=["Mock"],
            channel_name="Organelle",
            z_slice=16,
        )

        # Create data module with both databases
        dm = DynaCellDataModule(
            target_database=target_db,
            pred_database=pred_db,
            batch_size=1,
            num_workers=0,
        )
        dm.setup(stage="test")

        # Print a sample to verify metadata
        sample = next(iter(dm.test_dataloader()))
        print(f"Sample keys: {sample.keys()}")
        print(f"Cell type: {sample['cell_type']}")
        print(f"Organelle: {sample['organelle']}")
        print(f"Infection condition: {sample['infection_condition']}")

        # Run intensity metrics
        lm = IntensityMetrics()
        # Use the method name and timestamp for unique identification
        name = f"intensity_{timestamp}"
        # Set up output directory
        output_dir = tmp_path / "intensity"
        output_dir.mkdir(exist_ok=True)

        # Use the CSVLogger without version (we'll use our own naming)
        logger = CSVLogger(save_dir=output_dir, name=name)
        trainer = Trainer(logger=logger)
        trainer.test(lm, datamodule=dm)

        # Find the metrics file - this should be in output_dir/name/metrics.csv
        metrics_file = list(output_dir.glob(f"{name}/metrics.csv"))[0]
        metrics = pd.read_csv(metrics_file)
        print(f"Intensity metrics saved to: {metrics_file}")
        print(f"Intensity metrics columns: {metrics.columns.tolist()}")

        return metrics
    else:
        raise ValueError(f"Invalid method: {method}")


# %%
if __name__ == "__main__":
    print("Running intensity metrics...")
    intensity_metrics = main("intensity")
