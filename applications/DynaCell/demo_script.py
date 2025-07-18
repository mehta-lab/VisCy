"""
This script is a demo script for the DynaCell application.
It loads the ome-zarr 0.4v format, calculates metrics and saves the results as csv files
"""

import datetime
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger

from viscy.data.dynacell import DynaCellDataBase, DynaCellDataModule
from viscy.trainer import Trainer
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

# Set float32 matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision("high")

csv_database_path = Path(
    "/home/eduardo.hirata/repos/viscy/applications/DynaCell/dynacell_summary_table.csv"
).expanduser()
tmp_path = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/demo_metrics")
tmp_path.mkdir(parents=True, exist_ok=True)

database = pd.read_csv(csv_database_path, dtype={"FOV": str})

# Add prediction paths
pred_database = database[database['Organelle'] == "HIST2H2BE"].copy()
pred_database["Path"]= "path_to_prediction"


def main(
    method: Literal["segmentation2D", "segmentation3D", "intensity"],
    target_channel_name: str,
    prediction_channel_name: str,
    use_z_slice_range: bool = False,
):
    """
    Run DynaCell metrics computation.

    Parameters
    ----------
    method : Literal["segmentation2D", "segmentation3D", "intensity"], optional
        Type of metrics to compute, by default "intensity"
    use_z_slice_range : bool, optional
        Whether to use a z-slice range instead of a single slice, by default False
    """
    # Generate timestamp for unique versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set z_slice based on whether to use range or single slice
    z_slice_value = slice(15, 17) if use_z_slice_range else 16

    # Create target database
    target_db = DynaCellDataBase(
        database=database,
        cell_types=["HEK293T"],
        organelles=["HIST2H2BE"],
        infection_conditions=["Mock"],
        channel_name=target_channel_name,
        z_slice=z_slice_value,
    )

    if method == "segmentation2D":
        # For segmentation, use same channel for pred and target (self-comparison)
        pred_db = DynaCellDataBase(
            database=database,
            cell_types=["HEK293T"],
            organelles=["HIST2H2BE"],
            infection_conditions=["Mock"],
            channel_name=prediction_channel_name,
            z_slice=z_slice_value,
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
        version = "1"

        output_dir = tmp_path / "segmentation"
        output_dir.mkdir(exist_ok=True)

        # Use the CSVLogger without version (we'll use our own naming)
        logger = CSVLogger(save_dir=output_dir, name=name, version=version)
        trainer = Trainer(logger=logger)
        trainer.test(lm, datamodule=dm)

        # Find the metrics file - use the correct relative pattern
        metrics_file = output_dir / name / version / "metrics.csv"
        if metrics_file.exists():
            metrics = pd.read_csv(metrics_file)
            print(f"Segmentation metrics saved to: {metrics_file}")
            print(f"Segmentation metrics columns: {metrics.columns.tolist()}")
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            metrics = None

        return metrics

    elif method == "segmentation3D":
        raise NotImplementedError("Segmentation3D is not implemented yet")

    elif method == "intensity":
        # For intensity comparison, use the same channel to compare to itself
        pred_db = DynaCellDataBase(
            database=database,
            cell_types=["HEK293T"],
            organelles=["HIST2H2BE"],
            infection_conditions=["Mock"],
            channel_name=prediction_channel_name,
            z_slice=z_slice_value,
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
        # Indicate whether z-slice range was used in the name
        range_suffix = "_range" if use_z_slice_range else ""
        name = f"intensity{range_suffix}_{timestamp}"
        version = "1"

        output_dir = tmp_path / "intensity"
        output_dir.mkdir(exist_ok=True)

        # Use the CSVLogger without version (we'll use our own naming)
        logger = CSVLogger(save_dir=output_dir, name=name, version=version)
        trainer = Trainer(logger=logger)
        trainer.test(lm, datamodule=dm)

        # Find the metrics file - use the correct relative pattern
        metrics_file = output_dir / name / version / "metrics.csv"
        if metrics_file.exists():
            metrics = pd.read_csv(metrics_file)
            print(f"Intensity metrics saved to: {metrics_file}")
            print(f"Intensity metrics columns: {metrics.columns.tolist()}")
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            metrics = None

        return metrics
    else:
        raise ValueError(f"Invalid method: {method}")


# %%
if __name__ == "__main__":
    # print("Running intensity metrics with single z-slice...")
    # intensity_metrics = main("intensity", use_z_slice_range=False)

    print("\nRunning intensity metrics with z-slice range...")
    intensity_metrics_range = main(
        method="intensity",
        target_channel_name="GFP",
        prediction_channel_name="nuclei_prediction",
        use_z_slice_range=True
    )
