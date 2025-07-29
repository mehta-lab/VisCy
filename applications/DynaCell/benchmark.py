"""
This script is a demo script for the DynaCell application.
It loads the ome-zarr 0.4v format, calculates metrics and saves the results as csv files
"""

import datetime
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import CSVLogger
from monai.transforms import NormalizeIntensityd

from viscy.data.dynacell import DynaCellDatabase, DynaCellDataModule
from viscy.trainer import Trainer
from viscy.transforms import NormalizeSampled, ScaleIntensityRangePercentilesd
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics

# Set float32 matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision("high")


def compute_metrics(
    metrics_module: LightningModule,
    cell_types: list,
    organelles: list,
    infection_conditions: list,
    target_database: pd.DataFrame,
    target_channel_name: str,
    prediction_database: pd.DataFrame,
    prediction_channel_name: str,
    log_output_dir: Path,
    log_name: str = "dynacell_metrics",
    log_version: str = None,
    z_slice: slice = None,
    transforms: list = None,
):
    """
    Compute DynaCell metrics.

    """
    # Generate timestamp for unique versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_version is None:
        log_version = timestamp

    # Create target database
    target_db = DynaCellDatabase(
        database=target_database,
        cell_types=cell_types,
        organelles=organelles,
        infection_conditions=infection_conditions,
        channel_name=target_channel_name,
        z_slice=z_slice,
    )

    # For segmentation, use same channel for pred and target (self-comparison)
    pred_db = DynaCellDatabase(
        database=prediction_database,
        cell_types=cell_types,
        organelles=organelles,
        infection_conditions=infection_conditions,
        channel_name=prediction_channel_name,
        z_slice=z_slice,
    )

    # Create data module with both databases
    dm = DynaCellDataModule(
        target_database=target_db,
        pred_database=pred_db,
        batch_size=1,
        num_workers=0,
        transforms=transforms,
    )
    dm.setup(stage="test")

    # Print a sample to verify metadata
    sample = next(iter(dm.test_dataloader()))
    print(f"Sample keys: {sample.keys()}")
    print(f"Cell type: {sample['cell_type']}")
    print(f"Organelle: {sample['organelle']}")
    print(f"Infection condition: {sample['infection_condition']}")

    # Use the CSVLogger without version (we'll use our own naming)
    log_output_dir.mkdir(exist_ok=True)
    logger = CSVLogger(save_dir=log_output_dir, name=log_name, version=log_version)

    trainer = Trainer(
        logger=logger, accelerator="cpu", devices=1, precision="16-mixed", num_nodes=1
    )
    trainer.test(metrics_module, datamodule=dm)

    # Find the metrics file - use the correct relative pattern
    metrics_file = log_output_dir / log_name / log_version / "metrics.csv"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        print(f"Segmentation metrics saved to: {metrics_file}")
        print(f"Segmentation metrics columns: {metrics.columns.tolist()}")
    else:
        print(f"Warning: Metrics file not found at {metrics_file}")
        metrics = None

    return metrics


# %%
if __name__ == "__main__":
    csv_database_path = Path(
        "~/mydata/gdrive/dynacell/summary_table/dynacell_summary_table_2025_05_05.csv"
    ).expanduser()
    output_dir = Path("/home/eduardo.hirata/repos/viscy/applications/DynaCell/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    database = pd.read_csv(csv_database_path, dtype={"FOV": str})

    # Select test set only
    database = database[database["Test Set"] == "x"]

    print("\nRunning intensity metrics with z-slice range...")
    metrics = compute_metrics(
        metrics_module=IntensityMetrics(),  # IntensityMetrics(), or SegmentationMetrics(mode="2D"),
        cell_types=["HEK293T"],
        organelles=["HIST2H2BE"],
        infection_conditions=["Mock"],
        target_database=database,
        target_channel_name="Organelle",
        prediction_database=database,
        prediction_channel_name="Nuclei-prediction",
        log_output_dir=output_dir,
        log_name="intensity_metrics",
        z_slice=36,
        transforms=[
            NormalizeIntensityd(
                keys=["pred", "target"],
            )
        ],
    )
