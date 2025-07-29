import datetime
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from iohub import open_ome_zarr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized
from torch.onnx import OperatorExportTypes

from viscy.data.dynacell import DynaCellDatabase, DynaCellDataModule
from viscy.preprocessing.precompute import precompute_array
from viscy.translation.evaluation import IntensityMetrics, SegmentationMetrics
from viscy.utils.meta_utils import generate_normalization_metadata

_logger = logging.getLogger("lightning.pytorch")


class VisCyTrainer(Trainer):
    def preprocess(
        self,
        data_path: Path,
        channel_names: list[str] | Literal[-1] = -1,
        num_workers: int = 1,
        block_size: int = 32,
        model: LightningModule | None = None,
    ):
        """
        Compute dataset statistics before training or testing for normalization.

        Parameters
        ----------
        data_path : Path
            Path to the HCS OME-Zarr dataset
        channel_names : list[str] | Literal[-1], optional
            Channel names to compute statistics for, by default -1
        num_workers : int, optional
            Number of CPU workers, by default 1
        block_size : int, optional
            Block size to subsample images, by default 32
        model: LightningModule, optional
            Ignored placeholder, by default None
        """
        if model is not None:
            _logger.warning("Ignoring model configuration during preprocessing.")
        with open_ome_zarr(data_path, layout="hcs", mode="r") as dataset:
            channel_indices = (
                [dataset.channel_names.index(c) for c in channel_names]
                if channel_names != -1
                else channel_names
            )
        generate_normalization_metadata(
            zarr_dir=data_path,
            num_workers=num_workers,
            channel_ids=channel_indices,
            grid_spacing=block_size,
        )

    def export(
        self,
        model: LightningModule,
        export_path: Path,
        ckpt_path: Path,
        format: str = "onnx",
    ):
        """
        Export the model for deployment (currently only ONNX is supported).

        Parameters
        ----------
        model : LightningModule
            Module to export.
        export_path : Path
            Output file name.
        ckpt_path : Path
            Model checkpoint path.
        format : str, optional
            Format (currently only ONNX is supported), by default "onnx".
        """
        if not format.lower() == "onnx":
            raise NotImplementedError(f"Export format '{format}'")
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        model.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"])
        model.eval()
        model.to_onnx(
            export_path,
            input_sample=model.example_input_array,
            export_params=True,
            opset_version=18,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
                "output": {
                    0: "batch_size",
                    1: "channels",
                    3: "num_rows",
                    4: "num_cols",
                },
            },
        )
        _logger.info(f"ONNX exported at {export_path}")

    def precompute(
        self,
        data_path: Path,
        output_path: Path,
        channel_names: list[str],
        subtrahends: list[Literal["mean"] | float],
        divisors: list[Literal["std"] | tuple[float, float]],
        image_array_key: str = "0",
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        model: LightningModule | None = None,
    ):
        precompute_array(
            data_path=data_path,
            output_path=output_path,
            channel_names=channel_names,
            subtrahends=subtrahends,
            divisors=divisors,
            image_array_key=image_array_key,
            include_wells=include_wells,
            exclude_fovs=exclude_fovs,
        )

    def compute_dynacell_metrics(
        self,
        target_database: Path,
        pred_database: Path,
        output_dir: Path,
        method: str = "intensity",
        target_channel: str = "Organelle",
        pred_channel: str = "Organelle",
        target_z_slice: int | list[int] = 16,
        pred_z_slice: int | list[int] = 16,
        target_cell_types: list[str] = None,
        target_organelles: list[str] = None,
        target_infection_conditions: list[str] = None,
        pred_cell_types: list[str] = None,
        pred_organelles: list[str] = None,
        pred_infection_conditions: list[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        version: str = "1",
        transforms: list = None,
        model: LightningModule | None = None,
    ):
        """
        Compute metrics for DynaCell datasets.

        Parameters
        ----------
        target_database : Path
            Path to the target DynaCell database file
        pred_database : Path
            Path to the prediction DynaCell database file
        output_dir : Path
            Directory to save output metrics
        method : str, optional
            Type of metrics to compute ('intensity' or 'segmentation2D'), by default "intensity"
        target_channel : str, optional
            Channel name for target dataset, by default "Organelle"
        pred_channel : str, optional
            Channel name for prediction dataset, by default "Organelle"
        target_z_slice : int | list[int], optional
            Z-slice to use for target dataset, by default 16
        pred_z_slice : int | list[int], optional
            Z-slice to use for prediction dataset, by default 16
        target_cell_types : list[str], optional
            Cell types to include for target dataset, by default None (all available)
        target_organelles : list[str], optional
            Organelles to include for target dataset, by default None (all available)
        target_infection_conditions : list[str], optional
            Infection conditions to include for target dataset, by default None (all available)
        pred_cell_types : list[str], optional
            Cell types to include for prediction dataset, by default None (all available)
        pred_organelles : list[str], optional
            Organelles to include for prediction dataset, by default None (all available)
        pred_infection_conditions : list[str], optional
            Infection conditions to include for prediction dataset, by default None (all available)
        batch_size : int, optional
            Batch size for processing, by default 1
        num_workers : int, optional
            Number of workers for data loading, by default 0
        version : str, optional
            Version string for output directory, by default "1"
        transforms : list, optional
            List of transforms to apply to the data (e.g., normalization), by default None
        model : LightningModule | None, optional
            Ignored placeholder, by default None
        """
        if model is not None:
            _logger.warning("Ignoring model configuration for DynaCell metrics.")

        # Set default empty lists for filters
        target_cell_types = target_cell_types or []
        target_organelles = target_organelles or []
        target_infection_conditions = target_infection_conditions or []
        pred_cell_types = pred_cell_types or []
        pred_organelles = pred_organelles or []
        pred_infection_conditions = pred_infection_conditions or []

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique versioning
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle z_slice values (-1 means all slices)
        if isinstance(target_z_slice, list) and len(target_z_slice) == 2:
            # Use the list as a range [start, stop] for the slice
            if target_z_slice[1] - target_z_slice[0] == 1:
                # If range length is 1, just use the single integer
                target_z_slice_value = int(target_z_slice[0])
            else:
                target_z_slice_value = slice(target_z_slice[0], target_z_slice[1])
        else:
            target_z_slice_value = (
                slice(None) if target_z_slice == -1 else int(target_z_slice)
            )

        if isinstance(pred_z_slice, list) and len(pred_z_slice) == 2:
            # Use the list as a range [start, stop] for the slice
            if pred_z_slice[1] - pred_z_slice[0] == 1:
                # If range length is 1, just use the single integer
                pred_z_slice_value = int(pred_z_slice[0])
            else:
                pred_z_slice_value = slice(pred_z_slice[0], pred_z_slice[1])
        else:
            pred_z_slice_value = (
                slice(None) if pred_z_slice == -1 else int(pred_z_slice)
            )

        # Default to all available values if not specified for target database
        if (
            not target_cell_types
            or not target_organelles
            or not target_infection_conditions
        ):
            _logger.info("Loading target database to get available values...")
            df = pd.read_csv(target_database, dtype={"FOV": str})

            if not target_cell_types:
                target_cell_types = df["Cell type"].unique().tolist()
                _logger.info(
                    f"Using all available target cell types: {target_cell_types}"
                )

            if not target_organelles:
                target_organelles = df["Organelle"].unique().tolist()
                _logger.info(
                    f"Using all available target organelles: {target_organelles}"
                )

            if not target_infection_conditions:
                target_infection_conditions = df["Infection"].unique().tolist()
                _logger.info(
                    f"Using all available target infection conditions: {target_infection_conditions}"
                )

        # Default to all available values if not specified for prediction database
        if not pred_cell_types or not pred_organelles or not pred_infection_conditions:
            _logger.info("Loading prediction database to get available values...")
            df = pd.read_csv(pred_database, dtype={"FOV": str})

            if not pred_cell_types:
                pred_cell_types = df["Cell type"].unique().tolist()
                _logger.info(
                    f"Using all available prediction cell types: {pred_cell_types}"
                )

            if not pred_organelles:
                pred_organelles = df["Organelle"].unique().tolist()
                _logger.info(
                    f"Using all available prediction organelles: {pred_organelles}"
                )

            if not pred_infection_conditions:
                pred_infection_conditions = df["Infection"].unique().tolist()
                _logger.info(
                    f"Using all available prediction infection conditions: {pred_infection_conditions}"
                )

        # Create target database
        _logger.info(
            f"Creating target database from {target_database} with channel '{target_channel}'"
        )
        target_db = DynaCellDatabase(
            database=target_database,
            cell_types=target_cell_types,
            organelles=target_organelles,
            infection_conditions=target_infection_conditions,
            channel_name=target_channel,
            z_slice=target_z_slice_value,
        )

        # Create prediction database
        _logger.info(
            f"Creating prediction database from {pred_database} with channel '{pred_channel}'"
        )
        pred_db = DynaCellDatabase(
            database=pred_database,
            cell_types=pred_cell_types,
            organelles=pred_organelles,
            infection_conditions=pred_infection_conditions,
            channel_name=pred_channel,
            z_slice=pred_z_slice_value,
        )

        # Create datamodule
        _logger.info("Creating DynaCellDataModule...")
        dm = DynaCellDataModule(
            target_database=target_db,
            pred_database=pred_db,
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=transforms,
        )

        # Setup datamodule
        dm.setup(stage="test")

        # Determine run-specific output paths
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)

        # Unique name based on method and timestamp
        run_name = f"{method}_{timestamp}"

        # Create logger
        _logger.info(f"Creating logger for run '{run_name}' with version '{version}'")
        logger = CSVLogger(save_dir=method_dir, name=run_name, version=version)

        # Select and run appropriate metrics
        if method == "Segmentation2D":
            _logger.info("Running segmentation metrics...")
            metrics_module = SegmentationMetrics()
        elif method == "Intensity":
            _logger.info("Running intensity metrics...")
            metrics_module = IntensityMetrics()
        else:
            raise ValueError(f"Invalid method: {method}")

        # Run the metrics computation
        self.test(metrics_module, datamodule=dm)

        # Find the metrics file
        metrics_file = method_dir / run_name / version / "metrics.csv"

        if not metrics_file.exists():
            _logger.warning(f"No metrics file found at {metrics_file}")
            return None

        metrics_df = pd.read_csv(metrics_file)
        _logger.info(f"Metrics saved to: {metrics_file}")
        _logger.info(f"Computed {len(metrics_df)} metric rows")

        # Display columns in the metrics
        _logger.info(f"Metrics columns: {metrics_df.columns.tolist()}")

        # Display a preview of the metrics
        if not metrics_df.empty:
            _logger.info(f"Metrics preview:\n{repr(metrics_df.head())}")

        return metrics_file
