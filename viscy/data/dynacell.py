"""Test stage data modules for loading data from DynaCell benchmark datasets."""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, MapTransform
from torch.utils.data import ConcatDataset, DataLoader

from viscy.data.segmentation import TargetPredictionDataset
from viscy.data.typing import DynaCellSample

_logger = logging.getLogger("lightning.pytorch")


class DynaCellDataset(TargetPredictionDataset):
    """Return a DynaCellSample object with the cell type, organelle, and infection condition."""

    def __init__(
        self,
        cell_type: str,
        organelle: str,
        infection_condition: str,
        transforms: list[MapTransform] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cell_type = cell_type
        self.organelle = organelle
        self.infection_condition = infection_condition
        self.transforms = Compose(transforms) if transforms else None

    def __getitem__(self, idx) -> DynaCellSample:
        sample = super().__getitem__(idx)

        # Convert tensors to float32 for metrics compatibility
        sample["pred"] = sample["pred"].float()
        sample["target"] = sample["target"].float()

        # Add channel dimension if needed for the metrics (BxHxW -> BxCxHxW)
        if sample["pred"].ndim == 2:
            sample["pred"] = sample["pred"].unsqueeze(0)  # Add channel dimension
        elif sample["pred"].ndim == 3 and sample["pred"].shape[0] == 1:
            # If the first dimension is batch with size 1, reshape to [C,H,W]
            sample["pred"] = sample["pred"].squeeze(0).unsqueeze(0)

        if sample["target"].ndim == 2:
            sample["target"] = sample["target"].unsqueeze(0)  # Add channel dimension
        elif sample["target"].ndim == 3 and sample["target"].shape[0] == 1:
            # If the first dimension is batch with size 1, reshape to [C,H,W]
            sample["target"] = sample["target"].squeeze(0).unsqueeze(0)

        sample.update(
            {
                "cell_type": self.cell_type,
                "organelle": self.organelle,
                "infection_condition": self.infection_condition,
            }
        )

        # Apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)

        return sample


class DynaCellDatabase:
    """Database for DynaCell datasets filtered by cell types, organelles, and infection conditions."""

    def __init__(
        self,
        database: pd.DataFrame,
        cell_types: list[str],
        organelles: list[str],
        infection_conditions: list[str],
        channel_name: str,
        zarr_path_column_name: str = "Path",
        z_slice: int | slice | None = None,
    ):
        self.database = database
        self.cell_types = cell_types
        self.organelles = organelles
        self.infection_conditions = infection_conditions
        self.channel_name = channel_name
        self.z_slice = z_slice
        self.zarr_path_column_name = zarr_path_column_name

        required_columns = [
            "Cell type",
            "Organelle",
            "Infection",
            zarr_path_column_name,
        ]
        if not set(required_columns).issubset(self.database.columns):
            raise ValueError(f"Database must contains {required_columns}.")

        self._process_database()

    def _process_database(self):
        # Select the portion of the database that matches the provided criteria
        self._filtered_db = self.database[
            self.database["Cell type"].isin(self.cell_types)
            & self.database["Organelle"].isin(self.organelles)
            & self.database["Infection"].isin(self.infection_conditions)
        ].copy()

        # Extract zarr store paths
        self._filtered_db["Zarr path"] = self._filtered_db[
            self.zarr_path_column_name
        ].apply(lambda x: Path(*Path(x).parts[:-3]))
        self._filtered_db["FOV name"] = self._filtered_db[
            self.zarr_path_column_name
        ].apply(lambda x: Path(*Path(x).parts[-3:]).as_posix())
        self._filtered_db = self._filtered_db.drop_duplicates(subset=["Zarr path"])

        # Store values for later use
        self.zarr_paths = self._filtered_db["Zarr path"].values.tolist()
        self.position_names = self._filtered_db["FOV name"].values.tolist()
        self.cell_types_per_store = self._filtered_db["Cell type"].values.tolist()
        self.organelles_per_store = self._filtered_db["Organelle"].values.tolist()
        self.infection_per_store = self._filtered_db["Infection"].values.tolist()

    def __getitem__(self, idx) -> dict:
        return {
            "zarr_path": self.zarr_paths[idx],
            "position_names": [self.position_names[idx]],
            "cell_type": self.cell_types_per_store[idx],
            "organelle": self.organelles_per_store[idx],
            "infection_condition": self.infection_per_store[idx],
            "channel_name": self.channel_name,
            "z_slice": self.z_slice,
        }

    def __len__(self) -> int:
        return len(self.zarr_paths)


class DynaCellDataModule(LightningDataModule):
    def __init__(
        self,
        target_database: DynaCellDatabase,
        pred_database: DynaCellDatabase,
        batch_size: int,
        num_workers: int,
        transforms: list[MapTransform] | None = None,
    ) -> None:
        super().__init__()
        self.target_database = target_database
        self.pred_database = pred_database
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    def setup(self, stage: str) -> None:
        if stage != "test":
            raise NotImplementedError("Only test stage is supported!")

        # Verify both databases have the same length
        if len(self.target_database) != len(self.pred_database):
            raise ValueError(
                f"Target database length ({len(self.target_database)}) doesn't match "
                f"prediction database length ({len(self.pred_database)})"
            )

        # Create datasets
        datasets = []
        for i in range(len(self.target_database)):
            target_data = self.target_database[i]
            pred_data = self.pred_database[i]

            # Ensure target and prediction metadata match
            self._validate_matching_metadata(target_data, pred_data, i)

            datasets.append(
                DynaCellDataset(
                    cell_type=target_data["cell_type"],
                    organelle=target_data["organelle"],
                    infection_condition=target_data["infection_condition"],
                    pred_dataset=open_ome_zarr(pred_data["zarr_path"]),
                    target_dataset=open_ome_zarr(target_data["zarr_path"]),
                    position_names=target_data["position_names"],
                    pred_channel=pred_data["channel_name"],
                    target_channel=target_data["channel_name"],
                    pred_z_slice=pred_data["z_slice"],
                    target_z_slice=target_data["z_slice"],
                    transforms=self.transforms,
                )
            )

        self.test_dataset = ConcatDataset(datasets)

    def _validate_matching_metadata(
        self, target_data: dict, pred_data: dict, idx: int
    ) -> None:
        """Validate that target and prediction metadata match."""
        # Check cell type
        if target_data["cell_type"] != pred_data["cell_type"]:
            raise ValueError(
                f"Cell type mismatch at index {idx}: "
                f"target={target_data['cell_type']}, pred={pred_data['cell_type']}"
            )

        # Check organelle
        if target_data["organelle"] != pred_data["organelle"]:
            raise ValueError(
                f"Organelle mismatch at index {idx}: "
                f"target={target_data['organelle']}, pred={pred_data['organelle']}"
            )

        # Check infection condition
        if target_data["infection_condition"] != pred_data["infection_condition"]:
            raise ValueError(
                f"Infection condition mismatch at index {idx}: "
                f"target={target_data['infection_condition']}, pred={pred_data['infection_condition']}"
            )

        # Check zarr paths if they should match
        if target_data["zarr_path"] != pred_data["zarr_path"]:
            _logger.warning(
                f"Zarr path mismatch at index {idx}: "
                f"target={target_data['zarr_path']}, pred={pred_data['zarr_path']}"
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._custom_collate,
        )

    def _custom_collate(self, batch):
        """Custom collate function that preserves metadata strings."""
        assert len(batch) == 1, "Batch size must be 1 for DynaCellDataModule"
        # Extract metadata from first element in batch
        metadata = {
            "cell_type": batch[0]["cell_type"],
            "organelle": batch[0]["organelle"],
            "infection_condition": batch[0]["infection_condition"],
        }

        # Standard collate for tensors
        collated = torch.utils.data.default_collate(batch)

        # Add metadata back into collated batch
        collated.update(metadata)

        return collated
