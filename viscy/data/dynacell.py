"""Test stage data modules for loading data from DynaCell benchmark datasets."""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from viscy.data.segmentation import TargetPredictionDataset
from viscy.data.typing import DynaCellSample

_logger = logging.getLogger("lightning.pytorch")


class DynaCellDataset(TargetPredictionDataset):
    def __init__(
        self,
        cell_type: str,
        organelle: str,
        infection_condition: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cell_type = cell_type
        self.organelle = organelle
        self.infection_condition = infection_condition

    def __getitem__(self, idx) -> DynaCellSample:
        return (
            super()
            .__getitem__(idx)
            .update(
                {
                    "cell_type": self.cell_type,
                    "organelle": self.organelle,
                    "infection_condition": self.infection_condition,
                }
            )
        )


class DynaCellDataModule(LightningDataModule):
    def __init__(
        self,
        csv_database_path: Path,
        pred_channel: str,
        target_channel: str,
        pred_z_slice: int | slice | None,
        target_z_slice: int | slice | None,
        batch_size: int,
        num_workers: int,
        cell_types: Sequence | None = None,
        organelles: Sequence | None = None,
        infection_conditions: Sequence | None = None,
    ) -> None:
        super().__init__()
        self.csv_database_path = csv_database_path
        self.pred_channel = pred_channel
        self.target_channel = target_channel
        self.pred_z_slice = pred_z_slice
        self.target_z_slice = target_z_slice
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._cell_types = cell_types
        self._organelles = organelles
        self._infection_conditions = infection_conditions

    @staticmethod
    def _validate(
        database: pd.DataFrame, column: str, provided_values: Sequence | None
    ) -> list:
        all_columns = database[column].unique().tolist()

        if not provided_values:
            return all_columns

        if not set(provided_values).issubset(set(all_columns)):
            raise ValueError(
                f"Not all of {provided_values} are found in column {column} of the database."
            )

        return list(provided_values)

    def setup(self, stage: str) -> None:
        if stage != "test":
            raise NotImplementedError("Only test stage is supported!")

        database = pd.read_csv(self.csv_database_path, dtype={"FOV": str})
        self.cell_types = self._validate(database, "Cell type", self._cell_types)
        self.organelles = self._validate(database, "Organelle", self._organelles)
        self.infection_conditions = self._validate(
            database, "Infection", self._infection_conditions
        )

        # Select the portion of the database that matches the provided cell types, organelles, and infection conditions
        _database = database[
            database["Cell type"].isin(self.cell_types)
            & database["Organelle"].isin(self.organelles)
            & database["Infection"].isin(self.infection_conditions)
        ].copy()

        # Input Paths are at the FOV level, e.g. store.zarr/A/1/0, we'll extract the zarr store path
        # and remove duplicates at the FOV level since DynaCellDataset will iterate over the FOVs
        _database["Zarr path"] = _database["Path"].apply(
            lambda x: Path(*Path(x).parts[:-3])
        )
        _database = _database.drop_duplicates(subset=["Zarr path"])

        zarr_store_paths = _database["Zarr path"].values.tolist()
        cell_type_per_store = _database["Cell type"].values.tolist()
        organelle_per_store = _database["Organelle"].values.tolist()
        infection_per_store = _database["Infection"].values.tolist()

        self.test_dataset = ConcatDataset(
            (
                DynaCellDataset(
                    cell_type,
                    organelle,
                    infection_condition,
                    pred_dataset=open_ome_zarr(zarr_store_path),
                    target_dataset=open_ome_zarr(zarr_store_path),
                    pred_channel=self.pred_channel,
                    target_channel=self.target_channel,
                    pred_z_slice=self.pred_z_slice,
                    target_z_slice=self.target_z_slice,
                )
                for zarr_store_path, cell_type, organelle, infection_condition in zip(
                    zarr_store_paths[:3],
                    cell_type_per_store,
                    organelle_per_store,
                    infection_per_store,  # DEBUG
                )
            )
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
