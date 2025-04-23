"""Test stage data modules for loading data from DynaCell benchmark datasets."""

import logging
from pathlib import Path

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
        infection: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cell_type = cell_type
        self.organelle = organelle
        self.infection = infection

    def __getitem__(self, idx) -> DynaCellSample:
        return super().__getitem__(idx).update(
            {
                "cell_type": self.cell_type,
                "organelle": self.organelle,
                "infection": self.infection,
            }
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
        cell_type: str | None = None,
        organelle: str | None = None,
        infection: str | None = None,
    ) -> None:
        super().__init__()
        self.csv_database_path = csv_database_path
        self.pred_channel = pred_channel
        self.target_channel = target_channel
        self.pred_z_slice = pred_z_slice
        self.target_z_slice = target_z_slice
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cell_type = cell_type
        self.organelle = organelle
        self.infection = infection

    def setup(self, stage: str) -> None:
        if stage != "test":
            raise NotImplementedError("Only test stage is supported!")
        
        # Collect zarr store paths based on the requested cell_type, organelle, and infection conditions
        zarr_store_paths = [] # TODO
        cell_types = []
        organelles = []
        infections = []
        self.test_dataset = ConcatDataset(
            (
                DynaCellDataset(
                    cell_type,
                    organelle,
                    infection,
                    pred_dataset=open_ome_zarr(zarr_store_path),
                    target_dataset=open_ome_zarr(zarr_store_path),
                    pred_channel=self.pred_channel,
                    target_channel=self.target_channel,
                    pred_z_slice=self.pred_z_slice,
                    target_z_slice=self.target_z_slice,
                )
                for zarr_store_path, cell_type, organelle, infection in zip(
                    zarr_store_paths, cell_types, organelles, infections
                )
            )
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
