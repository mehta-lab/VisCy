import logging
from pathlib import Path

import numpy as np
from iohub.ngff import ImageArray, Plate, Position, TransformationMeta, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, MapTransform
from torch import Tensor
from torch.utils.data import DataLoader

from viscy.data.hcs import ChannelMap, SlidingWindowDataset


class CTMCv1DataModule(LightningDataModule):
    """
    Autoregression data module for the CTMCv1 dataset.
    Training and validation datasets are stored in separate HCS OME-Zarr stores.
    """

    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        train_transforms: list[MapTransform],
        val_transforms: list[MapTransform],
        batch_size: int = 16,
        num_workers: int = 8,
        channel_name: str = "DIC",
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.channel_map = ChannelMap(source=channel_name, target=channel_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        train_plate = open_ome_zarr(self.train_data_path, mode="r")
        val_plate = open_ome_zarr(self.val_data_path, mode="r")
        train_positions = [p for _, p in train_plate.positions()]
        val_positions = [p for _, p in val_plate.positions()]
        self.train_dataset = SlidingWindowDataset(
            train_positions,
            channels=self.channel_map,
            z_window_size=1,
            transform=Compose(self.train_transform),
        )
        self.val_dataset = SlidingWindowDataset(
            val_positions,
            channels=self.channel_map,
            z_window_size=1,
            transform=Compose(self.val_transform),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
