"""Test stage data module for evaluating segmentation."""

import logging
from pathlib import Path

import numpy as np
import torch
from iohub.ngff import ImageArray, Plate, open_ome_zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from viscy.data.typing import SegmentationSample

_logger = logging.getLogger("lightning.pytorch")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pred_dataset: Plate,
        target_dataset: Plate,
        pred_channel: str,
        target_channel: str,
        pred_z_slice: int | slice,
        target_z_slice: int | slice,
        img_name: str = "0",
    ) -> None:
        super().__init__()
        self.pred_dataset = pred_dataset
        self.target_dataset = target_dataset
        self.pred_channel = pred_dataset.get_channel_index(pred_channel)
        self.target_channel = target_dataset.get_channel_index(target_channel)
        self.pred_z_slice = pred_z_slice
        self.target_z_slice = target_z_slice
        self.img_name = img_name
        self._build_indices()

    def _build_indices(self) -> None:
        self._indices = []
        for p, (name, target_fov) in enumerate(self.target_dataset.positions()):
            pred_img: ImageArray = self.pred_dataset[name][self.img_name]
            target_img: ImageArray = target_fov[self.img_name]
            if not pred_img.shape[0] == target_img.shape[0]:
                raise ValueError(
                    "Shape mismatch between prediction and target: "
                    f"{pred_img.shape} vs {target_img.shape}"
                )
            for t in range(pred_img.shape[0]):
                self._indices.append((pred_img, target_img, p, t))
        _logger.info(f"Number of test samples: {len(self)}")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> SegmentationSample:
        pred_img, target_img, p, t = self._indices[idx]
        _logger.debug(f"Target image: {target_img.name}")
        pred = torch.from_numpy(
            pred_img[t, self.pred_channel, self.pred_z_slice].astype(np.int16)
        )
        target = torch.from_numpy(
            target_img[t, self.target_channel, self.target_z_slice].astype(np.int16)
        )
        return {"pred": pred, "target": target, "position_idx": p, "time_idx": t}


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        pred_dataset: Path,
        target_dataset: Path,
        pred_channel: str,
        target_channel: str,
        pred_z_slice: int,
        target_z_slice: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.pred_dataset = open_ome_zarr(pred_dataset)
        self.target_dataset = open_ome_zarr(target_dataset)
        self.pred_channel = pred_channel
        self.target_channel = target_channel
        self.pred_z_slice = pred_z_slice
        self.target_z_slice = target_z_slice
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage != "test":
            raise NotImplementedError("Only test stage is supported!")
        self.test_dataset = SegmentationDataset(
            self.pred_dataset,
            self.target_dataset,
            self.pred_channel,
            self.target_channel,
            self.pred_z_slice,
            self.target_z_slice,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
