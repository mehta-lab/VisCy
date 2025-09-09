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


class TargetPredictionDataset(Dataset):
    """
    A PyTorch Dataset providing paired target and prediction images / volumes from OME-Zarr
    datasets.

    Attributes:
        pred_dataset (Plate): The prediction dataset Plate object.
        target_dataset (Plate): The target dataset Plate object.
        pred_channel (str): The channel name in the prediction dataset.
        target_channel (str): The channel name in the target dataset.
        pred_z_slice (int | slice | None): The z-slice or range of z-slices for the 
            prediction dataset. Defaults to None which is converted to slice(None).
        target_z_slice (int | slice | None): The z-slice or range of z-slices for the 
            target dataset. Defaults to None which is converted to slice(None).
        img_name (str): The name of the image to retrieve from the datasets. Defaults to "0".
        dtype (np.dtype | None): The data type to cast the images to. Defaults to np.int16.
    """

    def __init__(
        self,
        pred_dataset: Plate,
        target_dataset: Plate,
        pred_channel: str,
        target_channel: str,
        pred_z_slice: int | slice | None = None,
        target_z_slice: int | slice | None = None,
        position_names: list[str] | None = None,
        img_name: str = "0",
        dtype: np.dtype | None = np.int16,
    ) -> None:
        super().__init__()
        self.pred_dataset = pred_dataset
        self.target_dataset = target_dataset
        self.pred_channel = pred_dataset.get_channel_index(pred_channel)
        self.target_channel = target_dataset.get_channel_index(target_channel)
        self.pred_z_slice = pred_z_slice if pred_z_slice is not None else slice(None)
        self.target_z_slice = (
            target_z_slice if target_z_slice is not None else slice(None)
        )
        self.img_name = img_name
        self.dtype = dtype
        self.position_names = position_names
        if not position_names:
            self.position_names = list([p[0] for p in self.target_dataset.positions()])

        self._build_indices()

    def _build_indices(self) -> None:
        self._indices = []
        for p, name in enumerate(self.position_names):
            pred_img: ImageArray = self.pred_dataset[name][self.img_name]
            target_img: ImageArray = self.target_dataset[name][self.img_name]
            if not pred_img.shape[0] == target_img.shape[0]:
                raise ValueError(
                    "Shape mismatch between prediction and target: "
                    f"{pred_img.shape} vs {target_img.shape}"
                )
            for t in range(pred_img.shape[0]):
                self._indices.append((pred_img, target_img, p, t))
        # Only log sample count once to reduce noise
        if hasattr(self, '_samples_logged'):
            pass  # Already logged for this dataset type
        else:
            _logger.info(f"Built dataset with {len(self)} samples across {len(self.position_names)} positions")
            type(self)._samples_logged = True

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> SegmentationSample:
        pred_img, target_img, p, t = self._indices[idx]
        _logger.debug(f"Target image: {target_img.name}")
        _pred = pred_img[t, self.pred_channel, self.pred_z_slice]
        _target = target_img[t, self.target_channel, self.target_z_slice]
        if self.dtype is not None:
            _pred = _pred.astype(self.dtype)
            _target = _target.astype(self.dtype)
        pred = torch.from_numpy(_pred.astype(self.dtype))
        target = torch.from_numpy(_target.astype(self.dtype))
        return {"pred": pred, "target": target, "position_idx": p, "time_idx": t}


class TargetPredictionDataModule(LightningDataModule):
    def __init__(
        self,
        pred_dataset: Path,
        target_dataset: Path,
        pred_channel: str,
        target_channel: str,
        pred_z_slice: int | slice | None,
        target_z_slice: int | slice | None,
        batch_size: int,
        num_workers: int,
        dtype: np.dtype | None = np.int16,
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
        self.dtype = dtype

    def setup(self, stage: str) -> None:
        if stage != "test":
            raise NotImplementedError("Only test stage is supported!")
        self.test_dataset = TargetPredictionDataset(
            self.pred_dataset,
            self.target_dataset,
            self.pred_channel,
            self.target_channel,
            self.pred_z_slice,
            self.target_z_slice,
            dtype=self.dtype,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
