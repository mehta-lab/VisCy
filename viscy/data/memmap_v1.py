from pathlib import Path

from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, MapTransform
from torch.utils.data import DataLoader

from viscy.data.hcs import ChannelMap, SlidingWindowDataset
from viscy.data.typing import Sample
import logging
import math
import os
import re
import tempfile
from glob import glob
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import torch
import zarr
from imageio import imread
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    MultiSampleTrait,
    RandAffined,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy.data.typing import ChannelMap, HCSStackIndex, NormMeta, Sample


class MemMapDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        train_transforms: list[MapTransform],
        val_transforms: list[MapTransform],
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self._setup_fit()

    def prepare_data(self):
        logger = logging.getLogger()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(self._data_log_path())
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.info(f"Caching dataset at {self.cache_path}.")
        tmp_store = zarr.NestedDirectoryStore(self.cache_path)
