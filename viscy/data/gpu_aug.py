from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from iohub.ngff import Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data.meta_obj import set_track_meta
from monai.transforms import Compose
from torch import Tensor
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, Dataset, Subset

from viscy.data.distributed import ShardedDistributedSampler
from viscy.data.hcs import _ensure_channel_list, _read_norm_meta
from viscy.data.typing import DictTransform

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy

_logger = getLogger("lightning.pytorch")


class CachedOmeZarrDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        channel_names: list[str],
        cache_map: DictProxy,
        transform: DictTransform | None = None,
        array_key: str = "0",
    ):
        key = 0
        self._metadata_map = {}
        for position in positions:
            img = position[array_key]
            norm_meta = _read_norm_meta(position)
            for time_idx in range(img.frames):
                cache_map[key] = None
                self._metadata_map[key] = (position, time_idx, norm_meta)
                key += 1
        self.channels = {ch: position.get_channel_index(ch) for ch in channel_names}
        self.array_key = array_key
        self._cache_map = cache_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self._cache_map)

    def __getitem__(self, idx: int) -> Tensor:
        position, time_idx, norm_meta = self._metadata_map[idx]
        cache = self._cache_map[idx]
        if cache is None:
            _logger.debug(f"Caching for index {idx}")
            volume = torch.from_numpy(
                position[self.array_key]
                .oindex[time_idx, list(self.channels.values())]
                .astype(np.float32)
            )
            self._cache_map[idx] = volume
        else:
            _logger.debug(f"Using cached volume for index {idx}")
            volume = cache
        sample = {name: img[None] for name, img in zip(self.channels.keys(), volume)}
        sample["norm_meta"] = norm_meta
        if self.transform:
            sample = self.transform(sample)
        if not isinstance(sample, list):
            sample = [sample]
        out_tensors = []
        for s in sample:
            s.pop("norm_meta")
            s_out = torch.cat(list(s.values()))
            out_tensors.append(s_out)
        return out_tensors


class CachedOmeZarrDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        channels: str | list[str],
        batch_size: int,
        num_workers: int,
        split_ratio: float,
        transforms: list[DictTransform],
    ):
        super().__init__()
        self.data_path = data_path
        self.channels = _ensure_channel_list(channels)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.transforms = Compose(transforms)

    def _set_fit_global_state(self, num_positions: int) -> list[int]:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions).tolist()

    def setup(self, stage: Literal["fit", "validate"]) -> None:
        cache_map = Manager().dict()
        plate: Plate = open_ome_zarr(self.data_path, mode="r", layout="hcs")
        positions = [p for _, p in plate.positions()]
        shuffled_indices = self._set_fit_global_state(len(positions))
        num_train_fovs = int(len(positions) * self.split_ratio)
        dataset = CachedOmeZarrDataset(
            positions, self.channels, cache_map, self.transforms
        )
        self.train_dataset = Subset(dataset, shuffled_indices[:num_train_fovs])
        self.val_dataset = Subset(dataset, shuffled_indices[num_train_fovs:])

    def _maybe_sampler(
        self, dataset: Dataset, shuffle: bool
    ) -> ShardedDistributedSampler | None:
        return (
            ShardedDistributedSampler(dataset, shuffle=shuffle)
            if torch.distributed.is_initialized()
            else None
        )

    def train_dataloader(self) -> DataLoader:
        sampler = self._maybe_sampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self._maybe_sampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            drop_last=False,
        )
