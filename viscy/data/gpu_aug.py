from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from iohub.ngff import Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data.meta_obj import set_track_meta
from monai.data.utils import list_data_collate
from monai.transforms.compose import Compose
from torch import Tensor
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, Dataset

from viscy.data.distributed import ShardedDistributedSampler
from viscy.data.hcs import _ensure_channel_list, _read_norm_meta
from viscy.data.typing import DictTransform, NormMeta
from viscy.preprocessing.precompute import _filter_fovs, _filter_wells

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy

_logger = getLogger("lightning.pytorch")

_CacheMetadata = tuple[Position, int, NormMeta | None]


class GPUTransformDataModule(ABC, LightningDataModule):
    """Abstract data module with GPU transforms."""

    train_dataset: Dataset
    val_dataset: Dataset
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int | None

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
        _logger.debug(f"Using training sampler {sampler}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=list_data_collate,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self._maybe_sampler(self.val_dataset, shuffle=False)
        _logger.debug(f"Using validation sampler {sampler}")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            persistent_workers=True if self.num_workers > 0 else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=list_data_collate,
            prefetch_factor=self.prefetch_factor,
        )

    @property
    @abstractmethod
    def train_cpu_transforms(self) -> Compose: ...

    @property
    @abstractmethod
    def train_gpu_transforms(self) -> Compose: ...

    @property
    @abstractmethod
    def val_cpu_transforms(self) -> Compose: ...

    @property
    @abstractmethod
    def val_gpu_transforms(self) -> Compose: ...


class CachedOmeZarrDataset(Dataset):
    """Dataset for cached OME-Zarr arrays.

    Parameters
    ----------
    positions : list[Position]
        List of FOVs to load images from.
    channel_names : list[str]
        List of channel names to load.
    cache_map : DictProxy
        Shared dictionary for caching loaded volumes.
    transform : Compose | None, optional
        Composed transforms to be applied on the CPU, by default None
    array_key : str, optional
        The image array key name (multi-scale level), by default "0"
    load_normalization_metadata : bool, optional
        Load normalization metadata in the sample dictionary, by default True
    skip_cache : bool, optional
        Skip caching to save RAM, by default False
    """

    def __init__(
        self,
        positions: list[Position],
        channel_names: list[str],
        cache_map: DictProxy,
        transform: Compose | None = None,
        array_key: str = "0",
        load_normalization_metadata: bool = True,
        skip_cache: bool = False,
    ):
        key = 0
        self._metadata_map: dict[int, _CacheMetadata] = {}
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
        self.load_normalization_metadata = load_normalization_metadata
        self.skip_cache = skip_cache

    def __len__(self) -> int:
        return len(self._metadata_map)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        position, time_idx, norm_meta = self._metadata_map[idx]
        cache = self._cache_map[idx]
        if cache is None:
            _logger.debug(f"Loading volume for index {idx}")
            volume = torch.from_numpy(
                position[self.array_key]
                .oindex[time_idx, list(self.channels.values())]
                .astype(np.float32)
            )
            if not self.skip_cache:
                _logger.debug(f"Caching for index {idx}")
                self._cache_map[idx] = volume
        else:
            _logger.debug(f"Using cached volume for index {idx}")
            volume = cache
        sample = {name: img[None] for name, img in zip(self.channels.keys(), volume)}
        if self.load_normalization_metadata:
            sample["norm_meta"] = norm_meta
        if self.transform:
            sample = self.transform(sample)
        if not isinstance(sample, list):
            sample = [sample]
        return sample


class SelectWell:
    _include_wells: list[str] | None
    _exclude_fovs: list[str] | None

    def _filter_fit_fovs(self, plate: Plate) -> list[Position]:
        positions = []
        for well in _filter_wells(plate, include_wells=self._include_wells):
            for fov in _filter_fovs(well, exclude_fovs=self._exclude_fovs):
                positions.append(fov)
        if len(positions) < 2:
            raise ValueError(
                "At least 2 FOVs are required for training and validation."
            )
        return positions


class CachedOmeZarrDataModule(GPUTransformDataModule, SelectWell):
    """Data module for cached OME-Zarr arrays.

    Parameters
    ----------
    data_path : Path
        Path to the HCS OME-Zarr dataset.
    channels : str | list[str]
        Channel names to load.
    batch_size : int
        Batch size for training and validation.
    num_workers : int
        Number of workers for data-loaders.
    split_ratio : float
        Fraction of the FOVs used for the training split.
        The rest will be used for validation.
    train_cpu_transforms : list[DictTransform]
        Transforms to be applied on the CPU during training.
    val_cpu_transforms : list[DictTransform]
        Transforms to be applied on the CPU during validation.
    train_gpu_transforms : list[DictTransform]
        Transforms to be applied on the GPU during training.
    val_gpu_transforms : list[DictTransform]
        Transforms to be applied on the GPU during validation.
    pin_memory : bool, optional
        Use page-locked memory in data-loaders, by default True
    skip_cache : bool, optional
        Skip caching for this dataset, by default False
    include_wells : list[str], optional
        List of well names to include in the dataset, by default None (all)
    include_wells : list[str], optional
        List of well names to include in the dataset, by default None (all)
    """

    def __init__(
        self,
        data_path: Path,
        channels: str | list[str],
        batch_size: int,
        num_workers: int,
        split_ratio: float,
        train_cpu_transforms: list[DictTransform],
        val_cpu_transforms: list[DictTransform],
        train_gpu_transforms: list[DictTransform],
        val_gpu_transforms: list[DictTransform],
        pin_memory: bool = True,
        skip_cache: bool = False,
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.channels = _ensure_channel_list(channels)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self._train_cpu_transforms = Compose(train_cpu_transforms)
        self._val_cpu_transforms = Compose(val_cpu_transforms)
        self._train_gpu_transforms = Compose(train_gpu_transforms)
        self._val_gpu_transforms = Compose(val_gpu_transforms)
        self.pin_memory = pin_memory
        self.skip_cache = skip_cache
        self._include_wells = include_wells
        self._exclude_fovs = exclude_fovs

    @property
    def train_cpu_transforms(self) -> Compose:
        return self._train_cpu_transforms

    @property
    def train_gpu_transforms(self) -> Compose:
        return self._train_gpu_transforms

    @property
    def val_cpu_transforms(self) -> Compose:
        return self._val_cpu_transforms

    @property
    def val_gpu_transforms(self) -> Compose:
        return self._val_gpu_transforms

    def _set_fit_global_state(self, num_positions: int) -> list[int]:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions).tolist()

    def _include_well_name(self, name: str) -> bool:
        if self._include_wells is None:
            return True
        else:
            return name in self._include_wells

    def _filter_fit_fovs(self, plate: Plate) -> list[Position]:
        positions = []
        for well_name, well in plate.wells():
            if self._include_well_name(well_name):
                for _, p in well.positions():
                    positions.append(p)
        if len(positions) < 2:
            raise ValueError(
                "At least 2 FOVs are required for training and validation."
            )
        return positions

    def setup(self, stage: Literal["fit", "validate"]) -> None:
        if stage not in ("fit", "validate"):
            raise NotImplementedError("Only fit and validate stages are supported.")
        cache_map = Manager().dict()
        plate: Plate = open_ome_zarr(self.data_path, mode="r", layout="hcs")
        positions = self._filter_fit_fovs(plate)
        shuffled_indices = self._set_fit_global_state(len(positions))
        num_train_fovs = int(len(positions) * self.split_ratio)
        train_fovs = [positions[i] for i in shuffled_indices[:num_train_fovs]]
        val_fovs = [positions[i] for i in shuffled_indices[num_train_fovs:]]
        _logger.debug(f"Training FOVs: {[p.zgroup.name for p in train_fovs]}")
        _logger.debug(f"Validation FOVs: {[p.zgroup.name for p in val_fovs]}")
        self.train_dataset = CachedOmeZarrDataset(
            train_fovs,
            self.channels,
            cache_map,
            transform=self.train_cpu_transforms,
            skip_cache=self.skip_cache,
        )
        self.val_dataset = CachedOmeZarrDataset(
            val_fovs,
            self.channels,
            cache_map,
            transform=self.val_cpu_transforms,
            skip_cache=self.skip_cache,
        )
