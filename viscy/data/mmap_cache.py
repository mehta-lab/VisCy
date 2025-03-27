from __future__ import annotations

import os
import tempfile
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from iohub.ngff import Plate, Position, open_ome_zarr
from monai.data.meta_obj import set_track_meta
from monai.transforms.compose import Compose
from tensordict.memmap import MemoryMappedTensor
from torch import Tensor
from torch.multiprocessing import Manager
from torch.utils.data import Dataset

from viscy.data.gpu_aug import GPUTransformDataModule, SelectWell
from viscy.data.hcs import _ensure_channel_list, _read_norm_meta
from viscy.data.typing import DictTransform, NormMeta

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy

_logger = getLogger("lightning.pytorch")

_CacheMetadata = tuple[Position, int, NormMeta | None]


class MmappedDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        channel_names: list[str],
        cache_map: DictProxy,
        buffer: MemoryMappedTensor,
        preprocess_transforms: Compose | None = None,
        cpu_transform: Compose | None = None,
        array_key: str = "0",
        load_normalization_metadata: bool = True,
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
        self._buffer = buffer
        self._cache_map = cache_map
        self.preprocess_transforms = preprocess_transforms
        self.cpu_transform = cpu_transform
        self.load_normalization_metadata = load_normalization_metadata

    def __len__(self) -> int:
        return len(self._metadata_map)

    def _split_channels(self, volume: Tensor) -> dict[str, Tensor]:
        return {name: img[None] for name, img in zip(self.channels.keys(), volume)}

    def _preprocess_volume(self, volume: Tensor, norm_meta) -> Tensor:
        if self.preprocess_transforms:
            orig_shape = volume.shape
            sample = self._split_channels(volume)
            if self.load_normalization_metadata:
                sample["norm_meta"] = norm_meta
            sample = self.preprocess_transforms(sample)
            volume = torch.cat([sample[name] for name in self.channels.keys()], dim=0)
            assert volume.shape == orig_shape, (volume.shape, orig_shape, sample.keys())
        return volume

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        position, time_idx, norm_meta = self._metadata_map[idx]
        if not self._cache_map[idx]:
            _logger.debug(f"Loading volume for index {idx}")
            volume = torch.from_numpy(
                position[self.array_key]
                .oindex[time_idx, list(self.channels.values())]
                .astype(np.float32)
            )
            volume = self._preprocess_volume(volume, norm_meta)
            _logger.debug(f"Caching for index {idx}")
            self._cache_map[idx] = True
            self._buffer[idx] = volume
        else:
            _logger.debug(f"Using cached volume for index {idx}")
            volume = self._buffer[idx]
        sample = self._split_channels(volume)
        if self.cpu_transform:
            sample = self.cpu_transform(sample)
        if not isinstance(sample, list):
            sample = [sample]
        return sample


class MmappedDataModule(GPUTransformDataModule, SelectWell):
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
    prefetch_factor : int | None, optional
        Prefetching ratio for the torch dataloader, by default None
    array_key : str, optional
        Name of the image arrays (multiscales level), by default "0"
    scratch_dir : Path | None, optional
        Path to the scratch directory,
        by default None (use OS temporary data directory)
    include_wells : list[str] | None, optional
        Include only a subset of wells, by default None (include all wells)
    exclude_fovs : list[str] | None, optional
        Exclude FOVs, by default None (do not exclude any FOVs)
    """

    def __init__(
        self,
        data_path: Path,
        channels: str | list[str],
        batch_size: int,
        num_workers: int,
        split_ratio: float,
        preprocess_transforms: list[DictTransform],
        train_cpu_transforms: list[DictTransform],
        val_cpu_transforms: list[DictTransform],
        train_gpu_transforms: list[DictTransform],
        val_gpu_transforms: list[DictTransform],
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        array_key: str = "0",
        scratch_dir: Path | None = None,
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.channels = _ensure_channel_list(channels)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self._preprocessing_transforms = Compose(preprocess_transforms)
        self._train_cpu_transforms = Compose(train_cpu_transforms)
        self._val_cpu_transforms = Compose(val_cpu_transforms)
        self._train_gpu_transforms = Compose(train_gpu_transforms)
        self._val_gpu_transforms = Compose(val_gpu_transforms)
        self.pin_memory = pin_memory
        self.array_key = array_key
        self.scratch_dir = scratch_dir
        self._include_wells = include_wells
        self._exclude_fovs = exclude_fovs
        self.prepare_data_per_node = True
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

    @property
    def preprocessing_transforms(self) -> Compose:
        return self._preprocessing_transforms

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

    @property
    def cache_dir(self) -> Path:
        scratch_dir = self.scratch_dir or Path(tempfile.gettempdir())
        cache_dir = Path(
            scratch_dir,
            os.getenv("SLURM_JOB_ID", "viscy_cache"),
            str(
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            ),
            self.data_path.name,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _set_fit_global_state(self, num_positions: int) -> list[int]:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions).tolist()

    def _buffer_shape(self, arr_shape, fovs) -> tuple[int, ...]:
        return (len(fovs) * arr_shape[0], len(self.channels), *arr_shape[2:])

    def setup(self, stage: Literal["fit", "validate"]) -> None:
        if stage not in ("fit", "validate"):
            raise NotImplementedError("Only fit and validate stages are supported.")
        plate: Plate = open_ome_zarr(self.data_path, mode="r", layout="hcs")
        positions = self._filter_fit_fovs(plate)
        arr_shape = positions[0][self.array_key].shape
        shuffled_indices = self._set_fit_global_state(len(positions))
        num_train_fovs = int(len(positions) * self.split_ratio)
        train_fovs = [positions[i] for i in shuffled_indices[:num_train_fovs]]
        val_fovs = [positions[i] for i in shuffled_indices[num_train_fovs:]]
        _logger.debug(f"Training FOVs: {[p.zgroup.name for p in train_fovs]}")
        _logger.debug(f"Validation FOVs: {[p.zgroup.name for p in val_fovs]}")
        train_buffer = MemoryMappedTensor.empty(
            self._buffer_shape(arr_shape, train_fovs),
            dtype=torch.float32,
            filename=self.cache_dir / "train.mmap",
        )
        val_buffer = MemoryMappedTensor.empty(
            self._buffer_shape(arr_shape, val_fovs),
            dtype=torch.float32,
            filename=self.cache_dir / "val.mmap",
        )
        cache_map_train = Manager().dict()
        self.train_dataset = MmappedDataset(
            positions=train_fovs,
            channel_names=self.channels,
            cache_map=cache_map_train,
            buffer=train_buffer,
            preprocess_transforms=self.preprocessing_transforms,
            cpu_transform=self.train_cpu_transforms,
            array_key=self.array_key,
        )
        cache_map_val = Manager().dict()
        self.val_dataset = MmappedDataset(
            positions=val_fovs,
            channel_names=self.channels,
            cache_map=cache_map_val,
            buffer=val_buffer,
            preprocess_transforms=self.preprocessing_transforms,
            cpu_transform=self.val_cpu_transforms,
            array_key=self.array_key,
        )
