import logging
from datetime import datetime
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    MultiSampleTrait,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy.data.hcs import _read_norm_meta
from viscy.data.typing import ChannelMap, DictTransform, Sample
from viscy.data.distributed import ShardedDistributedSampler
from torch.distributed import get_rank
import torch.distributed as dist

_logger = logging.getLogger("lightning.pytorch")

# TODO: cache the norm metadata when caching the dataset

# Map the NumPy dtype to the corresponding PyTorch dtype
numpy_to_torch_dtype = {
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("uint8"): torch.int8,
    np.dtype("uint16"): torch.int16,
}


def _stack_channels(
    sample_images: list[dict[str, Tensor]] | dict[str, Tensor],
    channels: ChannelMap,
    key: str,
) -> Tensor | list[Tensor]:
    """Stack single-channel images into a multi-channel tensor."""
    if not isinstance(sample_images, list):
        return torch.stack([sample_images[ch][0] for ch in channels[key]])
    # training time
    # sample_images is a list['Phase3D'].shape = (1,3,256,256)
    return [torch.stack([im[ch][0] for ch in channels[key]]) for im in sample_images]


def _collate_samples(batch: Sequence[Sample]) -> Sample:
    """Collate samples into a batch sample.

    :param Sequence[Sample] batch: a sequence of dictionaries,
        where each key may point to a value of a single tensor or a list of tensors,
        as is the case with ``train_patches_per_stack > 1``.
    :return Sample: Batch sample (dictionary of tensors)
    """
    collated: Sample = {}
    for key in batch[0].keys():
        data = []
        for sample in batch:
            if isinstance(sample[key], Sequence):
                data.extend(sample[key])
            else:
                data.append(sample[key])
        collated[key] = collate_meta_tensor(data)
    return collated

def is_ddp_enabled() -> bool:
    """Check if distributed data parallel (DDP) is initialized."""
    return dist.is_available() and dist.is_initialized()

class CachedDataset(Dataset):
    """
    A dataset that caches the data in RAM.
    It relies on the `__getitem__` method to load the data on the 1st epoch.
    """

    def __init__(
        self,
        shared_dict: DictProxy,
        positions: list[Position],
        channels: ChannelMap,
        transform: DictTransform | None = None,
    ):
        super().__init__()
        if is_ddp_enabled():
            self.rank = dist.get_rank()
            _logger.info(f"=== Initializing cache pool for rank {self.rank} ===")

        self.cache_dict = shared_dict
        self.positions = positions
        self.channels = channels
        self.transform = transform

        self.source_ch_idx = [
            positions[0].get_channel_index(c) for c in channels["source"]
        ]
        self.target_ch_idx = (
            [positions[0].get_channel_index(c) for c in channels["target"]]
            if "target" in channels
            else None
        )
        # Get total num channels
        self.total_ch_names = self.channels["source"].copy()
        self.total_ch_idx = self.source_ch_idx.copy()
        if self.target_ch_idx is not None:
            self.total_ch_names.extend(self.channels["target"])
            self.total_ch_idx.extend(self.target_ch_idx)
        self._position_mapping()

        # Cached dictionary with tensors
        # TODO: Delete after testing
        self._cached_pos = []

    def _position_mapping(self) -> None:
        self.position_keys = []
        self.norm_meta_dict = {}

        for pos in self.positions:
            self.position_keys.append(pos.data.name)
            self.norm_meta_dict[str(pos.data.name)] = _read_norm_meta(pos)

    def _cache_dataset(self, index: int, channel_index: list[int], t: int = 0) -> None:
        # Add the position to the cached_dict
        # TODO: hardcoding to t=0
        data = self.positions[index].data.oindex[slice(t, t + 1), channel_index, :]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        self.cache_dict[str(self.position_keys[index])] = torch.from_numpy(data)

    def _get_weight_map(self, position: Position) -> Tensor:
        # Get the weight map from the position for the MONAI weightedcrop transform
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Sample:
        # Check if the sample is in the cache else add it
        sample_id = self.position_keys[index]
        if sample_id not in self.cache_dict:
            _logger.info(f"Adding {sample_id} to cache")
            self._cached_pos.append(index)
            _logger.info(f"Cached positions: {self._cached_pos}")
            self._cache_dataset(index, channel_index=self.total_ch_idx)

        # Get the sample from the cache
        _logger.info("Getting sample from cache")
        start_time = datetime.now()
        images = self.cache_dict[sample_id].unbind(dim=1)
        norm_meta = self.norm_meta_dict[str(sample_id)]
        after_cache = datetime.now() - start_time
        sample_images = {k: v for k, v in zip(self.total_ch_names, images)}

        if self.target_ch_idx is not None:
            # FIXME: this uses the first target channel as weight for performance
            # since adding a reference to a tensor does not copy
            # maybe write a weight map in preprocessing to use more information?
            sample_images["weight"] = sample_images[self.channels["target"][0]]
        if norm_meta is not None:
            sample_images["norm_meta"] = norm_meta
        if self.transform:
            before_transform = datetime.now()
            sample_images = self.transform(sample_images)
            after_transform = datetime.now() - before_transform
        if "weight" in sample_images:
            del sample_images["weight"]
        sample = {
            "index": sample_id,
            "source": _stack_channels(sample_images, self.channels, "source"),
            "norm_meta": norm_meta,
        }
        if self.target_ch_idx is not None:
            sample["target"] = _stack_channels(sample_images, self.channels, "target")

        _logger.info(f"\nTime taken to cache: {after_cache}")
        _logger.info(f"Time taken to transform: {after_transform}")
        _logger.info(f"Time taken to get sample: {datetime.now() - start_time}\n")

        return sample

    def _load_sample(self, position: Position) -> Sample:
        source, target = self.channel_map.source, self.channel_map.target
        source_data = self._load_channel_data(position, source)
        target_data = self._load_channel_data(position, target)
        sample = {"source": source_data, "target": target_data}
        return sample


class CachedDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        source_channel: str | Sequence[str],
        target_channel: str | Sequence[str],
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        architecture: Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae"] = "UNeXt2",
        yx_patch_size: tuple[int, int] = (256, 256),
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        z_window_size: int = 1,
        timeout: int = 600,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_channel = source_channel
        self.target_channel = target_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_2d = False if architecture in ["UNeXt2", "3D", "fcmae"] else True
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.normalizations = normalizations
        self.augmentations = augmentations
        self.z_window_size = z_window_size
        self.timeout = timeout

    @property
    def _base_dataset_settings(self) -> dict[str, dict[str, list[str]] | int]:
        return {
            "channels": {"source": self.source_channel},
        }

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        dataset_settings = self._base_dataset_settings
        if stage in ("fit", "validate"):
            self._setup_fit(dataset_settings)
        elif stage == "test":
            raise NotImplementedError("Test stage is not supported")
        elif stage == "predict":
            raise NotImplementedError("Predict stage is not supported")
        else:
            raise NotImplementedError(f"Stage {stage} is not supported")

    def _train_transform(self) -> list[Callable]:
        """Set the train augmentations"""

        if self.augmentations:
            for aug in self.augmentations:
                if isinstance(aug, MultiSampleTrait):
                    num_samples = aug.cropper.num_samples
                    if self.batch_size % num_samples != 0:
                        raise ValueError(
                            "Batch size must be divisible by `num_samples` per stack. "
                            f"Got batch size {self.batch_size} and "
                            f"number of samples {num_samples} for "
                            f"transform type {type(aug)}."
                        )
                    self.train_patches_per_stack = num_samples
        else:
            self.augmentations = []

        _logger.debug(f"Training augmentations: {self.augmentations}")
        return list(self.augmentations)

    def _fit_transform(self) -> tuple[Compose, Compose]:
        """(normalization -> maybe augmentation -> center crop)
        Deterministic center crop as the last step of training and validation."""
        # TODO: These have a fixed order for now... ()
        final_crop = [
            CenterSpatialCropd(
                keys=self.source_channel + self.target_channel,
                roi_size=(
                    self.z_window_size,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                ),
            )
        ]
        train_transform = Compose(
            self.normalizations + self._train_transform() + final_crop
        )
        val_transform = Compose(self.normalizations + final_crop)
        return train_transform, val_transform

    def _set_fit_global_state(self, num_positions: int) -> torch.Tensor:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions)

    def _setup_fit(self, dataset_settings: dict) -> None:
        """
        Setup the train and validation datasets.
        """
        train_transform, val_transform = self._fit_transform()
        dataset_settings["channels"]["target"] = self.target_channel
        # Load the plate
        plate = open_ome_zarr(self.data_path)
        # shuffle positions, randomness is handled globally
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = self._set_fit_global_state(len(positions))
        positions = list(positions[i] for i in shuffled_indices)
        num_train_fovs = int(len(positions) * self.split_ratio)

        shared_dict = Manager().dict()
        self.train_dataset = CachedDataset(
            shared_dict,
            positions[:num_train_fovs],
            transform=train_transform,
            **dataset_settings,
        )
        self.val_dataset = CachedDataset(
            shared_dict,
            positions[num_train_fovs:],
            transform=val_transform,
            **dataset_settings,
        )

    def train_dataloader(self) -> DataLoader:
        sampler = ShardedDistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.train_patches_per_stack,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            pin_memory=True,
            shuffle=False,
            timeout=self.timeout,
            collate_fn=_collate_samples,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = ShardedDistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            pin_memory=True,
            shuffle=False,
            timeout=self.timeout,
            sampler=sampler
        )
