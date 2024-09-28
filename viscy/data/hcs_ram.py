import logging
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
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

_logger = logging.getLogger("lightning.pytorch")

# TODO: cache the norm metadata when caching the dataset


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



class CachedDataset(Dataset):
    """
    A dataset that caches the data in RAM.
    It relies on the `__getitem__` method to load the data on the 1st epoch.
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        transform: DictTransform | None = None,
    ):
        super().__init__()
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
        self._position_mapping()
        self.cache_dict = {}

    def _position_mapping(self) -> None:
        self.position_keys = []
        self.norm_meta_dict = {}

        for pos in self.positions:
            self.position_keys.append(pos.data.name)
            self.norm_meta_dict[str(pos.data.name)] = _read_norm_meta(pos)

    def _cache_dataset(self, index: int, channel_index: list[int], t: int = 0) -> None:
        # Add the position to the cached_dict
        # TODO: hardcoding to t=0
        self.cache_dict[str(self.position_keys[index])] = torch.from_numpy(
            self.positions[index]
            .data.oindex[slice(t, t + 1), channel_index, :]
            .astype(np.float32)
        )

    def _get_weight_map(self, position: Position) -> Tensor:
        # Get the weight map from the position for the MONAI weightedcrop transform
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Sample:
        ch_names = self.channels["source"].copy()
        ch_idx = self.source_ch_idx.copy()
        if self.target_ch_idx is not None:
            ch_names.extend(self.channels["target"])
            ch_idx.extend(self.target_ch_idx)

        # Check if the sample is in the cache else add it
        # Split the tensor into the channels
        sample_id = self.position_keys[index]
        if sample_id not in self.cache_dict:
            logging.info(f"Adding {sample_id} to cache")
            self._cache_dataset(index, channel_index=ch_idx)

        # Get the sample from the cache
        images = self.cache_dict[sample_id].unbind(dim=1)
        norm_meta = self.norm_meta_dict[str(sample_id)]

        sample_images = {k: v for k, v in zip(ch_names, images)}

        if self.target_ch_idx is not None:
            # FIXME: this uses the first target channel as weight for performance
            # since adding a reference to a tensor does not copy
            # maybe write a weight map in preprocessing to use more information?
            sample_images["weight"] = sample_images[self.channels["target"][0]]
        if norm_meta is not None:
            sample_images["norm_meta"] = norm_meta
        if self.transform:
            sample_images = self.transform(sample_images)
        if "weight" in sample_images:
            del sample_images["weight"]
        sample = {
            "index": sample_id,
            "source": _stack_channels(sample_images, self.channels, "source"),
            "norm_meta": norm_meta,
        }
        if self.target_ch_idx is not None:
            sample["target"] = _stack_channels(sample_images, self.channels, "target")
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

        self.train_dataset = CachedDataset(
            positions[:num_train_fovs],
            transform=train_transform,
            **dataset_settings,
        )
        self.val_dataset = CachedDataset(
            positions[num_train_fovs:],
            transform=val_transform,
            **dataset_settings,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.train_patches_per_stack,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            shuffle=True,
            timeout=self.timeout
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            shuffle=False,
            timeout=self.timeout
        )
