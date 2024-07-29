import logging
import math
import os
import re
import tempfile
from glob import glob
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

# import pytorch_lightning as pl
from monai.transforms import MapTransform
from collections import defaultdict
from torch.utils.data import Sampler, Dataset, DataLoader
from torch.multiprocessing import Manager
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import random
import numpy as np
import torch
import zarr
from imageio import imread
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr

# from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Rand3DElasticd,
    RandGaussianSharpend,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from viscy.data.typing import ChannelMap, HCSStackIndex, NormMeta, Sample
import random
from iohub import open_ome_zarr
import pandas as pd
import warnings
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

# from viscy.data.typing import Optional
from pathlib import Path

_logger = logging.getLogger("lightning.pytorch")


def _ensure_channel_list(str_or_seq: str | Sequence[str]) -> list[str]:
    """
    Ensure channel argument is a list of strings.

    :param Union[str, Sequence[str]] str_or_seq: channel name or list of channel names
    :return list[str]: list of channel names
    """
    if isinstance(str_or_seq, str):
        return [str_or_seq]
    try:
        return list(str_or_seq)
    except TypeError:
        raise TypeError(
            "Channel argument must be a string or sequence of strings. "
            f"Got {str_or_seq}."
        )


def _search_int_in_str(pattern: str, file_name: str) -> str:
    """Search image indices in a file name with regex patterns and strip leading zeros.
    E.g. ``'001'`` -> ``1``"""
    match = re.search(pattern, file_name)
    if match:
        return match.group()
    else:
        raise ValueError(f"Cannot find pattern {pattern} in {file_name}.")


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


def _read_norm_meta(fov: Position) -> NormMeta | None:
    """
    Read normalization metadata from the FOV.
    Convert to float32 tensors to avoid automatic casting to float64.
    """
    norm_meta = fov.zattrs.get("normalization", None)
    if norm_meta is None:
        return None
    for channel, channel_values in norm_meta.items():
        for level, level_values in channel_values.items():
            for stat, value in level_values.items():
                norm_meta[channel][level][stat] = torch.tensor(
                    value, dtype=torch.float32
                )
    return norm_meta


class SlidingWindowDataset(Dataset):
    """Torch dataset where each element is a window of
    (C, Z, Y, X) where C=2 (source and target) and Z is ``z_window_size``.

    :param list[Position] positions: FOVs to include in dataset
    :param ChannelMap channels: source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``
    :param int z_window_size: Z window size of the 2.5D U-Net, 1 for 2D
    :param Callable[[dict[str, Tensor]], dict[str, Tensor]] | None transform:
        a callable that transforms data, defaults to None
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.positions = positions
        self.channels = {k: _ensure_channel_list(v) for k, v in channels.items()}
        self.source_ch_idx = [
            positions[0].get_channel_index(c) for c in channels["source"]
        ]
        self.target_ch_idx = (
            [positions[0].get_channel_index(c) for c in channels["target"]]
            if "target" in channels
            else None
        )
        self.z_window_size = z_window_size
        self.transform = transform
        self._get_windows()

    def _get_windows(self) -> None:
        """Count the sliding windows along T and Z,
        and build an index-to-window LUT."""
        w = 0
        self.window_keys = []
        self.window_arrays = []
        self.window_norm_meta: list[NormMeta | None] = []
        for fov in self.positions:
            img_arr: ImageArray = fov["0"]
            ts = img_arr.frames
            zs = img_arr.slices - self.z_window_size + 1
            w += ts * zs
            self.window_keys.append(w)
            self.window_arrays.append(img_arr)
            self.window_norm_meta.append(_read_norm_meta(fov))
        self._max_window = w

    def _find_window(self, index: int) -> tuple[ImageArray, int, NormMeta | None]:
        """Look up window given index."""
        window_idx = sorted(self.window_keys + [index + 1]).index(index + 1)
        w = self.window_keys[window_idx]
        tz = index - self.window_keys[window_idx - 1] if window_idx > 0 else index
        norm_meta = self.window_norm_meta[self.window_keys.index(w)]
        return (self.window_arrays[self.window_keys.index(w)], tz, norm_meta)

    def _read_img_window(
        self, img: ImageArray, ch_idx: list[int], tz: int
    ) -> tuple[list[Tensor], HCSStackIndex]:
        """Read image window as tensor.

        :param ImageArray img: NGFF image array
        :param list[int] ch_idx: list of channel indices to read,
            output channel ordering will reflect the sequence
        :param int tz: window index within the FOV, counted Z-first
        :return list[Tensor], HCSStackIndex:
            list of (C=1, Z, Y, X) image tensors,
            tuple of image name, time index, and Z index
        """
        zs = img.shape[-3] - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        data = img.oindex[
            slice(t, t + 1),
            [int(i) for i in ch_idx],
            slice(z, z + self.z_window_size),
        ].astype(np.float32)
        return torch.from_numpy(data).unbind(dim=1), (img.name, t, z)

    def __len__(self) -> int:
        return self._max_window

    def _stack_channels(
        self,
        sample_images: list[dict[str, Tensor]] | dict[str, Tensor],
        key: str,
    ) -> Tensor | list[Tensor]:
        """Stack single-channel images into a multi-channel tensor."""
        if not isinstance(sample_images, list):
            return torch.stack([sample_images[ch][0] for ch in self.channels[key]])
        # training time
        return [
            torch.stack([im[ch][0] for ch in self.channels[key]])
            for im in sample_images
        ]

    def __getitem__(self, index: int) -> Sample:
        img, tz, norm_meta = self._find_window(index)
        ch_names = self.channels["source"].copy()
        ch_idx = self.source_ch_idx.copy()
        if self.target_ch_idx is not None:
            ch_names.extend(self.channels["target"])
            ch_idx.extend(self.target_ch_idx)
        images, sample_index = self._read_img_window(img, ch_idx, tz)
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
            "index": sample_index,
            "source": self._stack_channels(sample_images, "source"),
            "norm_meta": norm_meta,
        }
        if self.target_ch_idx is not None:
            sample["target"] = self._stack_channels(sample_images, "target")
        return sample


class MaskTestDataset(SlidingWindowDataset):
    """Torch dataset where each element is a window of
    (C, Z, Y, X) where C=2 (source and target) and Z is ``z_window_size``.
    This a testing stage version of :py:class:`viscy.light.data.SlidingWindowDataset`,
    and can only be used with batch size 1 for efficiency (no padding for collation),
    since the mask is not available for each stack.

    :param list[Position] positions: FOVs to include in dataset
    :param ChannelMap channels: source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``
    :param int z_window_size: Z window size of the 2.5D U-Net, 1 for 2D
    :param Callable[[dict[str, Tensor]], dict[str, Tensor]] transform:
        a callable that transforms data, defaults to None
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        ground_truth_masks: str = None,
    ) -> None:
        super().__init__(positions, channels, z_window_size, transform)
        self.masks = {}
        for img_path in Path(ground_truth_masks).glob("*cp_masks.png"):
            img_name = img_path.name
            position_name = _search_int_in_str(r"(?<=_p)\d{3}", img_name)
            # TODO: specify time index in the file name
            t_idx = 0
            # TODO: record channel name
            # channel_name = re.search(r"^.+(?=_p\d{3})", img_name).group()
            z_idx = _search_int_in_str(r"(?<=_z)\d+", img_name)
            self.masks[(int(position_name), int(t_idx), int(z_idx))] = img_path
        _logger.info(str(self.masks))

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        img_name, t_idx, z_idx = sample["index"]
        position_name = int(img_name.split("/")[-2])
        key = (position_name, int(t_idx), int(z_idx) + self.z_window_size // 2)
        if img_path := self.masks.get(key):
            sample["labels"] = torch.from_numpy(imread(img_path).astype(np.int16))
        return sample


class HCSDataModule(LightningDataModule):
    """Lightning data module for a preprocessed HCS NGFF Store.

    :param str data_path: path to the data store
    :param str | Sequence[str] source_channel: name(s) of the source channel,
        e.g. ``'Phase'``
    :param str | Sequence[str] target_channel: name(s) of the target channel,
        e.g. ``['Nuclei', 'Membrane']``
    :param int z_window_size: Z window size of the 2.5D U-Net, 1 for 2D
    :param float split_ratio: split ratio of the training subset in the fit stage,
        e.g. 0.8 means a 80/20 split between training/validation,
        by default 0.8
    :param int batch_size: batch size, defaults to 16
    :param int num_workers: number of data-loading workers, defaults to 8
    :param Literal["2D", "UNeXt2", "2.5D", "3D"] architecture: U-Net architecture,
        defaults to "2.5D"
    :param tuple[int, int] yx_patch_size: patch size in (Y, X),
        defaults to (256, 256)
    :param list[MapTransform] normalizations: MONAI dictionary transforms
        applied to selected channels, defaults to [] (no normalization)
    :param list[MapTransform] augmentations: MONAI dictionary transforms
        applied to the training set, defaults to [] (no augmentation)
    :param bool caching: whether to decompress all the images and cache the result,
        will store in ``/tmp/$SLURM_JOB_ID/`` if available,
        defaults to False
    :param Path | None ground_truth_masks: path to the ground truth masks,
        used in the test stage to compute segmentation metrics,
        defaults to None
    """

    def __init__(
        self,
        data_path: str,
        source_channel: str | Sequence[str],
        target_channel: str | Sequence[str],
        z_window_size: int,
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        architecture: Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae"] = "2.5D",
        yx_patch_size: tuple[int, int] = (256, 256),
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        caching: bool = False,
        ground_truth_masks: Path | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.source_channel = _ensure_channel_list(source_channel)
        self.target_channel = _ensure_channel_list(target_channel)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_2d = False if architecture in ["UNeXt2", "3D", "fcmae"] else True
        self.z_window_size = z_window_size
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.normalizations = normalizations
        self.augmentations = augmentations
        self.caching = caching
        self.ground_truth_masks = ground_truth_masks
        self.prepare_data_per_node = True

    @property
    def cache_path(self):
        return Path(
            tempfile.gettempdir(),
            os.getenv("SLURM_JOB_ID", "viscy_cache"),
            self.data_path.name,
        )

    def _data_log_path(self) -> Path:
        log_dir = Path.cwd()
        if self.trainer:
            if self.trainer.logger:
                if self.trainer.logger.log_dir:
                    log_dir = Path(self.trainer.logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "data.log"

    def prepare_data(self):
        if not self.caching:
            return
        # setup logger
        logger = logging.getLogger("viscy_data")
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
        with open_ome_zarr(self.data_path, mode="r") as lazy_plate:
            _, skipped, _ = zarr.copy(
                lazy_plate.zgroup,
                zarr.open(tmp_store, mode="a"),
                name="/",
                log=logger.debug,
                if_exists="skip_initialized",
                compressor=None,
            )
        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} items when caching. Check debug log for details."
            )

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        channels = {"source": self.source_channel}
        dataset_settings = dict(channels=channels, z_window_size=self.z_window_size)
        if stage in ("fit", "validate"):
            self._setup_fit(dataset_settings)
        elif stage == "test":
            self._setup_test(dataset_settings)
        elif stage == "predict":
            self._setup_predict(dataset_settings)
        else:
            raise NotImplementedError(f"{stage} stage")

    def _setup_fit(self, dataset_settings: dict):
        """Set up the training and validation datasets."""
        # Setup the transformations
        # TODO: These have a fixed order for now... (normalization->augmentation->fit_transform)
        fit_transform = self._fit_transform()
        train_transform = Compose(
            self.normalizations + self._train_transform() + fit_transform
        )
        val_transform = Compose(self.normalizations + fit_transform)

        dataset_settings["channels"]["target"] = self.target_channel
        data_path = self.cache_path if self.caching else self.data_path
        plate = open_ome_zarr(data_path, mode="r")

        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = torch.randperm(len(positions))
        positions = list(positions[i] for i in shuffled_indices)
        num_train_fovs = int(len(positions) * self.split_ratio)
        # training set needs to sample more Z range for augmentation
        train_dataset_settings = dataset_settings.copy()
        z_scale_low, z_scale_high = self.train_z_scale_range
        if z_scale_high <= 0.0:
            expanded_z = self.z_window_size
        else:
            expanded_z = math.ceil(self.z_window_size * (1 + z_scale_high))
            expanded_z -= expanded_z % 2
        train_dataset_settings["z_window_size"] = expanded_z
        # train/val split
        self.train_dataset = SlidingWindowDataset(
            positions[:num_train_fovs],
            transform=train_transform,
            **train_dataset_settings,
        )
        self.val_dataset = SlidingWindowDataset(
            positions[num_train_fovs:],
            transform=val_transform,
            **dataset_settings,
        )

    def _setup_test(self, dataset_settings: dict):
        """Set up the test stage."""
        if self.batch_size != 1:
            _logger.warning(f"Ignoring batch size {self.batch_size} in test stage.")

        dataset_settings["channels"]["target"] = self.target_channel
        data_path = self.cache_path if self.caching else self.data_path
        plate = open_ome_zarr(data_path, mode="r")
        test_transform = Compose(self.normalizations)
        if self.ground_truth_masks:
            self.test_dataset = MaskTestDataset(
                [p for _, p in plate.positions()],
                transform=test_transform,
                ground_truth_masks=self.ground_truth_masks,
                **dataset_settings,
            )
        else:
            self.test_dataset = SlidingWindowDataset(
                [p for _, p in plate.positions()],
                transform=test_transform,
                **dataset_settings,
            )

    def _setup_predict(
        self,
        dataset_settings: dict,
    ):
        """Set up the predict stage."""
        # track metadata for inverting transform
        set_track_meta(True)
        if self.caching:
            _logger.warning("Ignoring caching config in 'predict' stage.")
        dataset: Plate | Position = open_ome_zarr(self.data_path, mode="r")
        if isinstance(dataset, Position):
            try:
                plate_path = self.data_path.parent.parent.parent
                fov_name = self.data_path.relative_to(plate_path).as_posix()
                plate = open_ome_zarr(plate_path)
            except Exception:
                raise FileNotFoundError(
                    "Parent HCS store not found for single FOV input."
                )
            positions = [plate[fov_name]]
        elif isinstance(dataset, Plate):
            positions = [p for _, p in dataset.positions()]
        predict_transform = Compose(self.normalizations)
        self.predict_dataset = SlidingWindowDataset(
            positions=positions,
            transform=predict_transform,
            **dataset_settings,
        )

    def on_before_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Removes redundant Z slices if the target is 2D to save VRAM."""
        predicting = False
        if self.trainer:
            if self.trainer.predicting:
                predicting = True
        if predicting or isinstance(batch, Tensor):
            # skipping example input array
            return batch
        if self.target_2d:
            # slice the center during training or testing
            z_index = self.z_window_size // 2
            batch["target"] = batch["target"][:, :, slice(z_index, z_index + 1)]
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.train_patches_per_stack,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=bool(self.num_workers),
            prefetch_factor=4 if self.num_workers else None,
            collate_fn=_collate_samples,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=4 if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _fit_transform(self):
        """Deterministic center crop as the last step of training and validation."""
        return [
            CenterSpatialCropd(
                keys=self.source_channel + self.target_channel,
                roi_size=(
                    self.z_window_size,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                ),
            )
        ]

    def _train_transform(self) -> list[Callable]:
        """Setup training augmentations: check input values,
        and parse the number of Z slices and patches to sample per stack."""
        self.train_patches_per_stack = 1
        z_scale_range = None
        if self.augmentations:
            for aug in self.augmentations:
                if isinstance(aug, RandAffined):
                    if z_scale_range is not None:
                        raise ValueError(
                            "Only one RandAffined augmentation is allowed."
                        )
                    z_scale_range = aug.rand_affine.rand_affine_grid.scale_range[0]
                if isinstance(aug, MultiSampleTrait):
                    # e.g. RandWeightedCropd.cropper.num_samples
                    # this trait does not have any concrete interface
                    # so this attribute may not be the same for other transforms
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
        if z_scale_range is not None:
            if isinstance(z_scale_range, float):
                z_scale_range = (-z_scale_range, z_scale_range)
            if z_scale_range[0] > 0 or z_scale_range[1] < 0:
                raise ValueError(f"Invalid scaling range: {z_scale_range}")
            self.train_z_scale_range = z_scale_range
        else:
            self.train_z_scale_range = (0.0, 0.0)
        _logger.debug(f"Training augmentations: {self.augmentations}")
        return list(self.augmentations)


# dataloader for organelle phenotyping
class ContrastiveDataset(Dataset):
    def __init__(
        self,
        base_path,
        channels,
        x,
        y,
        timesteps_csv_path,
        channel_names,
        transform=None,
        z_range=None,
    ):
        self.base_path = base_path
        self.channels = channels
        self.x = x
        self.y = y
        self.z_range = z_range
        self.channel_names = channel_names
        self.transform = get_transforms()
        self.ds = self.open_zarr_store(self.base_path)
        self.positions = list(self.ds.positions())
        self.timesteps_df = pd.read_csv(timesteps_csv_path)
        self.channel_indices = [
            self.ds.channel_names.index(channel) for channel in self.channel_names
        ]
        print("channel indices!")
        print(self.channel_indices)
        print(f"Initialized dataset with {len(self.positions)} positions.")

        #self.statistics = self.compute_statistics()
        #self.print_channel_statistics()

    def print_statistics(self, data, data_label):
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        print(f"{data_label} Statistics:")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")
        print(f"  Min: {min_val}")
        print(f"  Max: {max_val}")

    def compute_statistics(self):
        stats = {
            channel: {"mean": 0, "sum_sq_diff": 0, "min": np.inf, "max": -np.inf}
            for channel in self.channel_names
        }
        count = 0
        total_elements = 0

        for idx in range(len(self.positions)):
            position_path = self.positions[idx][0]
            data = self.load_data(position_path)
            for i, channel in enumerate(self.channel_names):
                channel_data = data[i]
                mean = np.mean(channel_data)
                stats[channel]["mean"] += mean
                stats[channel]["min"] = min(stats[channel]["min"], np.min(channel_data))
                stats[channel]["max"] = max(stats[channel]["max"], np.max(channel_data))
                stats[channel]["sum_sq_diff"] += np.sum((channel_data - mean) ** 2)
            count += 1
            total_elements += np.prod(channel_data.shape)

        for channel in self.channel_names:
            stats[channel]["mean"] /= count
            stats[channel]["std"] = np.sqrt(
                stats[channel]["sum_sq_diff"] / total_elements
            )
            del stats[channel]["sum_sq_diff"]

        print("done!")
        return stats

    def print_channel_statistics(self):
        for channel, channel_stats in self.statistics.items():
            print(f"Channel: {channel}")
            print(f"  Mean: {channel_stats['mean']}")
            print(f"  Std: {channel_stats['std']}")
            print(f"  Min: {channel_stats['min']}")
            print(f"  Max: {channel_stats['max']}")

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        # print(f"Opening Zarr store at {path} with layout '{layout}' and mode '{mode}'")
        return open_ome_zarr(path, layout=layout, mode=mode)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        anchor_position_path = self.positions[idx][0]
        anchor_data_load = self.load_data(anchor_position_path)
        anchor_data = self.normalize_data(anchor_data_load)
        #self.print_statistics(anchor_data, "Anchor Data")

        positive_data = self.apply_channel_transforms(anchor_data_load)
        positive_data = self.normalize_data(positive_data)
        #self.print_statistics(positive_data, "Positive Data")

        # if self.transform:
        #     print("Positive transformation applied")

        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, self.__len__() - 1)
        negative_position_path = self.positions[negative_idx][0]
        negative_data = self.load_data(negative_position_path)
    
        negative_data = self.apply_channel_transforms(negative_data)
        negative_data = self.normalize_data(negative_data)

        # if self.transform:
        #     print("Negative transformation applied")

        # print("shapes of tensors")
        # print(torch.tensor(anchor_data).shape)
        # print(torch.tensor(positive_data).shape)
        # print(torch.tensor(negative_data).shape)

        #self.print_statistics(negative_data, "Negative Data")

        return (
            torch.tensor(anchor_data, dtype=torch.float32),
            torch.tensor(positive_data, dtype=torch.float32),
            torch.tensor(negative_data, dtype=torch.float32),
        )

    def load_data(self, position_path):
        position = self.ds[position_path]
        # print(f"Loading data from position: {position_path}")

        zarr_array = position["0"][:]
        # print("Shape before:", zarr_array.shape)
        data = self.restructure_data(zarr_array, position_path)
        data = data[self.channel_indices, self.z_range[0] : self.z_range[1], :, :]

        # print("shape after!")
        # print(data.shape)
        return data

    def restructure_data(self, data, position_path):
        # Extract row, column, fov, and cell_id from position_path
        parts = position_path.split("/")
        row = parts[0]
        column = parts[1]
        fov_cell = parts[2]

        fov = int(fov_cell.split("fov")[1].split("cell")[0])
        cell_id = int(fov_cell.split("cell")[1])

        extracted_combined = f"{row}/{column}/fov{fov}cell{cell_id}"

        matched_rows = self.timesteps_df[
            self.timesteps_df.apply(
                lambda x: f"{x['Row']}/{x['Column']}/fov{x['FOV']}cell{x['Cell ID']}",
                axis=1,
            )
            == extracted_combined
        ]

        if matched_rows.empty:
            raise ValueError(
                f"No matching entry found for position path: {position_path}"
            )

        start_time = matched_rows["Start Time"].values[0]
        end_time = matched_rows["End Time"].values[0]

        random_timestep = np.random.randint(start_time, end_time)

        reshaped_data = data[random_timestep]
        return reshaped_data

    def normalize_data(self, data):
        normalized_data = np.empty_like(data)
        for i in range(data.shape[0]):  # iterate over each channel
            channel_data = data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            normalized_data[i] = (channel_data - mean) / (std + 1e-6)
        return normalized_data

    def apply_channel_transforms(self, data):
        transformed_data = np.empty_like(data)
        for i, channel_name in enumerate(self.channel_names):
            channel_data = data[i]
            transform = self.transform[channel_name]
            transformed_data[i] = transform({"image": channel_data})["image"]
            # print(f"transformed {channel_name}")
        return transformed_data


def get_transforms():
    # phase_transforms = Compose(
    #     [
    #         RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.95, 1.05)),
    #         RandAffined(
    #             keys=["image"],
    #             prob=0.5,
    #             rotate_range=(0.05, 0.05),  # Reduced rotation range
    #             shear_range=(0.05, 0.05),   # Reduced shear range
    #             scale_range=(0.05, 0.05),   # Reduced scale range
    #         ),
    #         RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.01),  # Appropriate std
    #         RandGaussianSmoothd(
    #             keys=["image"],
    #             prob=0.5,
    #             sigma_x=(0.02, 0.05),  # Adjusted sigma values
    #             sigma_y=(0.02, 0.05),
    #             sigma_z=(0.02, 0.05),
    #         ),
    #         RandScaleIntensityd(keys=["image"], factors=(0.98, 1.02), prob=0.5),  # Subtle scaling factors
    #     ]
    # )

    rfp_transforms = Compose(
        [
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.8, 1.2)),
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0.05, 0.05),  
                shear_range=(0.05, 0.05),   
                scale_range=(0.05, 0.05),   
            ),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.5),  
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.5,
                sigma_x=(0.1, 0.2),  
                sigma_y=(0.1, 0.2),
                sigma_z=(0.1, 0.2),
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.9, 1.1), prob=0.5),  
        ]
    )

    return {
        "RFP": rfp_transforms
    }
    
    # return {
    #     "Phase3D": phase_transforms
    # }


class ContrastiveDataModule(LightningDataModule):
    def __init__(
        self,
        base_path: str,
        channels: int,
        x: int,
        y: int,
        timesteps_csv_path: str,
        channel_names: list,
        transform=None,
        predict_base_path: str = None,
        train_split_ratio: float = 0.70,
        val_split_ratio: float = 0.20,
        batch_size: int = 4,
        num_workers: int = 15,  # for analysis purposes reduced to 1
        z_range: tuple[int, int] = None,
        analysis: bool = False,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.channels = channels
        self.x = x
        self.y = y
        self.timesteps_csv_path = timesteps_csv_path
        self.channel_names = channel_names
        self.transform = transform
        self.predict_base_path = Path(predict_base_path) if predict_base_path else None
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.z_range = z_range
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.analysis = analysis
        self.position_to_timesteps = None

    def setup(self, stage: str = None):
        if stage == "fit":
            dataset = ContrastiveDataset(
                self.base_path,
                self.channels,
                self.x,
                self.y,
                self.timesteps_csv_path,
                channel_names=self.channel_names,
                transform=self.transform,
                z_range=self.z_range,
            )

            train_size = int(len(dataset) * self.train_split_ratio)
            val_size = int(len(dataset) * self.val_split_ratio)
            test_size = len(dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = (
                torch.utils.data.random_split(
                    dataset, [train_size, val_size, test_size]
                )
            )

        # # setup prediction dataset 
        # if stage == "predict" and self.predict_base_path and not self.analysis:
        #     print("setting up!")
        #     self.predict_dataset = PredictDataset(
        #         self.predict_base_path,
        #         self.channels,
        #         self.x,
        #         self.y,
        #         timesteps_csv_path=self.timesteps_csv_path,
        #         channel_names=self.channel_names,
        #         z_range=self.z_range,
        #     )

        # both should be on for extracting embeddings  
        if stage == "predict" and self.analysis == True:
            print("doing analysis set up!")
            self.predict_dataset = PredictDataset(
                self.predict_base_path,
                self.channels,
                self.x,
                self.y,
                timesteps_csv_path=self.timesteps_csv_path,
                channel_names=self.channel_names,
                z_range=self.z_range,
                analysis=True,
            )
            self.position_to_timesteps = self.predict_dataset.position_to_timesteps
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        print("running predict DataLoader!")
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset not set up. Call setup(stage='predict') first."
            )
        
        # Check if distributed training is initialized
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        else:
            rank = 0
            num_replicas = 1

        sampler = DistributedSampler(
            self.predict_dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False
        )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,  # Use DistributedSampler
            num_workers=self.num_workers,
            collate_fn=ContrastiveDataModule.custom_collate_fn,  # Use custom collate function
            prefetch_factor=2,
            persistent_workers=True
        )
    
    @staticmethod
    def custom_collate_fn(batch):
        # Sort the batch by position_path and timestep
        batch.sort(key=lambda x: (x[1], x[2]))
        data, positions, timesteps = zip(*batch)
        return torch.stack(data), positions, timesteps

class PredictDataset(Dataset):
    def __init__(
        self,
        base_path,
        channels,
        x,
        y,
        timesteps_csv_path,
        channel_names,
        z_range=None,
        analysis=False,
        embeddings_dict=None,  # Shared dict to store embeddings
        order_dict=None,  # Shared dict to store order
    ):
        self.base_path = base_path
        self.channels = channels
        self.x = x
        self.y = y
        self.z_range = z_range
        self.channel_names = channel_names
        self.ds = self.open_zarr_store(self.base_path)
        self.timesteps_csv_path = timesteps_csv_path
        self.timesteps_df = pd.read_csv(timesteps_csv_path)
        self.positions = list(self.ds.positions()) if not analysis else self.filter_positions_from_csv()
        self.channel_indices = [self.ds.channel_names.index(channel) for channel in self.channel_names]
        self.analysis = analysis
        self.embeddings_dict = embeddings_dict if embeddings_dict is not None else Manager().dict()
        self.order_dict = order_dict if order_dict is not None else Manager().dict()
        print("channel indices!")
        print(self.channel_indices)
        print(f"Initialized predict dataset with {len(self.positions)} positions.")

        # Vectorized creation of position_to_timesteps and position_timestep_pairs
        self.timesteps_df["Position"] = self.timesteps_df.apply(
            lambda x: f"{x['Row']}/{x['Column']}/fov{x['FOV']}cell{x['Cell ID']}",
            axis=1,
        )

        # Filter timesteps_df to only include positions that are in self.positions
        filtered_timesteps_df = self.timesteps_df[
            self.timesteps_df["Position"].isin(self.positions)
        ]

        # Group by position and collect timesteps
        position_groups = filtered_timesteps_df.groupby("Position")

        self.position_to_timesteps = {
            position: group["Timestep"].tolist()
            for position, group in position_groups
        }

        print("done position_to_timesteps!")

        self.position_timestep_pairs = [
            (position, timestep)
            for position, timesteps in self.position_to_timesteps.items()
            for timestep in timesteps
        ]

        print("done position_timestep_pairs!")

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        return open_ome_zarr(path, layout=layout, mode=mode)

    def filter_positions_from_csv(self):
        unique_positions = self.timesteps_df[['Row', 'Column', 'FOV', 'Cell ID']].drop_duplicates()
        valid_positions = [
            f"{row['Row']}/{row['Column']}/fov{row['FOV']}cell{row['Cell ID']}"
            for _, row in unique_positions.iterrows()
        ]
        return valid_positions

    def __len__(self):
        return len(self.position_timestep_pairs)

    def __getitem__(self, idx):
        position_path, timestep = self.position_timestep_pairs[idx]
        data = self.load_data(position_path, timestep)
        data = self.normalize_data(data)
        # Combine position path and timestep into a single identifier
        combined_pos_info = f"{position_path}/t{timestep}"
        return torch.tensor(data, dtype=torch.float32), combined_pos_info, timestep

    # double check printing order
    def load_data(self, position_path, timestep=None):
        position = self.ds[position_path]
        zarr_array = position["0"][:]
        data = zarr_array[timestep, self.channel_indices, self.z_range[0]:self.z_range[1], :, :]
        return data

    def normalize_data(self, data):
        normalized_data = np.empty_like(data)
        for i in range(data.shape[0]):  # iterate over each channel
            channel_data = data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            normalized_data[i] = (channel_data - mean) / (std + 1e-6)
        return normalized_data
    
