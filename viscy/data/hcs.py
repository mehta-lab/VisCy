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
        for img_path in glob(os.path.join(ground_truth_masks, "*cp_masks.png")):
            img_name = os.path.basename(img_path)
            position_name = _search_int_in_str(r"(?<=_p)\d{3}", img_name)
            # TODO: specify time index in the file name
            t_idx = 0
            # TODO: record channel name
            # channel_name = re.search(r"^.+(?=_p\d{3})", img_name).group()
            z_idx = _search_int_in_str(r"(?<=_z)\d+", img_name)
            self.masks[(int(position_name), int(t_idx), int(z_idx))] = img_path
        logging.info(str(self.masks))

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
    :param Union[str, Sequence[str]] source_channel: name(s) of the source channel,
        e.g. ``'Phase'``
    :param Union[str, Sequence[str]] target_channel: name(s) of the target channel,
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
    :param Optional[Path] ground_truth_masks: path to the ground truth masks,
        used in the test stage to compute segmentation metrics,
        defaults to None
    """

    def __init__(
        self,
        data_path: str,
        source_channel: Union[str, Sequence[str]],
        target_channel: Union[str, Sequence[str]],
        z_window_size: int,
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        architecture: Literal["2D", "UNeXt2", "2.5D", "3D", "fcmae"] = "2.5D",
        yx_patch_size: tuple[int, int] = (256, 256),
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        caching: bool = False,
        ground_truth_masks: Optional[Path] = None,
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
            logging.warning(f"Ignoring batch size {self.batch_size} in test stage.")

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
            logging.warning("Ignoring caching config in 'predict' stage.")
        dataset: Union[Plate, Position] = open_ome_zarr(self.data_path, mode="r")
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
        logging.debug(f"Training augmentations: {self.augmentations}")
        return list(self.augmentations)
