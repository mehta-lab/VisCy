import logging
import os
import re
import tempfile
from glob import glob
from typing import Callable, Iterable, Literal, Sequence, TypedDict, Union

import numpy as np
import torch
import zarr
from imageio import imread
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    InvertibleTransform,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianSmoothd,
    RandWeightedCropd,
)
from torch.utils.data import DataLoader, Dataset


def _ensure_channel_list(str_or_seq: Union[str, Sequence[str]]):
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


class ChannelMap(TypedDict, total=False):
    source: Union[str, Sequence[str]]
    # optional
    target: Union[str, Sequence[str]]


class Sample(TypedDict, total=False):
    index: tuple[str, int, int]
    # optional
    source: torch.Tensor
    target: torch.Tensor
    labels: torch.Tensor


class NormalizeSampled(MapTransform, InvertibleTransform):
    """Dictionary transform to only normalize target (fluorescence) channel.

    :param Union[str, Iterable[str]] keys: keys to normalize
    :param dict[str, dict] norm_meta: Plate normalization metadata
        written in preprocessing
    """

    def __init__(
        self, keys: Union[str, Iterable[str]], norm_meta: dict[str, dict]
    ) -> None:
        if set(keys) > set(norm_meta.keys()):
            raise KeyError(f"{keys} is not a subset of {norm_meta.keys()}")
        super().__init__(keys, allow_missing_keys=False)
        self.norm_meta = norm_meta

    def _stat(self, key: str) -> dict:
        return self.norm_meta[key]["dataset_statistics"]

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] - self._stat(key)["median"]) / self._stat(key)["iqr"]
        return d

    def inverse(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] * self._stat(key)["iqr"]) + self._stat(key)["median"]


class SlidingWindowDataset(Dataset):
    """Torch dataset where each element is a window of
    (C, Z, Y, X) where C=2 (source and target) and Z is ``z_window_size``.

    :param list[Position] positions: FOVs to include in dataset
    :param ChannelMap channels: source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``
    :param int z_window_size: Z window size of the 2.5D U-Net, 1 for 2D
    :param Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] transform:
        a callable that transforms data, defaults to None
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] = None,
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
        for fov in self.positions:
            img_arr = fov["0"]
            ts = img_arr.frames
            zs = img_arr.slices - self.z_window_size + 1
            w += ts * zs
            self.window_keys.append(w)
            self.window_arrays.append(img_arr)
        self._max_window = w

    def _find_window(self, index: int) -> tuple[int, int]:
        """Look up window given index."""
        window_idx = sorted(self.window_keys + [index + 1]).index(index + 1)
        w = self.window_keys[window_idx]
        tz = index - self.window_keys[window_idx - 1] if window_idx > 0 else index
        return self.window_arrays[self.window_keys.index(w)], tz

    def _read_img_window(
        self, img: ImageArray, ch_idx: list[str], tz: int
    ) -> tuple[tuple[torch.Tensor], tuple[str, int, int]]:
        """Read image window as tensor.

        :param ImageArray img: NGFF image array
        :param list[int] channels: list of channel indices to read,
            output channel ordering will reflect the sequence
        :param int tz: window index within the FOV, counted Z-first
        :return tuple[torch.Tensor], tuple[str, int, int]:
            tuple of (C=1, Z, Y, X) image tensors,
            tuple of image name, time index, and Z index
        """
        zs = img.shape[-3] - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        data = img.oindex[
            slice(t, t + 1),
            [int(i) for i in ch_idx],
            slice(z, z + self.z_window_size),
        ]
        return torch.from_numpy(data).unbind(dim=1), (img.name, t, z)

    def __len__(self) -> int:
        return self._max_window

    def _stack_channels(
        self, sample_images: dict[str, torch.Tensor], key: str
    ) -> torch.Tensor:
        return torch.stack([sample_images[ch][0] for ch in self.channels[key]])

    def __getitem__(self, index: int) -> Sample:
        img, tz = self._find_window(index)
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
        if self.transform:
            sample_images = self.transform(sample_images)
        if isinstance(sample_images, list):
            sample_images = sample_images[0]
        if "weight" in sample_images:
            del sample_images["weight"]
        sample = {
            "index": sample_index,
            "source": self._stack_channels(sample_images, "source"),
        }
        if self.target_ch_idx is not None:
            sample["target"] = self._stack_channels(sample_images, "target")
        return sample

    def __del__(self):
        """Close the Zarr store when the dataset instance gets GC'ed."""
        self.positions[0].zgroup.store.close()


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
    :param Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] transform:
        a callable that transforms data, defaults to None
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] = None,
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
        e.g. 0.8 means a 80/20 split between training/validation
    :param int batch_size: batch size, defaults to 16
    :param int num_workers: number of data-loading workers, defaults to 8
    :param Literal["2.5D", "2D", "3D"] architecture: U-Net architecture,
        defaults to "2.5D"
    :param tuple[int, int] yx_patch_size: patch size in (Y, X),
        defaults to (256, 256)
    :param bool augment: whether to apply augmentation in training,
        defaults to True
    :param bool caching: whether to decompress all the images and cache the result,
        defaults to False
    :param str ground_truth_masks: path to the ground truth segmentation masks,
        defaults to None
    """

    def __init__(
        self,
        data_path: str,
        source_channel: Union[str, Sequence[str]],
        target_channel: Union[str, Sequence[str]],
        z_window_size: int,
        split_ratio: float,
        batch_size: int = 16,
        num_workers: int = 8,
        architecture: Literal["2.5D", "2D", "3D"] = "2.5D",
        yx_patch_size: tuple[int, int] = (256, 256),
        augment: bool = True,
        caching: bool = False,
        normalize_source: bool = False,
        ground_truth_masks: str = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_channel = _ensure_channel_list(source_channel)
        self.target_channel = _ensure_channel_list(target_channel)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_2d = True if architecture == "2.5D" else False
        self.z_window_size = z_window_size
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.augment = augment
        self.caching = caching
        self.normalize_source = normalize_source
        self.ground_truth_masks = ground_truth_masks
        self.tmp_zarr = None

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
        os.mkdir(self.trainer.logger.log_dir)
        file_handler = logging.FileHandler(
            os.path.join(self.trainer.logger.log_dir, "data.log")
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        # cache in temporary directory
        self.tmp_zarr = os.path.join(
            tempfile.gettempdir(), os.path.basename(self.data_path)
        )
        logger.info(f"Caching dataset at {self.tmp_zarr}.")
        tmp_store = zarr.NestedDirectoryStore(self.tmp_zarr)
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

    def _setup_eval(self, dataset_settings: dict) -> tuple[Plate, MapTransform]:
        dataset_settings["channels"]["target"] = self.target_channel
        data_path = self.tmp_zarr if self.tmp_zarr else self.data_path
        plate = open_ome_zarr(data_path, mode="r")
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # define training stage transforms
        norm_keys = self.target_channel
        if self.normalize_source:
            norm_keys += self.source_channel
        normalize_transform = NormalizeSampled(
            norm_keys,
            plate.zattrs["normalization"],
        )
        return plate, normalize_transform

    def _setup_fit(self, dataset_settings: dict):
        plate, normalize_transform = self._setup_eval(dataset_settings)
        fit_transform = self._fit_transform()
        train_transform = Compose(
            [normalize_transform] + self._train_transform() + fit_transform
        )
        val_transform = Compose([normalize_transform] + fit_transform)
        # shuffle positions, randomness is handled globally
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = torch.randperm(len(positions))
        positions = list(positions[i] for i in shuffled_indices)
        num_train_fovs = int(len(positions) * self.split_ratio)
        # train/val split
        self.train_dataset = SlidingWindowDataset(
            positions[:num_train_fovs],
            transform=train_transform,
            **dataset_settings,
        )
        self.val_dataset = SlidingWindowDataset(
            positions[num_train_fovs:], transform=val_transform, **dataset_settings
        )

    def _setup_test(self, dataset_settings):
        if self.batch_size != 1:
            logging.warning(f"Ignoring batch size {self.batch_size} in test stage.")
        plate, normalize_transform = self._setup_eval(dataset_settings)
        self.test_dataset = MaskTestDataset(
            [p for _, p in plate.positions()],
            transform=normalize_transform,
            ground_truth_masks=self.ground_truth_masks,
            **dataset_settings,
        )

    def _setup_predict(self, dataset_settings: dict):
        # track metadata for inverting transform
        set_track_meta(True)
        if self.caching:
            logging.warning("Ignoring caching config in 'predict' stage.")
        plate = open_ome_zarr(self.data_path, mode="r")
        predict_transform = (
            NormalizeSampled(
                self.source_channel,
                plate.zattrs["normalization"],
            )
            if self.normalize_source
            else None
        )
        self.predict_dataset = SlidingWindowDataset(
            [p for _, p in plate.positions()],
            transform=predict_transform,
            **dataset_settings,
        )

    def on_before_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        if self.trainer.predicting or isinstance(batch, torch.Tensor):
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
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
        return [
            CenterSpatialCropd(
                keys=self.source_channel + self.target_channel,
                roi_size=(
                    -1,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                ),
            )
        ]

    def _train_transform(self) -> list[Callable]:
        transforms = [
            RandWeightedCropd(
                keys=self.source_channel + self.target_channel,
                w_key="weight",
                spatial_size=(-1, self.yx_patch_size[0] * 2, self.yx_patch_size[1] * 2),
                num_samples=1,
            )
        ]
        if self.augment:
            transforms.extend(
                [
                    RandAffined(
                        keys=self.source_channel + self.target_channel,
                        prob=0.5,
                        rotate_range=(np.pi, 0, 0),
                        shear_range=(0, (0.05), (0.05)),
                        scale_range=(0, 0.3, 0.3),
                    ),
                    RandAdjustContrastd(
                        keys=self.source_channel, prob=0.3, gamma=(0.75, 1.5)
                    ),
                    RandGaussianSmoothd(
                        keys=self.source_channel,
                        prob=0.3,
                        sigma_x=(0.05, 0.25),
                        sigma_y=(0.05, 0.25),
                        sigma_z=((0.05, 0.25)),
                    ),
                ]
            )
        return transforms
