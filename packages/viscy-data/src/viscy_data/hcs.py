"""Lightning data module for a preprocessed HCS NGFF Store."""

import logging
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from imageio import imread
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    MultiSampleTrait,
    RandAffined,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy_data._typing import ChannelMap, DictTransform, HCSStackIndex, NormMeta, Sample
from viscy_data._utils import (
    _collate_samples,
    _ensure_channel_list,
    _read_norm_meta,
    _search_int_in_str,
)

_logger = logging.getLogger("lightning.pytorch")


class SlidingWindowDataset(Dataset):
    """Sliding window dataset over HCS NGFF positions.

    Each element is a window of (C, Z, Y, X) where C=2 (source and target)
    and Z is ``z_window_size``.

    Parameters
    ----------
    positions : list[Position]
        FOVs to include in dataset.
    channels : ChannelMap
        Source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``.
    z_window_size : int
        Z window size of the 2.5D U-Net, 1 for 2D.
    array_key : str
        Name of the image arrays (multiscales level), by default "0".
    transform : DictTransform | None
        A callable that transforms data, defaults to None.
    load_normalization_metadata : bool
        Whether to load normalization metadata, defaults to True.
    min_nonzero_fraction : float
        Minimum fraction of voxels above ``nonzero_threshold`` for a patch
        to be used. Patches below this fraction are retried up to
        ``max_nonzero_retries`` times. Default 0.0 disables filtering.
    nonzero_threshold : float
        Intensity threshold for the nonzero fraction check.
        Default 0.0 means any nonzero voxel counts.
    nonzero_channel : str or None
        Channel name to check for nonzero fraction. ``None`` defaults
        to the first target channel.
    max_nonzero_retries : int
        Maximum number of random re-samples when a patch fails the
        nonzero fraction check. Default 100.
    fg_mask_key : str or None
        Zarr array key for precomputed foreground masks. When set,
        masks are loaded alongside images and included in the sample
        as ``"fg_mask"``. Default None disables mask loading.
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        array_key: str = "0",
        transform: DictTransform | None = None,
        load_normalization_metadata: bool = True,
        min_nonzero_fraction: float = 0.0,
        nonzero_threshold: float = 0.0,
        nonzero_channel: str | None = None,
        max_nonzero_retries: int = 100,
        fg_mask_key: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_nonzero_fraction <= 1.0:
            raise ValueError(f"min_nonzero_fraction must be in [0, 1], got {min_nonzero_fraction}")
        if max_nonzero_retries < 0:
            raise ValueError(f"max_nonzero_retries must be >= 0, got {max_nonzero_retries}")
        self.positions = positions
        self.channels = {k: _ensure_channel_list(v) for k, v in channels.items()}
        self.source_ch_idx = [positions[0].get_channel_index(c) for c in channels["source"]]
        self.target_ch_idx = (
            [positions[0].get_channel_index(c) for c in channels["target"]] if "target" in channels else None
        )
        self.z_window_size = z_window_size
        self.transform = transform
        self.array_key = array_key
        self.load_normalization_metadata = load_normalization_metadata
        self.min_nonzero_fraction = min_nonzero_fraction
        self.nonzero_threshold = nonzero_threshold
        self.nonzero_channel = nonzero_channel
        self.max_nonzero_retries = max_nonzero_retries
        self.fg_mask_key = fg_mask_key
        self._get_windows()
        if nonzero_channel is not None:
            all_channels = list(self.channels.get("source", [])) + list(self.channels.get("target", []))
            if nonzero_channel not in all_channels:
                raise ValueError(f"nonzero_channel '{nonzero_channel}' not found in channels: {all_channels}")

    def _get_windows(self) -> None:
        """Count the sliding windows along T and Z, and build an index-to-window LUT."""
        w = 0
        self.window_keys = []
        self.window_arrays = []
        self.window_norm_meta: list[NormMeta | None] = []
        self.window_fg_mask_arrays: list[ImageArray | None] = []
        for fov in self.positions:
            img_arr: ImageArray = fov[str(self.array_key)]
            ts = img_arr.frames
            zs = img_arr.slices - self.z_window_size + 1
            if zs < 1:
                raise IndexError(
                    f"Z window size {self.z_window_size} "
                    f"is larger than the number of Z slices ({img_arr.slices}) "
                    f"for FOV {fov.name}."
                )
            w += ts * zs
            self.window_keys.append(w)
            self.window_arrays.append(img_arr)
            self.window_norm_meta.append(_read_norm_meta(fov))
            if self.fg_mask_key is not None:
                if self.fg_mask_key not in fov:
                    raise FileNotFoundError(
                        f"Mask array '{self.fg_mask_key}' not found in position. "
                        "Run preprocessing with --compute_fg_masks first."
                    )
                self.window_fg_mask_arrays.append(fov[self.fg_mask_key])
            else:
                self.window_fg_mask_arrays.append(None)
        self._max_window = w

    def _find_window(self, index: int) -> tuple[ImageArray, int, NormMeta | None, ImageArray | None]:
        """Look up window given index."""
        window_idx = sorted(self.window_keys + [index + 1]).index(index + 1)
        w = self.window_keys[window_idx]
        tz = index - self.window_keys[window_idx - 1] if window_idx > 0 else index
        arr_idx = self.window_keys.index(w)
        norm_meta = self.window_norm_meta[arr_idx]
        fg_mask_arr = self.window_fg_mask_arrays[arr_idx]
        return (self.window_arrays[arr_idx], tz, norm_meta, fg_mask_arr)

    def _read_img_window(self, img: ImageArray, ch_idx: list[int], tz: int) -> tuple[list[Tensor], HCSStackIndex]:
        """Read image window as tensor.

        Parameters
        ----------
        img : ImageArray
            NGFF image array.
        ch_idx : list[int]
            List of channel indices to read,
            output channel ordering will reflect the sequence.
        tz : int
            Window index within the FOV, counted Z-first.

        Returns
        -------
        list[Tensor], HCSStackIndex
            List of (C=1, Z, Y, X) image tensors,
            tuple of image name, time index, and Z index.
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
        """Return total number of windows."""
        return self._max_window

    def _stack_channels(
        self,
        sample_images: list[dict[str, Tensor]] | dict[str, Tensor],
        key: str | None = None,
        keys: list[str] | None = None,
    ) -> Tensor | list[Tensor]:
        """Stack single-channel images into a multi-channel tensor."""
        ch_keys = keys if keys is not None else self.channels[key]
        if not isinstance(sample_images, list):
            return torch.stack([sample_images[ch][0] for ch in ch_keys])
        return [torch.stack([im[ch][0] for ch in ch_keys]) for im in sample_images]

    def __getitem__(self, index: int) -> Sample:
        """Return a sample for the given index."""
        check_key = (
            (self.nonzero_channel or self.channels.get("target", [None])[0]) if self.min_nonzero_fraction > 0 else None
        )
        idx = index
        for attempt in range(self.max_nonzero_retries + 1):
            img, tz, norm_meta, fg_mask_arr = self._find_window(idx)
            ch_names = self.channels["source"].copy()
            ch_idx = self.source_ch_idx.copy()
            if self.target_ch_idx is not None:
                ch_names.extend(self.channels["target"])
                ch_idx.extend(self.target_ch_idx)
            images, sample_index = self._read_img_window(img, ch_idx, tz)
            sample_images = {k: v for k, v in zip(ch_names, images)}
            # Read mask once — reused for both nonzero check and sample output
            mask_images = None
            if fg_mask_arr is not None and self.target_ch_idx is not None:
                mask_images, _ = self._read_img_window(fg_mask_arr, self.target_ch_idx, tz)
            if check_key is not None:
                if mask_images is not None and check_key in self.channels.get("target", []):
                    check_ch = self.channels["target"].index(check_key)
                    frac = mask_images[check_ch].sum().item() / mask_images[check_ch].numel()
                elif check_key in sample_images:
                    patch = sample_images[check_key]
                    frac = (patch >= self.nonzero_threshold).sum().item() / patch.numel()
                else:
                    break
                if frac < self.min_nonzero_fraction:
                    if attempt < self.max_nonzero_retries:
                        idx = random.randint(0, len(self) - 1)
                        continue
                    _logger.warning(
                        f"Exhausted {self.max_nonzero_retries} retries for nonzero fraction "
                        f">= {self.min_nonzero_fraction} on channel '{check_key}' "
                        f"(index {index}). Returning last sample."
                    )
            break
        # Inject mask as temp keys so MONAI spatial transforms (e.g. _final_crop) co-align them
        fg_mask_keys = []
        if mask_images is not None:
            for ch_name, mask_tensor in zip(self.channels["target"], mask_images):
                key = f"__fg_mask_{ch_name}"
                sample_images[key] = mask_tensor
                fg_mask_keys.append(key)
        if self.target_ch_idx is not None:
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
        }
        if self.target_ch_idx is not None:
            sample["target"] = self._stack_channels(sample_images, "target")
        if fg_mask_keys:
            sample["fg_mask"] = self._stack_channels(sample_images, keys=fg_mask_keys)
        if self.load_normalization_metadata and norm_meta is not None:
            sample["norm_meta"] = norm_meta
        return sample


class MaskTestDataset(SlidingWindowDataset):
    """Test dataset with optional ground truth masks.

    Each element is a window of (C, Z, Y, X) where C=2 (source and target)
    and Z is ``z_window_size``.

    This a testing stage version of
    :py:class:`viscy_data.hcs.SlidingWindowDataset`,
    and can only be used with batch size 1 for efficiency (no padding for collation),
    since the mask is not available for each stack.

    Parameters
    ----------
    positions : list[Position]
        FOVs to include in dataset.
    channels : ChannelMap
        Source and target channel names,
        e.g. ``{'source': 'Phase', 'target': ['Nuclei', 'Membrane']}``.
    z_window_size : int
        Z window size of the 2.5D U-Net, 1 for 2D.
    transform : DictTransform
        A callable that transforms data, defaults to None.
    ground_truth_masks : str | None
        Path to the ground truth masks.
    array_key : str, optional
        Name of the image arrays (multiscales level), by default "0".
    """

    def __init__(
        self,
        positions: list[Position],
        channels: ChannelMap,
        z_window_size: int,
        transform: DictTransform | None = None,
        ground_truth_masks: str | None = None,
        array_key: str = "0",
        **kwargs,
    ) -> None:
        super().__init__(positions, channels, z_window_size, array_key=array_key, transform=transform, **kwargs)
        self.masks = {}
        if ground_truth_masks is None:
            return
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
        """Return a sample with optional ground truth mask."""
        sample = super().__getitem__(index)
        img_name, t_idx, z_idx = sample["index"]
        position_name = int(img_name.split("/")[-2])
        key = (position_name, int(t_idx), int(z_idx) + self.z_window_size // 2)
        if img_path := self.masks.get(key):
            sample["labels"] = torch.from_numpy(imread(img_path).astype(np.int16))
        return sample


class HCSDataModule(LightningDataModule):
    """Lightning data module for a preprocessed HCS NGFF Store.

    Parameters
    ----------
    data_path : str
        Path to the data store.
    source_channel : str or Sequence[str]
        Name(s) of the source channel, e.g. 'Phase'.
    target_channel : str or Sequence[str]
        Name(s) of the target channel, e.g. ['Nuclei', 'Membrane'].
    z_window_size : int
        Z window size of the 2.5D U-Net, 1 for 2D.
    split_ratio : float, optional
        Split ratio of the training subset in the fit stage,
        e.g. 0.8 means an 80/20 split between training/validation,
        by default 0.8.
    batch_size : int, optional
        Batch size, defaults to 16.
    num_workers : int, optional
        Number of data-loading workers, defaults to 8.
    target_2d : bool, optional
        Whether the target is 2D (e.g. in a 2.5D model),
        defaults to False.
    yx_patch_size : tuple[int, int], optional
        Patch size in (Y, X), defaults to (256, 256).
    normalizations : list of MapTransform or None, optional
        MONAI dictionary transforms applied to selected channels,
        defaults to None (no normalization).
    augmentations : list of MapTransform or None, optional
        MONAI dictionary transforms applied to the training set,
        defaults to None (no augmentation).
    caching : bool, optional
        Whether to decompress all the images and cache the result,
        will store in `/tmp/$SLURM_JOB_ID/` if available,
        defaults to False.
    ground_truth_masks : Path or None, optional
        Path to the ground truth masks,
        used in the test stage to compute segmentation metrics,
        defaults to None.
    persistent_workers : bool, optional
        Whether to keep the workers alive between fitting epochs,
        defaults to False.
    prefetch_factor : int or None, optional
        Number of samples loaded in advance by each worker during fitting,
        defaults to None (2 per PyTorch default).
    array_key : str, optional
        Name of the image arrays (multiscales level), by default "0".
    min_nonzero_fraction : float, optional
        Minimum fraction of voxels above ``nonzero_threshold`` for training.
        Default 0.0 disables filtering.
    nonzero_threshold : float, optional
        Intensity threshold for the nonzero fraction check, by default 0.0.
    nonzero_channel : str or None, optional
        Channel to check. ``None`` defaults to the first target channel.
    max_nonzero_retries : int, optional
        Maximum retries when a patch fails the nonzero check, by default 100.
    fg_mask_key : str or None, optional
        Zarr array key for precomputed foreground masks, by default None.
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
        target_2d: bool = False,
        yx_patch_size: tuple[int, int] = (256, 256),
        normalizations: list[MapTransform] | None = None,
        augmentations: list[MapTransform] | None = None,
        caching: bool = False,
        ground_truth_masks: Path | None = None,
        persistent_workers=False,
        prefetch_factor=None,
        array_key: str = "0",
        pin_memory=False,
        min_nonzero_fraction: float = 0.0,
        nonzero_threshold: float = 0.0,
        nonzero_channel: str | None = None,
        max_nonzero_retries: int = 100,
        fg_mask_key: str | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.source_channel = _ensure_channel_list(source_channel)
        self.target_channel = _ensure_channel_list(target_channel)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_2d = target_2d
        self.z_window_size = z_window_size
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.normalizations = normalizations or []
        self.augmentations = augmentations or []
        self.caching = caching
        self.ground_truth_masks = ground_truth_masks
        self.prepare_data_per_node = True
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.array_key = array_key
        self.pin_memory = pin_memory
        self.min_nonzero_fraction = min_nonzero_fraction
        self.nonzero_threshold = nonzero_threshold
        self.nonzero_channel = nonzero_channel
        self.max_nonzero_retries = max_nonzero_retries
        self.fg_mask_key = fg_mask_key

    @property
    def cache_path(self):
        """Return the cache path for the dataset."""
        return Path(
            tempfile.gettempdir(),
            os.getenv("SLURM_JOB_ID", "viscy_cache"),
            self.data_path.name,
        )

    @property
    def maybe_cached_data_path(self):
        """Return the cached data path if caching is enabled."""
        return self.cache_path if self.caching else self.data_path

    def _data_log_path(self) -> Path:
        log_dir = Path.cwd()
        if self.trainer:
            if self.trainer.logger:
                if self.trainer.logger.log_dir:
                    log_dir = Path(self.trainer.logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "data.log"

    def prepare_data(self):
        """Cache dataset if caching is enabled."""
        if not self.caching:
            return
        # setup logger
        logger = logging.getLogger("viscy_data.hcs.cache")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(self._data_log_path())
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.info(f"Caching dataset at {self.cache_path}.")
        if self.cache_path.exists():
            logger.info("Cache already exists, skipping copy.")
            return
        shutil.copytree(self.data_path, self.cache_path)
        logger.info("Cached dataset.")

    @property
    def _base_dataset_settings(self) -> dict:
        """Return base dataset settings."""
        settings: dict = {
            "channels": {"source": self.source_channel},
            "z_window_size": self.z_window_size,
            "array_key": self.array_key,
        }
        if self.fg_mask_key is not None:
            settings["fg_mask_key"] = self.fg_mask_key
        return settings

    @property
    def _train_filter_settings(self) -> dict:
        """Return nonzero fraction filtering settings (training only)."""
        settings: dict = {}
        if self.min_nonzero_fraction > 0:
            settings["min_nonzero_fraction"] = self.min_nonzero_fraction
            settings["nonzero_threshold"] = self.nonzero_threshold
            settings["max_nonzero_retries"] = self.max_nonzero_retries
            if self.nonzero_channel is not None:
                settings["nonzero_channel"] = self.nonzero_channel
        return settings

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        """Set up datasets for the given stage."""
        dataset_settings = self._base_dataset_settings
        if stage in ("fit", "validate"):
            self._setup_fit(dataset_settings)
        elif stage == "test":
            self._setup_test(dataset_settings)
        elif stage == "predict":
            self._setup_predict(dataset_settings)
        else:
            raise NotImplementedError(f"{stage} stage")

    def _set_fit_global_state(self, num_positions: int) -> torch.Tensor:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions)

    def _setup_fit(self, dataset_settings: dict):
        """Set up the training and validation datasets."""
        train_transform, val_transform = self._fit_transform()
        dataset_settings["channels"]["target"] = self.target_channel
        data_path = self.maybe_cached_data_path
        plate = open_ome_zarr(data_path, mode="r")

        # shuffle positions, randomness is handled globally
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = self._set_fit_global_state(len(positions))
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
        train_dataset_settings.update(self._train_filter_settings)
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
        data_path = self.maybe_cached_data_path
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

    def _set_predict_global_state(self) -> None:
        # track metadata for inverting transform
        set_track_meta(True)
        if self.caching:
            _logger.warning("Ignoring caching config in 'predict' stage.")

    def _positions_maybe_single(self) -> list[Position]:
        dataset: Plate | Position = open_ome_zarr(self.data_path, mode="r")
        if isinstance(dataset, Position):
            try:
                plate_path = self.data_path.parent.parent.parent
                fov_name = self.data_path.relative_to(plate_path).as_posix()
                plate = open_ome_zarr(plate_path)
            except (OSError, ValueError):
                raise FileNotFoundError("Parent HCS store not found for single FOV input.")
            positions = [plate[fov_name]]
        elif isinstance(dataset, Plate):
            positions = [p for _, p in dataset.positions()]
        return positions

    def _setup_predict(
        self,
        dataset_settings: dict,
    ):
        """Set up the predict stage."""
        self._set_predict_global_state()
        predict_transform = Compose(self.normalizations)
        self.predict_dataset = SlidingWindowDataset(
            positions=self._positions_maybe_single(),
            transform=predict_transform,
            **dataset_settings,
        )

    def on_before_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Remove redundant Z slices if the target is 2D to save VRAM."""
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
            if "fg_mask" in batch:
                batch["fg_mask"] = batch["fg_mask"][:, :, slice(z_index, z_index + 1)]
        return batch

    def train_dataloader(self):
        """Return training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.train_patches_per_stack,
            num_workers=self.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            collate_fn=_collate_samples,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Return predict data loader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _fit_transform(self) -> tuple[Compose, Compose]:
        """Build training and validation transforms.

        Apply normalization, augmentation, then center crop as the last step.
        """
        # TODO: These have a fixed order for now... ()
        final_crop = [self._final_crop()]
        train_transform = Compose(self.normalizations + self._train_transform() + final_crop)
        val_transform = Compose(self.normalizations + final_crop)
        return train_transform, val_transform

    def _final_crop(self) -> CenterSpatialCropd:
        """Set up final cropping: center crop to the target size."""
        keys = self.source_channel + self.target_channel
        if self.fg_mask_key is not None:
            keys = keys + [f"__fg_mask_{ch}" for ch in self.target_channel]
        return CenterSpatialCropd(
            keys=keys,
            roi_size=(
                self.z_window_size,
                self.yx_patch_size[0],
                self.yx_patch_size[1],
            ),
            allow_missing_keys=True,
        )

    def _train_transform(self) -> list[Callable]:
        """Set up training augmentations.

        Check input values, and parse the number of Z slices and
        patches to sample per stack.
        """
        self.train_patches_per_stack = 1
        z_scale_range = None
        if self.augmentations:
            for aug in self.augmentations:
                if isinstance(aug, RandAffined):
                    if z_scale_range is not None:
                        raise ValueError("Only one RandAffined augmentation is allowed.")
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
            if isinstance(z_scale_range, (float, int)):
                z_scale_range = float(z_scale_range)
                z_scale_range = (-z_scale_range, z_scale_range)
            if z_scale_range[0] > 0 or z_scale_range[1] < 0:
                raise ValueError(f"Invalid scaling range: {z_scale_range}")
            self.train_z_scale_range = z_scale_range
        else:
            self.train_z_scale_range = (0.0, 0.0)
        _logger.debug(f"Training augmentations: {self.augmentations}")
        return list(self.augmentations)
