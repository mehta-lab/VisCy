"""Lightning data module for a preprocessed HCS NGFF Store."""

import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch
from iohub.ngff import Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.transforms import CenterSpatialCropd, Compose, MapTransform, MultiSampleTrait, RandAffined
from torch import Tensor
from torch.utils.data import DataLoader

from viscy_data._typing import Sample
from viscy_data._utils import _collate_samples, _ensure_channel_list
from viscy_data.foreground_masks import ForegroundMaskSupport
from viscy_data.sliding_window import MaskTestDataset, SlidingWindowDataset

_logger = logging.getLogger("lightning.pytorch")


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
        gpu_augmentations: list[MapTransform] | None = None,
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
        if gpu_augmentations and self.fg_mask_key is not None:
            ForegroundMaskSupport.patch_spatial_transforms(gpu_augmentations, ("target",), ("fg_mask",))
        self._gpu_augmentations = Compose(gpu_augmentations) if gpu_augmentations else None

    @staticmethod
    def _inject_mask_keys(
        transforms: list,
        target_keys: tuple[str, ...],
        mask_keys: tuple[str, ...],
    ) -> None:
        """Append mask keys to spatial transforms that operate on target keys.

        Delegates to :meth:`ForegroundMaskSupport.patch_spatial_transforms`.

        Parameters
        ----------
        transforms : list
            Mutable list of transform instances to patch in place.
        target_keys : tuple[str, ...]
            Keys that identify target channels (e.g. channel names or
            ``"target"``).
        mask_keys : tuple[str, ...]
            Keys to append (e.g. ``("__fg_mask_Nuclei",)`` or
            ``("fg_mask",)``).
        """
        ForegroundMaskSupport.patch_spatial_transforms(transforms, target_keys, mask_keys)

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
        with open_ome_zarr(data_path, mode="r") as plate:
            positions = [pos for _, pos in plate.positions()]

        # shuffle positions, randomness is handled globally
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
        with open_ome_zarr(data_path, mode="r") as plate:
            positions = [p for _, p in plate.positions()]
        test_transform = Compose(self.normalizations)
        if self.ground_truth_masks:
            self.test_dataset = MaskTestDataset(
                positions,
                transform=test_transform,
                ground_truth_masks=self.ground_truth_masks,
                **dataset_settings,
            )
        else:
            self.test_dataset = SlidingWindowDataset(
                positions,
                transform=test_transform,
                **dataset_settings,
            )

    def _set_predict_global_state(self) -> None:
        # track metadata for inverting transform
        set_track_meta(True)
        if self.caching:
            _logger.warning("Ignoring caching config in 'predict' stage.")

    def _positions_maybe_single(self) -> list[Position]:
        with open_ome_zarr(self.data_path, mode="r") as dataset:
            if isinstance(dataset, Position):
                try:
                    plate_path = self.data_path.parent.parent.parent
                    fov_name = self.data_path.relative_to(plate_path).as_posix()
                    with open_ome_zarr(plate_path, mode="r") as plate:
                        positions = [plate[fov_name]]
                except (OSError, ValueError):
                    raise FileNotFoundError("Parent HCS store not found for single FOV input.")
            elif isinstance(dataset, Plate):
                positions = [p for _, p in dataset.positions()]
        return positions

    def _setup_predict(
        self,
        dataset_settings: dict,
    ):
        """Set up the predict stage."""
        self._set_predict_global_state()
        # fg_mask is only used during training (Spotlight loss) — prediction
        # never reads it, and inference datasets may not have the array.
        dataset_settings.pop("fg_mask_key", None)
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

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Apply GPU augmentations after batch transfer to device.

        Parameters
        ----------
        batch : Sample
            Batch dict with ``source``, ``target`` keys as
            ``(B, C, Z, Y, X)`` tensors.
        dataloader_idx : int
            Dataloader index (unused).

        Returns
        -------
        Sample
            Augmented batch (training only; validation/test pass through).
        """
        if isinstance(batch, Tensor) or self._gpu_augmentations is None:
            return batch
        if self.trainer and not self.trainer.training:
            return batch
        return self._gpu_augmentations(batch)

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
        When ``fg_mask_key`` is set, spatial augmentations are patched to
        also transform the mask keys so they stay pixel-aligned with the target.
        """
        final_crop = [self._final_crop()]
        augmentations = self._train_transform()
        if self.fg_mask_key is not None:
            mask_keys = ForegroundMaskSupport.mask_temp_keys(list(self.target_channel))
            ForegroundMaskSupport.patch_spatial_transforms(augmentations, tuple(self.target_channel), mask_keys)
        train_transform = Compose(self.normalizations + augmentations + final_crop)
        val_transform = Compose(self.normalizations + final_crop)
        return train_transform, val_transform

    def _final_crop(self) -> CenterSpatialCropd:
        """Set up final cropping: center crop to the target size."""
        keys = self.source_channel + self.target_channel
        allow_missing = False
        if self.fg_mask_key is not None:
            keys = keys + list(ForegroundMaskSupport.mask_temp_keys(list(self.target_channel)))
            allow_missing = True
        return CenterSpatialCropd(
            keys=keys,
            roi_size=(
                self.z_window_size,
                self.yx_patch_size[0],
                self.yx_patch_size[1],
            ),
            allow_missing_keys=allow_missing,
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
