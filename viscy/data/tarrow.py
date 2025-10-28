from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch.nn as nn
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from tarrow.data.tarrow_dataset import TarrowDataset
from torch.utils.data import ConcatDataset, DataLoader

# FIXME: This module is not available in the viscy package,so shuffle the list of datasets manually.
# from viscy.utils.engine_state import set_fit_global_state
import random


class TarrowDataModule(LightningDataModule):
    """Lightning DataModule for TimeArrowNet training.

    Parameters
    ----------
    ome_zarr_path : str or Path
        Path to OME-Zarr file
    channel_name : str
        Name of the channel to load
    train_split : float, default=0.8
        Fraction of data to use for training (0.0 to 1.0)
    patch_size : tuple[int, int], default=(128, 128)
        Patch size for TarrowDataset
    visual_patch_size : tuple[int, int] | None, default=None
        Patch size for visualization dataset
    visual_batch_size : int | None, default=None
        Batch size for visualization dataloader
    batch_size : int, default=16
        Batch size for dataloaders
    num_workers : int, default=8
        Number of workers for dataloaders
    prefetch_factor : int, optional
        Prefetch factor for dataloaders
    include_fov_names : list[str], default=[]
        List of FOV names to include. If empty, use all FOVs
    train_samples_per_epoch : int, default=100000
        Number of training samples per epoch
    val_samples_per_epoch : int, default=10000
        Number of validation samples per epoch
    resolution : int, default=0
        Resolution level to load from OME-Zarr
    normalization : function, optional (default=None)
        Normalization function to apply to images
    z_slice : int, default=0
        Z-slice to load
    pin_memory : bool, default=True
        Whether to pin memory
    persistent_workers : bool, default=True
        Whether to keep the workers alive between epochs
    augmentations : list[nn.Module], default=[]
        List of Kornia augmentation transforms to apply during training
    **kwargs : dict
        Additional arguments passed to TarrowDataset
    """

    def __init__(
        self,
        ome_zarr_path: str | Path,
        channel_name: str,
        train_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        patch_size: tuple[int, int] = (128, 128),
        visual_patch_size: tuple[int, int] | None = None,
        visual_batch_size: int | None = None,
        prefetch_factor: int | None = None,
        include_fov_names: list[str] = [],
        train_samples_per_epoch: int = 100000,
        val_samples_per_epoch: int = 10000,
        resolution: int = 0,
        z_slice: int = 0,
        normalization: Callable[[np.ndarray], np.ndarray] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        augmentations: Sequence[nn.Module] = [],
        **kwargs,
    ):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.channel_name = channel_name
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.patch_size = patch_size
        self.visual_patch_size = visual_patch_size or tuple(4 * x for x in patch_size)
        self.visual_batch_size = visual_batch_size or min(4, batch_size)
        self.include_fov_names = include_fov_names
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch
        self.resolution = resolution
        self.z_slice = z_slice
        self.kwargs = kwargs
        self.normalization = normalization
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.augmentations = augmentations

        self._filter_positions()
        self._channel_idx = self._get_channel_index()

    def _get_channel_index(self) -> int:
        """Get the index of the specified channel from the plate metadata."""
        with open_ome_zarr(self.ome_zarr_path, mode="r") as plate:
            _, first_pos = next(plate.positions())
        return first_pos.channel_names.index(self.channel_name)

    def _create_augmentation_pipeline(self) -> nn.Sequential | None:
        """Create the augmentation pipeline for training.

        Returns
        -------
        nn.Sequential | None
            Sequential container of Kornia augmentations or None if no augmentations
        """
        if not self.augmentations:
            return None

        return nn.Sequential(*self.augmentations)

    def _load_images(self, position: Position, channel_idx: int) -> list[np.ndarray]:
        """Load all images from positions into memory.

        Parameters
        ----------
        position : Position
            Position to load
        channel_idx : int
            Index of channel to load

        Returns
        -------
        list[np.ndarray]
            List of 2D numpy arrays
        """
        imgs = []
        img_arr = position[str(self.resolution)]
        # Load all timepoints for this position
        for t in range(len(img_arr)):
            imgs.append(img_arr[t, channel_idx, self.z_slice])
        return imgs

    def setup(self, stage: str):
        """Set up the data module for a specific stage.

        Parameters
        ----------
        stage : str
            Stage to set up for ("fit", "test", or "predict")

        Raises
        ------
        NotImplementedError
            If stage is not "fit"
        """
        if stage == "fit":
            list_dataset = []
            list_visual_dataset = []

            # Create augmentation pipeline
            augmenter = self._create_augmentation_pipeline()

            for pos in self.positions:
                pos_imgs = self._load_images(pos, self._channel_idx)
                list_dataset.append(
                    TarrowDataset(
                        imgs=pos_imgs,
                        normalize=self.normalization,
                        size=self.patch_size,
                        augmenter=augmenter,  # Pass augmenter to dataset
                        **self.kwargs,
                    )
                )
                # Create visualization dataset with larger patches
                list_visual_dataset.append(
                    TarrowDataset(
                        imgs=pos_imgs,
                        normalize=self.normalization,
                        size=self.visual_patch_size,
                        **self.kwargs,
                    )
                )

            # Calculate split point
            split_idx = int(len(self.positions) * self.train_split)

            # Shuffle the list of datasets
            
            #FIXME: This module is not available in the viscy package,so shuffle the list of datasets manually.
            # shuffled_indices = set_fit_global_state(len(list_dataset))
            shuffled_indices = list(range(len(list_dataset)))
            random.shuffle(shuffled_indices)
            list_dataset = [list_dataset[i] for i in shuffled_indices]
            list_visual_dataset = [
                list_visual_dataset[i] for i in shuffled_indices
            ]  # Use same shuffling

            # Create training dataset with first train_split% of images
            self.train_dataset = ConcatDataset(list_dataset[:split_idx])
            self.val_dataset = ConcatDataset(list_dataset[split_idx:])

            # Take up to n_visual samples from validation set
            # NOTE fixed to take the first n_visual samples from validation set
            self.visual_batch_size = max(
                len(list_visual_dataset[split_idx:]), self.visual_batch_size
            )
            self.visual_dataset = ConcatDataset(
                list_visual_dataset[split_idx : split_idx + self.visual_batch_size]
            )

        elif stage == "test":
            raise NotImplementedError(f"Invalid stage: {stage}")
        elif stage == "predict":
            raise NotImplementedError(f"Invalid stage: {stage}")
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def _filter_positions(self):
        """Filter positions based on include_fov_names."""
        # Get the positions to load
        plate = open_ome_zarr(self.ome_zarr_path, mode="r")
        if self.include_fov_names:
            positions = []
            for fov_str, pos in plate.positions():
                normalized_include_fovs = [
                    f.lstrip("/") for f in self.include_fov_names
                ]
                if fov_str in normalized_include_fovs:
                    positions.append(pos)
        else:
            positions = [pos for _, pos in plate.positions()]

        self.positions = positions

    def train_dataloader(self):
        """Create the training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for training data
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def visual_dataloader(self):
        """Create the visualization dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for visualization data
        """
        return DataLoader(
            self.visual_dataset,
            batch_size=self.visual_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create the test dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for test data without shuffling

        Raises
        ------
        NotImplementedError
            Test stage is not implemented yet
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
