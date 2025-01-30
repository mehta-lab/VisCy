from pathlib import Path
from typing import Callable

import numpy as np
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from tarrow.data.tarrow_dataset import TarrowDataset
from torch.utils.data import ConcatDataset, DataLoader

from viscy.utils.engine_state import set_fit_global_state


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
        prefetch_factor: int | None = None,
        include_fov_names: list[str] = [],
        train_samples_per_epoch: int = 100000,
        val_samples_per_epoch: int = 10000,
        resolution: int = 0,
        z_slice: int = 0,
        normalization: Callable[[np.ndarray], np.ndarray] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.channel_name = channel_name
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.path_size = patch_size
        self.include_fov_names = include_fov_names
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch
        self.resolution = resolution
        self.z_slice = z_slice
        self.kwargs = kwargs
        self.normalization = normalization

        self._filter_positions()
        self._channel_idx = self._get_channel_index()

    def _get_channel_index(self, plate) -> int:
        """Get the index of the specified channel from the plate metadata.

        Parameters
        ----------
        plate : iohub.ngff.Plate
            OME-Zarr plate object

        Returns
        -------
        int
            Index of the specified channel

        Raises
        ------
        ValueError
            If channel_name is not found in available channels
        """
        # Get channel names from first position
        _, first_pos = next(plate.positions())
        try:
            return first_pos.channel_names.index(self.channel_name)
        except ValueError:
            available_channels = ", ".join(first_pos.channel_names)
            raise ValueError(
                f"Channel '{self.channel_name}' not found. Available channels: {available_channels}"
            )

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

        # Get channel index once

        if stage == "fit":
            list_dataset = []
            for pos in self.positions:
                pos_imgs = self._load_images(pos, self._channel_idx)
                list_dataset.append(
                    TarrowDataset(
                        imgs=pos_imgs,
                        normalize=self.normalization,
                        size=self.path_size,
                        **self.kwargs,
                    )
                )

            # Calculate split point
            split_idx = int(len(self.positions) * self.train_split)

            # Shuffle the list of datasets
            shuffled_indices = set_fit_global_state(len(list_dataset))
            list_dataset = [list_dataset[i] for i in shuffled_indices]

            # Create training dataset with first train_split% of images
            self.train_dataset = ConcatDataset(list_dataset[:split_idx])

            # Create validation dataset with remaining images
            self.val_dataset = ConcatDataset(list_dataset[split_idx:])

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

    def _get_channel_index(self):
        """Get the index of the specified channel from the plate metadata."""
        with open_ome_zarr(self.ome_zarr_path, mode="r") as plate:
            _, first_pos = next(plate.positions())
        return first_pos.channel_names.index(self.channel_name)

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
            pin_memory=True,
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
            pin_memory=True,
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
