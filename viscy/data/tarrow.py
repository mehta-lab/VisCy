from pathlib import Path

import numpy as np
import torch
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from tarrow.data.tarrow_dataset import TarrowDataset
from torch.utils.data import DataLoader


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
        prefetch_factor: int | None = None,
        include_fov_names: list[str] = [],
        train_samples_per_epoch: int = 100000,
        val_samples_per_epoch: int = 10000,
        resolution: int = 0,
        z_slice: int = 0,
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
        self.include_fov_names = include_fov_names
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch
        self.resolution = resolution
        self.z_slice = z_slice
        self.kwargs = kwargs

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

    def _load_images(
        self, positions: list[Position], channel_idx: int
    ) -> list[np.ndarray]:
        """Load all images from positions into memory.

        Parameters
        ----------
        positions : list[Position]
            List of positions to load
        channel_idx : int
            Index of channel to load

        Returns
        -------
        list[np.ndarray]
            List of 2D numpy arrays
        """
        imgs = []
        for pos in positions:
            img_arr = pos[str(self.resolution)]
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
        plate = open_ome_zarr(self.ome_zarr_path, mode="r")

        # Get channel index once
        channel_idx = self._get_channel_index(plate)

        # Get the positions to load
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

        # Load all images into memory using the pre-determined channel index
        imgs = self._load_images(positions, channel_idx)

        # Calculate split point
        split_idx = int(len(imgs) * self.train_split)

        if stage == "fit":
            # Create training dataset with first train_split% of images
            self.train_dataset = TarrowDataset(
                imgs=imgs[:split_idx],
                **self.kwargs,
            )

            # Create validation dataset with remaining images
            self.val_dataset = TarrowDataset(
                imgs=imgs[split_idx:],
                **{k: v for k, v in self.kwargs.items() if k != "augmenter"},
            )

        elif stage == "test":
            raise NotImplementedError(f"Invalid stage: {stage}")
        elif stage == "predict":
            raise NotImplementedError(f"Invalid stage: {stage}")
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        """Create the training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for training data with random sampling
        """
        return DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=self.train_samples_per_epoch,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
        )

    def val_dataloader(self):
        """Create the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for validation data with random sampling
        """
        return DataLoader(
            self.val_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.val_dataset,
                replacement=True,
                num_samples=self.val_samples_per_epoch,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
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
