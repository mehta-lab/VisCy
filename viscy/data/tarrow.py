from pathlib import Path
import numpy as np
from iohub.ngff import Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from tarrow.data.tarrow_dataset import TarrowDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch


class TarrowDataModule(LightningDataModule):
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
        **kwargs,
    ):
        """Initialize TarrowDataModule.

        Args:
            ome_zarr_path: Path to OME-Zarr file
            channel_name: Name of the channel to load
            train_split: Fraction of data to use for training (0.0 to 1.0)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            prefetch_factor: Prefetch factor for dataloaders
            include_fov_names: List of FOV names to include. If empty, use all FOVs.
            train_samples_per_epoch: Number of training samples per epoch
            val_samples_per_epoch: Number of validation samples per epoch
            resolution: Resolution level to load from OME-Zarr
            z_slice: Z-slice to load
            **kwargs: Additional arguments passed to TarrowDataset
        """
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

        Args:
            plate: OME-Zarr plate object

        Returns:
            Index of the specified channel

        Raises:
            ValueError: If channel_name is not found in available channels
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

        Args:
            positions: List of positions to load
            channel_idx: Index of channel to load

        Returns:
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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
