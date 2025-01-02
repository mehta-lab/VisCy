from pathlib import Path

import torch
from iohub.ngff import open_ome_zarr
from monai.transforms import Compose, MapTransform

from viscy.data.gpu_aug import CachedOmeZarrDataset, GPUTransformDataModule


class CTMCv1DataModule(GPUTransformDataModule):
    """
    Autoregression data module for the CTMCv1 dataset.
    Training and validation datasets are stored in separate HCS OME-Zarr stores.

    Parameters
    ----------
    train_data_path : str or Path
        Path to the training dataset.
    val_data_path : str or Path
        Path to the validation dataset.
    train_cpu_transforms : list of MapTransform
        List of CPU transforms for training.
    val_cpu_transforms : list of MapTransform
        List of CPU transforms for validation.
    train_gpu_transforms : list of MapTransform
        List of GPU transforms for training.
    val_gpu_transforms : list of MapTransform
        List of GPU transforms for validation.
    batch_size : int, optional
        Batch size, by default 16.
    num_workers : int, optional
        Number of dataloading workers, by default 8.
    val_subsample_ratio : int, optional
        Skip every N frames for validation to reduce redundancy in video,
        by default 30.
    channel_name : str, optional
        Name of the DIC channel, by default "DIC".
    pin_memory : bool, optional
        Pin memory for dataloaders, by default True.
    """

    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        train_cpu_transforms: list[MapTransform],
        val_cpu_transforms: list[MapTransform],
        train_gpu_transforms: list[MapTransform],
        val_gpu_transforms: list[MapTransform],
        batch_size: int = 16,
        num_workers: int = 8,
        val_subsample_ratio: int = 30,
        channel_name: str = "DIC",
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self._train_cpu_transforms = Compose(train_cpu_transforms)
        self._val_cpu_transforms = Compose(val_cpu_transforms)
        self._train_gpu_transforms = Compose(train_gpu_transforms)
        self._val_gpu_transforms = Compose(val_gpu_transforms)
        self.channel_names = [channel_name]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_subsample_ratio = val_subsample_ratio
        self.pin_memory = pin_memory

    @property
    def train_cpu_transforms(self) -> Compose:
        return self._train_cpu_transforms

    @property
    def val_cpu_transforms(self) -> Compose:
        return self._val_cpu_transforms

    @property
    def train_gpu_transforms(self) -> Compose:
        return self._train_gpu_transforms

    @property
    def val_gpu_transforms(self) -> Compose:
        return self._val_gpu_transforms

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self._setup_fit()

    def _setup_fit(self) -> None:
        cache_map = torch.multiprocessing.Manager().dict()
        train_plate = open_ome_zarr(self.train_data_path)
        val_plate = open_ome_zarr(self.val_data_path)
        train_positions = [p for _, p in train_plate.positions()]
        val_positions = [p for _, p in val_plate.positions()]
        self.train_dataset = CachedOmeZarrDataset(
            positions=train_positions,
            channel_names=self.channel_names,
            cache_map=cache_map,
            transform=self.train_cpu_transforms,
            load_normalization_metadata=False,
        )
        full_val_dataset = CachedOmeZarrDataset(
            positions=val_positions,
            channel_names=self.channel_names,
            cache_map=cache_map,
            transform=self.val_cpu_transforms,
            load_normalization_metadata=False,
        )
        subsample_indices = list(
            range(0, len(full_val_dataset), self.val_subsample_ratio)
        )
        self.val_dataset = torch.utils.data.Subset(full_val_dataset, subsample_indices)
