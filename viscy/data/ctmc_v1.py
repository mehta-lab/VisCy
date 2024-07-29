from pathlib import Path

from iohub.ngff import open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, MapTransform
from torch.utils.data import DataLoader

from viscy.data.hcs import ChannelMap, SlidingWindowDataset
from viscy.data.typing import Sample


class CTMCv1ValidationDataset(SlidingWindowDataset):
    def __len__(self, subsample_rate: int = 30) -> int:
        # sample every 30th frame in the videos
        return super().__len__() // self.subsample_rate

    def __getitem__(self, index: int) -> Sample:
        index = index * self.subsample_rate
        return super().__getitem__(index)


class CTMCv1DataModule(LightningDataModule):
    """
    Autoregression data module for the CTMCv1 dataset.
    Training and validation datasets are stored in separate HCS OME-Zarr stores.

    :param str | Path train_data_path: Path to the training dataset
    :param str | Path val_data_path: Path to the validation dataset
    :param list[MapTransform] train_transforms: List of transforms for training
    :param list[MapTransform] val_transforms: List of transforms for validation
    :param int batch_size: Batch size, defaults to 16
    :param int num_workers: Number of workers, defaults to 8
    :param str channel_name: Name of the DIC channel, defaults to "DIC"
    """

    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        train_transforms: list[MapTransform],
        val_transforms: list[MapTransform],
        batch_size: int = 16,
        num_workers: int = 8,
        channel_name: str = "DIC",
    ) -> None:
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.channel_map = ChannelMap(source=[channel_name], target=[channel_name])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self._setup_fit()

    def _setup_fit(self) -> None:
        train_plate = open_ome_zarr(self.train_data_path)
        val_plate = open_ome_zarr(self.val_data_path)
        train_positions = [p for _, p in train_plate.positions()]
        val_positions = [p for _, p in val_plate.positions()]
        self.train_dataset = SlidingWindowDataset(
            train_positions,
            channels=self.channel_map,
            z_window_size=1,
            transform=Compose(self.train_transforms),
        )
        self.val_dataset = CTMCv1ValidationDataset(
            val_positions,
            channels=self.channel_map,
            z_window_size=1,
            transform=Compose(self.val_transforms),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
            shuffle=False,
        )
