import json
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, Transform
from tifffile import imread
from torch.utils.data import DataLoader, Dataset

from viscy.data.typing import Sample


class LiveCellDataset(Dataset):
    """
    LiveCell dataset.

    :param list[Path] images: List of paths to single-page, single-channel TIFF files.
    :param Transform | Compose transform: Transform to apply to the dataset
    """

    def __init__(self, images: list[Path], transform: Transform | Compose) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        image = imread(self.images[idx])[None, None]
        image = torch.from_numpy(image).to(torch.float32)
        image = self.transform(image)
        return {"source": image, "target": image}


class LiveCellDataModule(LightningDataModule):
    def __init__(
        self,
        train_val_images: Path,
        train_annotations: Path,
        val_annotations: Path,
        train_transforms: list[Transform],
        val_transforms: list[Transform],
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_val_images = Path(train_val_images)
        if not self.train_val_images.is_dir():
            raise NotADirectoryError(str(train_val_images))
        self.train_annotations = Path(train_annotations)
        if not self.train_annotations.is_file():
            raise FileNotFoundError(str(train_annotations))
        self.val_annotations = Path(val_annotations)
        if not self.val_annotations.is_file():
            raise FileNotFoundError(str(val_annotations))
        self.train_transforms = Compose(train_transforms)
        self.val_transforms = Compose(val_transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise NotImplementedError("Only fit stage is supported")
        self._setup_fit()

    def _parse_image_names(self, annotations: Path) -> list[Path]:
        with open(annotations) as f:
            images = [f["file_name"] for f in json.load(f)["images"]]
        return sorted(images)

    def _setup_fit(self) -> None:
        train_images = self._parse_image_names(self.train_annotations)
        val_images = self._parse_image_names(self.val_annotations)
        self.train_dataset = LiveCellDataset(
            [self.train_val_images / f for f in train_images],
            transform=self.train_transforms,
        )
        self.val_dataset = LiveCellDataset(
            [self.train_val_images / f for f in val_images],
            transform=self.val_transforms,
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
        )
