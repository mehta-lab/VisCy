import json
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose, MapTransform
from pycocotools.coco import COCO
from tifffile import imread
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert

from viscy.data.typing import Sample


class LiveCellDataset(Dataset):
    """
    LiveCell dataset.

    :param list[Path] images: List of paths to single-page, single-channel TIFF files.
    :param MapTransform | Compose transform: Transform to apply to the dataset
    """

    def __init__(self, images: list[Path], transform: MapTransform | Compose) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        image = imread(self.images[idx])[None, None]
        image = torch.from_numpy(image).to(torch.float32)
        image = self.transform(image)
        return {"source": image, "target": image}


class LiveCellTestDataset(Dataset):
    """
    LiveCell dataset.

    :param list[Path] images: List of paths to single-page, single-channel TIFF files.
    :param MapTransform | Compose transform: Transform to apply to the dataset
    """

    def __init__(
        self,
        image_dir: Path,
        transform: MapTransform | Compose,
        annotations: Path,
        load_target: bool = False,
        load_labels: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(str(annotations))
        self.image_ids = list(self.coco.imgs.keys())
        self.load_target = load_target
        self.load_labels = load_labels

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Sample:
        image_id = self.image_ids[idx]
        file_name = self.coco.imgs[image_id]["file_name"]
        image_path = self.image_dir / file_name
        image = imread(image_path)[None, None]
        image = torch.from_numpy(image).to(torch.float32)
        sample = Sample(source=image)
        if self.load_target:
            sample["target"] = image
        if self.load_labels:
            anns = self.coco.loadAnns(self.coco.getAnnIds(image_id)) or []
            boxes = [torch.tensor(ann["bbox"]).to(torch.float32) for ann in anns]
            masks = [
                torch.from_numpy(self.coco.annToMask(ann)).to(torch.bool)
                for ann in anns
            ]
            dets = {
                "boxes": box_convert(torch.stack(boxes), in_fmt="xywh", out_fmt="xyxy"),
                "labels": torch.zeros(len(anns)).to(torch.uint8),
                "masks": torch.stack(masks),
            }
            sample["detections"] = dets
            sample["file_name"] = file_name
        self.transform(sample)
        return sample


class LiveCellDataModule(LightningDataModule):
    def __init__(
        self,
        train_val_images: Path | None = None,
        test_images: Path | None = None,
        train_annotations: Path | None = None,
        val_annotations: Path | None = None,
        test_annotations: Path | None = None,
        train_transforms: list[MapTransform] = [],
        val_transforms: list[MapTransform] = [],
        test_transforms: list[MapTransform] = [],
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_val_images = Path(train_val_images)
        if not self.train_val_images.is_dir():
            raise NotADirectoryError(str(train_val_images))
        self.test_images = Path(test_images)
        if not self.test_images.is_dir():
            raise NotADirectoryError(str(test_images))
        self.train_annotations = Path(train_annotations)
        if not self.train_annotations.is_file():
            raise FileNotFoundError(str(train_annotations))
        self.val_annotations = Path(val_annotations)
        if not self.val_annotations.is_file():
            raise FileNotFoundError(str(val_annotations))
        self.test_annotations = Path(test_annotations)
        if not self.test_annotations.is_file():
            raise FileNotFoundError(str(test_annotations))
        self.train_transforms = Compose(train_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._setup_fit()
        elif stage == "test":
            self._setup_test()

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

    def _setup_test(self) -> None:
        self.test_dataset = LiveCellTestDataset(
            self.test_images,
            transform=self.test_transforms,
            annotations=self.test_annotations,
            load_labels=True,
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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
