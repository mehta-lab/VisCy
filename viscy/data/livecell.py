from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from monai.transforms import Compose, MapTransform, Transform
from pycocotools.coco import COCO
from tifffile import imread
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert

from viscy.data.gpu_aug import GPUTransformDataModule
from viscy.data.typing import Sample

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy


class LiveCellDataset(Dataset):
    """
    LiveCell dataset.

    Parameters
    ----------
    images : list of Path
        List of paths to single-page, single-channel TIFF files.
    transform : Transform or Compose
        Transform to apply to the dataset.
    cache_map : DictProxy
        Shared dictionary for caching images.
    """

    def __init__(
        self,
        images: list[Path],
        transform: Transform | Compose,
        cache_map: DictProxy,
    ) -> None:
        self.images = images
        self.transform = transform
        self._cache_map = cache_map

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        name = self.images[idx]
        if name not in self._cache_map:
            image = imread(name)[None, None]
            image = torch.from_numpy(image).to(torch.float32)
            self._cache_map[name] = image
        else:
            image = self._cache_map[name]
        sample = Sample(source=image)
        sample = self.transform(sample)
        if not isinstance(sample, list):
            sample = [sample]
        return sample


class LiveCellTestDataset(Dataset):
    """
    LiveCell dataset.

    Parameters
    ----------
    image_dir : Path
        Directory containing the images.
    transform : MapTransform | Compose
        Transform to apply to the dataset.
    annotations : Path
        Path to the COCO annotations file.
    load_target : bool, optional
        Whether to load the target images (default is False).
    load_labels : bool, optional
        Whether to load the labels (default is False).
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


class LiveCellDataModule(GPUTransformDataModule):
    def __init__(
        self,
        train_val_images: Path | None = None,
        test_images: Path | None = None,
        train_annotations: Path | None = None,
        val_annotations: Path | None = None,
        test_annotations: Path | None = None,
        train_cpu_transforms: list[MapTransform] = [],
        val_cpu_transforms: list[MapTransform] = [],
        train_gpu_transforms: list[MapTransform] = [],
        val_gpu_transforms: list[MapTransform] = [],
        test_transforms: list[MapTransform] = [],
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        if train_val_images is not None:
            self.train_val_images = Path(train_val_images)
            if not self.train_val_images.is_dir():
                raise NotADirectoryError(str(train_val_images))
        if test_images is not None:
            self.test_images = Path(test_images)
            if not self.test_images.is_dir():
                raise NotADirectoryError(str(test_images))
        if train_annotations is not None:
            self.train_annotations = Path(train_annotations)
            if not self.train_annotations.is_file():
                raise FileNotFoundError(str(train_annotations))
        if val_annotations is not None:
            self.val_annotations = Path(val_annotations)
            if not self.val_annotations.is_file():
                raise FileNotFoundError(str(val_annotations))
        if test_annotations is not None:
            self.test_annotations = Path(test_annotations)
            if not self.test_annotations.is_file():
                raise FileNotFoundError(str(test_annotations))
        self._train_cpu_transforms = Compose(train_cpu_transforms)
        self._val_cpu_transforms = Compose(val_cpu_transforms)
        self._train_gpu_transforms = Compose(train_gpu_transforms)
        self._val_gpu_transforms = Compose(val_gpu_transforms)
        self.test_transforms = Compose(test_transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        if stage == "fit":
            self._setup_fit()
        elif stage == "test":
            self._setup_test()

    def _parse_image_names(self, annotations: Path) -> list[Path]:
        with open(annotations) as f:
            images = [f["file_name"] for f in json.load(f)["images"]]
        return sorted(images)

    def _setup_fit(self) -> None:
        cache_map = torch.multiprocessing.Manager().dict()
        train_images = self._parse_image_names(self.train_annotations)
        val_images = self._parse_image_names(self.val_annotations)
        self.train_dataset = LiveCellDataset(
            [self.train_val_images / f for f in train_images],
            transform=self.train_cpu_transforms,
            cache_map=cache_map,
        )
        self.val_dataset = LiveCellDataset(
            [self.train_val_images / f for f in val_images],
            transform=self.val_cpu_transforms,
            cache_map=cache_map,
        )

    def _setup_test(self) -> None:
        self.test_dataset = LiveCellTestDataset(
            self.test_images,
            transform=self.test_transforms,
            annotations=self.test_annotations,
            load_labels=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
