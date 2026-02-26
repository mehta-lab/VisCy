"""LiveCell dataset and data module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from monai.transforms import Compose, MapTransform, Transform
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

try:
    from tifffile import imread
except ImportError:
    imread = None

try:
    from torchvision.ops import box_convert
except ImportError:
    box_convert = None

from viscy_data._typing import Sample
from viscy_data.gpu_aug import GPUTransformDataModule

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy


class LiveCellDataset(Dataset):
    """LiveCell dataset.

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
        if COCO is None or imread is None or box_convert is None:
            missing = []
            if COCO is None:
                missing.append("pycocotools")
            if imread is None:
                missing.append("tifffile")
            if box_convert is None:
                missing.append("torchvision")
            raise ImportError(
                f"{', '.join(missing)} required for LiveCellDataset. Install with: pip install 'viscy-data[livecell]'"
            )
        self.images = images
        self.transform = transform
        self._cache_map = cache_map

    def __len__(self) -> int:
        """Return total number of images."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        """Return a sample for the given index, using cache when available."""
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
    """LiveCell test dataset.

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
        if COCO is None or imread is None or box_convert is None:
            missing = []
            if COCO is None:
                missing.append("pycocotools")
            if imread is None:
                missing.append("tifffile")
            if box_convert is None:
                missing.append("torchvision")
            raise ImportError(
                f"{', '.join(missing)} required for LiveCellTestDataset. "
                "Install with: pip install 'viscy-data[livecell]'"
            )
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(str(annotations))
        self.image_ids = list(self.coco.imgs.keys())
        self.load_target = load_target
        self.load_labels = load_labels

    def __len__(self) -> int:
        """Return total number of test images."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Sample:
        """Return a sample for the given index."""
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
            if anns:
                boxes = [torch.tensor(ann["bbox"]).to(torch.float32) for ann in anns]
                masks = [torch.from_numpy(self.coco.annToMask(ann)).to(torch.bool) for ann in anns]
                dets = {
                    "boxes": box_convert(torch.stack(boxes), in_fmt="xywh", out_fmt="xyxy"),
                    "labels": torch.zeros(len(anns)).to(torch.uint8),
                    "masks": torch.stack(masks),
                }
            else:
                h, w = self.coco.imgs[image_id]["height"], self.coco.imgs[image_id]["width"]
                dets = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.uint8),
                    "masks": torch.zeros((0, h, w), dtype=torch.bool),
                }
            sample["detections"] = dets
            sample["file_name"] = file_name
        sample = self.transform(sample)
        return sample


class LiveCellDataModule(GPUTransformDataModule):
    """Data module for LiveCell training and evaluation.

    Parameters
    ----------
    train_val_images : Path | None, optional
        Path to the training/validation images directory.
    test_images : Path | None, optional
        Path to the test images directory.
    train_annotations : Path | None, optional
        Path to the training COCO annotations file.
    val_annotations : Path | None, optional
        Path to the validation COCO annotations file.
    test_annotations : Path | None, optional
        Path to the test COCO annotations file.
    train_cpu_transforms : list[MapTransform], optional
        CPU transforms for training.
    val_cpu_transforms : list[MapTransform], optional
        CPU transforms for validation.
    train_gpu_transforms : list[MapTransform], optional
        GPU transforms for training.
    val_gpu_transforms : list[MapTransform], optional
        GPU transforms for validation.
    test_transforms : list[MapTransform], optional
        Transforms for test stage.
    batch_size : int, optional
        Batch size, by default 16.
    num_workers : int, optional
        Number of dataloading workers, by default 8.
    pin_memory : bool, optional
        Pin memory for dataloaders, by default True.
    """

    def __init__(
        self,
        train_val_images: Path | None = None,
        test_images: Path | None = None,
        train_annotations: Path | None = None,
        val_annotations: Path | None = None,
        test_annotations: Path | None = None,
        train_cpu_transforms: list[MapTransform] | None = None,
        val_cpu_transforms: list[MapTransform] | None = None,
        train_gpu_transforms: list[MapTransform] | None = None,
        val_gpu_transforms: list[MapTransform] | None = None,
        test_transforms: list[MapTransform] | None = None,
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
        self._train_cpu_transforms = Compose(train_cpu_transforms or [])
        self._val_cpu_transforms = Compose(val_cpu_transforms or [])
        self._train_gpu_transforms = Compose(train_gpu_transforms or [])
        self._val_gpu_transforms = Compose(val_gpu_transforms or [])
        self.test_transforms = Compose(test_transforms or [])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = None

    @property
    def train_cpu_transforms(self) -> Compose:
        """Return training CPU transforms."""
        return self._train_cpu_transforms

    @property
    def val_cpu_transforms(self) -> Compose:
        """Return validation CPU transforms."""
        return self._val_cpu_transforms

    @property
    def train_gpu_transforms(self) -> Compose:
        """Return training GPU transforms."""
        return self._train_gpu_transforms

    @property
    def val_gpu_transforms(self) -> Compose:
        """Return validation GPU transforms."""
        return self._val_gpu_transforms

    def setup(self, stage: str) -> None:
        """Set up datasets for the given stage."""
        if stage == "fit":
            self._setup_fit()
        elif stage == "test":
            self._setup_test()

    def _parse_image_names(self, annotations: Path) -> list[str]:
        """Parse image file names from COCO annotations."""
        with open(annotations) as f:
            images = [f["file_name"] for f in json.load(f)["images"]]
        return sorted(images)

    def _setup_fit(self) -> None:
        """Set up training and validation datasets."""
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
        """Set up test dataset."""
        self.test_dataset = LiveCellTestDataset(
            self.test_images,
            transform=self.test_transforms,
            annotations=self.test_annotations,
            load_labels=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
