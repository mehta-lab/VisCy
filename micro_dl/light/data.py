from typing import Callable, Literal, Union

import numpy as np
import torch
from iohub.ngff import ImageArray, Plate, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    NormalizeIntensity,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianSmoothd,
    RandWeightedCropd,
)
from torch.utils.data import DataLoader, Dataset, Subset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        plate: Plate,
        source_channel: str,
        target_channel: str,
        z_window_size: int,
        transform: Callable = None,
        target_center_slice_only: bool = True,
        normalize_intensity: bool = True,
    ) -> None:
        super().__init__()
        self.plate = plate
        self.source_ch_idx = plate.get_channel_index(source_channel)
        self.target_ch_idx = plate.get_channel_index(target_channel)
        self.z_window_size = z_window_size
        self.transform = transform
        self.target_center_slice_only = target_center_slice_only
        self.normalize_intensity = normalize_intensity
        self._count_windows()
        self._get_normalizer(source_channel, target_channel)

    def _count_windows(self) -> None:
        w = 0
        self.windows = {}
        for _, fov in self.plate.positions():
            ts = fov["0"].frames
            zs = fov["0"].slices - self.z_window_size + 1
            w += ts * zs
            self.windows[w] = fov
        self._max_window = w

    def _get_normalizer(self, source_channel: str, target_channel: str):
        norm_meta = self.plate.zattrs["normalization"]
        self.source_normalizer = NormalizeIntensity(
            subtrahend=norm_meta[source_channel]["dataset_statistics"]["median"],
            divisor=norm_meta[source_channel]["dataset_statistics"]["iqr"],
        )
        self.target_normalizer = NormalizeIntensity(
            subtrahend=norm_meta[target_channel]["dataset_statistics"]["median"],
            divisor=norm_meta[target_channel]["dataset_statistics"]["iqr"],
        )

    def _find_window(self, index: int) -> tuple[int, int]:
        window_keys = list(self.windows.keys())
        window_idx = sorted(window_keys + [index + 1]).index(index + 1)
        w = window_keys[window_idx]
        tz = index - window_keys[window_idx - 1] if window_idx > 0 else index
        return w, tz

    def _read_img_window(self, img: ImageArray, ch_idx: int, tz: int) -> torch.Tensor:
        zs = img.slices - self.z_window_size + 1
        t = (tz + zs) // zs - 1
        z = tz - t * zs
        selection = (int(t), int(ch_idx), slice(z, z + self.z_window_size))
        data = img[selection][np.newaxis]
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self._max_window

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        w, tz = self._find_window(index)
        img = self.windows[w]["0"]
        source = self.source_normalizer(
            self._read_img_window(img, self.source_ch_idx, tz)
        )
        target = self.target_normalizer(
            self._read_img_window(img, self.target_ch_idx, tz)
        )
        sample = {"source": source, "target": target}
        if self.transform:
            sample = self.transform(sample)
        if isinstance(sample, list):
            sample = sample[0]
        if self.target_center_slice_only:
            sample["target"] = sample["target"][:, self.z_window_size // 2][:, None]
        return sample

    def __del__(self):
        self.plate.close()


class HCSDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        source_channel: str,
        target_channel: str,
        z_window_size: int,
        split_ratio: float,
        batch_size: int = 16,
        num_workers: int = 8,
        yx_patch_size: tuple[int, int] = (256, 256),
        augment: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.source_channel = source_channel
        self.target_channel = target_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.z_window_size = z_window_size
        self.split_ratio = split_ratio
        self.yx_patch_size = yx_patch_size
        self.augment = augment

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        # train/val split
        if stage in (None, "fit", "validate"):
            # set training stage transforms
            fit_transform = self._fit_transform()
            train_transform = self._train_transform() + fit_transform
            plate = open_ome_zarr(self.data_path, mode="r")
            whole_train_dataset = SlidingWindowDataset(
                plate,
                source_channel=self.source_channel,
                target_channel=self.target_channel,
                z_window_size=self.z_window_size,
                transform=Compose(train_transform),
            )
            whole_val_dataset = SlidingWindowDataset(
                plate,
                source_channel=self.source_channel,
                target_channel=self.target_channel,
                z_window_size=self.z_window_size,
                transform=Compose(fit_transform),
            )
            # randomness is handled globally
            indices = torch.randperm(len(whole_train_dataset))
            self.sep = int(len(indices) * self.split_ratio)
            self.train_dataset = Subset(whole_train_dataset, indices[: self.sep])
            self.val_dataset = Subset(whole_val_dataset, indices[self.sep :])
        # test/predict stage
        else:
            raise NotImplementedError(f"{stage} stage")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def _fit_transform(self):
        return [
            CenterSpatialCropd(
                keys=["source", "target"],
                roi_size=(
                    -1,
                    self.yx_patch_size[0],
                    self.yx_patch_size[1],
                ),
            )
        ]

    def _train_transform(self) -> list[Callable]:
        transforms = [
            RandWeightedCropd(
                keys=["source", "target"],
                w_key="target",
                spatial_size=(-1, self.yx_patch_size[0] * 2, self.yx_patch_size[1] * 2),
                num_samples=1,
            )
        ]
        if self.augment:
            transforms.extend(
                [
                    RandAffined(
                        keys=["source", "target"],
                        prob=0.5,
                        rotate_range=(np.pi, 0, 0),
                        shear_range=(0, (0.05), (0.05)),
                        scale_range=(0, 0.2, 0.2),
                    ),
                    RandAdjustContrastd(keys=["source"], prob=0.1, gamma=(0.75, 1.5)),
                    RandGaussianSmoothd(
                        keys=["source"],
                        prob=0.2,
                        sigma_x=(0.05, 0.25),
                        sigma_y=(0.05, 0.25),
                        sigma_z=((0.05, 0.25)),
                    ),
                ]
            )
        return transforms
