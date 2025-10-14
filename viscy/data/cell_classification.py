from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from iohub.ngff import Plate, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy.data.hcs import _read_norm_meta
from viscy.data.triplet import INDEX_COLUMNS
from viscy.data.typing import AnnotationColumns


class ClassificationDataset(Dataset):
    def __init__(
        self,
        plate: Plate,
        annotation: pd.DataFrame,
        channel_name: str,
        z_range: tuple[int, int],
        transform: Callable | None,
        initial_yx_patch_size: tuple[int, int],
        return_indices: bool = False,
        label_column: AnnotationColumns = "infection_state",
    ):
        self.plate = plate
        self.z_range = z_range
        self.initial_yx_patch_size = initial_yx_patch_size
        self.transform = transform
        self.channel_name = channel_name
        self.channel_index = plate.get_channel_index(channel_name)
        self.return_indices = return_indices
        y_exclude, x_exclude = (
            self.initial_yx_patch_size[0] // 2,
            self.initial_yx_patch_size[1] // 2,
        )
        example_image_shape = next(plate.positions())[1]["0"].shape
        y_range = (y_exclude, example_image_shape[-2] - y_exclude)
        x_range = (x_exclude, example_image_shape[-1] - x_exclude)
        self.annotation = annotation[
            annotation["y"].between(*y_range, inclusive="neither")
            & annotation["x"].between(*x_range, inclusive="neither")
        ]
        self.label_column = label_column

    def __len__(self):
        return len(self.annotation)

    def __getitem__(
        self, idx
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, int | str]]:
        row = self.annotation.iloc[idx]
        fov_name, t, y, x = row["fov_name"], row["t"], row["y"], row["x"]
        fov = self.plate[fov_name]
        y_half, x_half = (s // 2 for s in self.initial_yx_patch_size)
        image = torch.from_numpy(
            fov["0"][
                t,
                self.channel_index,
                slice(*self.z_range),
                slice(y - y_half, y + y_half),
                slice(x - x_half, x + x_half),
            ]
        ).float()[None]
        norm_meta = _read_norm_meta(fov)[self.channel_name]["fov_statistics"]
        img = (image - norm_meta["mean"]) / norm_meta["std"]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(row[self.label_column]).float()[None]
        if self.return_indices:
            return img, label, row[INDEX_COLUMNS].to_dict()
        else:
            return img, label


class ClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        image_path: Path,
        annotation_path: Path,
        val_fovs: list[str] | None,
        channel_name: str,
        z_range: tuple[int, int],
        train_exclude_timepoints: list[int],
        train_transforms: list[Callable] | None,
        val_transforms: list[Callable] | None,
        initial_yx_patch_size: tuple[int, int],
        batch_size: int,
        num_workers: int,
        label_column: str = "infection_state",
    ):
        super().__init__()
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.val_fovs = val_fovs
        self.channel_name = channel_name
        self.z_range = z_range
        self.train_exclude_timepoints = train_exclude_timepoints
        self.train_transform = Compose(train_transforms)
        self.val_transform = Compose(val_transforms)
        self.initial_yx_patch_size = initial_yx_patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_column = label_column

    def _subset(
        self,
        plate: Plate,
        annotation: pd.DataFrame,
        fov_names: list[str],
        transform: Callable | None,
        exclude_timepoints: list[int] = [],
        return_indices: bool = False,
    ) -> ClassificationDataset:
        if exclude_timepoints:
            filter_timepoints = annotation["t"].isin(exclude_timepoints)
            annotation = annotation[~filter_timepoints]
        return ClassificationDataset(
            plate=plate,
            annotation=annotation[annotation["fov_name"].isin(fov_names)],
            channel_name=self.channel_name,
            z_range=self.z_range,
            transform=transform,
            initial_yx_patch_size=self.initial_yx_patch_size,
            return_indices=return_indices,
            label_column=self.label_column,
        )

    def setup(self, stage=None):
        plate = open_ome_zarr(self.image_path)
        annotation = pd.read_csv(self.annotation_path)
        all_fovs = [name for (name, _) in plate.positions()]
        if annotation["fov_name"].iloc[0].startswith("/"):
            all_fovs = ["/" + name for name in all_fovs]
        if all_fovs[0].startswith("/"):
            if not self.val_fovs[0].startswith("/"):
                self.val_fovs = ["/" + name for name in self.val_fovs]
        else:
            if self.val_fovs[0].startswith("/"):
                self.val_fovs = [name[1:] for name in self.val_fovs]
        for column in ("t", "y", "x"):
            annotation[column] = annotation[column].astype(int)
        if stage in (None, "fit", "validate"):
            train_fovs = list(set(all_fovs) - set(self.val_fovs))
            self.train_dataset = self._subset(
                plate,
                annotation,
                train_fovs,
                transform=self.train_transform,
                exclude_timepoints=self.train_exclude_timepoints,
            )
            self.val_dataset = self._subset(
                plate,
                annotation,
                self.val_fovs,
                transform=self.val_transform,
                exclude_timepoints=[],
            )
        elif stage == "predict":
            self.predict_dataset = ClassificationDataset(
                plate=plate,
                annotation=annotation,
                channel_name=self.channel_name,
                z_range=self.z_range,
                transform=None,
                initial_yx_patch_size=self.initial_yx_patch_size,
                return_indices=True,
            )
        elif stage == "test":
            raise NotImplementedError("Test stage not implemented.")
        else:
            raise (f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
