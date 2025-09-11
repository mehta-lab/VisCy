from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from iohub.ngff import Plate, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.transforms import Compose
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy.data.hcs import _read_norm_meta
from viscy.data.triplet import INDEX_COLUMNS


class ClassificationDataset(Dataset):
    """Dataset for cell classification tasks.

    A PyTorch Dataset that provides cell patches and labels for classification.
    Loads image patches from HCS OME-Zarr data based on cell annotations.

    Parameters
    ----------
    plate : Plate
        HCS OME-Zarr plate containing image data.
    annotation : pd.DataFrame
        DataFrame with cell annotations and labels.
    channel_name : str
        Name of the image channel to load.
    z_range : tuple[int, int]
        Range of Z slices to include (start, end).
    transform : Callable | None, optional
        Transform to apply to image patches.
    initial_yx_patch_size : tuple[int, int]
        Initial patch size in Y and X dimensions.
    return_indices : bool
        Whether to return cell indices with patches, by default False.
    """

    def __init__(
        self,
        plate: Plate,
        annotation: pd.DataFrame,
        channel_name: str,
        z_range: tuple[int, int],
        transform: Callable | None,
        initial_yx_patch_size: tuple[int, int],
        return_indices: bool = False,
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

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotation)

    def __getitem__(
        self, idx
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, int | str]]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[Tensor, Tensor] or tuple[Tensor, Tensor, dict[str, int | str]]
            Image tensor, label tensor, and optionally cell indices.
        """
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
        label = torch.tensor(row["infection_state"]).float()[None]
        if self.return_indices:
            return img, label, row[INDEX_COLUMNS].to_dict()
        else:
            return img, label


class ClassificationDataModule(LightningDataModule):
    """Lightning DataModule for cell classification tasks.

    Manages data loading and preprocessing for cell classification workflows.
    Handles train/validation splits and applies appropriate transforms.

    Parameters
    ----------
    image_path : Path
        Path to HCS OME-Zarr image data.
    annotation_path : Path
        Path to cell annotation CSV file.
    val_fovs : list[str], optional
        List of FOV names to use for validation.
    channel_name : str
        Name of the image channel to load.
    z_range : tuple[int, int]
        Range of Z slices to include (start, end).
    train_exlude_timepoints : list[int]
        Timepoints to exclude from training data.
    train_transforms : list[Callable], optional
        List of transforms to apply to training data.
    val_transforms : list[Callable], optional
        List of transforms to apply to validation data.
    initial_yx_patch_size : tuple[int, int]
        Initial patch size in Y and X dimensions.
    batch_size : int
        Batch size for data loading.
    num_workers : int
        Number of workers for data loading.
    """

    def __init__(
        self,
        image_path: Path,
        annotation_path: Path,
        val_fovs: list[str] | None,
        channel_name: str,
        z_range: tuple[int, int],
        train_exlude_timepoints: list[int],
        train_transforms: list[Callable] | None,
        val_transforms: list[Callable] | None,
        initial_yx_patch_size: tuple[int, int],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.val_fovs = val_fovs
        self.channel_name = channel_name
        self.z_range = z_range
        self.train_exlude_timepoints = train_exlude_timepoints
        self.train_transform = Compose(train_transforms)
        self.val_transform = Compose(val_transforms)
        self.initial_yx_patch_size = initial_yx_patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        )

    def setup(self, stage=None) -> None:
        """
        Set up datasets for the specified stage.

        Parameters
        ----------
        stage : str, optional
            Stage to set up for ('fit', 'validate', 'predict', 'test').

        Raises
        ------
        NotImplementedError
            If stage is 'test'.
        ValueError
            If stage is unknown.
        """
        plate = open_ome_zarr(self.image_path)
        all_fovs = ["/" + name for (name, _) in plate.positions()]
        annotation = pd.read_csv(self.annotation_path)
        for column in ("t", "y", "x"):
            annotation[column] = annotation[column].astype(int)
        if stage in (None, "fit", "validate"):
            train_fovs = list(set(all_fovs) - set(self.val_fovs))
            self.train_dataset = self._subset(
                plate,
                annotation,
                train_fovs,
                transform=self.train_transform,
                exclude_timepoints=self.train_exlude_timepoints,
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
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Create training data loader.

        Returns
        -------
        DataLoader
            Training data loader with shuffling enabled.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation data loader.

        Returns
        -------
        DataLoader
            Validation data loader without shuffling.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create prediction data loader.

        Returns
        -------
        DataLoader
            Prediction data loader without shuffling.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
