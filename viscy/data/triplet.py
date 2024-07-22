import logging
import random
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    MultiSampleTrait,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from viscy.data.typing import ChannelMap, HCSStackIndex, NormMeta, Sample

_logger = logging.getLogger("lightning.pytorch")


# dataloader for organelle phenotyping
class ContrastiveDataset(Dataset):
    def __init__(
        self,
        base_path,
        channels,
        x,
        y,
        timesteps_csv_path,
        channel_names,
        transform=None,
        z_range=None,
    ):
        self.base_path = base_path
        self.channels = channels
        self.x = x
        self.y = y
        self.z_range = z_range
        self.channel_names = channel_names
        self.transform = get_transforms()
        self.ds = self.open_zarr_store(self.base_path)
        self.positions = list(self.ds.positions())
        self.timesteps_df = pd.read_csv(timesteps_csv_path)
        self.channel_indices = [
            self.ds.channel_names.index(channel) for channel in self.channel_names
        ]
        _logger.debug(f"Initialized dataset with {len(self.positions)} positions.")
        _logger.debug(f"Channel indices: {self.channel_indices}")

    def compute_statistics(self):
        stats = {
            channel: {"mean": 0, "sum_sq_diff": 0, "min": np.inf, "max": -np.inf}
            for channel in self.channel_names
        }
        count = 0
        total_elements = 0

        for idx in range(len(self.positions)):
            position_path = self.positions[idx][0]
            data = self.load_data(position_path)
            for i, channel in enumerate(self.channel_names):
                channel_data = data[i]
                mean = np.mean(channel_data)
                stats[channel]["mean"] += mean
                stats[channel]["min"] = min(stats[channel]["min"], np.min(channel_data))
                stats[channel]["max"] = max(stats[channel]["max"], np.max(channel_data))
                stats[channel]["sum_sq_diff"] += np.sum((channel_data - mean) ** 2)
            count += 1
            total_elements += np.prod(channel_data.shape)

        for channel in self.channel_names:
            stats[channel]["mean"] /= count
            stats[channel]["std"] = np.sqrt(
                stats[channel]["sum_sq_diff"] / total_elements
            )
            del stats[channel]["sum_sq_diff"]

        _logger.debug("done!")
        return stats

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        # _logger.debug(f"Opening Zarr store at {path} with layout '{layout}' and mode '{mode}'")
        return open_ome_zarr(path, layout=layout, mode=mode)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        anchor_position_path = self.positions[idx][0]
        anchor_data = self.load_data(anchor_position_path)
        anchor_data = self.normalize_data(anchor_data)

        positive_data = self.apply_channel_transforms(anchor_data)
        positive_data = self.normalize_data(positive_data)

        # if self.transform:
        #     _logger.debug("Positive transformation applied")

        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, self.__len__() - 1)
        negative_position_path = self.positions[negative_idx][0]
        negative_data = self.load_data(negative_position_path)
        negative_data = self.normalize_data(negative_data)

        negative_data = self.apply_channel_transforms(negative_data)
        negative_data = self.normalize_data(negative_data)

        # if self.transform:
        #     _logger.debug("Negative transformation applied")

        # _logger.debug("shapes of tensors")
        # _logger.debug(torch.tensor(anchor_data).shape)
        # _logger.debug(torch.tensor(positive_data).shape)
        # _logger.debug(torch.tensor(negative_data).shape)
        return (
            torch.tensor(anchor_data, dtype=torch.float32),
            torch.tensor(positive_data, dtype=torch.float32),
            torch.tensor(negative_data, dtype=torch.float32),
        )

    def load_data(self, position_path):
        position = self.ds[position_path]
        # _logger.debug(f"Loading data from position: {position_path}")

        zarr_array = position["0"][:]
        # _logger.debug("Shape before:", zarr_array.shape)
        data = self.restructure_data(zarr_array, position_path)
        data = data[self.channel_indices, self.z_range[0] : self.z_range[1], :, :]

        # _logger.debug("shape after!")
        # _logger.debug(data.shape)
        return data

    def restructure_data(self, data, position_path):
        # Extract row, column, fov, and cell_id from position_path
        parts = position_path.split("/")
        row = parts[0]
        column = parts[1]
        fov_cell = parts[2]

        fov = int(fov_cell.split("fov")[1].split("cell")[0])
        cell_id = int(fov_cell.split("cell")[1])

        extracted_combined = f"{row}/{column}/fov{fov}cell{cell_id}"

        matched_rows = self.timesteps_df[
            self.timesteps_df.apply(
                lambda x: f"{x['Row']}/{x['Column']}/fov{x['FOV']}cell{x['Cell ID']}",
                axis=1,
            )
            == extracted_combined
        ]

        if matched_rows.empty:
            raise ValueError(
                f"No matching entry found for position path: {position_path}"
            )

        start_time = matched_rows["Start Time"].values[0]
        end_time = matched_rows["End Time"].values[0]

        random_timestep = np.random.randint(start_time, end_time)

        reshaped_data = data[random_timestep]
        return reshaped_data

    def normalize_data(self, data):
        normalized_data = np.empty_like(data)
        for i in range(data.shape[0]):  # iterate over each channel
            channel_data = data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            normalized_data[i] = (channel_data - mean) / (std + 1e-6)
        return normalized_data

    def apply_channel_transforms(self, data):
        transformed_data = np.empty_like(data)
        for i, channel_name in enumerate(self.channel_names):
            channel_data = data[i]
            transform = self.transform[channel_name]
            transformed_data[i] = transform({"image": channel_data})["image"]
            # _logger.debug(f"transformed {channel_name}")
        return transformed_data


def get_transforms():
    rfp_transforms = Compose(
        [
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.75, 1.25)),
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0.1, 0.1),
                shear_range=(0.1, 0.1),
                scale_range=(0.1, 0.1),
            ),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.5,
                sigma_x=(0.1, 0.3),
                sigma_y=(0.1, 0.3),
                sigma_z=(0.1, 0.3),
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.85, 1.15), prob=0.5),
        ]
    )

    phase_transforms = Compose(
        [
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.97, 1.03)),
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0.05, 0.05),
                shear_range=(0.05, 0.05),
                scale_range=(0.05, 0.05),
            ),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.005),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.5,
                sigma_x=(0.03, 0.05),
                sigma_y=(0.03, 0.05),
                sigma_z=(0.03, 0.05),
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.97, 1.03), prob=0.5),
        ]
    )

    return {"RFP": rfp_transforms, "Phase3D": phase_transforms}


class ContrastiveDataModule(LightningDataModule):
    def __init__(
        self,
        base_path: str,
        channels: int,
        x: int,
        y: int,
        timesteps_csv_path: str,
        channel_names: list,
        transform=None,
        predict_base_path: str = None,
        train_split_ratio: float = 0.64,
        val_split_ratio: float = 0.16,
        batch_size: int = 4,
        num_workers: int = 8,
        z_range: tuple[int, int] = None,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.channels = channels
        self.x = x
        self.y = y
        self.timesteps_csv_path = timesteps_csv_path
        self.channel_names = channel_names
        self.transform = get_transforms()
        self.predict_base_path = Path(predict_base_path) if predict_base_path else None
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.z_range = z_range
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str = None):
        if stage == "fit":
            dataset = ContrastiveDataset(
                self.base_path,
                self.channels,
                self.x,
                self.y,
                self.timesteps_csv_path,
                channel_names=self.channel_names,
                transform=self.transform,
                z_range=self.z_range,
            )

            train_size = int(len(dataset) * self.train_split_ratio)
            val_size = int(len(dataset) * self.val_split_ratio)
            test_size = len(dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = (
                torch.utils.data.random_split(
                    dataset, [train_size, val_size, test_size]
                )
            )

        # setup prediction dataset
        if stage == "predict" and self.predict_base_path:
            _logger.debug("setting up!")
            self.predict_dataset = PredictDataset(
                self.predict_base_path,
                self.channels,
                self.x,
                self.y,
                timesteps_csv_path=self.timesteps_csv_path,
                channel_names=self.channel_names,
                z_range=self.z_range,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        _logger.debug("running predict DataLoader!")
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset not set up. Call setup(stage='predict') first."
            )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # False shuffle for prediction
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )


class PredictDataset(Dataset):
    def __init__(
        self,
        base_path,
        channels,
        x,
        y,
        timesteps_csv_path,
        channel_names,
        z_range=None,
    ):
        self.base_path = base_path
        self.channels = channels
        self.x = x
        self.y = y
        self.z_range = z_range
        self.channel_names = channel_names
        self.ds = self.open_zarr_store(self.base_path)
        self.timesteps_csv_path = timesteps_csv_path
        self.timesteps_df = pd.read_csv(timesteps_csv_path)
        self.positions = list(self.ds.positions())
        self.channel_indices = [
            self.ds.channel_names.index(channel) for channel in self.channel_names
        ]
        _logger.debug("channel indices!")
        _logger.debug(self.channel_indices)
        _logger.debug(
            f"Initialized predict dataset with {len(self.positions)} positions."
        )

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        return open_ome_zarr(path, layout=layout, mode=mode)

    # def get_positions_from_csv(self):
    #     positions = []
    #     #self.timesteps_df = pd.read_csv(self.timesteps_csv_path)
    #     for idx, row in self.timesteps_df.iterrows():
    #         position_path = f"{row['Row']}/{row['Column']}/fov{row['FOV']}cell{row['Cell ID']}"
    #         positions.append((position_path, row['Random Timestep']))
    #     #_logger.debug(positions)
    #     return positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position_path = self.positions[idx][0]
        # _logger.debug(f"Position path: {position_path}")
        data = self.load_data(position_path)
        data = self.normalize_data(data)

        return torch.tensor(data, dtype=torch.float32), (position_path)

    # double check printing order
    def load_data(self, position_path):
        position = self.ds[position_path]
        # _logger.debug(f"Loading data for position path: {position_path}")
        zarr_array = position["0"][:]

        parts = position_path.split("/")
        row = parts[0]
        column = parts[1]
        fov_cell = parts[2]
        fov = int(fov_cell.split("fov")[1].split("cell")[0])
        cell_id = int(fov_cell.split("cell")[1])

        combined_id = f"{row}/{column}/fov{fov}cell{cell_id}"
        matched_rows = self.timesteps_df[
            self.timesteps_df.apply(
                lambda x: f"{x['Row']}/{x['Column']}/fov{x['FOV']}cell{x['Cell ID']}",
                axis=1,
            )
            == combined_id
        ]

        if matched_rows.empty:
            raise ValueError(
                f"No matching entry found for position path: {position_path}"
            )

        random_timestep = matched_rows["Random Timestep"].values[0]
        data = zarr_array[
            random_timestep,
            self.channel_indices,
            self.z_range[0] : self.z_range[1],
            :,
            :,
        ]
        return data

    def normalize_data(self, data):
        normalized_data = np.empty_like(data)
        for i in range(data.shape[0]):  # iterate over each channel
            channel_data = data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            normalized_data[i] = (channel_data - mean) / (std + 1e-6)
        return normalized_data
