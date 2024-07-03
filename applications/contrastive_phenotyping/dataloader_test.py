# %%
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from viscy.transforms import (
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
)
from monai.transforms import Compose
from iohub import open_ome_zarr
import pandas as pd
import warnings
import pytorch_lightning as pl

# from viscy.data.typing import Optional
from pathlib import Path

warnings.filterwarnings("ignore")


# %%
class OMEZarrDataset(Dataset):
    def __init__(
        self,
        base_path,
        channels,
        x,
        y,
        timesteps_csv_path,
        transform=None,
        z_range=None,
    ):
        self.base_path = base_path
        self.channels = channels
        self.x = x
        self.y = y
        self.z_range = z_range
        self.transform = transform
        self.ds = self.open_zarr_store(self.base_path)
        self.positions = list(self.ds.positions())
        self.timesteps_df = pd.read_csv(timesteps_csv_path)
        print(f"Initialized dataset with {len(self.positions)} positions.")

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        print(f"Opening Zarr store at {path} with layout '{layout}' and mode '{mode}'")
        return open_ome_zarr(path, layout=layout, mode=mode)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        anchor_position_path = self.positions[idx][0]
        anchor_data = self.load_data(anchor_position_path)

        positive_data = (
            self.transform({"image": anchor_data})["image"]
            if self.transform
            else anchor_data
        )
        if self.transform:
            print("Positive transformation applied")

        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, self.__len__() - 1)
        negative_position_path = self.positions[negative_idx][0]
        negative_data = self.load_data(negative_position_path)

        negative_data = (
            self.transform({"image": negative_data})["image"]
            if self.transform
            else negative_data
        )
        if self.transform:
            print("Negative transformation applied")

        print("shapes of tensors")
        print(torch.tensor(anchor_data).shape)
        print(torch.tensor(positive_data).shape)
        print(torch.tensor(negative_data).shape)
        return (
            torch.tensor(anchor_data),
            torch.tensor(positive_data),
            torch.tensor(negative_data),
        )

    def load_data(self, position_path):
        position = self.ds[position_path]
        print(f"Loading data from position: {position_path}")
        zarr_array = position["0"][:]
        print("Shape before:", zarr_array.shape)
        data = self.restructure_data(zarr_array, position_path)
        if self.z_range:
            data = data[:, self.z_range[0] : self.z_range[1], :, :]
        print("Shape after:", data.shape)
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


def get_transforms():
    transforms = Compose(
        [
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(0.2, 0.2),
                shear_range=(0.2, 0.2),
                scale_range=(0.2, 0.2),
            ),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.5,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.5, 2.0), prob=0.5),
        ]
    )
    return transforms


class OMEZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_path: str,
        channels: int,
        x: int,
        y: int,
        timesteps_csv_path: str,
        predict_base_path: str = None,
        train_split_ratio: float = 0.64,
        val_split_ratio: float = 0.16,
        batch_size: int = 4,
        num_workers: int = 8,
        z_range: tuple[int, int] = None,
        transform=None,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.channels = channels
        self.x = x
        self.y = y
        self.timesteps_csv_path = timesteps_csv_path
        self.predict_base_path = Path(predict_base_path) if predict_base_path else None
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.z_range = z_range
        self.transform = transform or get_transforms()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str = None):
        dataset = OMEZarrDataset(
            self.base_path,
            self.channels,
            self.x,
            self.y,
            self.timesteps_csv_path,
            transform=self.transform,
            z_range=self.z_range,
        )

        train_size = int(len(dataset) * self.train_split_ratio)
        val_size = int(len(dataset) * self.val_split_ratio)
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        )

        # setup prediction dataset (if needed)
        if stage == "predict" and self.predict_base_path:
            self.predict_dataset = OMEZarrDataset(
                self.predict_base_path,
                self.channels,
                self.x,
                self.y,
                self.timesteps_csv_path,
                transform=self.transform,
                z_range=self.z_range,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset not set up. Call setup(stage='predict') first."
            )
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# %%  Testing the DataModule

base_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/small_patch.zarr"
# predict_base_path = " "
channels = 2
x = 200
y = 200
z = 10
z_range = (0, 10)
batch_size = 4
timesteps_csv_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/6-patches/final_track_timesteps.csv"

data_module = OMEZarrDataModule(
    base_path=base_path,
    channels=channels,
    x=x,
    y=y,
    timesteps_csv_path=timesteps_csv_path,
    batch_size=batch_size,
    z_range=z_range,
)

# for train and val
data_module.setup()

print(
    f"Total dataset size: {len(data_module.train_dataset) + len(data_module.val_dataset) + len(data_module.test_dataset)}"
)
print(f"Training dataset size: {len(data_module.train_dataset)}")
print(f"Validation dataset size: {len(data_module.val_dataset)}")
print(f"Test dataset size: {len(data_module.test_dataset)}")

train_loader = data_module.train_dataloader()

print("Training DataLoader:")
for batch in train_loader:
    anchor_batch, positive_batch, negative_batch = batch
    print("Anchor batch shape:", anchor_batch.shape)
    print("Positive batch shape:", positive_batch.shape)
    print("Negative batch shape:", negative_batch.shape)
    break

val_loader = data_module.val_dataloader()

print("Validation DataLoader:")
for batch in val_loader:
    anchor_batch, positive_batch, negative_batch = batch
    print("Anchor batch shape:", anchor_batch.shape)
    print("Positive batch shape:", positive_batch.shape)
    print("Negative batch shape:", negative_batch.shape)
    break

test_loader = data_module.test_dataloader()

print("Test DataLoader:")
for batch in test_loader:
    anchor_batch, positive_batch, negative_batch = batch
    print("Anchor batch shape:", anchor_batch.shape)
    print("Positive batch shape:", positive_batch.shape)
    print("Negative batch shape:", negative_batch.shape)
    break

# Setup the DataModule for prediction
# data_module.setup(stage='predict')

# Get the predict DataLoader and print batch shapes
# predict_loader = data_module.predict_dataloader()
# print("Predict DataLoader:")
# for batch in predict_loader:
#     anchor_batch, positive_batch, negative_batch = batch
#     print("Anchor batch shape:", anchor_batch.shape)
#     print("Positive batch shape:", positive_batch.shape)
#     print("Negative batch shape:", negative_batch.shape)
#     break

# %%
