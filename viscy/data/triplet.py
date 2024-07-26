import logging
import random
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
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

from viscy.data.hcs import HCSDataModule
from viscy.data.typing import ChannelMap, DictTransform, HCSStackIndex, NormMeta, Sample

_logger = logging.getLogger("lightning.pytorch")


def _scatter_channels(channel_names: list[str], patch: Tensor) -> dict[str, Tensor]:
    return {name: data for name, data in zip(channel_names, patch)}


def _gather_channels(patch_channels: dict[str, Tensor]) -> Tensor:
    """
    :param dict[str, Tensor] patch_channels: dictionary of single-channel tensors
    :return Tensor: Multi-channel tensor
    """
    return torch.stack(list(patch_channels.values()), dim=1)


def _transform_channel_wise(
    transform: DictTransform, channel_names: list[str], patch: Tensor
) -> Tensor:
    return _gather_channels(transform(_scatter_channels(channel_names, patch)))


class TripletDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        tracks_tables: list[pd.DataFrame],
        channel_names: list[str],
        yx_patch_size: tuple[int, int],
        z_range: slice,
        anchor_transform: DictTransform | None = None,
        positive_transform: DictTransform | None = None,
        negative_transform: DictTransform | None = None,
        fit: bool = True,
    ) -> None:
        self.positions = positions
        self.channel_names = channel_names
        self.channel_indices = [
            positions[0].get_channel_index(ch) for ch in channel_names
        ]
        self.z_range = z_range
        self.anchor_transform = anchor_transform
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform
        self.fit = fit
        self.yx_patch_size = yx_patch_size
        self.tracks = self._filter_tracks(tracks_tables)

    def _filter_tracks(self, tracks_tables: list[pd.DataFrame]) -> pd.DataFrame:
        filtered_tracks = []
        y_exclude, x_exclude = (self.yx_patch_size[0] // 2, self.yx_patch_size[1] // 2)
        for pos, tracks in zip(self.positions, tracks_tables, strict=True):
            tracks["position"] = [pos] * len(tracks)
            tracks["fov_name"] = pos.zgroup.name
            tracks["global_track_id"] = tracks["fov_name"].str.cat(
                tracks["track_id"].astype(str), sep="_"
            )
            image: ImageArray = pos["0"]
            if self.z_range.stop > image.slices:
                raise ValueError(
                    f"Z range {self.z_range} exceeds image with Z={image.slices}"
                )
            y_range = (y_exclude, image.height - y_exclude)
            x_range = (x_exclude, image.width - x_exclude)
            filtered_tracks.append(
                tracks[
                    tracks["y"].between(*y_range, inclusive="neither")
                    & tracks["x"].between(*x_range, inclusive="neither")
                ]
            )
        return pd.concat(filtered_tracks).reset_index(drop=True)

    def __len__(self):
        return len(self.tracks)

    def _sample_negative(self, anchor_row: pd.Series) -> pd.Series:
        candidates: pd.DataFrame = self.tracks[
            (self.tracks["global_track_id"] != anchor_row["global_track_id"])
        ]
        # NOTE: Random sampling
        # reproducibility relies on setting a global seed for numpy
        return candidates.sample(n=1).iloc[0]

    def _slice_patch(self, track_row: pd.Series) -> Tensor:
        image: ImageArray = track_row["position"]["0"]
        t = track_row["t"]
        y_center = track_row["y"]
        x_center = track_row["x"]
        y_half, x_half = (d // 2 for d in self.yx_patch_size)
        patch = image.oindex[
            slice(t, t + 1),
            [int(i) for i in self.channel_indices],
            self.z_range,
            slice(y_center - y_half, y_center + y_half),
            slice(x_center - x_half, x_center + x_half),
        ]
        return torch.from_numpy(patch)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        anchor_row = self.tracks.iloc[index]
        anchor_patch = self._slice_patch(anchor_row)
        if self.fit:
            positive_patch = anchor_patch.clone()
            if self.positive_transform:
                positive_patch = _transform_channel_wise(
                    transform=self.positive_transform,
                    channel_names=self.channel_names,
                    patch=positive_patch,
                )
            negative_row = self._sample_negative(anchor_row)
            negative_patch = self._slice_patch(negative_row)
            if self.negative_transform:
                negative_patch = _transform_channel_wise(
                    transform=self.negative_transform,
                    channel_names=self.channel_names,
                    patch=negative_patch,
                )
        if self.anchor_transform:
            anchor_patch = _transform_channel_wise(
                transform=self.anchor_transform,
                channel_names=self.channel_names,
                patch=anchor_patch,
            )
        if self.fit:
            return (anchor_patch, positive_patch, negative_patch)
        else:
            return (anchor_patch,)


class TripletDataModule(HCSDataModule):
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        source_channel: str | Sequence[str],
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        z_range: tuple[int, int] = None,
        yx_patch_size: tuple[int, int] = (256, 256),
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        caching: bool = False,
    ):
        super().__init__(
            data_path=data_path,
            source_channel=source_channel,
            target_channel="",
            z_window_size=z_range[1] - z_range[0],
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            architecture="UNeXt2",
            yx_patch_size=yx_patch_size,
            normalizations=normalizations,
            augmentations=augmentations,
            caching=caching,
        )
        self.z_range = z_range

    def _setup_fit(self, dataset_settings: NormMeta):
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
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        )

    def _setup_predict(self, dataset_settings: NormMeta):
        # setup prediction dataset
        self.predict_dataset = PredictDataset(
            self.predict_base_path,
            self.channels,
            self.x,
            self.y,
            timesteps_csv_path=self.timesteps_csv_path,
            channel_names=self.channel_names,
            z_range=self.z_range,
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
