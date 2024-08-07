import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from iohub.ngff import ImageArray, Position, open_ome_zarr
from monai.transforms import Compose, MapTransform
from torch import Tensor
from torch.utils.data import Dataset

from viscy.data.hcs import HCSDataModule, _read_norm_meta
from viscy.data.typing import DictTransform, NormMeta, TripletSample

_logger = logging.getLogger("lightning.pytorch")


def _scatter_channels(
    channel_names: list[str], patch: Tensor, norm_meta: NormMeta | None
) -> dict[str, Tensor | NormMeta] | dict[str, Tensor]:
    channels = {name: data[None] for name, data in zip(channel_names, patch)}
    if norm_meta is not None:
        channels |= {"norm_meta": norm_meta}
    return channels


def _gather_channels(patch_channels: dict[str, Tensor | NormMeta]) -> Tensor:
    """
    :param dict[str, Tensor | NormMeta] patch_channels: dictionary of single-channel tensors
    :return Tensor: Multi-channel tensor
    """
    patch_channels.pop("norm_meta", None)
    return torch.cat(list(patch_channels.values()), dim=0)


def _transform_channel_wise(
    transform: DictTransform,
    channel_names: list[str],
    patch: Tensor,
    norm_meta: NormMeta | None,
) -> Tensor:
    return _gather_channels(
        transform(_scatter_channels(channel_names, patch, norm_meta))
    )


class TripletDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        tracks_tables: list[pd.DataFrame],
        channel_names: list[str],
        initial_yx_patch_size: tuple[int, int],
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
        self.yx_patch_size = initial_yx_patch_size
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
        # this is to avoid combinatorial length growth at fitting time
        # since each cell can pair with any other cell
        # (3e4 instances will make 1e9 pairs)
        # reproducibility relies on setting a global seed for numpy
        return candidates.sample(n=1).iloc[0]

    def _slice_patch(self, track_row: pd.Series) -> tuple[Tensor, NormMeta | None]:
        position: Position = track_row["position"]
        image = position["0"]
        time = track_row["t"]
        y_center = track_row["y"]
        x_center = track_row["x"]
        y_half, x_half = (d // 2 for d in self.yx_patch_size)
        patch = image.oindex[
            time,
            [int(i) for i in self.channel_indices],
            self.z_range,
            slice(y_center - y_half, y_center + y_half),
            slice(x_center - x_half, x_center + x_half),
        ]
        return torch.from_numpy(patch), _read_norm_meta(position)

    def __getitem__(self, index: int) -> TripletSample:
        anchor_row = self.tracks.iloc[index]
        anchor_patch, anchor_norm = self._slice_patch(anchor_row)
        if self.fit:
            positive_patch = anchor_patch.clone()
            if self.positive_transform:
                positive_patch = _transform_channel_wise(
                    transform=self.positive_transform,
                    channel_names=self.channel_names,
                    patch=positive_patch,
                    norm_meta=anchor_norm,
                )
            negative_row = self._sample_negative(anchor_row)
            negative_patch, negetive_norm = self._slice_patch(negative_row)
            if self.negative_transform:
                negative_patch = _transform_channel_wise(
                    transform=self.negative_transform,
                    channel_names=self.channel_names,
                    patch=negative_patch,
                    norm_meta=negetive_norm,
                )
        if self.anchor_transform:
            anchor_patch = _transform_channel_wise(
                transform=self.anchor_transform,
                channel_names=self.channel_names,
                patch=anchor_patch,
                norm_meta=anchor_norm,
            )

        sample = {
            "anchor": anchor_patch,
            "index": anchor_row[["fov_name", "id"]].to_dict(),
        }
        if self.fit:
            sample.update(
                {
                    "positive": positive_patch,
                    "negative": negative_patch,
                }
            )
        return sample


class TripletDataModule(HCSDataModule):
    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        source_channel: str | Sequence[str],
        z_range: tuple[int, int],
        initial_yx_patch_size: tuple[int, int] = (512, 512),
        final_yx_patch_size: tuple[int, int] = (224, 224),
        split_ratio: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 8,
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        caching: bool = False,
    ):
        """Lightning data module for triplet sampling of patches.

        :param str data_path: Image dataset path
        :param str tracks_path: Tracks labels dataset path
        :param str | Sequence[str] source_channel: list of input channel names
        :param tuple[int, int] z_range: range of valid z-slices
        :param tuple[int, int] initial_yx_patch_size:
            XY size of the initially sampled image patch,
            defaults to (384, 384)
        :param tuple[int, int] final_yx_patch_size: output patch size,
            defaults to (256, 256)
        :param float split_ratio: ratio of training samples, defaults to 0.8
        :param int batch_size: batch size, defaults to 16
        :param int num_workers: number of data-loading workers, defaults to 8
        :param list[MapTransform] normalizations: list of normalization transforms,
            defaults to []
        :param list[MapTransform] augmentations: list of augmentation transforms,
            defaults to []
        :param bool caching: whether to cache the dataset, defaults to False
        """
        super().__init__(
            data_path=data_path,
            source_channel=source_channel,
            target_channel=[],
            z_window_size=z_range[1] - z_range[0],
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            architecture="UNeXt2",
            yx_patch_size=final_yx_patch_size,
            normalizations=normalizations,
            augmentations=augmentations,
            caching=caching,
        )
        self.z_range = slice(*z_range)
        self.tracks_path = Path(tracks_path)
        self.initial_yx_patch_size = initial_yx_patch_size

    def _align_tracks_tables_with_positions(
        self,
    ) -> tuple[list[Position], list[pd.DataFrame]]:
        positions = []
        tracks_tables = []
        images_plate = open_ome_zarr(self.data_path)
        for fov_name, _ in open_ome_zarr(self.tracks_path).positions():
            positions.append(images_plate[fov_name])
            tracks_df = pd.read_csv(
                next((self.tracks_path / fov_name).glob("*.csv"))
            ).astype(int)
            tracks_tables.append(tracks_df)

        return positions, tracks_tables

    @property
    def _base_dataset_settings(self) -> dict:
        return {
            "channel_names": self.source_channel,
            "z_range": self.z_range,
        }

    def _setup_fit(self, dataset_settings: dict):
        augment_transform, no_aug_transform = self._fit_transform()
        positions, tracks_tables = self._align_tracks_tables_with_positions()
        shuffled_indices = self._set_fit_global_state(len(positions))
        positions = [positions[i] for i in shuffled_indices]
        tracks_tables = [tracks_tables[i] for i in shuffled_indices]

        num_train_fovs = int(len(positions) * self.split_ratio)
        train_positions = positions[:num_train_fovs]
        val_positions = positions[num_train_fovs:]
        train_tracks_tables = tracks_tables[:num_train_fovs]
        val_tracks_tables = tracks_tables[num_train_fovs:]

        print(f"Number of training FOVs: {len(train_positions)}")
        print(f"Number of validation FOVs: {len(val_positions)}")

        self.train_dataset = TripletDataset(
            positions=train_positions,
            tracks_tables=train_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            anchor_transform=no_aug_transform,
            positive_transform=augment_transform,
            negative_transform=augment_transform,
            fit=True,
            **dataset_settings,
        )

        self.val_dataset = TripletDataset(
            positions=val_positions,
            tracks_tables=val_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            anchor_transform=no_aug_transform,
            positive_transform=augment_transform,
            negative_transform=augment_transform,
            fit=True,
            **dataset_settings,
        )

    def _setup_predict(self, dataset_settings: dict):
        self._set_predict_global_state()
        positions, tracks_tables = self._align_tracks_tables_with_positions()
        self.predict_dataset = TripletDataset(
            positions=positions,
            tracks_tables=tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            anchor_transform=Compose(self.normalizations),
            fit=False,
            **dataset_settings,
        )

    def _setup_test(self, *args, **kwargs):
        raise NotImplementedError("Self-supervised model does not support testing")
