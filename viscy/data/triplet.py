import logging
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch
from iohub.ngff import ImageArray, Position, open_ome_zarr
from monai.transforms import Compose, MapTransform
from torch import Tensor
from torch.utils.data import Dataset

from viscy.data.hcs import HCSDataModule, _read_norm_meta
from viscy.data.typing import DictTransform, NormMeta, TripletSample

_logger = logging.getLogger("lightning.pytorch")

INDEX_COLUMNS = ["fov_name", "track_id", "t", "id", "parent_track_id", "parent_id"]


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
        predict_cells: bool = False,
        include_fov_names: list[str] | None = None,
        include_track_ids: list[int] | None = None,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
    ) -> None:
        """Dataset for triplet sampling of cells based on tracking.

        Parameters
        ----------
        positions : list[Position]
            OME-Zarr images with consistent channel order
        tracks_tables : list[pd.DataFrame]
            Data frames containing ultrack results
        channel_names : list[str]
            Input channel names
        initial_yx_patch_size : tuple[int, int]
            YX size of the initially sampled image patch before augmentation
        z_range : slice
            Range of Z-slices
        anchor_transform : DictTransform | None, optional
            Transforms applied to the anchor sample, by default None
        positive_transform : DictTransform | None, optional
            Transforms applied to the positve sample, by default None
        negative_transform : DictTransform | None, optional
            Transforms applied to the negative sample, by default None
        fit : bool, optional
            Fitting mode in which the full triplet will be sampled,
            only sample anchor if ``False``, by default True
        predict_cells : bool, optional
            Only predict on selected cells, by default False
        include_fov_names : list[str] | None, optional
            Only predict on selected FOVs, by default None
        include_track_ids : list[int] | None, optional
            Only predict on selected track IDs, by default None
        time_interval : Literal["any"] | int, optional
            Future time interval to sample positive and anchor from,
            by default "any"
            (sample negative from another track any time point
            and use the augmented anchor patch as positive)
        return_negative : bool, optional
            Whether to return the negative sample during the fit stage
            (can be set to False when using a loss function like NT-Xent),
            by default True
        """
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
        self.predict_cells = predict_cells
        self.include_fov_names = include_fov_names or []
        self.include_track_ids = include_track_ids or []
        self.time_interval = time_interval
        self.tracks = self._filter_tracks(tracks_tables)
        self.tracks = (
            self._specific_cells(self.tracks) if self.predict_cells else self.tracks
        )
        self.valid_anchors = self._filter_anchors(self.tracks)
        self.return_negative = return_negative

    def _filter_tracks(self, tracks_tables: list[pd.DataFrame]) -> pd.DataFrame:
        """Exclude tracks that are too close to the border or do not have the next time point.

        Parameters
        ----------
        tracks_tables : list[pd.DataFrame]
            List of tracks_tables returned by TripletDataModule._align_tracks_tables_with_positions

        Returns
        -------
        pd.DataFrame
            Filtered tracks table
        """
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
            # FIXME: Check if future time points are available after interval
            filtered_tracks.append(
                tracks[
                    tracks["y"].between(*y_range, inclusive="neither")
                    & tracks["x"].between(*x_range, inclusive="neither")
                ]
            )
        return pd.concat(filtered_tracks).reset_index(drop=True)

    def _filter_anchors(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Ensure that anchors have the next time point after a time interval."""
        if self.time_interval == "any" or not self.fit:
            return tracks
        return pd.concat(
            [
                track[(track["t"] + self.time_interval).isin(track["t"])]
                for (_, track) in tracks.groupby("global_track_id")
            ]
        )

    def _specific_cells(self, tracks: pd.DataFrame) -> pd.DataFrame:
        specific_tracks = pd.DataFrame()
        print(self.include_fov_names)
        print(self.include_track_ids)
        for fov_name, track_id in zip(self.include_fov_names, self.include_track_ids):
            filtered_tracks = tracks[
                (tracks["fov_name"] == fov_name) & (tracks["track_id"] == track_id)
            ]
            specific_tracks = pd.concat([specific_tracks, filtered_tracks])
        return specific_tracks.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.valid_anchors)

    def _sample_positive(self, anchor_row: pd.Series) -> pd.Series:
        """Select a positive sample from the same track in the next time point."""
        same_track = self.tracks[
            (self.tracks["global_track_id"] == anchor_row["global_track_id"])
        ]
        return same_track[
            same_track["t"] == (anchor_row["t"] + self.time_interval)
        ].iloc[0]

    def _sample_negative(self, anchor_row: pd.Series) -> pd.Series:
        """Select a negative sample from a different track in the next time point
        if an interval is specified, otherwise from any random time point."""
        if self.time_interval == "any":
            tracks = self.tracks
        else:
            tracks = self.tracks[
                self.tracks["t"] == anchor_row["t"] + self.time_interval
            ]
        candidates: pd.DataFrame = tracks[
            (tracks["global_track_id"] != anchor_row["global_track_id"])
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
        anchor_row = self.valid_anchors.iloc[index]
        anchor_patch, anchor_norm = self._slice_patch(anchor_row)
        if self.fit:
            if self.time_interval == "any":
                positive_patch = anchor_patch.clone()
                positive_norm = anchor_norm
            else:
                positive_row = self._sample_positive(anchor_row)
                positive_patch, positive_norm = self._slice_patch(positive_row)
            if self.positive_transform:
                positive_patch = _transform_channel_wise(
                    transform=self.positive_transform,
                    channel_names=self.channel_names,
                    patch=positive_patch,
                    norm_meta=positive_norm,
                )
            if self.return_negative:
                negative_row = self._sample_negative(anchor_row)
                negative_patch, negative_norm = self._slice_patch(negative_row)
                if self.negative_transform:
                    negative_patch = _transform_channel_wise(
                        transform=self.negative_transform,
                        channel_names=self.channel_names,
                        patch=negative_patch,
                        norm_meta=negative_norm,
                    )
        if self.anchor_transform:
            anchor_patch = _transform_channel_wise(
                transform=self.anchor_transform,
                channel_names=self.channel_names,
                patch=anchor_patch,
                norm_meta=anchor_norm,
            )
        sample = {"anchor": anchor_patch}
        if self.fit:
            if self.return_negative:
                sample.update({"positive": positive_patch, "negative": negative_patch})
            else:
                sample.update({"positive": positive_patch})
        else:
            sample.update({"index": anchor_row[INDEX_COLUMNS].to_dict()})
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
        predict_cells: bool = False,
        include_fov_names: list[str] | None = None,
        include_track_ids: list[int] | None = None,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
    ):
        """Lightning data module for triplet sampling of patches.

        Parameters
        ----------
        data_path : str
            Image dataset path
        tracks_path : str
            Tracks labels dataset path
        source_channel : str | Sequence[str]
            List of input channel names
        z_range : tuple[int, int]
            Range of valid z-slices
        initial_yx_patch_size : tuple[int, int], optional
            XY size of the initially sampled image patch, by default (512, 512)
        final_yx_patch_size : tuple[int, int], optional
            Output patch size, by default (224, 224)
        split_ratio : float, optional
            Ratio of training samples, by default 0.8
        batch_size : int, optional
            Batch size, by default 16
        num_workers : int, optional
            Number of data-loading workers, by default 8
        normalizations : list[MapTransform], optional
            Normalization transforms, by default []
        augmentations : list[MapTransform], optional
            Augmentation transforms, by default []
        caching : bool, optional
            Whether to cache the dataset, by default False
        predict_cells : bool, optional
            Only predict for selected cells, by default False
        include_fov_names : list[str] | None, optional
            Only predict for selected FOVs, by default None
        include_track_ids : list[int] | None, optional
            Only predict for selected tracks, by default None
        time_interval : Literal["any"] | int, optional
            Future time interval to sample positive and anchor from,
            "any" means sampling negative from another track any time point
            and using the augmented anchor patch as positive), by default "any"
        return_negative : bool, optional
            Whether to return the negative sample during the fit stage
            (can be set to False when using a loss function like NT-Xent),
            by default True
        """
        super().__init__(
            data_path=data_path,
            source_channel=source_channel,
            target_channel=[],
            z_window_size=z_range[1] - z_range[0],
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            target_2d=False,
            yx_patch_size=final_yx_patch_size,
            normalizations=normalizations,
            augmentations=augmentations,
            caching=caching,
        )
        self.z_range = slice(*z_range)
        self.tracks_path = Path(tracks_path)
        self.initial_yx_patch_size = initial_yx_patch_size
        self.predict_cells = predict_cells
        self.include_fov_names = include_fov_names
        self.include_track_ids = include_track_ids
        self.time_interval = time_interval
        self.return_negative = return_negative

    def _align_tracks_tables_with_positions(
        self,
    ) -> tuple[list[Position], list[pd.DataFrame]]:
        """Parse positions in ome-zarr store containing tracking information
        and assemble tracks tables for each position.

        Returns
        -------
        tuple[list[Position], list[pd.DataFrame]]
            List of positions and list of tracks tables for each position
        """
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
            "time_interval": self.time_interval,
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
        _logger.debug(f"Number of training FOVs: {len(train_positions)}")
        _logger.debug(f"Number of validation FOVs: {len(val_positions)}")
        anchor_transform = (
            no_aug_transform
            if (self.time_interval == "any" or self.time_interval == 0)
            else augment_transform
        )
        self.train_dataset = TripletDataset(
            positions=train_positions,
            tracks_tables=train_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            anchor_transform=anchor_transform,
            positive_transform=augment_transform,
            negative_transform=augment_transform,
            fit=True,
            return_negative=self.return_negative,
            **dataset_settings,
        )

        self.val_dataset = TripletDataset(
            positions=val_positions,
            tracks_tables=val_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            anchor_transform=anchor_transform,
            positive_transform=augment_transform,
            negative_transform=augment_transform,
            fit=True,
            return_negative=self.return_negative,
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
            predict_cells=self.predict_cells,
            include_fov_names=self.include_fov_names,
            include_track_ids=self.include_track_ids,
            **dataset_settings,
        )

    def _setup_test(self, *args, **kwargs):
        raise NotImplementedError("Self-supervised model does not support testing")
