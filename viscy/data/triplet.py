import logging
import os
import warnings
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import tensorstore as ts
import torch
from iohub.ngff import ImageArray, Position, open_ome_zarr
from monai.data.thread_buffer import ThreadDataLoader
from monai.data.utils import collate_meta_tensor
from monai.transforms import Compose, MapTransform
from torch import Tensor
from torch.utils.data import Dataset

from viscy.data.hcs import HCSDataModule, _read_norm_meta
from viscy.data.select import _filter_fovs, _filter_wells
from viscy.data.typing import DictTransform, NormMeta
from viscy.transforms import BatchedCenterSpatialCropd

_logger = logging.getLogger("lightning.pytorch")

INDEX_COLUMNS = [
    "fov_name",
    "track_id",
    "t",
    "id",
    "parent_track_id",
    "parent_id",
    "z",
    "y",
    "x",
]


def _scatter_channels(
    channel_names: list[str], patch: Tensor, norm_meta: NormMeta | None
) -> dict[str, Tensor | NormMeta] | dict[str, Tensor]:
    channels = {
        name: patch[:, c : c + 1]
        for name, c in zip(channel_names, range(patch.shape[1]))
    }
    if norm_meta is not None:
        channels["norm_meta"] = collate_meta_tensor(norm_meta)
    return channels


def _gather_channels(
    patch_channels: dict[str, Tensor | NormMeta],
) -> list[Tensor]:
    patch_channels.pop("norm_meta", None)
    return torch.cat(list(patch_channels.values()), dim=1)


def _transform_channel_wise(
    transform: DictTransform,
    channel_names: list[str],
    patch: Tensor,
    norm_meta: NormMeta | None,
) -> list[Tensor]:
    scattered_channels = _scatter_channels(channel_names, patch, norm_meta)
    transformed_channels = transform(scattered_channels)
    return _gather_channels(transformed_channels)


class TripletDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        tracks_tables: list[pd.DataFrame],
        channel_names: list[str],
        initial_yx_patch_size: tuple[int, int],
        z_range: slice,
        fit: bool = True,
        predict_cells: bool = False,
        include_fov_names: list[str] | None = None,
        include_track_ids: list[int] | None = None,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
        cache_pool_bytes: int = 0,
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
            YX size of the initially sampled image patch
        z_range : slice
            Range of Z-slices
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
        cache_pool_bytes : int, optional
            Size of the tensorstore cache pool in bytes, by default 0
        """
        self.positions = positions
        self.channel_names = channel_names
        self.channel_indices = [
            positions[0].get_channel_index(ch) for ch in channel_names
        ]
        self.z_range = z_range
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
        self._setup_tensorstore_context(cache_pool_bytes)

    def _setup_tensorstore_context(self, cache_pool_bytes: int):
        """Configure tensorstore context with CPU limits based on SLURM environment."""
        cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus_per_task is not None:
            cpus_per_task = int(cpus_per_task)
        else:
            cpus_per_task = os.cpu_count() or 4
        self.tensorstore_context = ts.Context(
            {
                "data_copy_concurrency": {"limit": cpus_per_task},
                "cache_pool": {"total_bytes_limit": cache_pool_bytes},
            }
        )
        self._tensorstores = {}

    def _get_tensorstore(self, position: Position) -> ts.TensorStore:
        """Get cached tensorstore object or create and cache new one."""
        fov_name = position.zgroup.name
        if fov_name not in self._tensorstores:
            self._tensorstores[fov_name] = position["0"].tensorstore(
                context=self.tensorstore_context,
                # assume immutable data to reduce metadata access
                recheck_cached_data="open",
            )
        return self._tensorstores[fov_name]

    def _filter_tracks(self, tracks_tables: list[pd.DataFrame]) -> pd.DataFrame:
        """Exclude tracks that are too close to the border
        or do not have the next time point.

        Parameters
        ----------
        tracks_tables : list[pd.DataFrame]
            List of tracks_tables returned by
            TripletDataModule._align_tracks_tables_with_positions

        Returns
        -------
        pd.DataFrame
            Filtered tracks table
        """
        filtered_tracks = []
        y_exclude, x_exclude = (self.yx_patch_size[0] // 2, self.yx_patch_size[1] // 2)
        for pos, tracks in zip(self.positions, tracks_tables, strict=True):
            tracks["position"] = [pos] * len(tracks)
            tracks["fov_name"] = pos.zgroup.name.strip("/")
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

    def _sample_positives(self, anchor_rows: pd.DataFrame) -> pd.DataFrame:
        """Select a positive sample from the same track in the next time point."""
        query = anchor_rows[["global_track_id", "t"]].copy()
        query["t"] += self.time_interval
        return query.merge(self.tracks, on=["global_track_id", "t"], how="inner")

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

    def _sample_negatives(self, anchor_rows: pd.DataFrame) -> pd.DataFrame:
        negative_samples = [
            self._sample_negative(row) for _, row in anchor_rows.iterrows()
        ]
        return pd.DataFrame(negative_samples).reset_index(drop=True)

    def _slice_patch(
        self, track_row: pd.Series
    ) -> tuple[ts.TensorStore, NormMeta | None]:
        position: Position = track_row["position"]

        # Get cached tensorstore object using FOV name
        image = self._get_tensorstore(position)

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
        return patch, _read_norm_meta(position)

    def _slice_patches(self, track_rows: pd.DataFrame):
        patches = []
        norms = []
        for _, row in track_rows.iterrows():
            patch, norm = self._slice_patch(row)
            patches.append(patch)
            norms.append(norm)
        results = ts.stack([p.translate_to[0] for p in patches]).read().result()
        return torch.from_numpy(results), norms

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        anchor_rows = self.valid_anchors.iloc[indices]
        anchor_patches, anchor_norms = self._slice_patches(anchor_rows)
        sample = {"anchor": anchor_patches, "anchor_norm_meta": anchor_norms}
        if self.fit:
            if self.time_interval == "any":
                positive_patches = anchor_patches.clone()
                positive_norms = anchor_norms
            else:
                positive_rows = self._sample_positives(anchor_rows)
                positive_patches, positive_norms = self._slice_patches(positive_rows)

            sample["positive"] = positive_patches
            sample["positive_norm_meta"] = positive_norms
            if self.return_negative:
                negative_rows = self._sample_negatives(anchor_rows)
                negative_patches, negative_norms = self._slice_patches(negative_rows)
                sample["negative"] = negative_patches
                sample["negative_norm_meta"] = negative_norms
        else:
            indices_list = []
            for _, anchor_row in anchor_rows.iterrows():
                index_dict = {}
                for col in INDEX_COLUMNS:
                    if col in anchor_row.index:
                        index_dict[col] = anchor_row[col]
                    elif col not in ["y", "x", "z"]:
                        raise KeyError(f"Required column '{col}' not found in data")
                indices_list.append(index_dict)
            sample["index"] = indices_list

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
        num_workers: int = 1,
        normalizations: list[MapTransform] = [],
        augmentations: list[MapTransform] = [],
        augment_validation: bool = True,
        caching: bool = False,
        fit_include_wells: list[str] | None = None,
        fit_exclude_fovs: list[str] | None = None,
        predict_cells: bool = False,
        include_fov_names: list[str] | None = None,
        include_track_ids: list[int] | None = None,
        time_interval: Literal["any"] | int = "any",
        return_negative: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
        z_window_size: int | None = None,
        cache_pool_bytes: int = 0,
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
            Number of thread workers.
            Set to 0 to disable threading. Using more than 1 is not recommended.
            by default 1
        normalizations : list[MapTransform], optional
            Normalization transforms, by default []
        augmentations : list[MapTransform], optional
            Augmentation transforms, by default []
        augment_validation : bool, optional
            Apply augmentations to validation data, by default True.
            Set to False for VAE training where clean validation is needed.
        caching : bool, optional
            Whether to cache the dataset, by default False
        fit_include_wells : list[str], optional
            Only include these wells for fitting, by default None
        fit_exclude_fovs : list[str], optional
            Exclude these FOVs for fitting, by default None
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
        persistent_workers : bool, optional
            Whether to keep worker processes alive between iterations, by default False
        prefetch_factor : int | None, optional
            Number of batches loaded in advance by each worker, by default None
        pin_memory : bool, optional
            Whether to pin memory in CPU for faster GPU transfer, by default False
        z_window_size : int, optional
            Size of the final Z window, by default None (inferred from z_range)
        cache_pool_bytes : int, optional
            Size of the per-process tensorstore cache pool in bytes, by default 0
        """
        if num_workers > 1:
            warnings.warn(
                "Using more than 1 thread worker will likely degrade performance."
            )
        super().__init__(
            data_path=data_path,
            source_channel=source_channel,
            target_channel=[],
            z_window_size=z_window_size or z_range[1] - z_range[0],
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            target_2d=False,
            yx_patch_size=final_yx_patch_size,
            normalizations=normalizations,
            augmentations=augmentations,
            caching=caching,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
        self.z_range = slice(*z_range)
        self.tracks_path = Path(tracks_path)
        self.initial_yx_patch_size = initial_yx_patch_size
        self._include_wells = fit_include_wells
        self._exclude_fovs = fit_exclude_fovs
        self.predict_cells = predict_cells
        self.include_fov_names = include_fov_names
        self.include_track_ids = include_track_ids
        self.time_interval = time_interval
        self.return_negative = return_negative
        self.augment_validation = augment_validation
        self._cache_pool_bytes = cache_pool_bytes
        self._augmentation_transform = Compose(
            self.normalizations + self.augmentations + [self._final_crop()]
        )
        self._no_augmentation_transform = Compose(
            self.normalizations + [self._final_crop()]
        )

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
        for well in _filter_wells(images_plate, include_wells=self._include_wells):
            for fov in _filter_fovs(well, exclude_fovs=self._exclude_fovs):
                positions.append(fov)
                tracks_df = pd.read_csv(
                    next((self.tracks_path / fov.zgroup.name.strip("/")).glob("*.csv"))
                ).astype(int)
                tracks_tables.append(tracks_df)

        return positions, tracks_tables

    @property
    def _base_dataset_settings(self) -> dict:
        return {
            "channel_names": self.source_channel,
            "z_range": self.z_range,
            "time_interval": self.time_interval,
            "cache_pool_bytes": self._cache_pool_bytes,
        }

    def _setup_fit(self, dataset_settings: dict):
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
        self.train_dataset = TripletDataset(
            positions=train_positions,
            tracks_tables=train_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
            fit=True,
            return_negative=self.return_negative,
            **dataset_settings,
        )

        self.val_dataset = TripletDataset(
            positions=val_positions,
            tracks_tables=val_tracks_tables,
            initial_yx_patch_size=self.initial_yx_patch_size,
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
            fit=False,
            predict_cells=self.predict_cells,
            include_fov_names=self.include_fov_names,
            include_track_ids=self.include_track_ids,
            **dataset_settings,
        )

    def _setup_test(self, *args, **kwargs):
        raise NotImplementedError("Self-supervised model does not support testing")

    def train_dataloader(self):
        return ThreadDataLoader(
            self.train_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return ThreadDataLoader(
            self.val_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def predict_dataloader(self):
        return ThreadDataLoader(
            self.predict_dataset,
            use_thread_workers=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def _final_crop(self) -> BatchedCenterSpatialCropd:
        """Setup final cropping: center crop to the target size."""
        return BatchedCenterSpatialCropd(
            keys=self.source_channel,
            roi_size=(
                self.z_window_size,
                self.yx_patch_size[0],
                self.yx_patch_size[1],
            ),
        )

    def _find_transform(self, key: str):
        if self.trainer:
            if self.trainer.predicting:
                return self._no_augmentation_transform
            if self.trainer.validating and not self.augment_validation:
                return self._no_augmentation_transform
        # NOTE: for backwards compatibility
        if key == "anchor" and self.time_interval in ("any", 0):
            return self._no_augmentation_transform
        return self._augmentation_transform

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Apply transforms after transferring to device."""
        if isinstance(batch, Tensor):
            # example array
            return batch
        for key in ["anchor", "positive", "negative"]:
            if key in batch:
                norm_meta_key = f"{key}_norm_meta"
                norm_meta = batch.get(norm_meta_key)
                transformed_patches = _transform_channel_wise(
                    transform=self._find_transform(key),
                    channel_names=self.source_channel,
                    patch=batch[key],
                    norm_meta=norm_meta,
                )
                batch[key] = transformed_patches
                if norm_meta_key in batch:
                    del batch[norm_meta_key]

        return batch
