"""Temporal-window sampling of tracked cells from an HCS plate.

Where :mod:`viscy_data.triplet` samples pairs of timepoints for contrastive
learning, this module samples a **contiguous run of frames** from a single track:
the movie a sequence model consumes. Patches follow the tracked object frame by
frame, so the object stays centered and the clip is spatially registered.

Clips vary in length — short tracks contribute one full-length clip rather than
being dropped, which recovers a substantial fraction of a tracking dataset. Pair
the dataset with :class:`~viscy_data.sampler.LengthBucketedBatchSampler` so every
batch holds a single length and no padding masks are needed.
"""

import logging
from pathlib import Path
from typing import Sequence

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import tensorstore as ts
except ImportError:
    ts = None

import numpy as np
import torch
from iohub.ngff import ImageArray, Position
from lightning.pytorch import LightningDataModule
from monai.data.thread_buffer import ThreadDataLoader
from monai.transforms import Compose, MapTransform
from torch import Tensor

from viscy_data._utils import _collate_norm_meta, _read_norm_meta
from viscy_data.foreground_masks import ForegroundMaskSupport
from viscy_data.sampler import LengthBucketedBatchSampler
from viscy_data.triplet import align_tracks_tables_with_positions
from viscy_transforms import BatchedCenterSpatialCropd

__all__ = ["TrackWindowDataModule", "TrackWindowDataset"]

_logger = logging.getLogger("lightning.pytorch")

#: Transform key the nucleus/object mask travels under. Private so it cannot
#: collide with a channel name, and stripped from the batch before the model.
MASK_KEY = "__track_mask"


def _contiguous_runs(times: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted time index into maximal runs of consecutive timepoints.

    A temporal window must be contiguous: the sequence model reads position from
    the index within the clip, so a gap would silently misrepresent the interval
    between frames.

    Parameters
    ----------
    times : np.ndarray
        Sorted 1-D array of timepoints.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, length)`` offsets into ``times`` for each contiguous run.
    """
    if times.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) != 1) + 1
    bounds = np.concatenate(([0], breaks, [times.size]))
    return [(int(start), int(stop - start)) for start, stop in zip(bounds[:-1], bounds[1:])]


class TrackWindowDataset(torch.utils.data.Dataset):
    """Contiguous temporal clips of a tracked object, read from an HCS plate.

    Parameters
    ----------
    positions : list[Position]
        OME-Zarr positions with a consistent channel order.
    tracks_tables : list[pd.DataFrame]
        Tracking tables aligned with ``positions``, ultrack column layout.
    source_channel : list[str]
        Input channel names.
    target_channel : list[str]
        Output channel names. Empty when only auxiliary heads are trained.
    z_range : slice
        Range of Z slices to read.
    initial_yx_patch_size : tuple[int, int]
        YX size of the patch read around the tracked object, before augmentation
        and the final crop.
    window_size : int
        Maximum clip length, by default 32.
    window_stride : int
        Step between clip start times within a track, by default 1.
    min_frames : int
        Shortest clip to keep. Tracks between ``min_frames`` and ``window_size``
        frames contribute one full-length clip, by default 1.
    mask_channel : str | None
        Channel holding an object mask. When set, it is carried through the
        spatial transforms so a pooled per-frame readout can be computed from
        the registered target, by default ``None``.
    fit : bool
        Fitting mode. When ``False``, per-clip index metadata is returned,
        by default ``True``.

    Notes
    -----
    Patch centers are **clamped** to keep the read inside the field of view
    rather than dropping border frames. Dropping them would fragment tracks into
    short runs and lose exactly the long trajectories a temporal model needs;
    clamping shifts the object off-center for those frames instead.
    """

    def __init__(
        self,
        positions: list[Position],
        tracks_tables: "list[pd.DataFrame]",
        source_channel: list[str],
        target_channel: list[str],
        z_range: slice,
        initial_yx_patch_size: tuple[int, int],
        window_size: int = 32,
        window_stride: int = 1,
        min_frames: int = 1,
        mask_channel: str | None = None,
        fit: bool = True,
    ) -> None:
        if pd is None:
            raise ImportError(
                "pandas is required for TrackWindowDataset. Install with: pip install 'viscy-data[triplet]'"
            )
        if ts is None:
            raise ImportError(
                "tensorstore is required for TrackWindowDataset. Install with: pip install 'viscy-data[triplet]'"
            )
        if min_frames > window_size:
            raise ValueError(f"min_frames ({min_frames}) must not exceed window_size ({window_size}).")
        self.positions = positions
        self.source_channel = list(source_channel)
        self.target_channel = list(target_channel)
        self.mask_channel = mask_channel
        self.z_range = z_range
        self.yx_patch_size = tuple(initial_yx_patch_size)
        self.window_size = window_size
        self.window_stride = window_stride
        self.min_frames = min_frames
        self.fit = fit

        self.channel_names = self.source_channel + self.target_channel
        if mask_channel is not None:
            self.channel_names = self.channel_names + [mask_channel]
        self.channel_indices = [positions[0].get_channel_index(name) for name in self.channel_names]

        self.tracks = self._index_tracks(tracks_tables)
        self._runs, self._windows = self._build_windows()
        if not self._windows:
            raise ValueError(
                f"No temporal clips of at least {min_frames} frames found; "
                "check min_frames, split_ratio, the tracking tables, and the well/FOV filters."
            )
        self._tensorstores: dict[str, "ts.TensorStore"] = {}

    def _index_tracks(self, tracks_tables: "list[pd.DataFrame]") -> "pd.DataFrame":
        """Concatenate tracking tables and attach position and clamped centers."""
        indexed = []
        y_half, x_half = (size // 2 for size in self.yx_patch_size)
        for position, tracks in zip(self.positions, tracks_tables, strict=True):
            tracks = tracks.copy()
            tracks["position"] = [position] * len(tracks)
            tracks["fov_name"] = position.zgroup.name.strip("/")
            tracks["global_track_id"] = tracks["fov_name"].str.cat(tracks["track_id"].astype(str), sep="_")
            image: ImageArray = position["0"]
            if self.z_range.stop > image.slices:
                raise ValueError(f"Z range {self.z_range} exceeds image with Z={image.slices}")
            if image.height < self.yx_patch_size[0] or image.width < self.yx_patch_size[1]:
                raise ValueError(
                    f"initial_yx_patch_size {self.yx_patch_size} exceeds the "
                    f"{image.height}x{image.width} field of view."
                )
            tracks["y_center"] = tracks["y"].clip(y_half, image.height - y_half)
            tracks["x_center"] = tracks["x"].clip(x_half, image.width - x_half)
            indexed.append(tracks)
        return pd.concat(indexed).sort_values(["global_track_id", "t"]).reset_index(drop=True)

    def _build_windows(self) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """Enumerate contiguous clips as ``(run, offset, length)`` triples."""
        runs: list[np.ndarray] = []
        windows: list[tuple[int, int, int]] = []
        for _, track in self.tracks.groupby("global_track_id", sort=True):
            row_positions = track.index.to_numpy()
            times = track["t"].to_numpy()
            for start, length in _contiguous_runs(times):
                if length < self.min_frames:
                    continue
                run_index = len(runs)
                runs.append(row_positions[start : start + length])
                if length >= self.window_size:
                    for offset in range(0, length - self.window_size + 1, self.window_stride):
                        windows.append((run_index, offset, self.window_size))
                else:
                    # Short track: one full-length clip instead of dropping it.
                    windows.append((run_index, 0, length))
        return runs, windows

    def window_lengths(self) -> np.ndarray:
        """Return the clip length of every dataset index, for the batch sampler."""
        return np.array([length for _, _, length in self._windows], dtype=np.int64)

    @property
    def n_runs(self) -> int:
        """Return the number of whole contiguous track segments."""
        return len(self._runs)

    def run_lengths(self) -> np.ndarray:
        """Return the frame count of every whole track segment."""
        return np.array([len(rows) for rows in self._runs], dtype=np.int64)

    def read_run(self, run_index: int, max_frames: int | None = None) -> dict:
        """Read one whole track segment as a single-clip batch.

        A sequence model with no length-dependent parameters can consume a whole
        track in one pass, which removes the window-boundary discontinuities that
        tiled inference introduces into a predicted trajectory.

        Parameters
        ----------
        run_index : int
            Index of the track segment, see :attr:`n_runs`.
        max_frames : int | None
            Truncate the segment to its first ``max_frames`` frames, by default
            ``None`` (whole segment).

        Returns
        -------
        dict
            Same keys as :meth:`__getitems__`, with a batch dimension of 1.
        """
        rows = self._runs[run_index]
        if max_frames is not None:
            rows = rows[:max_frames]
        return self._read_windows([rows])

    def _window_rows(self, index: int) -> np.ndarray:
        """Return the tracking-table row positions of one clip."""
        run_index, offset, length = self._windows[index]
        return self._runs[run_index][offset : offset + length]

    def __len__(self) -> int:
        """Return the number of clips."""
        return len(self._windows)

    def _get_tensorstore(self, position: Position) -> "ts.TensorStore":
        """Return a cached tensorstore handle for a position."""
        fov_name = position.zgroup.name
        if fov_name not in self._tensorstores:
            self._tensorstores[fov_name] = position["0"].native
        return self._tensorstores[fov_name]

    def _slice_patch(self, row: "pd.Series") -> "ts.TensorStore":
        """Slice one frame's patch, centered on the tracked object."""
        image = self._get_tensorstore(row["position"])
        y_half, x_half = (size // 2 for size in self.yx_patch_size)
        return image.oindex[
            row["t"],
            [int(i) for i in self.channel_indices],
            self.z_range,
            slice(row["y_center"] - y_half, row["y_center"] + y_half),
            slice(row["x_center"] - x_half, row["x_center"] + x_half),
        ]

    def __getitems__(self, indices: list[int]) -> dict:
        """Read a batch of clips in a single tensorstore call.

        Parameters
        ----------
        indices : list[int]
            Clip indices. All must have the same length, which
            :class:`~viscy_data.sampler.LengthBucketedBatchSampler` guarantees.

        Returns
        -------
        dict
            ``source`` ``(B, T, C_src, Z, Y, X)``, ``target``
            ``(B, T, C_tgt, Z, Y, X)`` when target channels are configured,
            the mask under :data:`MASK_KEY` when ``mask_channel`` is set,
            ``norm_meta`` (one entry per clip), and ``index`` when not fitting.
        """
        return self._read_windows([self._window_rows(i) for i in indices])

    def _read_windows(self, window_rows: list[np.ndarray]) -> dict:
        """Read the given clips, one tensorstore call for the whole batch."""
        lengths = {len(rows) for rows in window_rows}
        if len(lengths) != 1:
            raise ValueError(
                f"A batch must hold clips of one length, got {sorted(lengths)}. "
                "Use LengthBucketedBatchSampler as the DataLoader's batch_sampler."
            )
        frames = lengths.pop()

        rows = self.tracks.iloc[np.concatenate(window_rows)]
        patches = [self._slice_patch(row) for _, row in rows.iterrows()]
        stacked = torch.from_numpy(ts.stack([p.translate_to[0] for p in patches]).read().result())  # noqa: PD013
        stacked = stacked.view(len(window_rows), frames, *stacked.shape[1:])

        n_source = len(self.source_channel)
        n_target = len(self.target_channel)
        sample: dict = {"source": stacked[:, :, :n_source]}
        if n_target:
            sample["target"] = stacked[:, :, n_source : n_source + n_target]
        if self.mask_channel is not None:
            sample[MASK_KEY] = stacked[:, :, n_source + n_target : n_source + n_target + 1]
        sample["norm_meta"] = [_read_norm_meta(self.tracks.iloc[rows[0]]["position"]) for rows in window_rows]
        if not self.fit:
            sample["index"] = [
                {
                    "fov_name": self.tracks.iloc[rows[0]]["fov_name"],
                    "track_id": int(self.tracks.iloc[rows[0]]["track_id"]),
                    "t": self.tracks.iloc[rows]["t"].tolist(),
                }
                for rows in window_rows
            ]
        return sample


class TrackWindowDataModule(LightningDataModule):
    """Lightning data module for temporal clips of tracked cells.

    Splits **by track**, never by frame, so a clip can never straddle the
    train/validation boundary.

    Normalization and augmentation run on GPU in
    :meth:`on_after_batch_transfer`. Time is folded into the channel axis for
    that pass, so one geometric draw applies to every frame of a clip and the
    clip stays spatially registered.

    Parameters
    ----------
    data_path : str
        HCS OME-Zarr plate.
    tracks_path : str
        Directory of per-FOV tracking CSVs.
    source_channel : str | Sequence[str]
        Input channel names.
    target_channel : str | Sequence[str]
        Output channel names, by default ``()``.
    z_range : tuple[int, int]
        Range of valid Z slices, by default ``(0, 1)``.
    initial_yx_patch_size : tuple[int, int]
        YX size read around the tracked object, by default ``(384, 384)``.
    final_yx_patch_size : tuple[int, int]
        YX size the model sees, reached by a crop applied after augmentation,
        by default ``(256, 256)``.
    window_size : int
        Maximum clip length, by default 32.
    window_stride : int
        Step between clip start times, by default 1.
    min_frames : int
        Shortest clip to keep, by default 8.
    frames_per_batch : int
        Frame budget per batch. Batch size is ``frames_per_batch // T``, so GPU
        memory stays roughly constant across clip lengths, by default 64.
    mask_channel : str | None
        Channel holding an object mask, used to pool a per-frame readout from
        the target channels into ``scalar_target``, by default ``None``.
    split_ratio : float
        Fraction of tracks used for training, by default 0.8.
    num_workers : int
        Thread workers. More than 1 is not recommended, by default 1.
    normalizations : list[MapTransform] | None
        Normalization transforms keyed by channel name, by default ``None``.
    augmentations : list[MapTransform] | None
        Augmentation transforms keyed by channel name, by default ``None``.
    augment_validation : bool
        Apply augmentations during validation, by default ``False``.
    include_wells : list[str] | None
        Only include these wells, by default ``None``.
    exclude_fovs : list[str] | None
        Exclude these FOVs, by default ``None``.
    persistent_workers : bool
        Keep worker processes alive between epochs, by default ``False``.
    prefetch_factor : int | None
        Batches prefetched per worker, by default ``None``.
    pin_memory : bool
        Pin host memory for transfers, by default ``False``.
    cache_pool_bytes : int
        Tensorstore cache pool size, by default 0.
    seed : int
        Seed for the track split and the batch sampler, by default 42.
    """

    def __init__(
        self,
        data_path: str,
        tracks_path: str,
        source_channel: str | Sequence[str],
        target_channel: str | Sequence[str] = (),
        z_range: tuple[int, int] = (0, 1),
        initial_yx_patch_size: tuple[int, int] = (384, 384),
        final_yx_patch_size: tuple[int, int] = (256, 256),
        window_size: int = 32,
        window_stride: int = 1,
        min_frames: int = 8,
        frames_per_batch: int = 64,
        mask_channel: str | None = None,
        split_ratio: float = 0.8,
        num_workers: int = 1,
        normalizations: list[MapTransform] | None = None,
        augmentations: list[MapTransform] | None = None,
        augment_validation: bool = False,
        include_wells: list[str] | None = None,
        exclude_fovs: list[str] | None = None,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
        cache_pool_bytes: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tracks_path = Path(tracks_path)
        self.source_channel = [source_channel] if isinstance(source_channel, str) else list(source_channel)
        self.target_channel = [target_channel] if isinstance(target_channel, str) else list(target_channel)
        self.mask_channel = mask_channel
        self.z_range = slice(*z_range)
        self.z_window_size = z_range[1] - z_range[0]
        self.initial_yx_patch_size = tuple(initial_yx_patch_size)
        self.final_yx_patch_size = tuple(final_yx_patch_size)
        self.window_size = window_size
        self.window_stride = window_stride
        self.min_frames = min_frames
        self.frames_per_batch = frames_per_batch
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.augment_validation = augment_validation
        self.include_wells = include_wells
        self.exclude_fovs = exclude_fovs
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.cache_pool_bytes = cache_pool_bytes
        self.seed = seed

        self.normalizations = list(normalizations or [])
        self.augmentations = list(augmentations or [])
        if mask_channel is not None:
            # Keep the mask geometrically aligned with the images without asking
            # the config to list it: only spatial transforms are patched, so
            # intensity augmentation never touches the mask.
            ForegroundMaskSupport.patch_spatial_transforms(
                self.augmentations,
                target_keys=tuple(self.source_channel + self.target_channel),
                mask_keys=(MASK_KEY,),
            )
        self._augment_transform = Compose(self.normalizations + self.augmentations)
        self._normalize_transform = Compose(self.normalizations)

    @property
    def _crop_keys(self) -> list[str]:
        """Return the transform keys the final crop applies to."""
        keys = self.source_channel + self.target_channel
        return keys + [MASK_KEY] if self.mask_channel is not None else keys

    def _dataset_settings(self, fit: bool) -> dict:
        """Return constructor arguments shared by every stage's dataset."""
        return {
            "source_channel": self.source_channel,
            "target_channel": self.target_channel,
            "z_range": self.z_range,
            "initial_yx_patch_size": self.initial_yx_patch_size,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "min_frames": self.min_frames,
            "mask_channel": self.mask_channel,
            "fit": fit,
        }

    def setup(self, stage: str) -> None:
        """Build datasets for the requested stage."""
        positions, tracks_tables = align_tracks_tables_with_positions(
            data_path=self.data_path,
            tracks_path=self.tracks_path,
            include_wells=self.include_wells,
            exclude_fovs=self.exclude_fovs,
            cache_pool_bytes=self.cache_pool_bytes,
        )
        if stage in ("fit", "validate"):
            train_tables, val_tables = self._split_tracks_by_id(tracks_tables)
            self.train_dataset = TrackWindowDataset(positions, train_tables, **self._dataset_settings(fit=True))
            self.val_dataset = TrackWindowDataset(positions, val_tables, **self._dataset_settings(fit=True))
            _logger.info(
                f"Track clips: {len(self.train_dataset)} train / {len(self.val_dataset)} validate "
                f"(window {self.window_size}, min {self.min_frames})"
            )
        elif stage == "predict":
            self.predict_dataset = TrackWindowDataset(positions, tracks_tables, **self._dataset_settings(fit=False))
        else:
            raise NotImplementedError(f"Stage '{stage}' is not supported by TrackWindowDataModule.")

    def _split_tracks_by_id(
        self, tracks_tables: "list[pd.DataFrame]"
    ) -> "tuple[list[pd.DataFrame], list[pd.DataFrame]]":
        """Split every FOV's table into train and validation rows, by track.

        Splitting on the track identifier rather than on FOVs keeps both splits
        represented in every field of view, which matters when tracking datasets
        have few FOVs.
        """
        track_keys = sorted(
            {
                f"{fov_index}_{track_id}"
                for fov_index, tracks in enumerate(tracks_tables)
                for track_id in tracks["track_id"].unique()
            }
        )
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(track_keys)
        n_train = int(round(len(shuffled) * self.split_ratio))
        train_keys = set(shuffled[:n_train])

        train_tables, val_tables = [], []
        for fov_index, tracks in enumerate(tracks_tables):
            keys = tracks["track_id"].astype(str).radd(f"{fov_index}_")
            is_train = keys.isin(train_keys)
            train_tables.append(tracks[is_train])
            val_tables.append(tracks[~is_train])
        return train_tables, val_tables

    def _dataloader(self, dataset: TrackWindowDataset, shuffle: bool, drop_last: bool) -> ThreadDataLoader:
        """Build a loader with one clip length per batch."""
        sampler = LengthBucketedBatchSampler(
            dataset.window_lengths(),
            frames_per_batch=self.frames_per_batch,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.seed,
            num_replicas=self.trainer.world_size if self.trainer else 1,
            rank=self.trainer.global_rank if self.trainer else 0,
        )
        return ThreadDataLoader(
            dataset,
            use_thread_workers=True,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x,
        )

    def train_dataloader(self) -> ThreadDataLoader:
        """Return the training data loader."""
        return self._dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> ThreadDataLoader:
        """Return the validation data loader."""
        return self._dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> ThreadDataLoader:
        """Return the prediction data loader."""
        return self._dataloader(self.predict_dataset, shuffle=False, drop_last=False)

    def _clip_transform(self) -> Compose:
        """Return the transform pipeline for the current stage."""
        if self.trainer is None:
            return self._augment_transform
        if self.trainer.predicting:
            return self._normalize_transform
        validating = self.trainer.validating or self.trainer.sanity_checking
        if validating and not self.augment_validation:
            return self._normalize_transform
        return self._augment_transform

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Normalize, augment and crop a clip batch on the accelerator.

        Time is folded into the channel axis so each transform sees a
        ``(B, T, Z, Y, X)`` tensor per channel and draws its random parameters
        once per clip, keeping every frame of a clip registered to the others.

        Parameters
        ----------
        batch : dict | Tensor
            Batch from the loader. A bare tensor (the traced example input) is
            returned unchanged.
        dataloader_idx : int
            Index of the data loader.

        Returns
        -------
        dict
            ``source`` and ``target`` of shape ``(B, T, C, Z, Y, X)`` cropped to
            ``final_yx_patch_size``, plus ``scalar_target`` of shape
            ``(B, T, C_tgt)`` when a mask channel is configured.
        """
        if isinstance(batch, Tensor):
            return batch

        norm_meta = batch.get("norm_meta")
        if isinstance(norm_meta, list):
            norm_meta = None if any(meta is None for meta in norm_meta) else _collate_norm_meta(norm_meta)
        # Kept in the batch (rather than consumed here) so evaluation can map
        # predictions back to physical units.
        batch["norm_meta"] = norm_meta

        scattered: dict = {}
        for index, name in enumerate(self.source_channel):
            scattered[name] = batch["source"][:, :, index]
        for index, name in enumerate(self.target_channel):
            scattered[name] = batch["target"][:, :, index]
        if self.mask_channel is not None:
            scattered[MASK_KEY] = batch[MASK_KEY][:, :, 0]
        if norm_meta is not None:
            scattered["norm_meta"] = norm_meta

        transformed = self._clip_transform()(scattered)
        transformed = self._crop(transformed)

        batch["source"] = torch.stack([transformed[name] for name in self.source_channel], dim=2)
        if self.mask_channel is not None:
            batch["mask"] = transformed[MASK_KEY].unsqueeze(2)
        if self.target_channel:
            target = torch.stack([transformed[name] for name in self.target_channel], dim=2)
            batch["target"] = target
            if self.mask_channel is not None:
                batch["scalar_target"] = _pool_over_mask(target, transformed[MASK_KEY])
        batch.pop(MASK_KEY, None)

        expected = (self.z_window_size, *self.final_yx_patch_size)
        actual = tuple(batch["source"].shape[3:])
        if actual != expected:
            raise ValueError(f"Spatial shape mismatch after transforms: got {actual}, expected {expected}.")
        return batch

    def _crop(self, transformed: dict) -> dict:
        """Center-crop every image key to ``final_yx_patch_size``."""
        crop = BatchedCenterSpatialCropd(
            keys=self._crop_keys,
            roi_size=(self.z_window_size, *self.final_yx_patch_size),
            allow_missing_keys=True,
        )
        return crop(transformed)


def _pool_over_mask(target: Tensor, mask: Tensor) -> Tensor:
    """Average each target channel over the object mask, per frame.

    Frames whose mask is empty fall back to the whole patch, so a missing
    segmentation degrades to a frame-wide average instead of a division by zero.

    Parameters
    ----------
    target : Tensor
        ``(B, T, C, Z, Y, X)`` normalized target channels.
    mask : Tensor
        ``(B, T, Z, Y, X)`` object mask.

    Returns
    -------
    Tensor
        ``(B, T, C)`` per-frame pooled values.
    """
    binary = (mask > 0.5).to(target.dtype).unsqueeze(2)
    counts = binary.flatten(3).sum(-1)
    empty = counts == 0
    if empty.any():
        binary = torch.where(empty[..., None, None, None], torch.ones_like(binary), binary)
        counts = binary.flatten(3).sum(-1)
    return (target * binary).flatten(3).sum(-1) / counts
