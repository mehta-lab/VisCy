"""Temporal clips from pre-cropped per-track numpy arrays.

A dataset layout where every tracked cell is already cropped and saved as one file, alongside a
per-frame vector target:

```
root/
  images/<movie>.<track>.npy   (T, C_img, Y, X)
  labels/<movie>.<track>.npy   (C_tgt, T, 1)     -- note the transposed axis order
```

This is the layout of the Leger et al. (MELBA 2026) RPE1 cell-cycle dataset, where the images are
brightfield and H2B and the labels are two FUCCI channels per frame. It is the counterpart of
:mod:`viscy_data.track_window`, which re-sources the same kind of clip lazily from an HCS plate; use
this one when the crops already exist on disk.

Clips are indexed as ``(track, length)`` pairs over a fixed set of lengths, with the start offset
drawn afresh each epoch. Keeping the *length* static is what lets
:class:`~viscy_data.sampler.LengthBucketedBatchSampler` group a batch by exact length while the
model still sees a different crop of each track every epoch.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from monai.data import ThreadDataLoader
from torch import Tensor
from torch.utils.data import Dataset

from viscy_data.sampler import LengthBucketedBatchSampler

__all__ = ["NpyTrackDataset", "NpyTrackDataModule"]

#: Clip lengths sampled during training, following Leger et al. section 3.2.
DEFAULT_WINDOW_LENGTHS = (4, 16, 32, 64, 96, 128, 192)


def _track_names(root: Path) -> list[str]:
    """List the ``<movie>.<track>`` stems under ``root/images``, sorted."""
    names = sorted(p.stem for p in (root / "images").glob("*.npy"))
    if not names:
        raise FileNotFoundError(f"No .npy tracks found in {root / 'images'}.")
    return names


def _track_lengths(paths: list[Path]) -> list[int]:
    """Return the frame count of every array, from its file size.

    Opening each file to parse its header costs one metadata round trip plus a read; on a shared
    filesystem with a cold client that is minutes for a few thousand tracks, which looks
    indistinguishable from a hung job. Every track here has the same per-frame footprint, so one
    header read establishes the stride and the rest is a ``stat`` each -- measured 31x faster,
    with lengths identical to parsing every header.
    """
    with open(paths[0], "rb") as handle:
        np.lib.format.read_magic(handle)
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        header_bytes = handle.tell()
    frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize

    lengths = []
    for path in paths:
        size = path.stat().st_size - header_bytes
        if size % frame_bytes:
            # A track with a different frame shape or dtype breaks the stride assumption.
            raise ValueError(f"{path} is not a whole number of {frame_bytes}-byte frames; frame shapes differ.")
        lengths.append(size // frame_bytes)
    return lengths


class NpyTrackDataset(Dataset):
    """Clips drawn from per-track numpy arrays.

    Parameters
    ----------
    root : Path
        Directory holding ``images/`` and ``labels/``.
    names : list[str]
        Track stems this dataset owns, e.g. ``["0610.8138", ...]``.
    source_channel : int
        Image channel to feed the model. 0 is brightfield, 1 is H2B.
    target_channel : int | None
        Image channel to emit as ``target`` for image-to-image training, e.g. 1 for H2B. ``None``
        emits no image target, which is what the scalar-only regression task wants.
    window_lengths : Sequence[int]
        Clip lengths to emit. A track contributes one clip per length it is long enough for.
    whole_track : bool
        Emit exactly one full-length clip per track instead, for inference. Batches must then be
        one clip wide, since lengths differ between tracks.
    random_offset : bool
        Draw the clip start uniformly at read time. Disable for a deterministic validation set.
    seed : int
        Base seed for offset sampling.
    """

    def __init__(
        self,
        root: Path,
        names: list[str],
        source_channel: int = 0,
        target_channel: int | None = None,
        window_lengths: Sequence[int] = DEFAULT_WINDOW_LENGTHS,
        whole_track: bool = False,
        random_offset: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.names = list(names)
        self.source_channel = source_channel
        self.target_channel = target_channel
        self.whole_track = whole_track
        self.random_offset = random_offset
        self.seed = seed
        self.epoch = 0

        self.track_lengths = _track_lengths([self.root / "images" / f"{n}.npy" for n in self.names])

        if whole_track:
            self.windows = [(i, length) for i, length in enumerate(self.track_lengths)]
        else:
            self.windows = [
                (i, int(length))
                for i, track_length in enumerate(self.track_lengths)
                for length in sorted(window_lengths)
                if length <= track_length
            ]
        if not self.windows:
            raise ValueError(
                f"No clips of any length in {sorted(window_lengths)} fit the {len(self.names)} tracks "
                f"under {self.root} (longest track is {max(self.track_lengths, default=0)} frames)."
            )

    def __len__(self) -> int:
        """Return the number of clips."""
        return len(self.windows)

    def window_lengths(self) -> np.ndarray:
        """Clip length per index, as :class:`~viscy_data.sampler.LengthBucketedBatchSampler` expects."""
        return np.array([length for _, length in self.windows], dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the offset sampler so each epoch sees different crops."""
        self.epoch = epoch

    def _offset(self, index: int, track_length: int, length: int) -> int:
        span = track_length - length
        if span <= 0:
            return 0
        if not self.random_offset:
            # Centred, so a deterministic split still covers the informative middle of the cycle.
            return span // 2
        rng = np.random.default_rng((self.seed, self.epoch, index))
        return int(rng.integers(0, span + 1))

    def _read(self, window_index: int) -> tuple[Tensor, Tensor, Tensor | None]:
        track_index, length = self.windows[window_index]
        name = self.names[track_index]
        offset = self._offset(window_index, self.track_lengths[track_index], length)

        images = np.load(self.root / "images" / f"{name}.npy", mmap_mode="r")
        labels = np.load(self.root / "labels" / f"{name}.npy", mmap_mode="r")

        clip = np.asarray(images[offset : offset + length, self.source_channel], dtype=np.float32)
        scalars = np.asarray(labels[:, offset : offset + length, 0], dtype=np.float32).T

        image_target = None
        if self.target_channel is not None:
            stained = np.asarray(images[offset : offset + length, self.target_channel], dtype=np.float32)
            image_target = torch.from_numpy(stained)[:, None, None]

        # (T, Y, X) -> (T, C=1, Z=1, Y, X); the model requires a 6-D (B, T, C, Z, Y, X) movie.
        return torch.from_numpy(clip)[:, None, None], torch.from_numpy(scalars), image_target

    def __getitem__(self, index: int) -> dict:
        """Return one clip as a batch of size one."""
        return self.__getitems__([index])

    def __getitems__(self, indices: list[int]) -> dict:
        """Return a collated batch, paired with ``collate_fn=lambda x: x`` on the loader."""
        clips, scalars, stains, names = [], [], [], []
        for index in indices:
            clip, scalar, stain = self._read(index)
            clips.append(clip)
            scalars.append(scalar)
            if stain is not None:
                stains.append(stain)
            names.append(self.names[self.windows[index][0]])
        source = torch.stack(clips)
        scalar_target = torch.stack(scalars)
        if scalar_target.shape[:2] != source.shape[:2] or scalar_target.ndim != 3:
            # RegressionHead.compute_loss broadcasts silently, so a wrong shape would train on
            # nonsense rather than raise. Fail here instead.
            raise ValueError(
                f"scalar_target must be (B, T, C); got {tuple(scalar_target.shape)} against a source of "
                f"{tuple(source.shape)}."
            )
        batch = {"source": source, "scalar_target": scalar_target, "index": names}
        if stains:
            batch["target"] = torch.stack(stains)
        return batch


class NpyTrackDataModule(LightningDataModule):
    """Lightning data module over per-track numpy clips.

    Splits by movie, never by track: every clip from one movie lands in a single split, so no two
    splits share a field of view, illumination or local imaging artefact.

    Parameters
    ----------
    data_path : str
        Dataset root holding ``train/``, and whichever of ``test/`` and ``test_drugged/`` are used.
    train_split : str
        Subdirectory to draw fit and validation clips from, by default ``"train"``.
    predict_split : str
        Subdirectory to predict over, by default ``"test"``.
    source_channel : int
        Image channel index: 0 brightfield, 1 H2B.
    val_movies : Sequence[str]
        Movie prefixes held out of training for validation.
    window_lengths : Sequence[int]
        Training clip lengths.
    frames_per_batch : int
        Frame budget per batch; the batch size is ``frames_per_batch // clip_length``.
    num_workers : int
        Loader workers.
    seed : int
        Base seed for offset sampling.
    persistent_workers, prefetch_factor, pin_memory
        Passed through to the loader.
    """

    def __init__(
        self,
        data_path: str,
        train_split: str = "train",
        predict_split: str = "test",
        source_channel: int = 0,
        target_channel: int | None = None,
        val_movies: Sequence[str] = ("0608",),
        window_lengths: Sequence[int] = DEFAULT_WINDOW_LENGTHS,
        frames_per_batch: int = 1024,
        num_workers: int = 8,
        seed: int = 42,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.train_split = train_split
        self.predict_split = predict_split
        self.source_channel = source_channel
        self.target_channel = target_channel
        self.val_movies = tuple(val_movies)
        self.window_lengths = tuple(window_lengths)
        self.frames_per_batch = frames_per_batch
        self.num_workers = num_workers
        self.seed = seed
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets for a stage."""
        if stage in ("fit", "validate", None):
            root = self.data_path / self.train_split
            names = _track_names(root)
            val_names = [n for n in names if n.split(".")[0] in self.val_movies]
            train_names = [n for n in names if n.split(".")[0] not in self.val_movies]
            if not val_names:
                raise ValueError(
                    f"No tracks from movies {self.val_movies} in {root}; "
                    f"available movies are {sorted({n.split('.')[0] for n in names})}."
                )
            self.train_dataset = NpyTrackDataset(
                root,
                train_names,
                self.source_channel,
                self.target_channel,
                self.window_lengths,
                seed=self.seed,
            )
            # Validation uses whole tracks, which is how the model is actually evaluated.
            self.val_dataset = NpyTrackDataset(
                root, val_names, self.source_channel, self.target_channel, whole_track=True
            )
        if stage in ("predict", "test"):
            root = self.data_path / self.predict_split
            self.predict_dataset = NpyTrackDataset(
                root, _track_names(root), self.source_channel, self.target_channel, whole_track=True
            )

    def _dataloader(self, dataset: NpyTrackDataset, shuffle: bool, drop_last: bool) -> ThreadDataLoader:
        sampler = LengthBucketedBatchSampler(
            dataset.window_lengths(),
            # Whole tracks are long and all different, so bucketing yields one clip per batch anyway;
            # asking for a single frame budget keeps that explicit.
            frames_per_batch=1 if dataset.whole_track else self.frames_per_batch,
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
        """Return the training loader, reseeding clip offsets for the new epoch."""
        self.train_dataset.set_epoch(self.trainer.current_epoch if self.trainer else 0)
        return self._dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> ThreadDataLoader:
        """Return the validation loader over whole tracks."""
        return self._dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> ThreadDataLoader:
        """Return the prediction loader over whole tracks."""
        return self._dataloader(self.predict_dataset, shuffle=False, drop_last=False)
