"""Clips of the Leger et al. (MELBA 2026) RPE1 cell-cycle tracks from cached foundation-model embeddings.

The dataset (``data/README.md``) stores one file per full mitosis-to-mitosis track and split:

- ``<split>/images/<movie>.<track>.npy``: ``(T, 2, 64, 64)`` float16, brightfield and H2B, each frame z-scored;
- ``<split>/labels/<movie>.<track>.npy``: ``(2, T, 1)`` float16, FUCCI Geminin (channel 0) and Cdt1 (channel 1), each
  min-max scaled to ``[0, 1]`` over its track. The label axes are transposed relative to the images.

:func:`write_index` flattens every frame of every split into one row table, ``index.csv``, with the labels as
``labels.npy`` ``(N, 2)``; rows run over the splits in :data:`SPLITS` order, the tracks of a split by name and the
frames of a track in time order, so every track is a contiguous run of rows. The embedding caches written by
:mod:`dynaformer.leger_embed` follow these rows and record the index's fingerprint, which
:class:`LegerDataModule` checks before reading them.

Clips follow :mod:`dynaformer.cellcycle_data`: training draws one clip per track and epoch at a random offset and in
a random view, cut to a multiple of ``length_quantum`` when the track is shorter than the window; evaluation runs every
track whole in the identity view. The published splits have no validation set: the movies in ``val_movies`` are held
out of ``train`` for it, so ``test`` and ``test_drugged`` stay untouched until scoring.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dynaformer.cellcycle_data import N_VIEWS, bucketed_loader, training_clip_length
from viscy_data.npy_track import _track_lengths

__all__ = [
    "REPORTERS",
    "SPLITS",
    "LegerClips",
    "LegerDataModule",
    "LegerTrackFrames",
    "build_index",
    "index_fingerprint",
    "load_index",
    "plan_chunks",
    "track_table",
    "write_index",
]

#: Published splits, in row order.
SPLITS = ("train", "test", "test_drugged")
#: Label channels in stored order: FUCCI Geminin (the paper's FUCCI2), then FUCCI Cdt1 (FUCCI1).
REPORTERS = ("geminin", "cdt1")
INDEX_COLUMNS = ["row", "name", "movie", "track", "split", "t", "track_index"]
_INDEX_DTYPES = {"name": str, "movie": str, "split": str}


def build_index(data_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per frame of every split, and the frame's two labels.

    Parameters
    ----------
    data_dir : str | Path
        Dataset root holding ``train/``, ``test/`` and ``test_drugged/``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        The index (columns :data:`INDEX_COLUMNS`) and the labels ``(N, 2)`` float32 in stored channel order.

    Raises
    ------
    ValueError
        If a split's images and labels do not name the same tracks, or a track's images and labels differ in length.
    """
    data_dir = Path(data_dir)
    tables, labels = [], []
    track_index = 0
    for split in SPLITS:
        names = sorted(p.stem for p in (data_dir / split / "labels").glob("*.npy"))
        image_names = sorted(p.stem for p in (data_dir / split / "images").glob("*.npy"))
        if not names or names != image_names:
            raise ValueError(f"{data_dir / split}: images/ and labels/ must hold the same non-empty set of tracks.")
        lengths = _track_lengths([data_dir / split / "images" / f"{name}.npy" for name in names])
        for name, length in zip(names, lengths, strict=True):
            label = np.load(data_dir / split / "labels" / f"{name}.npy")
            if label.shape != (2, length, 1):
                raise ValueError(f"{split}/{name}: labels {label.shape} do not match {length} image frames.")
            labels.append(label[:, :, 0].T.astype(np.float32))
            movie, track = name.split(".")
            tables.append(
                pd.DataFrame(
                    {"name": name, "movie": movie, "track": int(track), "split": split, "t": np.arange(length)}
                ).assign(track_index=track_index)
            )
            track_index += 1
    index = pd.concat(tables, ignore_index=True)
    index.insert(0, "row", np.arange(len(index)))
    return index[INDEX_COLUMNS], np.concatenate(labels)


def write_index(data_dir: str | Path, out_dir: str | Path) -> pd.DataFrame:
    """Write ``index.csv`` and ``labels.npy`` for every frame of the dataset to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index, labels = build_index(data_dir)
    index.to_csv(out_dir / "index.csv", index=False)
    np.save(out_dir / "labels.npy", labels)
    return index


def index_fingerprint(index_dir: str | Path) -> str:
    """SHA-1 of ``index.csv``: what an embedding cache's rows are aligned to."""
    return hashlib.sha1((Path(index_dir) / "index.csv").read_bytes()).hexdigest()


def load_index(index_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the index and the labels written by :func:`write_index`."""
    index_dir = Path(index_dir)
    index = pd.read_csv(index_dir / "index.csv", dtype=_INDEX_DTYPES)
    labels = np.load(index_dir / "labels.npy")
    if list(index.columns) != INDEX_COLUMNS or not (index.row.to_numpy() == np.arange(len(index))).all():
        raise ValueError(f"{index_dir / 'index.csv'} is not an index written by write_index.")
    if labels.shape != (len(index), 2):
        raise ValueError(f"labels.npy {labels.shape} does not match {len(index)} index rows.")
    return index, labels


def track_table(index: pd.DataFrame) -> pd.DataFrame:
    """One row per track: ``track_index``, ``name``, ``split``, first ``row`` and ``length``, in row order."""
    tracks = index.groupby("track_index", sort=True).agg(
        name=("name", "first"), split=("split", "first"), row=("row", "first"), length=("row", "size")
    )
    return tracks.reset_index()


def plan_chunks(tracks: pd.DataFrame, frames_per_chunk: int) -> list[tuple[int, int]]:
    """Cut the rows into ``[start, end)`` chunks of whole tracks, each of at most ``frames_per_chunk`` frames.

    A track longer than ``frames_per_chunk`` gets a chunk of its own.
    """
    if frames_per_chunk < 1:
        raise ValueError(f"frames_per_chunk must be positive, got {frames_per_chunk}.")
    chunks: list[tuple[int, int]] = []
    start = end = int(tracks.row.iloc[0])
    for row, length in zip(tracks.row, tracks.length, strict=True):
        if row != end:
            raise ValueError(f"Track rows are not contiguous at row {row}.")
        if end > start and end + length - start > frames_per_chunk:
            chunks.append((start, end))
            start = end
        end = row + length
    chunks.append((start, end))
    return chunks


class LegerTrackFrames(Dataset):
    """The frames of whole tracks, one track per item, for embedding.

    Parameters
    ----------
    data_dir : str | Path
        Dataset root.
    tracks : pd.DataFrame
        Rows of :func:`track_table` to read, in row order.
    channel : int
        Image channel: 0 brightfield, 1 H2B, by default 0.
    """

    def __init__(self, data_dir: str | Path, tracks: pd.DataFrame, channel: int = 0) -> None:
        self.data_dir = Path(data_dir)
        self.tracks = tracks.reset_index(drop=True)
        self.channel = channel

    def __len__(self) -> int:
        """Return the number of tracks."""
        return len(self.tracks)

    def __getitem__(self, i: int) -> tuple[int, Tensor]:
        """Return the track's first row and its frames ``(T, 1, Y, X)``, float32, as stored."""
        track = self.tracks.iloc[i]
        images = np.load(self.data_dir / track.split / "images" / f"{track['name']}.npy", mmap_mode="r")
        if len(images) != track.length:
            raise ValueError(f"{track.split}/{track['name']}: {len(images)} frames, the index says {track.length}.")
        frames = np.asarray(images[:, self.channel : self.channel + 1], dtype=np.float32)
        return int(track.row), torch.from_numpy(frames)


class LegerClips(Dataset):
    """Clips of cached embeddings with their two labels per frame.

    Parameters
    ----------
    index : pd.DataFrame
        Index rows of the tracks to draw clips from; whole tracks only.
    labels : np.ndarray
        Labels of every index row, ``(N, 2)``.
    embeddings : str | Path
        Embedding cache ``(N, 8, D)`` aligned with the index.
    feature_slice : slice
        Part of the cached embedding to read.
    train : bool
        Training clips (windows at random offsets, random views) or evaluation clips (whole tracks, identity view).
    window_size : int
        Training only: longest clip.
    length_quantum : int
        Training only: clips shorter than ``window_size`` are cut to a multiple of this many frames.
    min_frames : int
        Tracks shorter than this give no clips.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        labels: np.ndarray,
        embeddings: str | Path,
        feature_slice: slice,
        train: bool,
        window_size: int,
        length_quantum: int,
        min_frames: int,
    ) -> None:
        self.embeddings_path = Path(embeddings)
        self.feature_slice = feature_slice
        self.train = train
        self.labels = labels
        self.clips: list[tuple[int, int, int, int]] = []  # (first row, length, clip length, track index)
        for track in track_table(index).itertuples():
            if track.length < min_frames:
                continue
            clip_length = training_clip_length(track.length, window_size, length_quantum) if train else track.length
            self.clips.append((track.row, track.length, clip_length, track.track_index))
        if not self.clips:
            raise ValueError(f"No clips of at least {min_frames} frames.")
        self._cache: np.ndarray | None = None

    def __getstate__(self) -> dict:
        """Drop the open memory map; each worker reopens it."""
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def window_lengths(self) -> np.ndarray:
        """Clip length of every index, for the length-bucketed batch sampler."""
        return np.array([clip_length for _, _, clip_length, _ in self.clips], dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of clips, one per track."""
        return len(self.clips)

    def __getitems__(self, indices: list[int]) -> dict:
        """Read a batch of clips of one length.

        Returns
        -------
        dict
            ``embeddings`` ``(B, T, D)``, ``target`` ``(B, T, 2)``, ``rows`` ``(B, T)``, ``track_id`` ``(B,)`` (the
            track index), ``t`` ``(B, T)`` and ``view`` ``(B,)``.
        """
        clips = [self.clips[i] for i in indices]
        lengths = {clip_length for _, _, clip_length, _ in clips}
        if len(lengths) != 1:
            raise ValueError(f"A batch must hold clips of one length, got {sorted(lengths)}.")
        frames = lengths.pop()
        if self._cache is None:
            self._cache = np.load(self.embeddings_path, mmap_mode="r")
        rng = np.random.default_rng(int(torch.randint(2**31 - 1, (1,)).item()))
        offsets = np.array(
            [int(rng.integers(0, length - frames + 1)) if self.train else 0 for _, length, _, _ in clips]
        )
        starts = np.array([row for row, _, _, _ in clips]) + offsets
        views = rng.integers(N_VIEWS, size=len(clips)) if self.train else np.zeros(len(clips), dtype=np.int64)
        rows = starts[:, None] + np.arange(frames)[None, :]
        embeddings = np.stack(
            [
                np.asarray(self._cache[start : start + frames, view, self.feature_slice])
                for start, view in zip(starts, views, strict=True)
            ]
        )
        return {
            "embeddings": torch.from_numpy(embeddings.astype(np.float32)),
            "target": torch.from_numpy(self.labels[rows]),
            "rows": torch.from_numpy(rows),
            "track_id": torch.tensor([track for _, _, _, track in clips]),
            "t": torch.from_numpy(offsets[:, None] + np.arange(frames)[None, :]),
            "view": torch.from_numpy(views),
        }


class LegerDataModule(LightningDataModule):
    """Train, validation and test clips of the Leger et al. RPE1 tracks, from one cached encoder.

    The index and the labels are read from the cache's folder (:func:`write_index`), and the cache must have been
    written against that same index (its ``.json`` records the fingerprint).

    Parameters
    ----------
    embeddings : str
        Embedding cache ``<encoder>.npy`` written by :mod:`dynaformer.leger_embed`.
    embedding_features : {"all", "token", "center"}
        Part of the cached embedding to use, by default ``"all"``.
    val_movies : Sequence[str]
        Training movies held out for validation, by default ``("0608",)``, as in the August dynaformer runs.
    window_size : int
        Longest training clip, by default 192 (the longest training window of Leger et al.).
    length_quantum : int
        Shorter training clips are cut to a multiple of this many frames, by default 8.
    min_frames : int
        Tracks shorter than this are skipped, by default 8.
    frames_per_batch : int
        Training frame budget per batch; batch size is ``frames_per_batch // T``, by default 6144.
    eval_frames_per_batch : int
        Validation and test frame budget per batch, by default 4096.
    test_split : {"test", "test_drugged", "val", "train"}
        Split the test stage evaluates, by default ``"test"``.
    num_workers : int
        Data loader worker processes, by default 8.
    seed : int
        Seed of the batch sampler, by default 42.

    Attributes
    ----------
    reporters : tuple[str, ...]
        Names of the two targets in stored order, Geminin and Cdt1.
    stats : dict
        Scales of the targets (1: the labels are already in ``[0, 1]``) and no PIP-FUCCI thresholds.
    """

    reporters = REPORTERS

    def __init__(
        self,
        embeddings: str,
        embedding_features: Literal["all", "token", "center"] = "all",
        val_movies: Sequence[str] = ("0608",),
        window_size: int = 192,
        length_quantum: int = 8,
        min_frames: int = 8,
        frames_per_batch: int = 6144,
        eval_frames_per_batch: int = 4096,
        test_split: Literal["test", "test_drugged", "val", "train"] = "test",
        num_workers: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.embeddings = Path(embeddings)
        self.val_movies = list(val_movies)
        self.window_size = window_size
        self.length_quantum = length_quantum
        self.min_frames = min_frames
        self.frames_per_batch = frames_per_batch
        self.eval_frames_per_batch = eval_frames_per_batch
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed
        self.stats = {"fucci": {"scale": [1.0, 1.0]}, "thresholds": None}

        meta_path = self.embeddings.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} is missing: the cache is not finalized (dynaformer.leger_embed).")
        meta = json.loads(meta_path.read_text())
        index_dir = self.embeddings.parent
        if meta["index_sha1"] != index_fingerprint(index_dir):
            raise ValueError(f"{self.embeddings} was written against a different {index_dir / 'index.csv'}.")
        self.index, self.labels = load_index(index_dir)
        token_dim = meta["token_dim"]
        self.feature_slice = {"all": slice(None), "token": slice(0, token_dim), "center": slice(token_dim, None)}[
            embedding_features
        ]
        self.embed_dim = len(range(meta["embed_dim"])[self.feature_slice])
        train_movies = set(self.index.movie[self.index.split == "train"])
        if not set(self.val_movies) <= train_movies:
            raise ValueError(f"Validation movies {self.val_movies} must be training movies {sorted(train_movies)}.")

    def _rows(self, split: str) -> pd.DataFrame:
        """Index rows of a split; ``val`` is the held-out training movies and ``train`` the rest."""
        index = self.index
        if split in ("train", "val"):
            held_out = index.movie.isin(self.val_movies)
            return index[(index.split == "train") & (held_out if split == "val" else ~held_out)]
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}.")
        return index[index.split == split]

    def _dataset(self, split: str, train: bool) -> LegerClips:
        return LegerClips(
            index=self._rows(split),
            labels=self.labels,
            embeddings=self.embeddings,
            feature_slice=self.feature_slice,
            train=train,
            window_size=self.window_size,
            length_quantum=self.length_quantum,
            min_frames=self.min_frames,
        )

    def setup(self, stage: str) -> None:
        """Build the datasets the stage needs."""
        if stage in ("fit", "validate"):
            self.train_dataset = self._dataset("train", train=True)
            self.val_dataset = self._dataset("val", train=False)
        elif stage == "test":
            self.test_dataset = self._dataset(self.test_split, train=False)
        else:
            raise NotImplementedError(f"Stage {stage!r} is not supported; use test to write predictions.")

    def train_dataloader(self) -> DataLoader:
        """Return the training loader."""
        return bucketed_loader(
            self.train_dataset, self.frames_per_batch, True, self.seed, self.num_workers, self.trainer
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation loader: every validation track, whole."""
        return bucketed_loader(
            self.val_dataset, self.eval_frames_per_batch, False, self.seed, self.num_workers, self.trainer
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test loader: every track of ``test_split``, whole."""
        return bucketed_loader(
            self.test_dataset, self.eval_frames_per_batch, False, self.seed, self.num_workers, self.trainer
        )
