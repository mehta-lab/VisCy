"""Clips of tracked nuclei for cell-cycle prediction from cached foundation-model embeddings.

Reads a dataset folder written by ``Bill/5-cellcycle`` (``extract.py`` and ``split.py``):

- ``points.csv``: one row per track point, sorted by ``(track_id, t)``, with the nucleus's Geminin-mKO and
  PIP-mVenus medians minus the frame's background (``mko``, ``mvenus``) and that background (``bg_mko``,
  ``bg_mvenus``);
- ``split_tracks.csv``: the split of every track;
- ``stats.json``: phase decoding and statistics, FUCCI scales, PIP-FUCCI thresholds;
- ``crops/{phase,fucci,mask}.npy``: crops centred on every track point, row-aligned with ``points.csv``.

The frozen encoder E never runs during training. Its embedding of every track point, under each of the 8
rotations/flips of the square (:func:`d4_view`), is read from a cache written by :mod:`dynaformer.embed`. A clip
draws one view and applies it to both its embeddings and its images, exactly as augmenting the movie would. An encoder
trained with the model cannot be cached: with ``phase_frames`` a batch carries the phase crop of every frame instead,
the same centred crop the cache encodes, in the clip's view, and E runs inside the model.

Embeddings are cheap and images are not, so a batch carries the embeddings of every frame (the temporal context and
the dense regression targets) but images of only some frames: ``images_per_batch`` random frames in training, every
``val_image_stride``-th frame of each track in validation, and every frame in testing.

Training draws one clip per track per epoch, at a random offset: tracks, not frames, are the independent samples, so
long tracks are not counted several times. Clip lengths are rounded down to a multiple of ``length_quantum`` so that
the length-bucketed batches (one length per batch, no padding) fill up: with a bucket per exact length, most lengths
have fewer tracks than a batch holds and ``drop_last`` would silently discard them.
"""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dynaformer.cellcycle_metrics import REPORTERS
from viscy_data.sampler import LengthBucketedBatchSampler

__all__ = [
    "CellCycleDataModule",
    "CellCycleDataset",
    "bucketed_loader",
    "d4_view",
    "decode_phase",
    "load_points",
    "training_clip_length",
]

_logger = logging.getLogger("lightning.pytorch")

N_VIEWS = 8
TARGET_COLUMNS = ["mko", "mvenus"]
BACKGROUND_COLUMNS = ["bg_mko", "bg_mvenus"]


def d4_view(x, view: int):
    """Apply one of the 8 rotations/flips of the square to the last two axes.

    Views 0-3 rotate by ``view`` quarter turns; views 4-7 flip X first. Works on NumPy arrays and tensors alike, so
    the embedding cache and the data loader transform images identically.
    """
    if not 0 <= view < N_VIEWS:
        raise ValueError(f"view must lie in [0, {N_VIEWS}), got {view}.")
    if isinstance(x, Tensor):
        if view >= 4:
            x = torch.flip(x, dims=(-1,))
        return torch.rot90(x, k=view % 4, dims=(-2, -1))
    if view >= 4:
        x = x[..., ::-1]
    return np.rot90(x, k=view % 4, axes=(-2, -1))


def decode_phase(raw: np.ndarray, phase_stats: dict) -> np.ndarray:
    """Decode stored uint16 phase to radians and z-score it; mosaic padding (0) becomes 0."""
    radians = raw.astype(np.float32) / phase_stats["scale"] - phase_stats["offset_neg"]
    z = (radians - phase_stats["mean"]) / phase_stats["std"]
    return np.where(raw == phase_stats["padding_value"], np.float32(0), z).astype(np.float32)


def training_clip_length(length: int, window_size: int, length_quantum: int) -> int:
    """Length of a track's training clip.

    A track at least ``window_size`` long gives a window; a shorter one is cut to a multiple of ``length_quantum``
    (keeping at least ``min(length, length_quantum)`` frames), so that the length buckets fill their batches.
    """
    if length >= window_size:
        return window_size
    return max(length - length % length_quantum, min(length, length_quantum))


def load_points(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the track points with their split, and the dataset statistics."""
    data_dir = Path(data_dir)
    points = pd.read_csv(data_dir / "points.csv")
    if not (points.row.to_numpy() == np.arange(len(points))).all():
        raise ValueError("points.csv rows are not in crop order.")
    split = pd.read_csv(data_dir / "split_tracks.csv", usecols=["track_id", "split"])
    points = points.merge(split, on="track_id", how="left", validate="many_to_one")
    if points.split.isna().any():
        raise ValueError("Some tracks in points.csv have no split.")
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    return points, stats


class CellCycleDataset(Dataset):
    """Clips of tracked nuclei: cached embeddings for every frame, images for some.

    Parameters
    ----------
    data_dir : str | Path
        Dataset folder (see the module docstring).
    points : pd.DataFrame
        Rows of ``points.csv`` to draw clips from; whole tracks only.
    stats : dict
        Contents of ``stats.json``.
    embeddings : str | Path | None
        Embedding cache ``(N, 8, D)`` written by :mod:`dynaformer.embed`. ``None`` for a model that reads no
        embeddings (the unconditioned virtual-staining baseline).
    embedding_features : {"all", "token", "center"}
        Part of the cached embedding to use: the global token, the central patch-token mean, or both.
    phase_frames : bool
        Load the centred phase crop of every frame, for an encoder that runs inside the model. Excludes
        ``embeddings``.
    train : bool
        Training clips (windows, random views, random image frames, crop jitter) or evaluation clips (whole tracks,
        identity view, fixed image frames, centred crops).
    window_size : int
        Training only: longest clip.
    length_quantum : int
        Training only: clips shorter than ``window_size`` are cut to a multiple of this many frames.
    min_frames : int
        Tracks shorter than this give no clips.
    images_per_batch : int
        Training only: frames per batch that get images. 0 loads no images.
    image_stride : int
        Evaluation only: every ``image_stride``-th frame of a track gets an image; 0 loads no images.
    image_size : int
        Side of the image crop fed to the model.
    jitter : int
        Training only: largest shift of the image crop from the centre, in pixels.
    """

    def __init__(
        self,
        data_dir: str | Path,
        points: pd.DataFrame,
        stats: dict,
        embeddings: str | Path | None,
        embedding_features: Literal["all", "token", "center"] = "all",
        phase_frames: bool = False,
        train: bool = True,
        window_size: int = 64,
        length_quantum: int = 8,
        min_frames: int = 8,
        images_per_batch: int = 0,
        image_stride: int = 0,
        image_size: int = 224,
        jitter: int = 16,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.stats = stats
        self.embeddings_path = None if embeddings is None else Path(embeddings)
        if phase_frames and embeddings is not None:
            raise ValueError("Pass an embedding cache or phase_frames, not both.")
        self.phase_frames = phase_frames
        self.train = train
        self.images_per_batch = images_per_batch
        self.image_stride = image_stride
        self.image_size = image_size
        self.jitter = jitter
        crop = stats["crop"]
        if image_size > crop:
            raise ValueError(f"image_size {image_size} exceeds the stored crop size {crop}.")
        if 2 * jitter > crop - image_size:
            raise ValueError(f"jitter {jitter} does not fit: crops are {crop} px and images {image_size} px.")

        self.feature_slice = slice(None)
        if self.embeddings_path is not None:
            with open(self.embeddings_path.with_suffix(".json")) as f:
                meta = json.load(f)
            token_dim = meta["token_dim"]
            self.feature_slice = {
                "all": slice(None),
                "token": slice(0, token_dim),
                "center": slice(token_dim, None),
            }[embedding_features]
            self.embed_dim = len(range(meta["embed_dim"])[self.feature_slice])

        scale = np.asarray(stats["fucci"]["scale"], dtype=np.float32)
        self.scale = scale
        # Positions in self.points are "local"; `row` is the global row of the crop and embedding arrays.
        self.points = points.reset_index(drop=True)
        self._targets = (self.points[TARGET_COLUMNS].to_numpy(np.float32) / scale).astype(np.float32)
        self._background = self.points[BACKGROUND_COLUMNS].to_numpy(np.float32)
        self._rows = self.points.row.to_numpy()

        # One clip per track: (first local position of the track, track length, clip length).
        self.clips: list[tuple[int, int, int]] = []
        for _, track in self.points.groupby("track_id", sort=True):
            local = track.index.to_numpy()
            start, length = int(local[0]), len(local)
            contiguous = (np.diff(track.t.to_numpy()) == 1).all() and (np.diff(local) == 1).all()
            if not contiguous or not (np.diff(self._rows[local]) == 1).all():
                raise ValueError(f"Track {track.track_id.iloc[0]} is not a contiguous run of frames and rows.")
            if length < min_frames:
                continue
            clip_length = training_clip_length(length, window_size, length_quantum) if train else length
            self.clips.append((start, length, clip_length))
        if not self.clips:
            raise ValueError(f"No clips of at least {min_frames} frames.")
        self._arrays: dict | None = None

    def __getstate__(self) -> dict:
        """Drop the open memory maps; each worker reopens them."""
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open(self) -> dict:
        """Open the crop and embedding arrays as read-only memory maps."""
        if self._arrays is None:
            crops = self.data_dir / "crops"
            self._arrays = {name: np.load(crops / f"{name}.npy", mmap_mode="r") for name in ("phase", "fucci", "mask")}
            if self.embeddings_path is not None:
                self._arrays["embeddings"] = np.load(self.embeddings_path, mmap_mode="r")
        return self._arrays

    def window_lengths(self) -> np.ndarray:
        """Clip length of every index, for the length-bucketed batch sampler."""
        return np.array([clip_length for _, _, clip_length in self.clips], dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of clips, one per track."""
        return len(self.clips)

    def _image_frames(self, n_clips: int, frames: int, rng: np.random.Generator) -> np.ndarray:
        """Flat indices into the ``(n_clips, frames)`` grid of the frames that get images."""
        if self.train:
            if self.images_per_batch == 0:
                return np.zeros(0, dtype=np.int64)
            chosen = rng.choice(n_clips * frames, size=min(self.images_per_batch, n_clips * frames), replace=False)
            return np.sort(chosen)
        if self.image_stride == 0:
            return np.zeros(0, dtype=np.int64)
        frame_index = np.arange(0, frames, self.image_stride)
        return (np.arange(n_clips)[:, None] * frames + frame_index[None, :]).ravel()

    def __getitems__(self, indices: list[int]) -> dict:
        """Read a batch of clips of one length.

        Returns
        -------
        dict
            ``embeddings`` ``(B, T, D)`` (when an embedding cache is set) or ``frames`` ``(B, T, 1, S, S)`` (with
            ``phase_frames``), ``target`` ``(B, T, 2)``, ``rows``,
            ``track_id``, ``t``, ``view``; and for the frames with images, ``image_index`` ``(M,)`` (flat indices into
            ``B * T``), ``source`` ``(M, 1, S, S)``, ``target_image`` ``(M, 2, S, S)``, ``mask`` ``(M, 1, S, S)``.
        """
        lengths = {self.clips[i][2] for i in indices}
        if len(lengths) != 1:
            raise ValueError(f"A batch must hold clips of one length, got {sorted(lengths)}.")
        frames = lengths.pop()
        arrays = self._open()
        rng = np.random.default_rng(int(torch.randint(2**31 - 1, (1,)).item()))
        # Training reads each track's clip at a random offset; evaluation reads whole tracks.
        starts = np.array(
            [
                start + (int(rng.integers(0, length - frames + 1)) if self.train else 0)
                for start, length, _ in (self.clips[i] for i in indices)
            ]
        )
        local = starts[:, None] + np.arange(frames)[None, :]  # rows of self.points
        views = rng.integers(N_VIEWS, size=len(indices)) if self.train else np.zeros(len(indices), dtype=np.int64)

        batch: dict = {
            "rows": torch.from_numpy(self._rows[local]),
            "track_id": torch.from_numpy(self.points.track_id.to_numpy()[starts]),
            "t": torch.from_numpy(self.points.t.to_numpy()[local]),
            "view": torch.from_numpy(views),
            "target": torch.from_numpy(self._targets[local]),
        }
        if self.embeddings_path is not None:
            cache = arrays["embeddings"]
            batch["embeddings"] = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(cache[self._rows[start] : self._rows[start] + frames, view, self.feature_slice])
                        for start, view in zip(starts, views, strict=True)
                    ]
                ).astype(np.float32)
            )
        if self.phase_frames:
            batch["frames"] = self._read_frames(starts, frames, views)

        image_index = self._image_frames(len(indices), frames, rng)
        if image_index.size:
            batch.update(self._read_images(local.ravel()[image_index], views[image_index // frames], rng))
            batch["image_index"] = torch.from_numpy(image_index)
        return batch

    def _read_frames(self, starts: np.ndarray, frames: int, views: np.ndarray) -> Tensor:
        """Every frame's centred phase crop, ``(B, T, 1, S, S)``: what :mod:`dynaformer.embed` encodes."""
        arrays = self._open()
        margin = (self.stats["crop"] - self.image_size) // 2
        window = slice(margin, margin + self.image_size)
        clips = []
        for start, view in zip(starts, views, strict=True):
            row = self._rows[start]
            phase = decode_phase(np.asarray(arrays["phase"][row : row + frames, window, window]), self.stats["phase"])
            clips.append(np.ascontiguousarray(d4_view(phase, int(view))))
        return torch.from_numpy(np.stack(clips)[:, :, None])

    def _read_images(self, local_rows: np.ndarray, views: np.ndarray, rng: np.random.Generator) -> dict:
        """Read, crop, rotate and normalize the images of the given rows."""
        arrays = self._open()
        crop, size = self.stats["crop"], self.image_size
        margin = (crop - size) // 2
        sources, targets, masks = [], [], []
        for local_row, view in zip(local_rows, views, strict=True):
            row = self._rows[local_row]
            y0, x0 = (margin + rng.integers(-self.jitter, self.jitter + 1, size=2)) if self.train else (margin, margin)
            window = (slice(y0, y0 + size), slice(x0, x0 + size))
            phase = decode_phase(np.asarray(arrays["phase"][row][window]), self.stats["phase"])
            fucci = np.asarray(arrays["fucci"][row][:, window[0], window[1]], dtype=np.float32)
            fucci = (fucci - self._background[local_row][:, None, None]) / self.scale[:, None, None]
            mask = np.asarray(arrays["mask"][row][window]) > 0
            sources.append(np.ascontiguousarray(d4_view(phase, int(view)))[None])
            targets.append(np.ascontiguousarray(d4_view(fucci, int(view))))
            masks.append(np.ascontiguousarray(d4_view(mask, int(view)))[None])
        return {
            "source": torch.from_numpy(np.stack(sources)),
            "target_image": torch.from_numpy(np.stack(targets).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(masks)),
        }


def _identity_collate(batch: dict) -> dict:
    return batch


def bucketed_loader(
    dataset: Dataset, frames_per_batch: int, shuffle: bool, seed: int, num_workers: int, trainer=None
) -> DataLoader:
    """Batches of clips of one length each (``dataset.window_lengths()``), read with ``dataset.__getitems__``.

    Partial batches are kept: a dropped one would take its tracks out of the epoch. Under DDP every rank reads its
    own slice of the same batch sequence.
    """
    sampler = LengthBucketedBatchSampler(
        dataset.window_lengths(),
        frames_per_batch=frames_per_batch,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
        num_replicas=trainer.world_size if trainer else 1,
        rank=trainer.global_rank if trainer else 0,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )


class CellCycleDataModule(LightningDataModule):
    """Train, validation and test clips of tracked nuclei with cell-cycle reporter targets.

    The split comes from ``split_tracks.csv``: training draws one clip of up to ``window_size`` frames per training
    track and epoch, while validation and testing run every track whole, which is how the model is used.

    Parameters
    ----------
    data_dir : str
        Dataset folder written by ``Bill/5-cellcycle``.
    embeddings : str | None
        Embedding cache written by :mod:`dynaformer.embed`. ``None`` reads no embeddings.
    embedding_features : {"all", "token", "center"}
        Part of the cached embedding to use, by default ``"all"``.
    phase_frames : bool
        Load the centred phase crop of every frame for an encoder trained inside the model, instead of ``embeddings``,
        by default ``False``.
    window_size : int
        Longest training clip, by default 64.
    length_quantum : int
        Shorter training clips are cut to a multiple of this many frames, so every length bucket fills batches,
        by default 8.
    min_frames : int
        Tracks shorter than this are skipped, by default 8.
    frames_per_batch : int
        Training frame budget per batch; batch size is ``frames_per_batch // T``, by default 1024.
    eval_frames_per_batch : int
        Validation and test frame budget per batch, by default 512.
    images_per_batch : int
        Training frames per batch that get images; 0 loads no images anywhere (a regression-only model), by default 0.
    val_image_stride : int
        Every this many frames of a validation track gets an image, by default 4. Testing images every frame.
    image_size : int
        Image side fed to the model, by default 224.
    jitter : int
        Largest training shift of the image crop from the centre, by default 16.
    test_split : {"test", "val", "train"}
        Split the test stage evaluates, by default ``"test"``.
    num_workers : int
        Data loader worker processes, by default 8.
    seed : int
        Seed of the batch sampler, by default 42.

    Attributes
    ----------
    reporters : tuple[str, ...]
        Names of the two targets in order, Geminin-mKO and PIP-mVenus.
    stats : dict
        Contents of ``stats.json``, with the reporters' scales and the PIP-FUCCI thresholds.
    """

    reporters = REPORTERS

    def __init__(
        self,
        data_dir: str,
        embeddings: str | None = None,
        embedding_features: Literal["all", "token", "center"] = "all",
        phase_frames: bool = False,
        window_size: int = 64,
        length_quantum: int = 8,
        min_frames: int = 8,
        frames_per_batch: int = 1024,
        eval_frames_per_batch: int = 512,
        images_per_batch: int = 0,
        val_image_stride: int = 4,
        image_size: int = 224,
        jitter: int = 16,
        test_split: Literal["test", "val", "train"] = "test",
        num_workers: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.embeddings = embeddings
        self.embedding_features = embedding_features
        self.phase_frames = phase_frames
        self.window_size = window_size
        self.length_quantum = length_quantum
        self.min_frames = min_frames
        self.frames_per_batch = frames_per_batch
        self.eval_frames_per_batch = eval_frames_per_batch
        self.images_per_batch = images_per_batch
        self.val_image_stride = val_image_stride
        self.image_size = image_size
        self.jitter = jitter
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed
        self.points, self.stats = load_points(self.data_dir)

    def _dataset(self, split: str, train: bool, image_stride: int = 0) -> CellCycleDataset:
        return CellCycleDataset(
            data_dir=self.data_dir,
            points=self.points[self.points.split == split],
            stats=self.stats,
            embeddings=self.embeddings,
            embedding_features=self.embedding_features,
            phase_frames=self.phase_frames,
            train=train,
            window_size=self.window_size,
            length_quantum=self.length_quantum,
            min_frames=self.min_frames,
            images_per_batch=self.images_per_batch,
            image_stride=image_stride if self.images_per_batch else 0,
            image_size=self.image_size,
            jitter=self.jitter,
        )

    def setup(self, stage: str) -> None:
        """Build the datasets the stage needs."""
        if stage in ("fit", "validate"):
            self.train_dataset = self._dataset("train", train=True)
            self.val_dataset = self._dataset("val", train=False, image_stride=self.val_image_stride)
            _logger.info(
                f"Tracks: {len(self.train_dataset)} train (clips of up to {self.window_size} frames), "
                f"{len(self.val_dataset)} validation"
            )
        elif stage == "test":
            self.test_dataset = self._dataset(self.test_split, train=False, image_stride=1)
        else:
            raise NotImplementedError(f"Stage {stage!r} is not supported; use test to write predictions.")

    def _loader(self, dataset: CellCycleDataset, frames_per_batch: int, shuffle: bool) -> DataLoader:
        return bucketed_loader(dataset, frames_per_batch, shuffle, self.seed, self.num_workers, self.trainer)

    def train_dataloader(self) -> DataLoader:
        """Return the training loader."""
        return self._loader(self.train_dataset, self.frames_per_batch, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation loader: every validation track, whole."""
        return self._loader(self.val_dataset, self.eval_frames_per_batch, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test loader: every track of ``test_split``, whole, with an image for every frame."""
        return self._loader(self.test_dataset, self.eval_frames_per_batch, shuffle=False)
