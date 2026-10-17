"""Whole-track trajectory metrics.

A model that scores well on per-frame reconstruction can still produce a
trajectory that jitters frame to frame, which is useless as a readout of a
process that unfolds over time. These metrics score the **shape of the predicted
trajectory** against ground truth, per track, which is what the temporal
conditioning is there to fix.

Inference runs a whole track in one pass rather than tiling fixed windows. A
sequence model with no length-dependent parameters supports that directly, and
it removes the discontinuities that appear at window boundaries.
"""

import logging
from typing import Literal

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor

from viscy_data.track_window import _pool_over_mask

_logger = logging.getLogger("lightning.pytorch")

__all__ = ["TrackTrajectoryMetrics", "channel_readout", "track_trajectory_metrics"]


def channel_readout(
    values: np.ndarray,
    readout: Literal["ratio", "identity"] = "ratio",
    channels: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """Reduce per-frame channel values to a single trajectory.

    Parameters
    ----------
    values : np.ndarray
        ``(T, C)`` per-frame channel values in physical units.
    readout : {"ratio", "identity"}
        ``"ratio"`` returns ``a / (a + b)`` over the two named channels, a
        normalized balance between two markers that is insensitive to overall
        brightness. ``"identity"`` returns the first named channel unchanged.
        By default ``"ratio"``.
    channels : tuple[int, ...]
        Channel indices the readout uses, by default ``(0, 1)``.

    Returns
    -------
    np.ndarray
        ``(T,)`` trajectory.
    """
    if readout == "identity":
        return values[:, channels[0]]
    if readout == "ratio":
        if len(channels) != 2:
            raise ValueError(f"The 'ratio' readout needs exactly two channels, got {channels}.")
        numerator = values[:, channels[0]]
        denominator = numerator + values[:, channels[1]]
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)
    raise ValueError(f"Unknown readout {readout!r}, expected 'ratio' or 'identity'.")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when either series is constant."""
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denominator == 0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denominator)


def _concordance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient.

    Unlike Pearson, this penalizes a predicted trajectory that has the right
    shape but the wrong offset or scale.
    """
    denominator = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    if denominator == 0:
        return float("nan")
    return float(2 * ((a - a.mean()) * (b - b.mean())).mean() / denominator)


def track_trajectory_metrics(
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
    truth_values: list[np.ndarray] | None = None,
    predicted_values: list[np.ndarray] | None = None,
    channels: tuple[int, ...] = (0, 1),
) -> dict[str, float]:
    """Score predicted trajectories against ground truth, averaged over tracks.

    Averaging per track rather than pooling all frames keeps long tracks from
    dominating and matches how the result is read: how well does this model
    follow *a cell*.

    Parameters
    ----------
    truths : list[np.ndarray]
        Ground-truth trajectories, one ``(T,)`` array per track.
    predictions : list[np.ndarray]
        Predicted trajectories, aligned with ``truths``.
    truth_values : list[np.ndarray] | None
        Ground-truth ``(T, C)`` channel values. Together with
        ``predicted_values`` this enables the dominance metric, which asks
        whether the model gets the *larger* channel right — a coarse but
        threshold-free check, by default ``None``.
    predicted_values : list[np.ndarray] | None
        Predicted ``(T, C)`` channel values, by default ``None``.
    channels : tuple[int, ...]
        Channel indices compared by the dominance metric, by default ``(0, 1)``.

    Returns
    -------
    dict[str, float]
        ``pearson``, ``ccc``, ``mae``, ``n_tracks``, and ``dominance`` when the
        per-channel values are supplied.
    """
    if len(truths) != len(predictions):
        raise ValueError(f"Got {len(truths)} truths and {len(predictions)} predictions.")
    pearson, ccc, mae = [], [], []
    for truth, prediction in zip(truths, predictions, strict=True):
        finite = np.isfinite(truth) & np.isfinite(prediction)
        if finite.sum() < 2:
            continue
        truth, prediction = truth[finite], prediction[finite]
        pearson.append(_pearson(truth, prediction))
        ccc.append(_concordance(truth, prediction))
        mae.append(float(np.abs(truth - prediction).mean()))

    metrics = {
        "pearson": float(np.nanmean(pearson)) if pearson else float("nan"),
        "ccc": float(np.nanmean(ccc)) if ccc else float("nan"),
        "mae": float(np.nanmean(mae)) if mae else float("nan"),
        "n_tracks": float(len(pearson)),
    }
    if truth_values is not None and predicted_values is not None:
        agreements = [
            float(
                (
                    (truth[:, channels[0]] > truth[:, channels[1]])
                    == (predicted[:, channels[0]] > predicted[:, channels[1]])
                ).mean()
            )
            for truth, predicted in zip(truth_values, predicted_values, strict=True)
        ]
        metrics["dominance"] = float(np.mean(agreements)) if agreements else float("nan")
    return metrics


def _denormalize(
    values: Tensor,
    norm_meta: dict,
    channel_names: list[str],
    level: str,
    subtrahend: str,
    divisor: str,
) -> Tensor:
    """Undo channel normalization so a nonlinear readout is computed in physical units.

    Parameters
    ----------
    values : Tensor
        ``(B, T, C)`` normalized per-frame channel values.
    norm_meta : dict
        Batched normalization statistics as produced by the data module.
    channel_names : list[str]
        Channel names, aligned with the last axis of ``values``.
    level : str
        Statistics level, e.g. ``"fov_statistics"``.
    subtrahend : str
        Statistic subtracted during normalization, e.g. ``"mean"``.
    divisor : str
        Statistic divided by during normalization, e.g. ``"std"``.

    Returns
    -------
    Tensor
        ``(B, T, C)`` values in physical units.
    """
    scales, offsets = [], []
    for name in channel_names:
        stats = norm_meta[name][level]
        scales.append(stats[divisor].to(values))
        offsets.append(stats[subtrahend].to(values))
    scale = torch.stack(scales, dim=-1).reshape(-1, 1, len(channel_names))
    offset = torch.stack(offsets, dim=-1).reshape(-1, 1, len(channel_names))
    return values * scale + offset


class TrackTrajectoryMetrics(Callback):
    """Score whole-track trajectories during validation.

    Two readouts are compared against ground truth: one pooled from the decoded
    image over the object mask, and one taken straight from a regression head.
    Comparing them separates *what the image says* from *what the head says*,
    which is how a decoder that ignores temporal context gets caught — its image
    readout jitters while its head readout stays smooth.

    Logs ``val/track_pearson/image`` and ``val/track_pearson/scalar`` (plus CCC,
    MAE and dominance). Monitor ``val/track_pearson/image`` for checkpoint
    selection: it is the quantity the model is ultimately judged on, and it does
    not move with the validation loss.

    Parameters
    ----------
    channel_names : list[str]
        Target channel names, in the order the model predicts them. Used to look
        up normalization statistics.
    readout : {"ratio", "identity"}
        Trajectory reduction, by default ``"ratio"``.
    channels : tuple[int, ...]
        Channel indices the readout uses, by default ``(0, 1)``.
    norm_level : str
        Statistics level used by the data module's normalization, by default
        ``"fov_statistics"``.
    norm_subtrahend : str
        Statistic subtracted by the data module's normalization, by default ``"mean"``.
    norm_divisor : str
        Statistic divided by the data module's normalization, by default ``"std"``.
    scalar_head_name : str
        Name of the regression head to read, by default ``"scalar"``.
    every_n_epochs : int
        Run every this many validation epochs. Whole-track inference is not
        cheap, by default 1.
    max_tracks : int
        Cap on the number of tracks scored. The longest tracks are used, since
        short tracks give unstable correlations, by default 32.
    max_frames : int
        Cap on frames per track, bounding peak memory, by default 128.
    min_frames : int
        Skip tracks shorter than this, by default 8.

    Notes
    -----
    ``norm_level``, ``norm_subtrahend`` and ``norm_divisor`` must match the
    ``NormalizeSampled`` transform in the data configuration. The readout is
    nonlinear, so it has to be computed in physical units; getting these wrong
    silently changes the metric rather than raising.
    """

    def __init__(
        self,
        channel_names: list[str],
        readout: Literal["ratio", "identity"] = "ratio",
        channels: tuple[int, ...] = (0, 1),
        norm_level: str = "fov_statistics",
        norm_subtrahend: str = "mean",
        norm_divisor: str = "std",
        scalar_head_name: str = "scalar",
        every_n_epochs: int = 1,
        max_tracks: int = 32,
        max_frames: int = 128,
        min_frames: int = 8,
    ) -> None:
        super().__init__()
        self.channel_names = list(channel_names)
        self.readout = readout
        self.channels = tuple(channels)
        self.norm_level = norm_level
        self.norm_subtrahend = norm_subtrahend
        self.norm_divisor = norm_divisor
        self.scalar_head_name = scalar_head_name
        self.every_n_epochs = every_n_epochs
        self.max_tracks = max_tracks
        self.max_frames = max_frames
        self.min_frames = min_frames

    def _select_runs(self, dataset) -> list[int]:
        """Return the indices of the longest eligible track segments."""
        lengths = dataset.run_lengths()
        eligible = np.flatnonzero(lengths >= self.min_frames)
        if eligible.size == 0:
            return []
        order = eligible[np.argsort(-lengths[eligible], kind="stable")]
        return order[: self.max_tracks].tolist()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run whole-track inference and log trajectory metrics."""
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        datamodule = trainer.datamodule
        dataset = getattr(datamodule, "val_dataset", None)
        if dataset is None:
            raise ValueError("TrackTrajectoryMetrics needs a data module exposing 'val_dataset'.")
        run_indices = self._select_runs(dataset)
        if not run_indices:
            _logger.warning(f"No validation tracks with at least {self.min_frames} frames; skipping metrics.")
            return

        truths: list[np.ndarray] = []
        truth_values: list[np.ndarray] = []
        image_readouts: list[np.ndarray] = []
        image_values: list[np.ndarray] = []
        scalar_readouts: list[np.ndarray] = []
        scalar_values: list[np.ndarray] = []

        was_training = pl_module.training
        pl_module.eval()
        for run_index in run_indices:
            raw = dataset.read_run(run_index, max_frames=self.max_frames)
            batch = pl_module.transfer_batch_to_device(raw, pl_module.device, 0)
            batch = datamodule.on_after_batch_transfer(batch, 0)
            if "scalar_target" not in batch:
                raise ValueError("TrackTrajectoryMetrics needs 'scalar_target'; set mask_channel on the data module.")
            prediction, head_outputs = pl_module.model(batch["source"])
            norm_meta = batch["norm_meta"]

            truth = self._to_values(batch["scalar_target"], norm_meta)
            truths.append(channel_readout(truth, self.readout, self.channels))
            truth_values.append(truth)

            if prediction is not None:
                pooled = _pool_over_mask(prediction, batch["mask"][:, :, 0])
                values = self._to_values(pooled, norm_meta)
                image_readouts.append(channel_readout(values, self.readout, self.channels))
                image_values.append(values)

            if self.scalar_head_name in head_outputs:
                values = self._to_values(head_outputs[self.scalar_head_name], norm_meta)
                scalar_readouts.append(channel_readout(values, self.readout, self.channels))
                scalar_values.append(values)
        pl_module.train(was_training)

        for name, readouts, values in (
            ("image", image_readouts, image_values),
            ("scalar", scalar_readouts, scalar_values),
        ):
            if not readouts:
                continue
            metrics = track_trajectory_metrics(truths, readouts, truth_values, values, self.channels)
            pl_module.log_dict(
                {f"val/track_{metric}/{name}": value for metric, value in metrics.items()},
                rank_zero_only=True,
                sync_dist=False,
            )

    def _to_values(self, values: Tensor, norm_meta: dict | None) -> np.ndarray:
        """Denormalize a ``(1, T, C)`` tensor and return it as ``(T, C)``."""
        if norm_meta is None:
            raise ValueError(
                "TrackTrajectoryMetrics needs 'norm_meta' to map predictions back to physical units; "
                "the readout is nonlinear, so it cannot be computed on normalized values."
            )
        denormalized = _denormalize(
            values, norm_meta, self.channel_names, self.norm_level, self.norm_subtrahend, self.norm_divisor
        )
        return denormalized[0].float().cpu().numpy()
