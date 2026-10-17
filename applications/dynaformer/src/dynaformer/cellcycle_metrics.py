"""Scoring per-nucleus PIP-FUCCI reporter predictions.

PIP-FUCCI (Grant et al. 2018, Cell Cycle 17:2496) reads the phase from two reporters together: PIP-mVenus is on in
G1, drops within minutes at S onset and comes back in G2; Geminin-mKO rises around G1/S and stays on until anaphase.
So G1 = PIP on and Geminin off, S = PIP off and Geminin on, G2/M = both on. The thresholds come from the valley
between each reporter's off and on peaks (``stats.json``, as in ``Bill/data_analysis/pipfucci_qc``).

Three kinds of score, all from per-frame tables of true and predicted reporter levels:

- frame level, pooled over frames: Pearson r, R^2 and mean absolute error per reporter;
- track level: Pearson r per reporter along each track, averaged over tracks, which is what a readout of the cycle
  needs and what frame-to-frame jitter destroys;
- phase: G1 / S / G2/M accuracy and macro F1. True phases use the strict rule (off below 0.6x, on above 1.5x the
  valley; frames in between are unclear and left out). Predictions get a forced choice with the same cut between S
  and G2/M (PIP at its off level: PIP comes back gradually from near zero in G2) and Geminin split at its valley, so
  a perfect prediction scores 1.

The frame- and track-level scores apply to any pair of reporters (``reporters``); the phase scores need the
PIP-FUCCI thresholds and are skipped without them.
"""

import numpy as np
import pandas as pd
import torch
from torch import Tensor

__all__ = [
    "PHASES",
    "REPORTERS",
    "masked_median",
    "phase_forced",
    "phase_strict",
    "pixel_pearson",
    "score_frames",
]

PHASES = ("G1", "S", "G2/M")
REPORTERS = ("mko", "mvenus")  # Geminin-mKO, PIP-mVenus: the order of the targets everywhere


def phase_strict(gem: np.ndarray, pip: np.ndarray, thresholds: dict) -> np.ndarray:
    """Return the true PIP-FUCCI phase per frame (0 G1, 1 S, 2 G2/M, -1 unclear) by the pipfucci_qc rules."""
    phase = np.full(np.shape(gem), -1)
    gem_on = gem >= thresholds["gem_on"]
    pip_present = pip >= thresholds["pip_off"]
    phase[(gem < thresholds["gem_off"]) & pip_present] = 0
    phase[gem_on & ~pip_present] = 1
    phase[gem_on & pip_present] = 2
    return phase


def phase_forced(gem: np.ndarray, pip: np.ndarray, thresholds: dict) -> np.ndarray:
    """Return the predicted PIP-FUCCI phase per frame (0 G1, 1 S, 2 G2/M), with no unclear class.

    Geminin above its valley means S or G2/M, told apart by PIP at its off level as in :func:`phase_strict`.
    """
    high_gem = gem >= thresholds["gem_valley"]
    pip_present = pip >= thresholds["pip_off"]
    return np.where(high_gem, np.where(pip_present, 2, 1), 0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def score_frames(
    frames: pd.DataFrame,
    scale: np.ndarray,
    thresholds: dict | None,
    min_track_frames: int = 8,
    reporters: tuple[str, ...] = REPORTERS,
) -> dict:
    """Score predicted reporter levels against the truth.

    Parameters
    ----------
    frames : pd.DataFrame
        One row per frame with ``track_id`` and ``true_<reporter>``, ``pred_<reporter>`` for every reporter, in
        normalized units (counts / ``scale``). Rows with a missing prediction are left out.
    scale : np.ndarray
        Per-reporter scale mapping normalized units back to counts.
    thresholds : dict | None
        PIP-FUCCI thresholds in counts (``stats.json``), read with the reporters as (Geminin, PIP). ``None`` skips
        the phase scores, for reporters without a PIP-FUCCI phase rule.
    min_track_frames : int
        Tracks shorter than this are left out of the track-level score, by default 8.
    reporters : tuple[str, ...]
        Reporter names in target order, by default :data:`REPORTERS`.

    Returns
    -------
    dict
        ``pearson``, ``r2``, ``mae`` (each per reporter and averaged), ``track_pearson`` (same), ``n_frames``,
        ``n_tracks``; with thresholds also ``phase_acc``, ``phase_f1``, ``n_phase_frames``.
    """
    frames = frames.dropna(subset=[f"pred_{r}" for r in reporters])
    metrics: dict[str, float] = {"n_frames": float(len(frames))}
    if len(frames) < 2:
        return metrics
    for name in ("pearson", "r2", "mae", "track_pearson"):
        metrics[name] = 0.0
    for reporter in reporters:
        true, pred = frames[f"true_{reporter}"].to_numpy(), frames[f"pred_{reporter}"].to_numpy()
        metrics[f"pearson_{reporter}"] = _pearson(true, pred)
        metrics[f"r2_{reporter}"] = float(1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum())
        metrics[f"mae_{reporter}"] = float(np.abs(true - pred).mean())
        per_track = [
            _pearson(track[f"true_{reporter}"].to_numpy(), track[f"pred_{reporter}"].to_numpy())
            for _, track in frames.groupby("track_id")
            if len(track) >= min_track_frames
        ]
        metrics[f"track_pearson_{reporter}"] = float(np.nanmean(per_track)) if per_track else float("nan")
        for name in ("pearson", "r2", "mae", "track_pearson"):
            metrics[name] += metrics[f"{name}_{reporter}"] / len(reporters)
    metrics["n_tracks"] = float(sum(len(t) >= min_track_frames for _, t in frames.groupby("track_id")))
    if thresholds is None:
        return metrics

    true_counts = frames[[f"true_{r}" for r in reporters]].to_numpy() * scale
    pred_counts = frames[[f"pred_{r}" for r in reporters]].to_numpy() * scale
    true_phase = phase_strict(true_counts[:, 0], true_counts[:, 1], thresholds)
    pred_phase = phase_forced(pred_counts[:, 0], pred_counts[:, 1], thresholds)
    clear = true_phase >= 0
    metrics["n_phase_frames"] = float(clear.sum())
    if clear.any():
        true_phase, pred_phase = true_phase[clear], pred_phase[clear]
        metrics["phase_acc"] = float((true_phase == pred_phase).mean())
        f1 = []
        for k, name in enumerate(PHASES):
            tp = float(((pred_phase == k) & (true_phase == k)).sum())
            denominator = float((pred_phase == k).sum() + (true_phase == k).sum())
            f1.append(2 * tp / denominator if denominator else float("nan"))
            metrics[f"phase_f1_{name}"] = f1[-1]
        metrics["phase_f1"] = float(np.nanmean(f1))
    return metrics


def masked_median(images: Tensor, mask: Tensor) -> Tensor:
    """Per-channel median of ``(M, C, Y, X)`` images over a ``(M, 1, Y, X)`` mask; NaN where the mask is empty."""
    values = images.float().flatten(2)
    inside = mask.flatten(2).expand_as(values)
    return torch.where(inside, values, torch.full_like(values, float("nan"))).nanmedian(dim=-1).values  # noqa: PD011


def pixel_pearson(pred: Tensor, target: Tensor) -> Tensor:
    """Per-image, per-channel Pearson r between ``(M, C, Y, X)`` images."""
    pred = pred.float().flatten(2)
    target = target.float().flatten(2)
    pred = pred - pred.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    denominator = (pred.pow(2).sum(-1) * target.pow(2).sum(-1)).sqrt()
    return (pred * target).sum(-1) / denominator.clamp_min(1e-12)
