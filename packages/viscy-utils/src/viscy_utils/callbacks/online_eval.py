"""Online evaluation callback for contrastive training.

Logs three lightweight metrics every N epochs during validation:

1. **k-NN accuracy** (k=20) on ``marker`` or a configurable label key —
   non-parametric probe of representation quality.
2. **Effective rank** of the embedding covariance matrix —
   detects dimensional collapse without labels.
3. **Temporal smoothness** — Spearman correlation between embedding
   distance and time distance within cell tracks. Measures whether the
   representation captures temporal dynamics (the core DynaCLR goal).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_warn
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from viscy_data._typing import TripletSample
from viscy_utils.tensor_utils import to_numpy

_logger = logging.getLogger("lightning.pytorch")


def effective_rank(features: np.ndarray) -> float:
    """Compute the effective rank of a feature matrix.

    Defined as the exponential of the Shannon entropy of the
    normalized singular values (Roy & Bhattacharya, 2007).
    A value close to 1 indicates complete collapse; a value
    close to ``min(n_samples, n_features)`` indicates full rank.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    float
        Effective rank (scalar >= 1).
    """
    # Guard against NaN/Inf in features — np.linalg.svd raises
    # "SVD did not converge" on non-finite input, which crashes the whole
    # run from inside a validation callback. Drop affected rows and return
    # NaN when no finite rows remain. Under DDP every rank computes on the
    # all-gathered full set, so the warning would otherwise fire once per
    # rank with identical content — emit only from rank 0.
    finite_mask = np.isfinite(features).all(axis=1)
    if not finite_mask.all():
        rank_zero_warn(f"effective_rank: {(~finite_mask).sum()}/{len(features)} rows contain NaN/Inf; skipping those")
        features = features[finite_mask]
    if features.shape[0] < 2:
        return float("nan")
    _, s, _ = np.linalg.svd(features, full_matrices=False)
    s = s[s > 1e-10]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def temporal_smoothness(
    features: np.ndarray,
    track_ids: np.ndarray,
    timepoints: np.ndarray,
) -> float:
    """Compute temporal smoothness as Spearman correlation.

    For every pair of cells within the same track, computes the
    Spearman rank correlation between cosine distance in embedding
    space and absolute time difference. A positive correlation means
    nearby timepoints produce nearby embeddings.

    Parameters
    ----------
    features : np.ndarray
        Embedding matrix of shape ``(n_samples, n_features)``.
    track_ids : np.ndarray
        Track identifier for each sample.
    timepoints : np.ndarray
        Integer timepoint for each sample.

    Returns
    -------
    float
        Spearman rho (in [-1, 1]). Higher is better.
        Returns NaN if fewer than 3 valid pairs.
    """
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    embedding_dists: list[float] = []
    time_dists: list[float] = []

    unique_tracks = np.unique(track_ids)
    for tid in unique_tracks:
        mask = track_ids == tid
        if mask.sum() < 2:
            continue
        f_track = features_norm[mask]
        t_track = timepoints[mask]
        n = len(f_track)
        for i in range(n):
            for j in range(i + 1, n):
                cos_dist = 1.0 - float(f_track[i] @ f_track[j])
                dt = abs(float(t_track[i]) - float(t_track[j]))
                embedding_dists.append(cos_dist)
                time_dists.append(dt)

    if len(embedding_dists) < 3:
        return float("nan")

    rho, _ = spearmanr(time_dists, embedding_dists)
    return float(rho)


class OnlineEvalCallback(Callback):
    """Evaluate representation quality during training.

    Accumulates validation embeddings every ``every_n_epochs`` epochs
    and computes three metrics:

    - ``metrics/knn_acc/{label_key}/val`` — k-NN accuracy (5-fold CV or
      stratified holdout, configurable via ``knn_eval_mode``)
    - ``metrics/effective_rank/val`` — effective rank of covariance
    - ``metrics/temporal_smoothness/val`` — Spearman rho (distance vs dt)

    Under DDP, features and metadata are ``all_gather``-ed across ranks
    so every rank computes metrics on the full validation set; the
    resulting per-rank scalars are identical and ``sync_dist=True``
    averages them as a no-op. Safe for single-GPU and DDP training.

    Parameters
    ----------
    every_n_epochs : int
        Evaluation frequency.
    label_key : str
        Metadata key used as k-NN target. Must be present in
        ``anchor_meta[i]["labels"]``.
    k : int
        Number of neighbors for k-NN.
    track_id_key : str
        Metadata key for track identity (temporal smoothness).
    timepoint_key : str
        Metadata key for timepoint (temporal smoothness).
    knn_eval_mode : {"cv", "holdout"}
        How to score the k-NN probe. ``"cv"`` runs 5-fold stratified CV
        (default; good for few-class probes like 40 markers). ``"holdout"``
        runs a single stratified 80/20 train/test split — ~5x cheaper and
        tolerates classes with only 2 samples, which is the right choice
        for many-class probes (e.g. 1001-gene perturbation).
    holdout_test_size : float
        Fraction of samples held out for scoring when
        ``knn_eval_mode="holdout"``. Ignored in CV mode.
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        label_key: str = "marker",
        k: int = 20,
        track_id_key: str = "global_track_id",
        timepoint_key: str = "t",
        knn_eval_mode: Literal["cv", "holdout"] = "cv",
        holdout_test_size: float = 0.2,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.label_key = label_key
        self.k = k
        self.track_id_key = track_id_key
        self.timepoint_key = timepoint_key
        self.knn_eval_mode = knn_eval_mode
        self.holdout_test_size = holdout_test_size
        self._collecting = False
        self._features: list[torch.Tensor] = []
        self._meta: list[dict] = []

    def _should_collect(self, trainer: Trainer) -> bool:
        return trainer.current_epoch % self.every_n_epochs == 0 and not trainer.sanity_checking

    def _reset(self) -> None:
        self._collecting = False
        self._features = []
        self._meta = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Enable collection if current epoch matches eval frequency."""
        if self._should_collect(trainer):
            self._collecting = True

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: TripletSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate embeddings and metadata from every validation batch."""
        if not self._collecting:
            return
        with torch.no_grad():
            features, _ = pl_module(batch["anchor"])
        self._features.append(features.detach().cpu())
        self._meta.extend(batch.get("anchor_meta", []))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log metrics on the full validation set under DDP.

        The validation DataLoader is sharded across ranks (each rank sees
        N/world_size samples), so per-rank metrics are not the metrics of
        the full set: ``effective_rank``, k-NN CV/holdout accuracy, and
        Spearman ``rho`` are non-linear in the sample set, and averaging
        per-shard values via ``sync_dist=True`` is statistically wrong.

        Fix: ``all_gather`` features and the per-sample arrays needed for
        each metric across ranks, then compute on the full set on every
        rank. Every rank produces the same scalar, so ``sync_dist=True``
        becomes a no-op average of identical values — but keeping it
        avoids the rank-0-only ``pl_module.log`` DDP deadlock (rank 0
        registers the metric, other ranks don't, and Lightning's
        epoch-end all-reduce never completes).
        """
        if not self._collecting or not self._features:
            self._reset()
            return

        features_local = torch.cat(self._features)
        labels_local = self._extract_array(self.label_key, source="labels")
        track_ids_local = self._extract_array(self.track_id_key, source="meta")
        timepoints_local = self._extract_array(self.timepoint_key, source="meta")

        features_np, labels, track_ids, timepoints = self._gather_across_ranks(
            pl_module,
            features_local,
            labels_local,
            track_ids_local,
            timepoints_local,
        )

        n_samples = features_np.shape[0]
        epoch = trainer.current_epoch
        is_rank_zero = trainer.global_rank == 0

        # --- Effective rank (always computable) ---
        erank = effective_rank(features_np)
        pl_module.log(
            "metrics/effective_rank/val",
            erank,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        if is_rank_zero:
            _logger.info(
                f"[OnlineEval epoch {epoch}] effective_rank={erank:.1f} (n={n_samples}, d={features_np.shape[1]})"
            )

        # --- k-NN accuracy (requires labels) ---
        if labels is not None and len(np.unique(labels)) >= 2:
            k = min(self.k, n_samples - 1)
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            min_class_count = int(min(np.bincount(labels)))
            mode = self.knn_eval_mode
            # Auto-degrade CV -> holdout when the smallest class has < 2
            # samples (CV would skip silently). Holdout mode still requires
            # >= 2 per class for stratified splitting.
            if mode == "cv" and min_class_count < 2:
                mode = "holdout"
            if mode == "cv":
                cv_folds = min(5, min_class_count)
                scores = cross_val_score(knn, features_np, labels, cv=cv_folds)
                knn_acc = float(scores.mean())
                eval_desc = f"cv={cv_folds}"
            elif mode == "holdout" and min_class_count >= 2:
                x_train, x_test, y_train, y_test = train_test_split(
                    features_np,
                    labels,
                    test_size=self.holdout_test_size,
                    stratify=labels,
                    random_state=0,
                )
                knn.fit(x_train, y_train)
                knn_acc = float(knn.score(x_test, y_test))
                eval_desc = f"holdout={self.holdout_test_size:.2f}"
            else:
                knn_acc = None
                if is_rank_zero:
                    _logger.debug(
                        f"[OnlineEval epoch {epoch}] Skipping k-NN: "
                        f"smallest class has {min_class_count} samples (need >=2)."
                    )
            if knn_acc is not None:
                pl_module.log(
                    f"metrics/knn_acc/{self.label_key}/val",
                    knn_acc,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                if is_rank_zero:
                    _logger.info(
                        f"[OnlineEval epoch {epoch}] knn_acc({self.label_key}, k={k})={knn_acc:.3f} ({eval_desc})"
                    )

        # --- Temporal smoothness (requires track_id + timepoint) ---
        if track_ids is not None and timepoints is not None:
            rho = temporal_smoothness(features_np, track_ids, timepoints)
            if not np.isnan(rho):
                pl_module.log(
                    "metrics/temporal_smoothness/val",
                    rho,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                if is_rank_zero:
                    _logger.info(f"[OnlineEval epoch {epoch}] temporal_smoothness={rho:.3f}")

        self._reset()

    def _extract_array(self, key: str, source: Literal["labels", "meta"] = "meta") -> np.ndarray | None:
        """Extract a per-sample array from accumulated metadata.

        Parameters
        ----------
        key : str
            The metadata key to extract.
        source : {"labels", "meta"}
            If ``"labels"``, looks in ``meta[i]["labels"][key]``.
            If ``"meta"``, looks in ``meta[i][key]``.

        Returns
        -------
        np.ndarray or None
            Array of length n_samples, or None if key is missing.
        """
        values = []
        for m in self._meta:
            if source == "labels":
                v = m.get("labels", {}).get(key)
            else:
                v = m.get(key)
            if v is None:
                return None
            values.append(v)
        return np.array(values)

    def _gather_across_ranks(
        self,
        pl_module: LightningModule,
        features_local: torch.Tensor,
        labels_local: np.ndarray | None,
        track_ids_local: np.ndarray | None,
        timepoints_local: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Gather per-rank validation arrays into a full-set view on every rank.

        On single-GPU runs this is a pure passthrough that converts the
        feature tensor to numpy. Under DDP, every rank participates in
        ``pl_module.all_gather`` so each ends up with the same
        concatenated arrays — the metrics computed downstream are then
        deterministic and identical on every rank, and the subsequent
        ``sync_dist=True`` log call averages identical scalars (a no-op
        that still pleases Lightning's epoch-end all-reduce).

        Per-rank shard sizes can differ by one batch when the dataset
        size is not divisible by ``world_size * batch_size``; equalize
        by truncating to the minimum length before the gather to keep
        ``all_gather`` happy with a fixed-shape tensor.

        Parameters
        ----------
        pl_module : LightningModule
            Used for ``pl_module.all_gather`` (Lightning routes to the
            correct backend) and ``pl_module.device``.
        features_local : torch.Tensor
            ``(n_local, d)`` features collected on this rank.
        labels_local, track_ids_local, timepoints_local : np.ndarray or None
            Per-rank metadata arrays, ``None`` if the corresponding key
            was missing.

        Returns
        -------
        features_np : np.ndarray
            Full-set features ``(N, d)`` (same on every rank under DDP).
        labels, track_ids, timepoints : np.ndarray or None
            Full-set metadata arrays. ``None`` if any rank reported
            ``None`` (handled via an explicit availability all-reduce).
        """
        world_size = getattr(pl_module.trainer, "world_size", 1) or 1
        if world_size <= 1:
            return (
                to_numpy(features_local),
                labels_local,
                track_ids_local,
                timepoints_local,
            )

        # Equalize shard sizes — all_gather requires identical shapes per rank.
        n_local = torch.tensor([features_local.shape[0]], device=pl_module.device)
        n_per_rank = pl_module.all_gather(n_local).flatten()
        n_min = int(n_per_rank.min().item())
        features_local = features_local[:n_min].to(pl_module.device)

        # Reduce a per-rank availability flag for each metadata array so
        # all ranks agree on whether to compute the dependent metric.
        # If any rank is missing the key, treat as missing globally.
        def _gather_optional(arr: np.ndarray | None) -> np.ndarray | None:
            available = torch.tensor(
                [1 if arr is not None else 0],
                device=pl_module.device,
            )
            available = pl_module.all_gather(available).flatten()
            if int(available.min().item()) == 0:
                return None
            arr_local = arr[:n_min]
            tensor = torch.as_tensor(arr_local, device=pl_module.device)
            gathered = pl_module.all_gather(tensor)
            # all_gather returns shape (world_size, n_min, *rest) for
            # 1D inputs — collapse to (world_size * n_min, *rest).
            gathered = gathered.reshape(-1, *tensor.shape[1:]) if tensor.ndim > 1 else gathered.reshape(-1)
            return gathered.detach().cpu().numpy()

        features_gathered = pl_module.all_gather(features_local)
        # all_gather on (n_min, d) returns (world_size, n_min, d) — flatten.
        features_gathered = features_gathered.reshape(-1, features_gathered.shape[-1])

        return (
            features_gathered.detach().cpu().numpy(),
            _gather_optional(labels_local),
            _gather_optional(track_ids_local),
            _gather_optional(timepoints_local),
        )
