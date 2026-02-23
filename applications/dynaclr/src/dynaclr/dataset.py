"""Multi-experiment triplet dataset with lineage-aware positive sampling.

Provides :class:`MultiExperimentTripletDataset` which reads cell patches from
multi-experiment OME-Zarr stores, samples temporal positives following lineage
through division events, and produces the exact batch format expected by
:class:`dynaclr.engine.ContrastiveModule`.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import tensorstore as ts
except ImportError:
    ts = None

from viscy_data._typing import INDEX_COLUMNS, NormMeta
from viscy_data._utils import _read_norm_meta

from dynaclr.index import MultiExperimentIndex
from dynaclr.tau_sampling import sample_tau

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentTripletDataset"]


class MultiExperimentTripletDataset(Dataset):
    """Dataset for multi-experiment triplet sampling with lineage-aware positives.

    Works with :class:`~dynaclr.index.MultiExperimentIndex` to sample
    anchor/positive cell patches across multiple experiments, following lineage
    through division events.

    The batch dict produced by :meth:`__getitems__` is directly compatible
    with :meth:`dynaclr.engine.ContrastiveModule.training_step`:

    * ``batch["anchor"]`` -- ``Tensor (B, C, Z, Y, X)``
    * ``batch["positive"]`` -- ``Tensor (B, C, Z, Y, X)`` (fit mode only)
    * ``batch["anchor_norm_meta"]`` / ``batch["positive_norm_meta"]`` --
      ``list[NormMeta | None]``
    * ``batch["index"]`` -- ``list[dict]`` (predict mode only)

    Parameters
    ----------
    index : MultiExperimentIndex
        Validated multi-experiment index with ``valid_anchors`` and ``tracks``.
    fit : bool
        If ``True`` (default), return anchor + positive. If ``False``,
        return anchor + index metadata for prediction.
    tau_range_hours : tuple[float, float]
        ``(min_hours, max_hours)`` converted to frames per experiment.
    tau_decay_rate : float
        Exponential decay rate for :func:`~dynaclr.tau_sampling.sample_tau`.
    return_negative : bool
        Reserved for future use.  Currently unused (NTXentLoss uses
        in-batch negatives).
    cache_pool_bytes : int
        Tensorstore cache pool size in bytes.
    """

    def __init__(
        self,
        index: MultiExperimentIndex,
        fit: bool = True,
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        return_negative: bool = False,
        cache_pool_bytes: int = 0,
    ) -> None:
        if ts is None:
            raise ImportError(
                "tensorstore is required for MultiExperimentTripletDataset. "
                "Install with: pip install tensorstore"
            )
        self.index = index
        self.fit = fit
        self.tau_range_hours = tau_range_hours
        self.tau_decay_rate = tau_decay_rate
        self.return_negative = return_negative

        self._rng = np.random.default_rng()
        self._setup_tensorstore_context(cache_pool_bytes)
        self._build_lineage_lookup()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _setup_tensorstore_context(self, cache_pool_bytes: int) -> None:
        """Configure tensorstore context with CPU limits based on SLURM env."""
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        cpus = int(cpus) if cpus is not None else (os.cpu_count() or 4)
        self._ts_context = ts.Context(
            {
                "data_copy_concurrency": {"limit": cpus},
                "cache_pool": {"total_bytes_limit": cache_pool_bytes},
            }
        )
        self._tensorstores: dict[str, ts.TensorStore] = {}

    def _build_lineage_lookup(self) -> None:
        """Build ``_lineage_timepoints`` for O(1) positive candidate lookup.

        Structure: ``{(experiment, lineage_id): {t: [row_indices_in_tracks]}}``
        """
        self._lineage_timepoints: dict[
            tuple[str, str], dict[int, list[int]]
        ] = defaultdict(lambda: defaultdict(list))

        for idx, row in self.index.tracks.iterrows():
            key = (row["experiment"], row["lineage_id"])
            self._lineage_timepoints[key][row["t"]].append(idx)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of valid anchor samples."""
        return len(self.index.valid_anchors)

    def __getitems__(self, indices: list[int]) -> dict:
        """Return a batch of triplet samples for the given indices.

        Parameters
        ----------
        indices : list[int]
            Row indices into ``self.index.valid_anchors``.

        Returns
        -------
        dict
            In fit mode: ``{"anchor": Tensor, "positive": Tensor,
            "anchor_norm_meta": list, "positive_norm_meta": list}``.
            In predict mode: ``{"anchor": Tensor, "index": list[dict]}``.
        """
        anchor_rows = self.index.valid_anchors.iloc[indices]
        anchor_patches, anchor_norms = self._slice_patches(anchor_rows)
        sample: dict = {
            "anchor": anchor_patches,
            "anchor_norm_meta": anchor_norms,
        }

        if self.fit:
            positive_rows = self._sample_positives(anchor_rows)
            positive_patches, positive_norms = self._slice_patches(positive_rows)
            sample["positive"] = positive_patches
            sample["positive_norm_meta"] = positive_norms
        else:
            indices_list = []
            for _, anchor_row in anchor_rows.iterrows():
                idx_dict: dict = {}
                for col in INDEX_COLUMNS:
                    if col in anchor_row.index:
                        idx_dict[col] = anchor_row[col]
                    elif col not in ["y", "x", "z"]:
                        # optional columns
                        pass
                indices_list.append(idx_dict)
            sample["index"] = indices_list

        return sample

    # ------------------------------------------------------------------
    # Positive sampling
    # ------------------------------------------------------------------

    def _sample_positives(self, anchor_rows: pd.DataFrame) -> pd.DataFrame:
        """Sample one positive for each anchor using lineage-aware lookup.

        Parameters
        ----------
        anchor_rows : pd.DataFrame
            Rows from ``valid_anchors`` for the current batch.

        Returns
        -------
        pd.DataFrame
            One row per anchor from ``self.index.tracks``.
        """
        pos_rows = []
        for _, row in anchor_rows.iterrows():
            pos = self._find_positive(row, self._rng)
            pos_rows.append(pos)
        return pd.DataFrame(pos_rows).reset_index(drop=True)

    def _find_positive(
        self,
        anchor_row: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series | None:
        """Find a positive sample for a given anchor.

        Searches for a row in ``self.index.tracks`` with the same
        ``lineage_id`` at ``t + tau``.  When multiple candidates exist
        (e.g. parent and daughter at the same timepoint), one is chosen
        randomly.

        Parameters
        ----------
        anchor_row : pd.Series
            A single row from ``valid_anchors``.
        rng : numpy.random.Generator
            Random number generator for tau sampling and tie-breaking.

        Returns
        -------
        pd.Series or None
            A track row for the positive, or ``None`` if no positive found.
        """
        exp_name = anchor_row["experiment"]
        lineage_id = anchor_row["lineage_id"]
        anchor_t = anchor_row["t"]

        # Convert tau range to frames for this experiment
        tau_min, tau_max = self.index.registry.tau_range_frames(
            exp_name, self.tau_range_hours
        )

        # Get lineage-timepoint lookup
        lt_key = (exp_name, lineage_id)
        lt_map = self._lineage_timepoints.get(lt_key)
        if lt_map is None:
            return None

        # Sample tau and search for positive
        # Try sampled tau first, then scan the full range as fallback
        sampled_tau = sample_tau(tau_min, tau_max, rng, self.tau_decay_rate)
        target_t = anchor_t + sampled_tau
        candidates = lt_map.get(target_t, [])
        if candidates:
            chosen_idx = candidates[rng.integers(len(candidates))]
            return self.index.tracks.iloc[chosen_idx]

        # Fallback: try all taus in range (skip tau=0)
        for tau in range(tau_min, tau_max + 1):
            if tau == 0:
                continue
            target_t_fb = anchor_t + tau
            candidates_fb = lt_map.get(target_t_fb, [])
            if candidates_fb:
                chosen_idx = candidates_fb[rng.integers(len(candidates_fb))]
                return self.index.tracks.iloc[chosen_idx]

        return None

    # ------------------------------------------------------------------
    # Patch extraction (tensorstore I/O)
    # ------------------------------------------------------------------

    def _get_tensorstore(
        self, position, fov_name: str
    ) -> "ts.TensorStore":
        """Get or create a cached tensorstore object for the given FOV.

        Parameters
        ----------
        position : iohub.ngff.Position
            Position object from the OME-Zarr store.
        fov_name : str
            FOV name used as cache key.

        Returns
        -------
        ts.TensorStore
        """
        if fov_name not in self._tensorstores:
            self._tensorstores[fov_name] = position["0"].tensorstore(
                context=self._ts_context,
                recheck_cached_data="open",
            )
        return self._tensorstores[fov_name]

    def _slice_patch(
        self, track_row: pd.Series
    ) -> tuple["ts.TensorStore", NormMeta | None]:
        """Slice a patch from the image store for a given track row.

        Uses per-experiment ``channel_maps`` for channel index remapping
        and ``y_clamp`` / ``x_clamp`` for border-safe centering.

        Parameters
        ----------
        track_row : pd.Series
            A single row from ``tracks`` or ``valid_anchors``.

        Returns
        -------
        tuple[ts.TensorStore, NormMeta | None]
            The sliced patch (lazy tensorstore) and normalization metadata.
        """
        position = track_row["position"]
        fov_name = track_row["fov_name"]
        exp_name = track_row["experiment"]

        image = self._get_tensorstore(position, fov_name)

        t = track_row["t"]
        y_center = int(track_row["y_clamp"])
        x_center = int(track_row["x_clamp"])

        y_half = self.index.yx_patch_size[0] // 2
        x_half = self.index.yx_patch_size[1] // 2

        # Per-experiment channel remapping
        channel_map = self.index.registry.channel_maps[exp_name]
        channel_indices = [channel_map[i] for i in sorted(channel_map.keys())]

        patch = image.oindex[
            t,
            [int(c) for c in channel_indices],
            self.index.z_range,
            slice(y_center - y_half, y_center + y_half),
            slice(x_center - x_half, x_center + x_half),
        ]
        return patch, _read_norm_meta(position)

    def _slice_patches(
        self, track_rows: pd.DataFrame
    ) -> tuple[torch.Tensor, list[NormMeta | None]]:
        """Slice and stack patches for multiple track rows.

        Parameters
        ----------
        track_rows : pd.DataFrame
            Multiple rows from ``tracks`` / ``valid_anchors``.

        Returns
        -------
        tuple[torch.Tensor, list[NormMeta | None]]
            Stacked tensor ``(B, C, Z, Y, X)`` and per-sample norm metadata.
        """
        patches = []
        norms = []
        for _, row in track_rows.iterrows():
            patch, norm = self._slice_patch(row)
            patches.append(patch)
            norms.append(norm)
        results = ts.stack(
            [p.translate_to[0] for p in patches]
        ).read().result()  # noqa: PD013
        return torch.from_numpy(results), norms
