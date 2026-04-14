"""Multi-experiment triplet dataset with flexible positive sampling.

Provides :class:`MultiExperimentTripletDataset` which reads cell patches from
multi-experiment OME-Zarr stores and samples positives via three strategies:

* ``positive_cell_source="lookup"`` with ``positive_match_columns=["lineage_id"]``
  (default) — temporal positive: same lineage at ``t+tau``.
* ``positive_cell_source="lookup"`` with other columns (e.g. ``["gene_name",
  "reporter"]``) — perturbation positive: different cell with same column values.
* ``positive_cell_source="self"`` — SimCLR-style: anchor and positive are the
  same crop; augmentation creates two views.

Produces the exact batch format expected by
:class:`dynaclr.engine.ContrastiveModule`.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

try:
    import tensorstore as ts
except ImportError:
    ts = None

from iohub.ngff import open_ome_zarr

from dynaclr.data.index import MultiExperimentIndex
from dynaclr.data.tau_sampling import sample_tau
from viscy_data._typing import ULTRACK_INDEX_COLUMNS, NormMeta, SampleMeta
from viscy_data._utils import _read_norm_meta

_META_COLUMNS = [
    "experiment",
    "perturbation",
    "microscope",
    "fov_name",
    "store_path",
    "global_track_id",
    "t",
    "hours_post_perturbation",
    "lineage_id",
    "marker",
    "y_clamp",
    "x_clamp",
]

_logger = logging.getLogger(__name__)

__all__ = ["MultiExperimentTripletDataset"]


def _rescale_patch(patch: Tensor, scale: tuple[float, float, float], target: tuple[int, int, int]) -> Tensor:
    """Rescale a ``(C, Z, Y, X)`` patch to *target* size using nearest-exact interpolation.

    Parameters
    ----------
    patch : Tensor
        Patch tensor of shape ``(C, Z, Y, X)``.
    scale : tuple[float, float, float]
        ``(scale_z, scale_y, scale_x)`` — 1.0 means no rescaling needed.
    target : tuple[int, int, int]
        Target spatial size ``(z, y, x)``.

    Returns
    -------
    Tensor
        Rescaled patch of shape ``(C, *target)``.
    """
    sz, sy, sx = scale
    if sz == 1.0 and sy == 1.0 and sx == 1.0:
        return patch
    return F.interpolate(
        patch.unsqueeze(0).float(),
        size=target,
        mode="nearest-exact",
    ).squeeze(0)


class MultiExperimentTripletDataset(Dataset):
    """Dataset for multi-experiment triplet sampling with flexible positive strategies.

    Works with :class:`~dynaclr.index.MultiExperimentIndex` to sample
    anchor/positive cell patches across multiple experiments.

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
        Only used when ``positive_cell_source="lookup"`` and
        ``"lineage_id"`` is in ``positive_match_columns``.
    tau_decay_rate : float
        Exponential decay rate for :func:`~dynaclr.tau_sampling.sample_tau`.
    return_negative : bool
        Reserved for future use.  Currently unused (NTXentLoss uses
        in-batch negatives).
    cache_pool_bytes : int
        Tensorstore cache pool size in bytes.
    channels_per_sample : int | list[str] | None
        Controls how many source channels to read per sample.
        ``None`` (default) — read all source channels, output ``(B, C, Z, Y, X)``.
        ``1`` — randomly select one channel per sample, output ``(B, 1, Z, Y, X)``.
        ``["labelfree", "reporter_gfp"]`` — read those specific channels by label.
        Integer values > 1 are not supported (use list form).
    positive_cell_source : str
        ``"self"`` — SimCLR: positive is the same crop as anchor (augmentation
        creates two views).  ``"lookup"`` (default) — find a different cell
        using ``positive_match_columns``.
    positive_match_columns : list[str] | None
        Columns that define "same identity" for positive lookup.  Defaults to
        ``["lineage_id"]`` (temporal matching).  For OPS perturbation use
        ``["gene_name", "reporter"]``.  When ``"lineage_id"`` is in the list,
        tau constraint is applied.
    positive_channel_source : str
        ``"same"`` (default) — anchor and positive use the same channel index.
        ``"any"`` — positive draws its channel independently from anchor.
        Only meaningful when ``channels_per_sample=1``.
    label_columns : dict[str, str] | None
        Mapping from ``batch_key`` (used by classification heads) to
        dataframe column name.  E.g. ``{"gene_label": "condition"}`` builds
        a label encoder from unique values in the ``condition`` column and
        populates ``anchor_meta[i]["labels"]["gene_label"]`` with integer
        class IDs.  Default: ``None`` (no labels).
    """

    def __init__(
        self,
        index: MultiExperimentIndex,
        fit: bool = True,
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        return_negative: bool = False,
        cache_pool_bytes: int = 0,
        channels_per_sample: int | list[str] | None = None,
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
        positive_channel_source: str = "same",
        label_columns: dict[str, str] | None = None,
    ) -> None:
        if ts is None:
            raise ImportError(
                "tensorstore is required for MultiExperimentTripletDataset. Install with: pip install tensorstore"
            )
        self.index = index
        self.fit = fit
        self.tau_range_hours = tau_range_hours
        self.tau_decay_rate = tau_decay_rate
        self.return_negative = return_negative
        # Resolve channel selection mode.
        # In all-channels mode with a flat parquet (one row per cell×channel),
        # deduplicate valid_anchors to one row per cell to avoid redundant
        # batches — each row reads all channels anyway.
        if channels_per_sample is None:
            self._channel_mode = "all"
            va = index.valid_anchors
            if va["cell_id"].duplicated().any():
                before = len(va)
                index.valid_anchors = va.drop_duplicates(subset="cell_id").reset_index(drop=True)
                _logger.info(
                    "All-channels dedup: %d → %d valid_anchors (flat parquet)",
                    before,
                    len(index.valid_anchors),
                )
        elif isinstance(channels_per_sample, int):
            if channels_per_sample != 1:
                raise ValueError(
                    f"channels_per_sample as int must be 1, got {channels_per_sample}. "
                    "Use a list of labels for multiple specific channels."
                )
            self._channel_mode = "from_index"
        elif isinstance(channels_per_sample, list):
            self._channel_mode = "fixed"
            self._fixed_channel_names = channels_per_sample
        else:
            raise TypeError(f"channels_per_sample must be int, list[str], or None, got {type(channels_per_sample)}")
        self.channels_per_sample = channels_per_sample
        self.positive_cell_source = positive_cell_source
        self.positive_match_columns = positive_match_columns if positive_match_columns is not None else ["lineage_id"]
        self.positive_channel_source = positive_channel_source

        self._label_encoders: dict[str, tuple[str, dict[str, int]]] = {}
        if label_columns:
            for batch_key, col in label_columns.items():
                unique_vals = sorted(index.valid_anchors[col].dropna().unique())
                encoder = {v: i for i, v in enumerate(unique_vals)}
                self._label_encoders[batch_key] = (col, encoder)
                _logger.info("Label encoder '%s' (%s): %d classes", batch_key, col, len(encoder))

        self._rng = np.random.default_rng()
        self._setup_tensorstore_context(cache_pool_bytes)
        if self.fit:
            self._build_match_lookup()

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
        self._store_cache: dict[str, object] = {}  # store_path -> Plate
        self._position_cache: dict[str, object] = {}  # fov_name -> Position
        self._norm_meta_cache: dict[str, NormMeta | None] = {}

    def _build_match_lookup(self) -> None:
        """Build lookup structures for O(1) positive candidate lookup.

        For ``positive_cell_source="self"``, no lookup is needed.

        For temporal mode (``"lineage_id"`` in ``positive_match_columns``),
        builds ``_lineage_timepoints``:
        ``{(experiment, lineage_id): {t: [row_indices_in_tracks]}}``.

        For generic column-match mode, builds ``_match_lookup``:
        ``{match_key_tuple: [row_indices_in_tracks]}``.
        """
        if self.positive_cell_source == "self":
            return

        tracks = self.index.tracks
        if "lineage_id" in self.positive_match_columns:
            grouped = tracks.groupby(["experiment", "lineage_id", "t"]).indices
            self._lineage_timepoints: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for (exp, lid, t), row_indices in grouped.items():
                self._lineage_timepoints[(exp, lid)][int(t)] = row_indices.tolist()
        else:
            cols = self.positive_match_columns
            grouped = tracks.groupby(cols).indices
            self._match_lookup: dict[tuple, list[int]] = {k: v.tolist() for k, v in grouped.items()}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of valid anchor samples."""
        return len(self.index.valid_anchors)

    def __getitem__(self, idx: int) -> None:  # noqa: D105
        raise NotImplementedError(
            "MultiExperimentTripletDataset only supports batched access via __getitems__. "
            "Use a batch sampler with collate_fn=lambda x: x."
        )

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
            "anchor_norm_meta": list, "positive_norm_meta": list,
            "anchor_meta": list[dict], "positive_meta": list[dict]}``.
            In predict mode: ``{"anchor": Tensor, "index": list[dict]}``.
        """
        anchor_rows = self.index.valid_anchors.iloc[indices]

        # Pre-compute per-sample channel names based on channel_mode.
        if self._channel_mode == "from_index":
            forced_channel_names = [[row["channel_name"]] for _, row in anchor_rows.iterrows()]
        elif self._channel_mode == "fixed":
            forced_channel_names = [self._fixed_channel_names] * len(indices)
        else:
            forced_channel_names = None

        anchor_patches, anchor_norms = self._slice_patches(anchor_rows, forced_channel_names)
        sample: dict = {
            "anchor": anchor_patches,
            "anchor_norm_meta": anchor_norms,
            "anchor_meta": self._extract_meta(anchor_rows),
        }

        if self.fit:
            positive_rows = self._sample_positives(anchor_rows)
            if self._channel_mode == "from_index":
                pos_forced_channel_names = [[row["channel_name"]] for _, row in positive_rows.iterrows()]
            else:
                pos_forced_channel_names = forced_channel_names
            positive_patches, positive_norms = self._slice_patches(positive_rows, pos_forced_channel_names)
            sample["positive"] = positive_patches
            sample["positive_norm_meta"] = positive_norms
            sample["positive_meta"] = self._extract_meta(positive_rows)
        else:
            indices_list = []
            for _, anchor_row in anchor_rows.iterrows():
                idx_dict: dict = {}
                for col in ULTRACK_INDEX_COLUMNS:
                    if col in anchor_row.index:
                        idx_dict[col] = anchor_row[col]
                    elif col not in ["y", "x", "z"]:
                        # optional columns
                        pass
                for col in [
                    "experiment",
                    "marker",
                    "perturbation",
                    "hours_post_perturbation",
                    "organelle",
                    "well",
                    "microscope",
                ]:
                    if col in anchor_row.index:
                        idx_dict[col] = anchor_row[col]
                indices_list.append(idx_dict)
            sample["index"] = indices_list

        return sample

    def _extract_meta(self, rows: pd.DataFrame) -> list[SampleMeta]:
        """Extract lightweight metadata dicts from track rows.

        Parameters
        ----------
        rows : pd.DataFrame
            Rows from ``valid_anchors`` or ``tracks``.

        Returns
        -------
        list[dict]
            One dict per row with keys from ``_META_COLUMNS``.
            If ``label_columns`` was set, each dict also contains a
            ``"labels"`` sub-dict with integer class IDs.
        """
        cols = [c for c in _META_COLUMNS if c in rows.columns]
        records = rows[cols].to_dict(orient="records")
        if self._label_encoders:
            for i, (_, row) in enumerate(rows.iterrows()):
                labels = {}
                for batch_key, (col, encoder) in self._label_encoders.items():
                    val = row.get(col)
                    if val is not None and val in encoder:
                        labels[batch_key] = encoder[val]
                records[i]["labels"] = labels
        return records

    # ------------------------------------------------------------------
    # Positive sampling
    # ------------------------------------------------------------------

    def _sample_positives(self, anchor_rows: pd.DataFrame) -> pd.DataFrame:
        """Sample one positive for each anchor.

        When ``positive_cell_source="self"``, returns a copy of ``anchor_rows``
        (same crop; augmentation creates two views).  Otherwise delegates to
        :meth:`_find_positive`.

        Parameters
        ----------
        anchor_rows : pd.DataFrame
            Rows from ``valid_anchors`` for the current batch.

        Returns
        -------
        pd.DataFrame
            One row per anchor from ``self.index.tracks``.
        """
        if self.positive_cell_source == "self":
            return anchor_rows.copy().reset_index(drop=True)

        pos_rows = []
        for _, row in anchor_rows.iterrows():
            pos = self._find_positive(row, self._rng)
            if pos is None:
                raise RuntimeError(
                    f"No positive found for anchor (experiment={row.get('experiment')}, "
                    f"match_key={tuple(row.get(c) for c in self.positive_match_columns)}, "
                    f"t={row.get('t')}). "
                    "This anchor should have been filtered out by valid_anchors."
                )
            pos_rows.append(pos)
        return pd.DataFrame(pos_rows).reset_index(drop=True)

    def _find_positive(
        self,
        anchor_row: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series | None:
        """Find a positive sample for a given anchor.

        Dispatches to temporal or generic column-match lookup based on
        ``positive_match_columns``.

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
        if "lineage_id" in self.positive_match_columns:
            return self._find_temporal_positive(anchor_row, rng)
        return self._find_column_match_positive(anchor_row, rng)

    def _find_temporal_positive(
        self,
        anchor_row: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series | None:
        """Find a temporal positive: same lineage at ``t + tau``.

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

        tau_min, tau_max = self.index.registry.tau_range_frames(exp_name, self.tau_range_hours)

        lt_key = (exp_name, lineage_id)
        lt_map = self._lineage_timepoints.get(lt_key)
        if lt_map is None:
            return None

        # In from_index mode (flat parquet), filter candidates to same marker.
        # NOTE:The parquet SHOULD guarantee one channel_name per marker per experiment,
        # so marker filtering is equivalent to channel_name filtering.
        anchor_marker = anchor_row.get("marker") if self._channel_mode == "from_index" else None

        def _pick(candidate_indices: list[int]) -> pd.Series | None:
            if not candidate_indices:
                return None
            if anchor_marker is not None:
                filtered = [
                    idx for idx in candidate_indices if self.index.tracks.iloc[idx].get("marker") == anchor_marker
                ]
                if filtered:
                    candidate_indices = filtered
            chosen_idx = candidate_indices[rng.integers(len(candidate_indices))]
            return self.index.tracks.iloc[chosen_idx]

        # Try sampled tau first, then scan full range as fallback
        sampled_tau = sample_tau(tau_min, tau_max, rng, self.tau_decay_rate)
        target_t = anchor_t + sampled_tau
        result = _pick(lt_map.get(target_t, []))
        if result is not None:
            return result

        for tau in range(tau_min, tau_max + 1):
            if tau == 0:
                continue
            result = _pick(lt_map.get(anchor_t + tau, []))
            if result is not None:
                return result

        return None

    def _find_column_match_positive(
        self,
        anchor_row: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series | None:
        """Find a positive by matching column values, excluding the anchor itself.

        Parameters
        ----------
        anchor_row : pd.Series
            A single row from ``valid_anchors``.
        rng : numpy.random.Generator
            Random number generator for tie-breaking.

        Returns
        -------
        pd.Series or None
            A track row for the positive, or ``None`` if no candidates found.
        """
        cols = self.positive_match_columns
        key = tuple(anchor_row[c] for c in cols)
        all_candidates = self._match_lookup.get(key, [])
        # Exclude the anchor row itself by integer index
        candidates = [i for i in all_candidates if i != anchor_row.name]
        if not candidates:
            return None
        chosen_idx = candidates[rng.integers(len(candidates))]
        return self.index.tracks.iloc[chosen_idx]

    # ------------------------------------------------------------------
    # Patch extraction (tensorstore I/O)
    # ------------------------------------------------------------------

    def _get_position(self, store_path: str, fov_name: str):
        """Get or create a cached Position object for the given FOV.

        Parameters
        ----------
        store_path : str
            Path to the OME-Zarr plate store.
        fov_name : str
            FOV name (e.g. ``"A/1/0"``).

        Returns
        -------
        iohub.ngff.Position
        """
        if fov_name not in self._position_cache:
            if store_path not in self._store_cache:
                self._store_cache[store_path] = open_ome_zarr(store_path, mode="r")
            plate = self._store_cache[store_path]
            self._position_cache[fov_name] = plate[fov_name]
        return self._position_cache[fov_name]

    def _get_tensorstore(self, store_path: str, fov_name: str) -> "ts.TensorStore":
        """Get or create a cached tensorstore object for the given FOV.

        Parameters
        ----------
        store_path : str
            Path to the OME-Zarr plate store.
        fov_name : str
            FOV name used as cache key.

        Returns
        -------
        ts.TensorStore
        """
        if fov_name not in self._tensorstores:
            position = self._get_position(store_path, fov_name)
            self._tensorstores[fov_name] = position["0"].tensorstore(
                context=self._ts_context,
                recheck_cached_data="open",
            )
        return self._tensorstores[fov_name]

    def _build_norm_meta(
        self,
        track_row: pd.Series,
        forced_channel_names: list[str] | None,
    ) -> NormMeta | None:
        """Build per-sample normalization metadata from parquet columns.

        When the parquet has ``norm_mean`` / ``norm_std`` columns (written by
        ``preprocess-cell-index``), reads stats directly from the row — no
        zarr zattrs access needed. Falls back to zarr zattrs for old parquets.

        Parameters
        ----------
        track_row : pd.Series
            A single row from ``tracks`` or ``valid_anchors``.
        forced_channel_names : list[str] or None
            Zarr channel names being read for this sample.

        Returns
        -------
        NormMeta or None
        """
        # Parquet path: norm columns present
        if "norm_mean" in track_row.index and pd.notna(track_row.get("norm_mean")):
            tp_stats = {
                "mean": torch.tensor(track_row["norm_mean"], dtype=torch.float32),
                "std": torch.tensor(track_row["norm_std"], dtype=torch.float32),
                "median": torch.tensor(track_row["norm_median"], dtype=torch.float32),
                "iqr": torch.tensor(track_row["norm_iqr"], dtype=torch.float32),
            }
            if self._channel_mode == "from_index":
                return {"channel_0": {"timepoint_statistics": tp_stats}}
            else:
                ch_name = track_row.get("channel_name", "channel_0")
                return {ch_name: {"timepoint_statistics": tp_stats}}

        # Fallback: read from zarr zattrs (old parquets without norm columns)
        store_path = track_row["store_path"]
        fov_name = track_row["fov_name"]
        t = track_row["t"]
        cache_key = (store_path, fov_name)
        if cache_key not in self._norm_meta_cache:
            position = self._get_position(store_path, fov_name)
            self._norm_meta_cache[cache_key] = _read_norm_meta(position)
        cached = self._norm_meta_cache[cache_key]
        if cached is None:
            return None
        raw_norm_meta = {}
        for ch, ch_meta in cached.items():
            resolved = {}
            for level, level_stats in ch_meta.items():
                if level == "timepoint_statistics" and isinstance(level_stats, dict):
                    resolved[level] = level_stats.get(str(t))
                else:
                    resolved[level] = level_stats
            raw_norm_meta[ch] = resolved
        if forced_channel_names is not None and self._channel_mode == "from_index":
            ch = forced_channel_names[0]
            if ch in raw_norm_meta:
                return {"channel_0": raw_norm_meta[ch]}
            return None
        if forced_channel_names is not None and self._channel_mode == "fixed":
            raw_norm_meta = {name: raw_norm_meta[name] for name in forced_channel_names if name in raw_norm_meta}
            return raw_norm_meta or None
        return raw_norm_meta

    def _slice_patch(
        self, track_row: pd.Series, forced_channel_names: list[str] | None = None
    ) -> tuple[
        "ts.TensorStore",
        NormMeta | None,
        tuple[float, float, float],
        tuple[int, int, int],
    ]:
        """Slice a patch from the image store for a given track row.

        Resolves zarr channel indices directly from channel names. Uses
        ``y_clamp`` / ``x_clamp`` for border-safe centering, and scale factors
        from the registry for physical-space normalization.

        Parameters
        ----------
        track_row : pd.Series
            A single row from ``tracks`` or ``valid_anchors``.
        forced_channel_names : list[str] or None
            Zarr channel names to read. When provided, only these channels
            are sliced from the zarr. None reads all channels.

        Returns
        -------
        tuple[ts.TensorStore, NormMeta | None, tuple[float, float, float], tuple[int, int, int]]
            The sliced patch (lazy tensorstore), normalization metadata,
            scale factors ``(scale_z, scale_y, scale_x)``, and target size
            ``(z_window, patch_h, patch_w)``.
        """
        store_path = track_row["store_path"]
        fov_name = track_row["fov_name"]
        exp_name = track_row["experiment"]

        image = self._get_tensorstore(store_path, fov_name)

        t = track_row["t"]
        y_center = int(track_row["y_clamp"])
        x_center = int(track_row["x_clamp"])

        # Per-experiment scale factors for physical-space normalization
        scale_z, scale_y, scale_x = self.index.registry.scale_factors[exp_name]
        y_half = round((self.index.yx_patch_size[0] // 2) * scale_y)
        x_half = round((self.index.yx_patch_size[1] // 2) * scale_x)

        # Resolve zarr channel indices from channel names
        exp = self.index.registry._name_map[exp_name]
        if forced_channel_names is not None:
            channel_names_to_read = forced_channel_names
        else:
            channel_names_to_read = exp.channel_names
        channel_indices = [exp.channel_names.index(name) for name in channel_names_to_read]

        # Per-experiment z_range (scale-adjusted window size centered on z_range center)
        z_start_base, z_end_base = self.index.registry.z_ranges[exp_name]
        z_window_size = z_end_base - z_start_base
        z_count = round(z_window_size * scale_z)
        z_focus = (z_start_base + z_end_base) // 2
        z_start = z_focus - z_count // 2
        z_end = z_start + z_count
        patch = image.oindex[
            t,
            [int(c) for c in channel_indices],
            slice(z_start, z_end),
            slice(y_center - y_half, y_center + y_half),
            slice(x_center - x_half, x_center + x_half),
        ]

        # Build norm_meta from parquet columns (preferred) or zarr zattrs (fallback).
        raw_norm_meta = self._build_norm_meta(track_row, forced_channel_names)

        # Use the configured extraction window as uniform target Z,
        # not the per-experiment capped range. This ensures all patches
        # in a mixed-experiment batch rescale to the same Z depth.
        # The random/center crop in on_after_batch_transfer then crops
        # to the final z_window.
        z_target = self.index.registry.z_extraction_window or z_window_size
        target_size = (
            z_target,
            self.index.yx_patch_size[0],
            self.index.yx_patch_size[1],
        )
        return patch, raw_norm_meta, (scale_z, scale_y, scale_x), target_size

    def _slice_patches(
        self,
        track_rows: pd.DataFrame,
        forced_channel_names: list[list[str]] | None = None,
    ) -> tuple[torch.Tensor, list[NormMeta | None]]:
        """Slice and stack patches for multiple track rows.

        Parameters
        ----------
        track_rows : pd.DataFrame
            Multiple rows from ``tracks`` / ``valid_anchors``.
        forced_channel_names : list[list[str]] or None
            Per-sample zarr channel names to read. Each inner list
            contains the channel names for that sample.
            None reads all channels for every sample.

        Returns
        -------
        tuple[torch.Tensor, list[NormMeta | None]]
            Stacked tensor ``(B, C, Z, Y, X)`` and per-sample norm metadata.
        """
        patches = []
        norms = []
        scales = []
        targets = []
        for i, (_, row) in enumerate(track_rows.iterrows()):
            forced = forced_channel_names[i] if forced_channel_names is not None else None
            patch, norm, scale, target = self._slice_patch(row, forced_channel_names=forced)
            patches.append(patch)
            norms.append(norm)
            scales.append(scale)
            targets.append(target)
        # Group patches by shape so ts.stack works within each group,
        # then read and rescale. This handles mixed-experiment batches
        # where different pixel sizes produce different native crop sizes.
        shape_groups: dict[tuple, list[int]] = defaultdict(list)
        for i, p in enumerate(patches):
            shape_groups[tuple(p.shape)].append(i)
        read_tensors: list[Tensor | None] = [None] * len(patches)
        for idxs in shape_groups.values():
            group_patches = [patches[i] for i in idxs]
            group_result = ts.stack([p.translate_to[0] for p in group_patches]).read().result()  # noqa: PD013
            for j, idx in enumerate(idxs):
                read_tensors[idx] = torch.from_numpy(group_result[j])
        # Rescale each patch to the uniform target size
        rescaled = []
        for i in range(len(patches)):
            rescaled.append(_rescale_patch(read_tensors[i], scales[i], targets[i]))
        return torch.stack(rescaled), norms
