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


def _pick_temporal_candidate(
    timepoints: dict[int, list[int]],
    anchor_t: int,
    tau_min: int,
    tau_max: int,
    tau_decay_rate: float,
    rng: np.random.Generator,
    tr_marker_arr: np.ndarray | None,
    anchor_marker: object | None,
) -> int | None:
    """Pick one positive tracks-index for a temporal anchor.

    Mirrors the legacy ``_find_temporal_positive._pick`` logic but
    operates on pre-computed NumPy arrays. Returns ``None`` if no
    candidate is found in the ``[tau_min, tau_max]`` window.
    """

    def _filter_and_pick(cand_indices: list[int]) -> int | None:
        if not cand_indices:
            return None
        if tr_marker_arr is not None:
            # NumPy fancy-index filter: O(n) with n = number of candidates,
            # single vectorized array op.
            idx_arr = np.asarray(cand_indices, dtype=np.int64)
            mask = tr_marker_arr[idx_arr] == anchor_marker
            filtered = idx_arr[mask]
            if len(filtered) > 0:
                return int(filtered[rng.integers(len(filtered))])
        return int(cand_indices[rng.integers(len(cand_indices))])

    sampled_tau = sample_tau(tau_min, tau_max, rng, tau_decay_rate)
    result = _filter_and_pick(timepoints.get(anchor_t + sampled_tau, []))
    if result is not None:
        return result
    for tau in range(tau_min, tau_max + 1):
        if tau == 0:
            continue
        result = _filter_and_pick(timepoints.get(anchor_t + tau, []))
        if result is not None:
            return result
    return None


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
        self._build_anchor_cache()

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
            # observed=True skips unobserved Categorical cross-products;
            # without it groupby yields empty groups for every Categorical
            # combination, exploding memory and time. Keys are coerced to
            # str so the lookup works regardless of dtype (Categorical vs
            # object vs ArrowString).
            grouped = tracks.groupby(["experiment", "lineage_id", "t"], observed=True).indices
            self._lineage_timepoints: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for (exp, lid, t), row_indices in grouped.items():
                self._lineage_timepoints[(str(exp), str(lid))][int(t)] = row_indices.tolist()
        else:
            cols = self.positive_match_columns
            grouped = tracks.groupby(cols).indices
            # Store candidate indices as ndarray for O(1) random choice without list copy.
            self._match_lookup: dict[tuple, np.ndarray] = {
                (k if isinstance(k, tuple) else (k,)): v for k, v in grouped.items()
            }

    def _build_anchor_cache(self) -> None:
        """Cache valid_anchors/tracks columns as NumPy arrays for fast per-sample access.

        Avoids pandas ``.iloc[idx][col]`` in the hot path, which constructs a
        Series per call (~9 ms per anchor on 81M-row indices). NumPy indexing
        is ~20 ns. Measured end-to-end speedup: ~3000× on positive-lookup.

        Both ``_va_arrays`` (for anchors) and ``_tr_arrays`` (for positives)
        cache the full set of columns needed by ``_slice_patch`` and
        ``_build_norm_meta``: ``store_path``, ``fov_name``, ``experiment``,
        ``t``, ``y_clamp``, ``x_clamp``, plus ``norm_*`` columns for the
        parquet-norm fast path.

        Cache is in-process RAM only — rebuilt on every dataset instantiation
        from ``self.index.valid_anchors`` / ``self.index.tracks``. Parquet
        remains the source of truth.

        Also precomputes per-experiment tau range (frames) to avoid a registry
        lookup per anchor inside ``_sample_positives_temporal``.
        """

        # High-cardinality string columns (store_path, fov_name, experiment,
        # marker, channel_name, lineage_id) have few unique values relative to
        # row count, so cache them as category codes + categories lookup instead
        # of object arrays. Object arrays of strings are ~40-80 bytes/entry; a
        # categorical code is 4-8 bytes. On 81M rows this is the difference
        # between an OOM and a healthy init.
        #
        # Access pattern: array[idx] still works if array is a pandas Categorical
        # (returns the underlying string); downstream code doesn't care.
        def _cache_columns(df: pd.DataFrame, columns: list[str]) -> dict:
            out = {}
            for col in columns:
                if col not in df.columns:
                    continue
                s = df[col]
                if s.dtype == object or pd.api.types.is_string_dtype(s):
                    out[col] = s.astype("category").array  # pd.Categorical
                else:
                    out[col] = s.to_numpy()
            return out

        # Whitelist columns actually read in the hot path. Caching every
        # column of valid_anchors (81M+ rows × ~20 cols × 4 DDP ranks) blows
        # the node memory budget; holding only the read set keeps per-rank
        # RSS in the low tens of GiB. `positive_match_columns` (user-defined)
        # and label column values must also be cached because they drive the
        # SupCon key construction and per-sample label lookup respectively.
        hot_cols: set[str] = {
            "channel_name",
            "experiment",
            "lineage_id",
            "t",
            "marker",
            "store_path",
            "fov_name",
            "y_clamp",
            "x_clamp",
            "norm_mean",
            "norm_std",
            "norm_median",
            "norm_iqr",
        }
        if self.positive_match_columns:
            hot_cols.update(self.positive_match_columns)
        if getattr(self, "_label_encoders", None):
            for col, _encoder in self._label_encoders.values():
                hot_cols.add(col)

        self._va_arrays: dict = _cache_columns(self.index.valid_anchors, sorted(hot_cols))
        self._tr_arrays: dict = _cache_columns(self.index.tracks, sorted(hot_cols))

        # Precompute per-experiment tau range in frames to avoid a per-anchor
        # registry call inside _sample_positive_indices_temporal. Skip
        # experiments with interval_minutes == 0 (static/snapshot datasets like
        # OPS) — they never go through the temporal path (positive_match_columns
        # wouldn't include lineage_id), so missing entries are harmless and
        # computing tau_range_frames for them would ZeroDivisionError.
        self._tau_range_frames_cache: dict[str, tuple[int, int]] = {}
        for name, exp in self.index.registry._name_map.items():
            if getattr(exp, "interval_minutes", 0):
                self._tau_range_frames_cache[name] = self.index.registry.tau_range_frames(name, self.tau_range_hours)

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
        # Use the NumPy cache to avoid a pandas Series construction per row.
        if self._channel_mode == "from_index":
            chan_arr = self._va_arrays["channel_name"]
            forced_channel_names = [[chan_arr[i]] for i in indices]
        elif self._channel_mode == "fixed":
            forced_channel_names = [self._fixed_channel_names] * len(indices)
        else:
            forced_channel_names = None

        anchor_patches, anchor_norms = self._slice_patches(self._va_arrays, indices, forced_channel_names)
        sample: dict = {
            "anchor": anchor_patches,
            "anchor_norm_meta": anchor_norms,
            "anchor_meta": self._extract_meta(anchor_rows),
        }

        if self.fit:
            if self.positive_cell_source == "self":
                # SimCLR: anchor and positive share the same patch pre-augmentation.
                # Skip the second zarr read + meta extraction entirely — augmentation
                # (applied independently downstream in on_after_batch_transfer) is
                # what creates the two views. This roughly halves per-batch wall
                # time for SimCLR baselines.
                # clone the tensor so augmentation has an independent buffer to
                # mutate without leaking into the anchor.
                sample["positive"] = sample["anchor"].clone()
                sample["positive_norm_meta"] = sample["anchor_norm_meta"]
                sample["positive_meta"] = sample["anchor_meta"]
            else:
                pos_track_indices = self._sample_positive_indices(anchor_positions=indices)
                if self._channel_mode == "from_index":
                    tr_chan_arr = self._tr_arrays["channel_name"]
                    pos_forced_channel_names = [[tr_chan_arr[i]] for i in pos_track_indices]
                else:
                    pos_forced_channel_names = forced_channel_names
                positive_patches, positive_norms = self._slice_patches(
                    self._tr_arrays, pos_track_indices, pos_forced_channel_names
                )
                positive_rows = self.index.tracks.iloc[pos_track_indices].reset_index(drop=True)
                sample["positive"] = positive_patches
                sample["positive_norm_meta"] = positive_norms
                sample["positive_meta"] = self._extract_meta(positive_rows)
        else:
            # Build per-sample index dicts via NumPy column arrays (no .iterrows).
            all_cols = list(ULTRACK_INDEX_COLUMNS) + [
                "experiment",
                "marker",
                "perturbation",
                "hours_post_perturbation",
                "organelle",
                "well",
                "microscope",
            ]
            present_cols = [c for c in all_cols if c in anchor_rows.columns]
            col_arrays = {c: anchor_rows[c].to_numpy() for c in present_cols}
            sample["index"] = [{c: col_arrays[c][i] for c in present_cols} for i in range(len(anchor_rows))]

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
            # Pre-extract label columns as NumPy arrays once (avoids per-row
            # Series construction in .iterrows()).
            label_arrays = {
                batch_key: (encoder, rows[col].to_numpy() if col in rows.columns else None)
                for batch_key, (col, encoder) in self._label_encoders.items()
            }
            for i in range(len(records)):
                labels = {}
                for batch_key, (encoder, arr) in label_arrays.items():
                    if arr is None:
                        continue
                    val = arr[i]
                    if val is not None and val in encoder:
                        labels[batch_key] = encoder[val]
                records[i]["labels"] = labels
        return records

    # ------------------------------------------------------------------
    # Positive sampling
    # ------------------------------------------------------------------

    def _sample_positive_indices(
        self,
        anchor_positions: list[int],
    ) -> np.ndarray:
        """Sample one positive tracks-index for each anchor.

        Returns positional indices into ``self.index.tracks`` / ``self._tr_arrays``
        — callers can slice patches directly from the cached NumPy arrays without
        materializing a DataFrame. The DataFrame is still constructed downstream
        for metadata extraction.

        Parameters
        ----------
        anchor_positions : list[int]
            Positional indices into ``valid_anchors`` (same as the sampler output).

        Returns
        -------
        np.ndarray
            One tracks-positional-index per anchor, shape ``(len(anchor_positions),)``.
        """
        # Temporal lineage mode — vectorized NumPy fast path
        # (used by DynaCLR-2D-MIP, DynaCLR-3D-BagOfChannels).
        if "lineage_id" in self.positive_match_columns:
            return self._sample_positive_indices_temporal(anchor_positions)

        # Column-match mode (SupCon) — vectorized NumPy fast path.
        cols = self.positive_match_columns
        va_col_arrs = [self._va_arrays[c] for c in cols]

        pos_track_indices = np.empty(len(anchor_positions), dtype=np.int64)
        match_lookup = self._match_lookup
        rng = self._rng
        for i, ai in enumerate(anchor_positions):
            key = tuple(arr[ai] for arr in va_col_arrs)
            cands = match_lookup.get(key)
            if cands is None or len(cands) == 0:
                raise RuntimeError(
                    f"No positive found for anchor at position {ai} key={key}. "
                    "This anchor should have been filtered out by valid_anchors."
                )
            # Random pick from candidates. Note: the anchor's own tracks-index
            # may be in `cands`; we don't filter it out explicitly because the
            # anchor's valid_anchors-position and its tracks-index are in
            # independent index spaces after reset_index(drop=True), and the
            # original per-row implementation made the same loose comparison.
            # For typical group sizes (>100), the self-as-positive probability
            # is <1% — functionally equivalent to `positive_cell_source="self"`.
            pos_track_indices[i] = cands[rng.integers(len(cands))]

        return pos_track_indices

    def _sample_positive_indices_temporal(self, anchor_positions: list[int]) -> np.ndarray:
        """Vectorized temporal positive lookup (lineage + tau range).

        Uses pre-computed NumPy caches instead of per-row pandas ``.iloc``.
        Uses ``self._tau_range_frames_cache`` to avoid a registry call per anchor.

        Parameters
        ----------
        anchor_positions : list[int]
            Positional indices into ``valid_anchors`` for the batch.

        Returns
        -------
        np.ndarray
            Positional indices into ``self.index.tracks``, one per anchor.
        """
        rng = self._rng
        exp_arr = self._va_arrays["experiment"]
        lid_arr = self._va_arrays["lineage_id"]
        t_arr = self._va_arrays["t"]
        tau_cache = self._tau_range_frames_cache

        # In from_index mode (flat parquet), we filter candidates to same marker.
        marker_filter = self._channel_mode == "from_index"
        if marker_filter:
            anchor_marker_arr = self._va_arrays["marker"]
            tr_marker_arr = self._tr_arrays["marker"]

        pos_track_indices = np.empty(len(anchor_positions), dtype=np.int64)
        lt_map = self._lineage_timepoints

        for i, ai in enumerate(anchor_positions):
            # Coerce to str: _va_arrays columns come back as Categorical
            # scalars after _materialize_strings, which hash differently
            # from the str keys in _lineage_timepoints / _tau_range_frames_cache.
            exp_name = str(exp_arr[ai])
            lineage_id = str(lid_arr[ai])
            anchor_t = int(t_arr[ai])

            tau_min, tau_max = tau_cache[exp_name]
            timepoints = lt_map.get((exp_name, lineage_id))
            if timepoints is None:
                raise RuntimeError(
                    f"No positive found for anchor at position {ai} "
                    f"(experiment={exp_name}, lineage_id={lineage_id}, t={anchor_t}). "
                    "This anchor should have been filtered out by valid_anchors."
                )

            anchor_marker = anchor_marker_arr[ai] if marker_filter else None
            chosen = _pick_temporal_candidate(
                timepoints,
                anchor_t,
                tau_min,
                tau_max,
                self.tau_decay_rate,
                rng,
                tr_marker_arr if marker_filter else None,
                anchor_marker,
            )
            if chosen is None:
                raise RuntimeError(
                    f"No positive found for anchor at position {ai} "
                    f"(experiment={exp_name}, lineage_id={lineage_id}, t={anchor_t}). "
                    "This anchor should have been filtered out by valid_anchors."
                )
            pos_track_indices[i] = chosen

        return pos_track_indices

    # ------------------------------------------------------------------
    # Patch extraction (tensorstore I/O)
    # ------------------------------------------------------------------

    def _get_position(self, store_path: str, fov_name: str):
        """Get or create a cached Position object for the given FOV.

        Cache is keyed by ``(store_path, fov_name)`` — critical for OPS
        where the same FOV name (e.g. ``"A/3/0"``) appears across multiple
        experiments.

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
        key = (store_path, fov_name)
        if key not in self._position_cache:
            if store_path not in self._store_cache:
                self._store_cache[store_path] = open_ome_zarr(store_path, mode="r")
            plate = self._store_cache[store_path]
            self._position_cache[key] = plate[fov_name]
        return self._position_cache[key]

    def _get_tensorstore(self, store_path: str, fov_name: str) -> "ts.TensorStore":
        """Get or create a cached tensorstore object for the given FOV.

        Cache is keyed by ``(store_path, fov_name)`` — critical for OPS
        where the same FOV name appears across multiple experiments.

        Parameters
        ----------
        store_path : str
            Path to the OME-Zarr plate store.
        fov_name : str
            FOV name used together with ``store_path`` as cache key.

        Returns
        -------
        ts.TensorStore
        """
        key = (store_path, fov_name)
        if key not in self._tensorstores:
            position = self._get_position(store_path, fov_name)
            self._tensorstores[key] = position["0"].tensorstore(
                context=self._ts_context,
                recheck_cached_data="open",
            )
        return self._tensorstores[key]

    def _build_norm_meta(
        self,
        arrays: dict[str, np.ndarray],
        idx: int,
        forced_channel_names: list[str] | None,
    ) -> NormMeta | None:
        """Build per-sample normalization metadata from parquet columns.

        When the parquet has ``norm_mean`` / ``norm_std`` columns (written by
        ``preprocess-cell-index``), reads stats directly from the cached
        NumPy arrays — no zarr zattrs access and no pandas Series construction.
        Falls back to zarr zattrs for old parquets.

        Parameters
        ----------
        arrays : dict[str, np.ndarray]
            Pre-cached NumPy column arrays (``_va_arrays`` or ``_tr_arrays``).
        idx : int
            Positional row index into ``arrays``.
        forced_channel_names : list[str] or None
            Zarr channel names being read for this sample.

        Returns
        -------
        NormMeta or None
        """
        # Parquet path: norm columns present and value is not NA
        norm_mean_arr = arrays.get("norm_mean")
        if norm_mean_arr is not None:
            norm_mean = norm_mean_arr[idx]
            if norm_mean is not None and not (isinstance(norm_mean, float) and np.isnan(norm_mean)):
                tp_stats = {
                    "mean": torch.tensor(norm_mean, dtype=torch.float32),
                    "std": torch.tensor(arrays["norm_std"][idx], dtype=torch.float32),
                    "median": torch.tensor(arrays["norm_median"][idx], dtype=torch.float32),
                    "iqr": torch.tensor(arrays["norm_iqr"][idx], dtype=torch.float32),
                }
                if self._channel_mode == "from_index":
                    return {"channel_0": {"timepoint_statistics": tp_stats}}
                else:
                    ch_arr = arrays.get("channel_name")
                    ch_name = ch_arr[idx] if ch_arr is not None else "channel_0"
                    return {ch_name: {"timepoint_statistics": tp_stats}}

        # Fallback: read from zarr zattrs (old parquets without norm columns)
        store_path = arrays["store_path"][idx]
        fov_name = arrays["fov_name"][idx]
        t = arrays["t"][idx]
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
        self,
        arrays: dict[str, np.ndarray],
        idx: int,
        forced_channel_names: list[str] | None = None,
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
        arrays : dict[str, np.ndarray]
            Pre-cached NumPy column arrays (``_va_arrays`` or ``_tr_arrays``).
        idx : int
            Positional row index into ``arrays``.
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
        store_path = arrays["store_path"][idx]
        fov_name = arrays["fov_name"][idx]
        exp_name = arrays["experiment"][idx]

        image = self._get_tensorstore(store_path, fov_name)

        t = int(arrays["t"][idx])
        y_center = int(arrays["y_clamp"][idx])
        x_center = int(arrays["x_clamp"][idx])

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
        raw_norm_meta = self._build_norm_meta(arrays, idx, forced_channel_names)

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
        arrays: dict[str, np.ndarray],
        indices: list[int] | np.ndarray,
        forced_channel_names: list[list[str]] | None = None,
    ) -> tuple[torch.Tensor, list[NormMeta | None]]:
        """Slice and stack patches for multiple track rows.

        Parameters
        ----------
        arrays : dict[str, np.ndarray]
            Pre-cached NumPy column arrays (``_va_arrays`` or ``_tr_arrays``).
        indices : list[int] or np.ndarray
            Positional row indices into ``arrays``.
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
        for i, idx in enumerate(indices):
            forced = forced_channel_names[i] if forced_channel_names is not None else None
            patch, norm, scale, target = self._slice_patch(arrays, int(idx), forced_channel_names=forced)
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
