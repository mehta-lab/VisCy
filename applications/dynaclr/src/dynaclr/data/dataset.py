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

from dynaclr.data.index import MultiExperimentIndex
from dynaclr.data.tau_sampling import sample_tau
from viscy_data._typing import ULTRACK_INDEX_COLUMNS, NormMeta, SampleMeta
from viscy_data._utils import _read_norm_meta

_META_COLUMNS = [
    "experiment",
    "condition",
    "microscope",
    "fov_name",
    "global_track_id",
    "t",
    "hours_post_perturbation",
    "lineage_id",
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
    bag_of_channels : bool
        If ``True``, randomly select one source channel per sample instead
        of reading all source channels.  Output shape is ``(B, 1, Z, Y, X)``
        instead of ``(B, C, Z, Y, X)``.
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
        ``"same"`` (default) — anchor and positive use the same channel index
        (bag-of-channels mode).  ``"any"`` — positive draws its channel
        independently from anchor.  Only meaningful when
        ``bag_of_channels=True``.
    cross_scope_fraction : float
        Deprecated. Use ``positive_match_columns=["condition"]`` instead.
        Fraction of positives sampled as cross-microscope positives. Default: 0.0.
    hpi_window : float
        Deprecated. Used only with ``cross_scope_fraction > 0``. Default: 1.0.
    """

    def __init__(
        self,
        index: MultiExperimentIndex,
        fit: bool = True,
        tau_range_hours: tuple[float, float] = (0.5, 2.0),
        tau_decay_rate: float = 2.0,
        return_negative: bool = False,
        cache_pool_bytes: int = 0,
        bag_of_channels: bool = False,
        positive_cell_source: str = "lookup",
        positive_match_columns: list[str] | None = None,
        positive_channel_source: str = "same",
        cross_scope_fraction: float = 0.0,
        hpi_window: float = 1.0,
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
        self.bag_of_channels = bag_of_channels
        self.positive_cell_source = positive_cell_source
        self.positive_match_columns = positive_match_columns if positive_match_columns is not None else ["lineage_id"]
        self.positive_channel_source = positive_channel_source
        self.cross_scope_fraction = cross_scope_fraction
        self.hpi_window = hpi_window

        if cross_scope_fraction > 0:
            import warnings

            warnings.warn(
                "cross_scope_fraction is deprecated. Use positive_match_columns=['condition'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            missing_microscope = [e.name for e in index.registry.experiments if not e.microscope]
            if missing_microscope:
                raise ValueError(
                    f"cross_scope_fraction > 0 but experiments are missing microscope field: {missing_microscope}"
                )

        self._rng = np.random.default_rng()
        self._setup_tensorstore_context(cache_pool_bytes)
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

        if "lineage_id" in self.positive_match_columns:
            self._lineage_timepoints: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for idx, row in self.index.tracks.iterrows():
                key = (row["experiment"], row["lineage_id"])
                self._lineage_timepoints[key][row["t"]].append(idx)
        else:
            cols = self.positive_match_columns
            self._match_lookup: dict[tuple, list[int]] = defaultdict(list)
            for idx, row in self.index.tracks.iterrows():
                key = tuple(row[c] for c in cols)
                self._match_lookup[key].append(idx)

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

        # In bag-of-channels mode, pre-sample one channel index per item so that
        # anchor and positive use the same channel (when positive_channel_source="same").
        if self.bag_of_channels:
            n_channels = len(self.index.registry.source_channel_labels)
            forced_channel_indices = list(self._rng.integers(n_channels, size=len(indices)))
        else:
            forced_channel_indices = None

        anchor_patches, anchor_norms = self._slice_patches(anchor_rows, forced_channel_indices)
        sample: dict = {
            "anchor": anchor_patches,
            "anchor_norm_meta": anchor_norms,
            "anchor_meta": self._extract_meta(anchor_rows),
        }

        if self.fit:
            positive_rows = self._sample_positives(anchor_rows)
            # positive_channel_source="any": positive draws its channel independently
            if self.bag_of_channels and self.positive_channel_source == "any":
                n_channels = len(self.index.registry.source_channel_labels)
                pos_forced_channel_indices = list(self._rng.integers(n_channels, size=len(indices)))
            else:
                pos_forced_channel_indices = forced_channel_indices
            positive_patches, positive_norms = self._slice_patches(positive_rows, pos_forced_channel_indices)
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
                indices_list.append(idx_dict)
            sample["index"] = indices_list

        return sample

    @staticmethod
    def _extract_meta(rows: pd.DataFrame) -> list[SampleMeta]:
        """Extract lightweight metadata dicts from track rows.

        Parameters
        ----------
        rows : pd.DataFrame
            Rows from ``valid_anchors`` or ``tracks``.

        Returns
        -------
        list[dict]
            One dict per row with keys from ``_META_COLUMNS``.
        """
        cols = [c for c in _META_COLUMNS if c in rows.columns]
        return rows[cols].to_dict(orient="records")

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

        # Try sampled tau first, then scan full range as fallback
        sampled_tau = sample_tau(tau_min, tau_max, rng, self.tau_decay_rate)
        target_t = anchor_t + sampled_tau
        candidates = lt_map.get(target_t, [])
        if candidates:
            chosen_idx = candidates[rng.integers(len(candidates))]
            return self.index.tracks.iloc[chosen_idx]

        for tau in range(tau_min, tau_max + 1):
            if tau == 0:
                continue
            candidates_fb = lt_map.get(anchor_t + tau, [])
            if candidates_fb:
                chosen_idx = candidates_fb[rng.integers(len(candidates_fb))]
                return self.index.tracks.iloc[chosen_idx]

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

    def _find_cross_scope_positive(
        self,
        anchor_row: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series | None:
        """Find a cross-microscope positive for a given anchor.

        Deprecated. Use ``positive_match_columns=["condition"]`` instead.

        Parameters
        ----------
        anchor_row : pd.Series
            A single row from ``valid_anchors``.
        rng : numpy.random.Generator
            Random number generator for tie-breaking.

        Returns
        -------
        pd.Series or None
            A track row for the cross-scope positive, or ``None`` if no candidate found.
        """
        tracks = self.index.tracks
        candidates = tracks[
            (tracks["microscope"] != anchor_row["microscope"])
            & (tracks["condition"] == anchor_row["condition"])
            & ((tracks["hours_post_perturbation"] - anchor_row["hours_post_perturbation"]).abs() <= self.hpi_window)
        ]
        if candidates.empty:
            return None
        return candidates.iloc[rng.integers(len(candidates))]

    # ------------------------------------------------------------------
    # Patch extraction (tensorstore I/O)
    # ------------------------------------------------------------------

    def _get_tensorstore(self, position, fov_name: str) -> "ts.TensorStore":
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
        self, track_row: pd.Series, forced_source_idx: int | None = None
    ) -> tuple[
        "ts.TensorStore",
        NormMeta | None,
        tuple[float, float, float],
        tuple[int, int, int],
    ]:
        """Slice a patch from the image store for a given track row.

        Uses per-experiment ``channel_maps`` for channel index remapping,
        ``y_clamp`` / ``x_clamp`` for border-safe centering, and scale factors
        from the registry for physical-space normalization.

        Parameters
        ----------
        track_row : pd.Series
            A single row from ``tracks`` or ``valid_anchors``.

        Returns
        -------
        tuple[ts.TensorStore, NormMeta | None, tuple[float, float, float], tuple[int, int, int]]
            The sliced patch (lazy tensorstore), normalization metadata,
            scale factors ``(scale_z, scale_y, scale_x)``, and target size
            ``(z_window, patch_h, patch_w)``.
        """
        position = track_row["position"]
        fov_name = track_row["fov_name"]
        exp_name = track_row["experiment"]

        image = self._get_tensorstore(position, fov_name)

        t = track_row["t"]
        y_center = int(track_row["y_clamp"])
        x_center = int(track_row["x_clamp"])

        # Per-experiment scale factors for physical-space normalization
        scale_z, scale_y, scale_x = self.index.registry.scale_factors[exp_name]
        y_half = round((self.index.yx_patch_size[0] // 2) * scale_y)
        x_half = round((self.index.yx_patch_size[1] // 2) * scale_x)

        # Per-experiment channel remapping
        channel_map = self.index.registry.channel_maps[exp_name]
        source_labels = self.index.registry.source_channel_labels
        if self.bag_of_channels:
            source_idx = int(
                forced_source_idx if forced_source_idx is not None else self._rng.integers(len(channel_map))
            )
            channel_indices = [channel_map[source_idx]]
            selected_label = source_labels[source_idx]
        else:
            channel_indices = [channel_map[i] for i in sorted(channel_map.keys())]

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

        # Remap norm_meta keys from zarr channel names to source labels
        # and pre-resolve timepoint_statistics for this sample's timepoint
        raw_norm_meta = _read_norm_meta(position)
        if raw_norm_meta is not None:
            key_map = self.index.registry.norm_meta_key_maps[exp_name]
            remapped = {key_map[k]: v for k, v in raw_norm_meta.items() if k in key_map}
            for label, ch_meta in remapped.items():
                if "timepoint_statistics" in ch_meta:
                    tp_stats = ch_meta["timepoint_statistics"].get(str(t))
                    ch_meta["timepoint_statistics"] = tp_stats
            if self.bag_of_channels:
                if selected_label in remapped:
                    raw_norm_meta = {"channel": remapped[selected_label]}
                else:
                    raw_norm_meta = None
            else:
                raw_norm_meta = remapped

        target_size = (
            z_window_size,
            self.index.yx_patch_size[0],
            self.index.yx_patch_size[1],
        )
        return patch, raw_norm_meta, (scale_z, scale_y, scale_x), target_size

    def _slice_patches(
        self,
        track_rows: pd.DataFrame,
        forced_channel_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[NormMeta | None]]:
        """Slice and stack patches for multiple track rows.

        Parameters
        ----------
        track_rows : pd.DataFrame
            Multiple rows from ``tracks`` / ``valid_anchors``.
        forced_channel_indices : list[int] or None
            Per-sample source channel indices to use (bag-of-channels mode).
            When provided, overrides the random draw in ``_slice_patch``.

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
            forced = forced_channel_indices[i] if forced_channel_indices is not None else None
            patch, norm, scale, target = self._slice_patch(row, forced_source_idx=forced)
            patches.append(patch)
            norms.append(norm)
            scales.append(scale)
            targets.append(target)
        results = ts.stack([p.translate_to[0] for p in patches]).read().result()  # noqa: PD013
        tensor = torch.from_numpy(results)
        # Rescale patches that have non-unity scale factors
        rescaled = []
        for i in range(tensor.shape[0]):
            rescaled.append(_rescale_patch(tensor[i], scales[i], targets[i]))
        return torch.stack(rescaled), norms
