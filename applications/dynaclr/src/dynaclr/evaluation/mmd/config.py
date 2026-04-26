"""Pydantic configuration for the MMD perturbation evaluation step."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator


class ComparisonSpec(BaseModel):
    """One pairwise comparison to run MMD on.

    Parameters
    ----------
    cond_a : str
        Value of ``obs[group_by]`` for group A (typically the reference/control).
    cond_b : str
        Value of ``obs[group_by]`` for group B (typically the treatment).
    label : str
        Human-readable label for this comparison (used in output filenames and plots).
    """

    cond_a: str
    cond_b: str
    label: str


class MMDSettings(BaseModel):
    """Kernel MMD algorithm settings, shared across per-experiment and combined modes.

    Parameters
    ----------
    n_permutations : int
        Number of permutations for the significance test. Default: 1000.
    max_cells : int or None
        Subsample each group to at most this many cells before computing MMD.
        Controls memory and compute cost. Default: 2000.
    min_cells : int
        Minimum cells required per group. Groups below this produce NaN. Default: 20.
    seed : int
        Random seed for subsampling and permutations. Default: 42.
    balance_samples : bool
        Subsample the larger group to match the smaller group's size before
        computing MMD. Prevents sample-size imbalance from inflating test statistics.
        Applied after the ``max_cells`` cap. Default: False.
    share_bandwidth_from : str or None
        Label of a comparison whose bandwidth should be reused for all other
        comparisons within the same (marker, time_bin) group. Typically the
        baseline comparison (e.g. ``"uninf1 vs uninf2"``). If None, each
        comparison computes its own bandwidth independently. Default: None.
    """

    n_permutations: int = 1000
    max_cells: Optional[int] = 2000
    min_cells: int = 20
    seed: int = 42
    balance_samples: bool = False
    share_bandwidth_from: Optional[str] = None


class MAPSettings(BaseModel):
    """Settings for the copairs-based mean Average Precision metric.

    Parameters
    ----------
    enabled : bool
        Compute mAP alongside MMD. Requires the ``copairs`` package. Default: False.
    distance : str
        Distance metric passed to copairs (e.g. ``"cosine"``). Default: ``"cosine"``.
    null_size : int
        Number of null pairs for the mAP permutation test. Default: 10000.
    seed : int
        Random seed. Default: 0.
    """

    enabled: bool = False
    distance: str = "cosine"
    null_size: int = 10000
    seed: int = 0


class _MMDBaseConfig(BaseModel):
    """Shared fields for all MMD analysis modes.

    Parameters
    ----------
    output_dir : str
        Directory for CSV results and plots.
    group_by : str
        obs column used to select condition groups. Default: ``"perturbation"``.
    obs_filter : dict[str, str] or None
        Restrict analysis to rows where ``obs[key] == value``. Default: None.
    embedding_key : str or None
        obsm key to use. None = raw ``.X`` backbone embeddings. Default: None.
    mmd : MMDSettings
        Kernel MMD algorithm settings.
    map_settings : MAPSettings
        copairs-based mAP settings. Default: disabled.
    temporal_bin_size : float or None
        Width of each temporal bin in hours, starting from 0.
        Bin edges: ``[0, size, 2*size, ..., max_hours]``.
        Mutually exclusive with ``temporal_bins``. Default: None (aggregate).
    temporal_bins : list[float] or None
        Explicit bin edges in hours (e.g. ``[0, 6, 12, 24]``). Takes precedence
        over ``temporal_bin_size``. Default: None (aggregate).
    save_plots : bool
        Generate plots after computing metrics. Default: True.
    """

    output_dir: str
    group_by: str = "perturbation"
    obs_filter: Optional[dict[str, str]] = None
    embedding_key: Optional[str] = None
    mmd: MMDSettings = MMDSettings()
    map_settings: MAPSettings = MAPSettings()
    temporal_bin_size: Optional[float] = None
    temporal_bins: Optional[list[float]] = None
    save_plots: bool = True

    @model_validator(mode="after")
    def _validate_temporal(self) -> "_MMDBaseConfig":
        if self.temporal_bin_size is not None and self.temporal_bins is not None:
            raise ValueError("temporal_bin_size and temporal_bins are mutually exclusive")
        return self


def _resolve_bin_edges(
    temporal_bin_size: Optional[float],
    temporal_bins: Optional[list[float]],
    max_hours: float,
) -> Optional[list[tuple[float, float]]]:
    """Return a list of (start, end) bin edge pairs, or None if no temporal binning.

    Parameters
    ----------
    temporal_bin_size : float or None
        Uniform bin width. Generates edges ``[0, size, 2*size, ..., max_hours]``.
    temporal_bins : list[float] or None
        Explicit bin edges (e.g. ``[0, 6, 12, 24]``). Takes precedence over
        ``temporal_bin_size``.
    max_hours : float
        Maximum hours value in the data, used only when ``temporal_bin_size`` is set.

    Returns
    -------
    list[tuple[float, float]] or None
        Ordered list of ``(bin_start, bin_end)`` pairs, or ``None`` for aggregate mode.
    """
    if temporal_bins is not None:
        edges = temporal_bins
    elif temporal_bin_size is not None:
        edges = list(np.arange(0, max_hours + temporal_bin_size, temporal_bin_size))
    else:
        return None
    return list(zip(edges[:-1], edges[1:]))


class MMDEvalConfig(_MMDBaseConfig):
    """Per-experiment MMD analysis with explicit pairwise comparisons.

    Parameters
    ----------
    input_path : str
        Path to a single per-experiment AnnData zarr store.
    comparisons : list[ComparisonSpec]
        Explicit list of pairwise comparisons to run (required).
    """

    input_path: str
    comparisons: list[ComparisonSpec]

    @model_validator(mode="after")
    def _validate(self) -> "MMDEvalConfig":
        if not self.comparisons:
            raise ValueError("comparisons must not be empty")
        return self


class MMDCombinedConfig(_MMDBaseConfig):
    """Pairwise cross-experiment MMD for batch-effect detection.

    Conditions are auto-discovered from the data intersection — no explicit
    comparisons needed. For each marker shared between a pair of experiments,
    runs MMD per (condition, time_bin) after per-experiment mean centering.

    Parameters
    ----------
    input_paths : list[str]
        Paths to per-experiment AnnData zarr stores.
    """

    input_paths: list[str]


class MMDPooledConfig(_MMDBaseConfig):
    """Pooled multi-experiment phenotypic analysis.

    Concatenates cells from all input experiments before computing MMD/mAP,
    faceted by marker and temporal bin. Unlike ``MMDCombinedConfig`` (pairwise
    batch-effect detection), this pools all experiments for a single biological
    comparison.

    Parameters
    ----------
    input_paths : list[str]
        Paths to per-experiment AnnData zarr stores to pool.
    comparisons : list[ComparisonSpec]
        Explicit list of pairwise comparisons to run (required).
    condition_aliases : dict[str, list[str]] or None
        Mapping from canonical condition name to variant strings found in the
        data. E.g. ``{"uninfected": ["uninfected", "uninfected1", "uninfected2"]}``.
        Applied to ``obs[group_by]`` before comparisons are evaluated.
    """

    input_paths: list[str]
    comparisons: list[ComparisonSpec]
    condition_aliases: Optional[dict[str, list[str]]] = None

    @model_validator(mode="after")
    def _validate(self) -> "MMDPooledConfig":
        if not self.comparisons:
            raise ValueError("comparisons must not be empty")
        return self
