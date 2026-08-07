"""Pydantic configuration for the MMD perturbation evaluation step."""

from __future__ import annotations

from typing import Literal

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
    max_cells: int | None = 2000
    min_cells: int = 20
    seed: int = 42
    balance_samples: bool = False
    share_bandwidth_from: str | None = None


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
    obs_filter: dict[str, str] | None = None
    embedding_key: str | None = None
    mmd: MMDSettings = MMDSettings()
    map_settings: MAPSettings = MAPSettings()
    temporal_bin_size: float | None = None
    temporal_bins: list[float] | None = None
    save_plots: bool = True

    @model_validator(mode="after")
    def _validate_temporal(self) -> "_MMDBaseConfig":
        if self.temporal_bin_size is not None and self.temporal_bins is not None:
            raise ValueError("temporal_bin_size and temporal_bins are mutually exclusive")
        return self


def _resolve_bin_edges(
    temporal_bin_size: float | None,
    temporal_bins: list[float] | None,
    max_hours: float,
) -> list[tuple[float, float]] | None:
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
    runs MMD per (condition, time_bin).

    Parameters
    ----------
    input_paths : list[str]
        Paths to per-experiment AnnData zarr stores.
    center_per_experiment : bool
        Subtract each experiment's own mean embedding before computing MMD.
        Default True detects *residual* batch effects independent of a global
        offset. Set False to keep the raw mean shift between experiments — this
        is required to validate a LOT correction whose main job is removing that
        offset (centering would delete the very effect being measured, so a
        genuine platform separation would collapse to a small MMD). Default: True.
    """

    input_paths: list[str]
    center_per_experiment: bool = True


class MMDOverTimeConfig(MMDCombinedConfig):
    """Pre/post batch-effect MMD over time in a single run.

    Runs combined cross-experiment MMD on the pre-LOT coordinates stored in
    ``corrected_paths[*].obsm["X_pre_lot"]`` and post-LOT coordinates in
    ``corrected_paths[*].X``. Both therefore use the same scaler/PCA space.
    ``input_paths`` identify the expected experiment-marker populations.

    Parameters
    ----------
    corrected_paths : list[str]
        Paths to the LOT-corrected per-experiment AnnData zarr stores. Should
        cover the same experiments as ``input_paths`` (matched by
        ``obs["experiment"]``, not list order).
    target_experiments : list[str] or None
        ``obs["experiment"]`` value(s) of the target/reference platform (v2).
        Used to tag each experiment pair as ``pair_kind="cross"`` (source↔target,
        the batch effect being corrected) vs ``"within"`` (source↔source, the
        within-platform baseline). When None, all pairs are ``"cross"``.
        Default: None.
    """

    corrected_paths: list[str]
    target_experiments: list[str] | None = None

    @model_validator(mode="after")
    def _validate_over_time(self) -> "MMDOverTimeConfig":
        if not self.corrected_paths:
            raise ValueError("corrected_paths must not be empty")
        return self


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
    condition_aliases: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "MMDPooledConfig":
        if not self.comparisons:
            raise ValueError("comparisons must not be empty")
        return self


class EmbeddingConsistencyConfig(_MMDBaseConfig):
    """Per-marker dataset-to-dataset embedding-consistency QC.

    Enumerates the input embedding zarrs for one model/run/checkpoint across
    datasets via :func:`dynaclr.evaluation.paths.iter_embeddings`, runs pairwise
    cross-dataset MMD on control cells only (``obs_filter``), and aggregates the
    long-form output into a symmetric per-marker dataset x dataset MMD matrix.
    A diagonal-dominant matrix (low off-diagonal MMD) means the embedding space
    is comparable across acquisitions; large off-diagonal MMD flags a batch
    effect that LOT correction must fix before downstream tasks trust the
    embeddings. This QC only *detects and reports* — it does not correct.

    Parameters
    ----------
    model_family : str
        Model-family identity to pool over (path component).
    run : str
        Training-run identity to pool over (path component).
    ckpt_name : str
        Checkpoint identity to pool over (path component).
    datasets_root : str or None
        Base under which datasets live. None uses the canonical
        :data:`dynaclr.evaluation.paths.DATASETS_ROOT`. Default: None.
    center_per_experiment : bool
        Subtract each dataset's own mean embedding before computing MMD, so the
        matrix reports *residual* batch effects independent of a global offset.
        Default: True.
    split_by : str or None
        Per-dataset obs column (constant within a dataset, e.g. ``"microscope"``)
        that partitions datasets into groups. When set, the QC emits, per group,
        a within-group matrix, plus one cross-group matrix per pair of groups
        (only the across-group dataset pairs). Blocks that are degenerate (a
        within-group block with <2 datasets, or a cross block with an empty
        side) are skipped with a log line. None (default) keeps the single
        pooled matrix over all datasets.

    Notes
    -----
    ``obs_filter`` (inherited) selects the control cells, e.g.
    ``{"perturbation": "uninfected"}`` — so perturbation biology cannot
    masquerade as a batch effect.
    """

    model_family: str
    run: str
    ckpt_name: str
    datasets_root: str | None = None
    center_per_experiment: bool = True
    split_by: str | None = None
    metrics: list[Literal["pearson", "mmd", "frechet"]] = ["pearson", "mmd", "frechet"]
    """Which per-marker matrices to compute/write. Default is all three; set to
    e.g. ``["pearson"]`` to start with the cheap mean-only matrix and skip the
    heavier MMD (permutation) and Fréchet (covariance) passes."""
    pearson_hpi_bin_hours: float | None = None
    """Time-pooling for the Pearson matrix. None (default): one grand mean over
    all control cells per dataset. When set, take the mean embedding per
    ``hours_post_perturbation`` bin of this width, then average the bin-means —
    so each HPI bin contributes equally and uneven time sampling (different
    intervals / frame counts across acquisitions) cannot masquerade as a batch
    effect. Bins are anchored at 0 h and shared across datasets, so acquisitions
    with different ``start_hpi`` still align on a common biological timeline."""
