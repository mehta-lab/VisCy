"""Stage A: turn MMD-witness + GMM into an annotation file.

A witness→GMM label *is an annotation*: its meaning is named by the modality it
is computed from (a witness over ``viral_sensor`` produces ``infection_state``;
over an organelle marker, ``organelle_remodeling_state``). This module scores
cells with the MMD witness, gates the perturbed cells per condition with a
two-component GMM, and writes a **named biological-state column** with the real
class vocabulary — a file indistinguishable from a hand annotation.

The downstream classifier is then trained by the unchanged annotation path
(``run-linear-classifiers`` with ``label_source="annotations"``), which joins the
file onto an embeddings zarr by cell key. Same-modality trains on the labeled
zarr; teacher/student points the training at a different modality's zarr.

Pipeline per marker:

1. Build a control reference (X) and a perturbed reference (Y) from
   per-experiment control/perturbed wells or filters.
2. Fit the empirical MMD witness on (X, Y) and score every cell.
3. Per perturbed condition, fit the configured 2-component GMM. The validated
   pooled contract fits one joint control+perturbed, tied-covariance GMM and
   applies symmetric posterior gates; ambiguous cells abstain. Near-noise
   markers (unimodal GMM) are skipped.
4. Map positive/negative to the config's class vocabulary and write the
   annotation file (CSV or parquet by extension).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import click
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from dynaclr.evaluation.linear_classifiers.witness_gmm_plots import (
    plot_mmd_null,
    plot_mmd_vs_hpi,
    plot_percentile_gate,
    plot_remodeling_vs_time,
    plot_witness_gmm,
)
from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.mmd import median_heuristic, mmd_permutation_test, subsample, witness_function
from viscy_utils.evaluation.witness_gmm import (
    PercentileLabelResult,
    fit_gmm_labels,
    fit_percentile_labels,
)

if TYPE_CHECKING:
    from dynaclr.evaluation.evaluate_config import WitnessGmmExperiment, WitnessGmmLabelsConfig

_logger = logging.getLogger(__name__)


def _extract_embedding_matrix(adata: ad.AnnData, embedding_key: str | None) -> np.ndarray:
    """Return the configured row-aligned representation as float32."""
    values = adata.X if embedding_key is None else adata.obsm[embedding_key]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _balanced_reference_indices(
    obs: pd.DataFrame,
    control_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    experiments: list[WitnessGmmExperiment],
    *,
    max_per_experiment_class: int | None,
    rng: np.random.Generator,
    condition: object | None = None,
    condition_column: str = "perturbation",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample equal control/perturbed references within every experiment.

    A shared feasible cap makes every experiment contribute equally. If the cap
    exceeds a sparse stratum, that experiment contributes its smaller available
    count; the saved configuration must therefore declare the intended cap.
    """
    experiment_values = obs["experiment"].astype(str).to_numpy()
    condition_values = obs[condition_column].to_numpy()
    control_parts: list[np.ndarray] = []
    perturbed_parts: list[np.ndarray] = []
    for source in experiments:
        exp_mask = experiment_values == str(source.experiment)
        control_rows = np.flatnonzero(control_mask & exp_mask)
        condition_mask = perturbed_mask & exp_mask
        if condition is not None:
            condition_mask &= condition_values == condition
        perturbed_rows = np.flatnonzero(condition_mask)
        n = min(len(control_rows), len(perturbed_rows))
        if max_per_experiment_class is not None:
            n = min(n, max_per_experiment_class)
        if n < 1:
            raise ValueError(
                f"{source.experiment}: balanced reference sampling needs both "
                f"control and perturbed rows (condition={condition!r})"
            )
        control_parts.append(rng.choice(control_rows, n, replace=False))
        perturbed_parts.append(rng.choice(perturbed_rows, n, replace=False))
    return np.concatenate(control_parts), np.concatenate(perturbed_parts)


def _well_prefix_mask(fov_name: pd.Series, wells: list[str]) -> np.ndarray:
    """Boolean mask of fov_name entries whose path prefix matches any well id.

    ``fov_name`` values look like ``"C/1/000000"``; a well id ``"C/1"`` matches
    any fov whose leading path components equal it. Matching on the ``/``-joined
    prefix (rather than ``startswith``) avoids ``"C/1"`` spuriously matching
    ``"C/10/..."``.

    Parameters
    ----------
    fov_name : pd.Series
        obs["fov_name"] values, e.g. ``"C/1/000000"``.
    wells : list[str]
        Well ids, e.g. ``["C/1", "C/2"]``.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (len(fov_name),).
    """
    stripped = fov_name.astype(object).str.strip("/")
    well_set = {w.strip("/") for w in wells}
    n_parts = {w.count("/") + 1 for w in well_set}

    def _matches(fov: str) -> bool:
        parts = fov.split("/")
        return any("/".join(parts[:k]) in well_set for k in n_parts)

    return stripped.map(_matches).to_numpy(dtype=bool)


_RANGE_OPS = {
    "lt": lambda s, v: s < v,
    "le": lambda s, v: s <= v,
    "gt": lambda s, v: s > v,
    "ge": lambda s, v: s >= v,
}


def obs_filter_mask(obs: pd.DataFrame, filter_dict: dict) -> np.ndarray:
    """Boolean mask of rows matching an obs filter (AND across keys).

    Each ``col -> spec`` entry contributes a condition; a row is kept only if it
    matches every entry. Spec forms:

    - scalar → equality (``obs[col] == spec``);
    - list/tuple → membership (``obs[col].isin(spec)``);
    - range dict → any of ``{lt, le, gt, ge}`` combined (one bound = half-line,
      two = window), e.g. ``{"ge": 24, "le": 36}`` → ``24 <= col <= 36``.

    A ``well`` or ``fov_name`` key routes to :func:`_well_prefix_mask` so wells
    are just another filterable column.

    Parameters
    ----------
    obs : pd.DataFrame
        The AnnData ``obs`` table.
    filter_dict : dict
        Mapping of obs column name to a scalar / list / range-dict spec.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (len(obs),).
    """
    mask = np.ones(len(obs), dtype=bool)
    for col, spec in filter_dict.items():
        if col in ("well", "fov_name"):
            wells = spec if isinstance(spec, (list, tuple)) else [spec]
            mask &= _well_prefix_mask(obs["fov_name"], list(wells))
            continue
        if col not in obs.columns:
            raise KeyError(f"obs_filter column '{col}' not found. Available: {list(obs.columns)}")
        series = obs[col]
        if isinstance(spec, dict):
            unknown = set(spec) - set(_RANGE_OPS)
            if unknown:
                raise ValueError(
                    f"range filter for '{col}' has unknown ops {sorted(unknown)}; use {sorted(_RANGE_OPS)}"
                )
            for op, val in spec.items():
                mask &= _RANGE_OPS[op](series, val).to_numpy(dtype=bool)
        elif isinstance(spec, (list, tuple)):
            mask &= series.isin(list(spec)).to_numpy(dtype=bool)
        else:
            mask &= (series == spec).to_numpy(dtype=bool)
    return mask


_KEY_PRIMARY = ["experiment", "fov_name", "id"]
_KEY_FALLBACK = ["experiment", "fov_name", "t", "track_id"]

# Tracking / spatial / biological metadata carried through to the annotation file
# in addition to the join key, so the label CSV is self-describing (an
# annotation-format file that keeps track lineage, position, marker, and
# perturbation rather than only exp/fov/id/t/state).
_METADATA_COLUMNS = [
    "t",
    "track_id",
    "parent_track_id",
    "parent_id",
    "y",
    "x",
    "marker",
    "perturbation",
    "hours_post_perturbation",
]


def _annotation_key_columns(obs: pd.DataFrame) -> list[str]:
    """Pick the annotation join key present in ``obs`` (``id`` primary, else t+track_id)."""
    if all(c in obs.columns for c in _KEY_PRIMARY):
        return _KEY_PRIMARY
    if all(c in obs.columns for c in _KEY_FALLBACK):
        return _KEY_FALLBACK
    raise KeyError(
        "embeddings obs lacks a usable annotation key: need (experiment, fov_name, id) "
        f"or (experiment, fov_name, t, track_id). Available: {list(obs.columns)}"
    )


@dataclass
class _MmdResult:
    """MMD permutation-test outcome for one condition (cached for diagnostics)."""

    mmd2: float
    p_value: float
    null: np.ndarray


@dataclass
class _MarkerScores:
    """Cached per-marker witness scores + reference masks for two-pass labeling."""

    obs: pd.DataFrame
    control_mask: np.ndarray
    perturbed_mask: np.ndarray
    scores: np.ndarray  # witness score per cell
    conditions: np.ndarray  # obs[condition_column] as array
    cond_mmd: dict  # condition -> _MmdResult (raw MMD² + p-value + null)
    # Optional MMD-vs-HPI kinetics: condition -> list of (hpi_bin_center, mmd2, p).
    # Key "__control_null__" holds the control-vs-control baseline (should be ~0).
    hpi_mmd: dict | None = None

    @property
    def cond_pvalues(self) -> dict:
        """condition -> raw MMD permutation p-value."""
        return {cond: res.p_value for cond, res in self.cond_mmd.items()}


@dataclass
class _MarkerLabels:
    """Result of GMM-labeling one marker: the annotation frame plus, for
    diagnostics, the per-condition GMM fits, the scores they were fit on, and the
    time vector over the *full* perturbed population (labeled + GMM-dropped) so
    the remodeling-vs-time plot has the whole-well denominator, not just the
    confident positives."""

    frame: pd.DataFrame
    cond_gmm: dict  # condition -> GmmLabelResult
    cond_scores: dict  # condition -> witness scores (all perturbed cells for the condition)
    cond_time: dict  # condition -> time value per perturbed cell (hpp if present, else t)
    control_time: np.ndarray | None  # time value per control-reference cell
    control_scores: np.ndarray  # witness scores of the control-reference cells (the clean negative reference)
    control_pos: np.ndarray  # per control cell: clears the GMM posterior threshold (empirical false positive)


def compute_marker_scores(
    adata: ad.AnnData,
    experiments: list[WitnessGmmExperiment],
    config: WitnessGmmLabelsConfig,
) -> _MarkerScores | None:
    """Pass 1: build references, witness-score every cell, and MMD-test each condition.

    Returns the cached scores/masks and the raw per-condition MMD permutation
    p-values. The caller pools these p-values across all markers and applies
    Benjamini-Yekutieli FDR control before deciding which conditions are
    significant (see :func:`generate_witness_gmm_annotation`). ``None`` when the
    marker has no control or perturbed reference cells.
    """
    obs = adata.obs
    rng = np.random.default_rng(config.random_seed)

    control_mask = np.zeros(len(obs), dtype=bool)
    perturbed_mask = np.zeros(len(obs), dtype=bool)
    for src in experiments:
        exp_mask = (obs["experiment"] == src.experiment).to_numpy(dtype=bool)
        if not exp_mask.any():
            continue
        if src.control_wells is not None:
            ctrl = _well_prefix_mask(obs["fov_name"], src.control_wells)
            pert = _well_prefix_mask(obs["fov_name"], src.perturbed_wells)
        else:
            ctrl = obs_filter_mask(obs, src.control_filter)
            pert = obs_filter_mask(obs, src.perturbed_filter)
        control_mask |= exp_mask & ctrl
        perturbed_mask |= exp_mask & pert

    X_all = _extract_embedding_matrix(adata, config.embedding_key)
    if control_mask.sum() == 0 or perturbed_mask.sum() == 0:
        _logger.warning("No control/perturbed reference cells found; skipping marker.")
        return None

    if config.reference_sampling == "balanced_by_experiment":
        control_ref_idx, perturbed_ref_idx = _balanced_reference_indices(
            obs,
            control_mask,
            perturbed_mask,
            experiments,
            max_per_experiment_class=config.max_reference_cells,
            rng=rng,
            condition_column=config.condition_column,
        )
        X_ref = X_all[control_ref_idx]
        Y_ref = X_all[perturbed_ref_idx]
    else:
        X_ref = subsample(X_all[control_mask], config.max_reference_cells, rng)
        Y_ref = subsample(X_all[perturbed_mask], config.max_reference_cells, rng)
    # Global bandwidth from the pooled reference — kept even in time-matched mode
    # so witness scores are on one comparable scale across timepoints.
    bandwidth = config.bandwidth if config.bandwidth is not None else median_heuristic(X_ref, Y_ref)

    if config.witness_time_bin_hours is None:
        scores = witness_function(X_all, X_ref, Y_ref, bandwidth=bandwidth)
    else:
        scores = _time_matched_witness_scores(
            X_all, obs, control_mask, perturbed_mask, X_ref, Y_ref, bandwidth, config, rng
        )

    # Per-condition MMD significance: is this condition's cloud distinct from the
    # control reference? Raw p-values here; FDR-corrected run-wide by the caller.
    conditions = obs[config.condition_column].to_numpy()
    cond_mmd: dict = {}
    for cond in pd.unique(conditions[perturbed_mask]):
        cond_mask = perturbed_mask & (conditions == cond)
        if cond_mask.sum() < 5:
            continue
        if config.reference_sampling == "balanced_by_experiment":
            control_test_idx, perturbed_test_idx = _balanced_reference_indices(
                obs,
                control_mask,
                perturbed_mask,
                experiments,
                max_per_experiment_class=config.max_reference_cells,
                rng=rng,
                condition=cond,
                condition_column=config.condition_column,
            )
            X_test = X_all[control_test_idx]
            Y_test = X_all[perturbed_test_idx]
        else:
            X_test = X_ref
            Y_test = subsample(X_all[cond_mask], config.max_reference_cells, rng)
        mmd2, p_value, null = mmd_permutation_test(
            X_test,
            Y_test,
            n_permutations=config.mmd_n_permutations,
            bandwidth=bandwidth,
            seed=config.random_seed,
        )
        cond_mmd[cond] = _MmdResult(mmd2=float(mmd2), p_value=float(p_value), null=np.asarray(null))

    hpi_mmd = None
    if config.mmd_hpi_bin_hours is not None:
        hpi_mmd = _compute_hpi_mmd(X_all, obs, control_mask, perturbed_mask, conditions, bandwidth, config, rng)

    return _MarkerScores(obs, control_mask, perturbed_mask, scores, conditions, cond_mmd, hpi_mmd)


def _time_matched_witness_scores(
    X_all: np.ndarray,
    obs: pd.DataFrame,
    control_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    X_ref_pooled: np.ndarray,
    Y_ref_pooled: np.ndarray,
    bandwidth: float,
    config: WitnessGmmLabelsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Witness score per cell using per-HPI-bin (time-matched) references.

    For each ``witness_time_bin_hours``-wide window, cells in that window are
    scored against control/perturbed reference cells **from the same window**, so
    the witness axis reflects perturbation rather than culture-time (the embedding
    has a strong time axis; see :func:`_compute_hpi_mmd`). Bins lacking enough
    reference cells on either side fall back to the pooled reference, so every
    cell is scored. The bandwidth is the shared global value for a comparable
    scale across bins.
    """
    if "hours_post_perturbation" not in obs.columns:
        _logger.warning("witness_time_bin_hours set but no hours_post_perturbation column; using pooled reference.")
        return witness_function(X_all, X_ref_pooled, Y_ref_pooled, bandwidth=bandwidth)

    hpi = obs["hours_post_perturbation"].to_numpy(dtype=float)
    width = config.witness_time_bin_hours
    scores = np.full(len(obs), np.nan, dtype=np.float64)
    finite = np.isfinite(hpi)
    lo0 = np.floor(hpi[finite].min() / width) * width if finite.any() else 0.0
    edges = np.arange(lo0, (hpi[finite].max() if finite.any() else 0.0) + width, width)
    min_ref = 5
    n_fallback = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = finite & (hpi >= lo) & (hpi < hi)
        if not in_bin.any():
            continue
        ctrl_bin = X_all[control_mask & in_bin]
        pert_bin = X_all[perturbed_mask & in_bin]
        if len(ctrl_bin) >= min_ref and len(pert_bin) >= min_ref:
            xr = subsample(ctrl_bin, config.max_reference_cells, rng)
            yr = subsample(pert_bin, config.max_reference_cells, rng)
        else:
            xr, yr = X_ref_pooled, Y_ref_pooled  # sparse bin → pooled fallback
            n_fallback += int(in_bin.sum())
        scores[in_bin] = witness_function(X_all[in_bin], xr, yr, bandwidth=bandwidth)
    # Cells with non-finite HPI (never assigned) get the pooled score.
    missing = np.isnan(scores)
    if missing.any():
        scores[missing] = witness_function(X_all[missing], X_ref_pooled, Y_ref_pooled, bandwidth=bandwidth)
    if n_fallback:
        _logger.info("Time-matched witness: %d cells in sparse bins used the pooled reference.", n_fallback)
    return scores


def _compute_hpi_mmd(
    X_all: np.ndarray,
    obs: pd.DataFrame,
    control_mask: np.ndarray,
    perturbed_mask: np.ndarray,
    conditions: np.ndarray,
    bandwidth: float,
    config: WitnessGmmLabelsConfig,
    rng: np.random.Generator,
) -> dict:
    """Time-matched MMD²(control, condition) per HPI bin — infection kinetics.

    For each ``mmd_hpi_bin_hours``-wide window of ``hours_post_perturbation``,
    tests each condition's cells against the **control cells in the SAME window**
    (not the pooled all-timepoint control reference). This time-matching is
    essential: the DynaCLR embedding carries a strong time/culture axis (uninfected
    cells drift substantially over a 36 h timelapse), so comparing a bin's
    perturbed cells to a time-pooled control conflates infection with that time
    axis. Matching control and perturbed within the bin cancels the shared time
    component and isolates the infection difference. Also emits a
    control-vs-control null (random half-split of the bin's control cells) under
    ``"__control_null__"`` as the no-difference floor. Bins with too few cells on
    either side are skipped. Returns ``{key: [(hpi_center, mmd2, p), ...]}``.
    """
    if "hours_post_perturbation" not in obs.columns:
        _logger.warning("mmd_hpi_bin_hours set but no hours_post_perturbation column; skipping HPI-MMD.")
        return {}
    hpi = obs["hours_post_perturbation"].to_numpy(dtype=float)
    width = config.mmd_hpi_bin_hours
    finite = hpi[np.isfinite(hpi)]
    if finite.size == 0:
        return {}
    edges = np.arange(np.floor(finite.min() / width) * width, finite.max() + width, width)

    def _mmd(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        mmd2, p, _null = mmd_permutation_test(
            subsample(a, config.max_reference_cells, rng),
            subsample(b, config.max_reference_cells, rng),
            n_permutations=config.mmd_n_permutations,
            bandwidth=bandwidth,
            seed=config.random_seed,
        )
        return float(mmd2), float(p)

    out: dict = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (hpi >= lo) & (hpi < hi)
        center = float(lo + width / 2)
        ctrl_bin = X_all[control_mask & in_bin]
        if len(ctrl_bin) < 5:
            continue  # need a time-matched control reference for this bin
        for cond in pd.unique(conditions[perturbed_mask]):
            cells = X_all[perturbed_mask & (conditions == cond) & in_bin]
            if len(cells) < 5:
                continue
            mmd2, p = _mmd(ctrl_bin, cells)
            out.setdefault(str(cond), []).append((center, mmd2, p))
        # Control-vs-control null: split this bin's control cells in half.
        if len(ctrl_bin) >= 10:
            perm = rng.permutation(len(ctrl_bin))
            half = len(ctrl_bin) // 2
            mmd2, p = _mmd(ctrl_bin[perm[:half]], ctrl_bin[perm[half:]])
            out.setdefault("__control_null__", []).append((center, mmd2, p))
    return out


def _balanced_control_perturbed_fit_scores(
    control_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    *,
    max_per_population: int | None,
    random_seed: int,
) -> np.ndarray:
    """Return an equal-size, deterministic control + perturbed GMM fit pool."""
    n = min(len(control_scores), len(perturbed_scores))
    if max_per_population is not None:
        n = min(n, max_per_population)
    if n < 1:
        raise ValueError("control and perturbed GMM fit populations must be non-empty")
    rng = np.random.default_rng(random_seed)
    control_idx = rng.choice(len(control_scores), size=n, replace=False)
    perturbed_idx = rng.choice(len(perturbed_scores), size=n, replace=False)
    return np.concatenate([control_scores[control_idx], perturbed_scores[perturbed_idx]])


def label_marker(
    marker_scores: _MarkerScores,
    significant_conditions: set,
    config: WitnessGmmLabelsConfig,
) -> _MarkerLabels | None:
    """Pass 2: GMM-gate each significant condition and assemble the annotation frame.

    A condition is labeled only if it is in ``significant_conditions`` (cleared
    the FDR-controlled MMD gate, decided run-wide) AND its GMM is bimodal.

    In legacy perturbed-only mode, negatives include every control-well cell by
    well identity plus perturbed-well cells whose posterior is confidently
    negative. In control-inclusive GMM modes, the same symmetric posterior gates
    are applied to every control and perturbed cell; ambiguous cells from either
    population stay unlabeled. With multiple perturbed conditions, a control is
    positive if any separated condition model is confidently positive and is
    negative only if every separated model is confidently negative.

    ``None`` when no condition survives both gates. The returned
    :class:`_MarkerLabels` also carries the per-condition GMM fits and the scores
    they were fit on, for the Stage-A diagnostic plots.
    """
    obs = marker_scores.obs
    key_cols = _annotation_key_columns(obs)
    scores = marker_scores.scores
    control_mask = marker_scores.control_mask
    perturbed_mask = marker_scores.perturbed_mask
    conditions = marker_scores.conditions
    pos_label = config.class_map["positive"]
    neg_label = config.class_map["negative"]

    labels = np.full(len(obs), None, dtype=object)
    control_inclusive_gmm = config.gate == "gmm" and config.gmm_fit_population != "perturbed"
    if not control_inclusive_gmm:
        labels[control_mask] = neg_label

    # Per-cell provenance carried into the annotation file. Legacy modes retain
    # forced control negatives with confidence 1.0. Control-inclusive GMM modes
    # fill these values from the same posterior gates used for perturbed cells.
    gmm_posterior = np.full(len(obs), np.nan, dtype=float)
    if not control_inclusive_gmm:
        gmm_posterior[control_mask] = 1.0

    time_col = "hours_post_perturbation" if "hours_post_perturbation" in obs.columns else "t"
    time_values = obs[time_col].to_numpy() if time_col in obs.columns else None

    control_scores_all = scores[control_mask]
    cond_gmm: dict = {}
    cond_scores: dict = {}
    cond_time: dict = {}
    control_gmm_posteriors: list[np.ndarray] = []
    any_separated = False
    for cond in pd.unique(conditions[perturbed_mask]):
        cond_mask = perturbed_mask & (conditions == cond)
        if cond_mask.sum() < 5:
            continue
        if cond not in significant_conditions:
            _logger.warning("MMD not significant (FDR) for condition %r; no positives labeled.", cond)
            continue
        cond_scores[cond] = scores[cond_mask]
        if time_values is not None:
            cond_time[cond] = time_values[cond_mask]
        cond_idx = np.flatnonzero(cond_mask)
        if config.gate == "percentile":
            # Symmetric quantiles of the CONTROL scores: positive below the
            # control_fp_target quantile, negative above the (1 - target) one,
            # ambiguous between. Fits nothing, so it neither requires the perturbed
            # pool to be bimodal nor invents a second mode when the perturbation
            # merely SHIFTS a unimodal population. Never abstains.
            res = fit_percentile_labels(
                scores[cond_mask],
                control_scores_all,
                control_fp_target=config.control_fp_target,
                negative_quantile=config.negative_quantile,
            )
            cond_gmm[cond] = res
            any_separated = True
        else:
            fit_scores = None
            if config.gmm_fit_population == "joint_control_perturbed":
                fit_scores = np.concatenate([control_scores_all, scores[cond_mask]])
            elif config.gmm_fit_population == "balanced_control_perturbed":
                fit_scores = _balanced_control_perturbed_fit_scores(
                    control_scores_all,
                    scores[cond_mask],
                    max_per_population=config.max_reference_cells,
                    random_seed=config.random_seed,
                )
            res = fit_gmm_labels(
                scores[cond_mask],
                pos_threshold=config.gmm_pos_threshold,
                random_state=config.random_seed,
                fit_scores_1d=fit_scores,
                covariance_type=config.gmm_covariance_type,
            )
            cond_gmm[cond] = res
            if not res.separated:
                _logger.warning("GMM unimodal for condition %r; no positives labeled.", cond)
                continue
            any_separated = True
            if control_inclusive_gmm:
                control_gmm_posteriors.append(
                    res.gmm.predict_proba(control_scores_all.reshape(-1, 1))[:, res.remod_component]
                )
        idx = cond_idx[res.hard_label == 1]
        labels[idx] = pos_label
        gmm_posterior[idx] = res.posterior[res.hard_label == 1]

        # Cells the gate is equally confident are NOT positive are real negatives:
        # pre-onset cells (in a timelapse, a cell in a perturbed well genuinely is
        # unperturbed before onset) and bystanders (infection is never 100%).
        # Dropping them — the old behavior — meant a perturbed FOV contributed
        # ONLY positives, so every negative came from the control well and the
        # classifier never saw a batch-matched negative (same well, plate,
        # illumination, segmentation as the positives).
        #
        # Both gates hold negatives to the same bar as positives and leave a
        # middle band unlabeled, but they express it differently: the GMM has a
        # posterior to threshold symmetrically, while the percentile gate marks
        # negatives directly (hard_label == 0) from its own control quantile,
        # since its posterior is 0 for every cell above the positive cut.
        if isinstance(res, PercentileLabelResult):
            neg_sel = res.hard_label == 0
        else:
            neg_sel = res.posterior <= 1.0 - config.gmm_pos_threshold
        idx_neg = cond_idx[neg_sel]
        labels[idx_neg] = neg_label
        # Same convention as control-well negatives: negatives are not
        # distinguished by provenance.
        gmm_posterior[idx_neg] = 1.0

    if not any_separated:
        return None

    if control_inclusive_gmm:
        if not control_gmm_posteriors:
            return None
        max_control_posterior = np.max(np.vstack(control_gmm_posteriors), axis=0)
        control_idx = np.flatnonzero(control_mask)
        control_positive = max_control_posterior >= config.gmm_pos_threshold
        control_negative = max_control_posterior <= 1.0 - config.gmm_pos_threshold
        labels[control_idx[control_positive]] = pos_label
        labels[control_idx[control_negative]] = neg_label
        gmm_posterior[control_idx[control_positive]] = max_control_posterior[control_positive]
        gmm_posterior[control_idx[control_negative]] = 1.0 - max_control_posterior[control_negative]

    keep = labels != None  # noqa: E711 — object-array null test
    carry = key_cols + [c for c in _METADATA_COLUMNS if c in obs.columns and c not in key_cols]
    frame = obs.loc[keep, carry].copy()
    frame[config.label_column] = labels[keep]
    frame["witness_score"] = scores[keep]
    frame["gmm_posterior"] = gmm_posterior[keep]
    # Normalize fov_name to match the annotation-loader convention.
    frame["fov_name"] = frame["fov_name"].astype(object).str.strip("/")
    control_time = time_values[control_mask] if time_values is not None else None
    control_scores = scores[control_mask]

    # Score the CONTROL cells against the fitted GMM(s) — a control cell is a
    # false positive if its remodel-mode posterior clears the threshold under any
    # separated condition's GMM (the same rule perturbed cells are gated by). This
    # drives the remodeling-vs-time control line as an empirical FALSE-POSITIVE
    # rate, not a hardcoded zero. NaN posterior for controls when no GMM separated.
    control_pos = np.zeros(control_mask.sum(), dtype=bool)
    ctrl_s = scores[control_mask]
    if control_inclusive_gmm:
        control_pos = np.max(np.vstack(control_gmm_posteriors), axis=0) >= config.gmm_pos_threshold
    for res in cond_gmm.values():
        if isinstance(res, PercentileLabelResult):
            # Percentile: the gate is a raw-score cut, so controls are scored by
            # the identical rule. This FP is ≈ control_fp_target by construction
            # (the gate is a quantile of these very scores) — it is a consistency
            # check on the plot, not a discovered quantity.
            control_pos |= ctrl_s < res.gate
            continue
        if not res.separated or control_inclusive_gmm:
            continue
        post = res.gmm.predict_proba(ctrl_s.reshape(-1, 1))[:, res.remod_component]
        control_pos |= post >= config.gmm_pos_threshold
    return _MarkerLabels(
        frame.reset_index(drop=True),
        cond_gmm,
        cond_scores,
        cond_time,
        control_time,
        control_scores,
        control_pos,
    )


@dataclass(frozen=True)
class PooledWitnessGmmFit:
    """In-memory result of pooled witness scoring, MMD/FDR, and GMM gates."""

    annotations: pd.DataFrame
    marker_scores: dict[str, _MarkerScores]
    marker_labels: dict[str, _MarkerLabels | None]
    adjusted_pvalues: dict[tuple[str, object], float]
    significant_conditions: dict[str, set]


def fit_pooled_witness_gmm(
    adata: ad.AnnData,
    config: WitnessGmmLabelsConfig,
) -> PooledWitnessGmmFit:
    """Fit the pooled witness and joint/tied GMM stack without file I/O.

    This is the canonical script-level API. It scores the selected representation,
    applies run-wide Benjamini-Yekutieli correction to marker/condition MMD tests,
    then gates each significant condition with the configured GMM. The YAML/CLI
    entrypoint loads AnnData and writes diagnostics around this function.
    """
    if "marker" not in adata.obs:
        raise KeyError("pooled witness input needs obs['marker']")
    markers = config.marker_filters or list(pd.unique(adata.obs["marker"]))

    marker_scores: dict[str, _MarkerScores] = {}
    pval_keys: list[tuple[str, object]] = []
    pvalues: list[float] = []
    for marker in markers:
        sub = adata[adata.obs["marker"] == marker]
        if sub.n_obs == 0:
            _logger.warning("No cells for marker %r; skipping.", marker)
            continue
        scores = compute_marker_scores(
            sub.copy(),
            config.experiments,
            config,
        )
        if scores is None:
            continue
        marker_scores[str(marker)] = scores
        for condition, pvalue in scores.cond_pvalues.items():
            pval_keys.append((str(marker), condition))
            pvalues.append(pvalue)

    if not pvalues:
        raise RuntimeError("No testable conditions; check references and condition_column.")

    adjusted = false_discovery_control(
        np.asarray(pvalues),
        method="by",
    )
    adjusted_pvalues: dict[tuple[str, object], float] = {}
    significant: dict[str, set] = {}
    for (marker, condition), adjusted_pvalue in zip(pval_keys, adjusted, strict=True):
        adjusted_pvalues[(marker, condition)] = float(adjusted_pvalue)
        if adjusted_pvalue <= config.mmd_pvalue_threshold:
            significant.setdefault(marker, set()).add(condition)
        else:
            _logger.info(
                "Condition %r (marker %r): BY-adjusted p=%.3g > %.3g; skipped.",
                condition,
                marker,
                adjusted_pvalue,
                config.mmd_pvalue_threshold,
            )

    marker_labels = {
        marker: label_marker(
            scores,
            significant.get(marker, set()),
            config,
        )
        for marker, scores in marker_scores.items()
    }
    frames = [result.frame for result in marker_labels.values() if result is not None]
    annotations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return PooledWitnessGmmFit(
        annotations=annotations,
        marker_scores=marker_scores,
        marker_labels=marker_labels,
        adjusted_pvalues=adjusted_pvalues,
        significant_conditions=significant,
    )


def generate_witness_gmm_annotation(config: WitnessGmmLabelsConfig) -> Path | None:
    """Run Stage A end to end and write the annotation file plus diagnostics.

    Loads each experiment's embeddings zarr, pools per marker, labels via
    :func:`label_marker`, concatenates, and writes the annotation to
    ``<output_dir>/labels/<label_column>.<annotation_format>``. Diagnostic plots
    (the label-decision evidence) go to ``<output_dir>/labels/plots/``:

    - ``mmd_null_<marker>_<condition>.png`` — MMD permutation-null + observed +
      p-values (the significance gate), for every tested condition.
    - ``witness_gmm_<marker>_<condition>.png`` — witness-score histogram + fitted
      GMM (the bimodality gate), for every FDR-significant condition.
    - ``remodeling_vs_time_<marker>.png`` — fraction of cells in the positive
      class vs time per condition, for every labeled marker.

    Parameters
    ----------
    config : WitnessGmmLabelsConfig
        Stage-A configuration.

    Returns
    -------
    Path
        The written annotation-file path.
    """
    parts: list[ad.AnnData] = []
    sources_by_path: dict[str, list[WitnessGmmExperiment]] = {}
    for source in config.experiments:
        sources_by_path.setdefault(source.embeddings_zarr, []).append(source)
    for embeddings_zarr, sources in sources_by_path.items():
        _logger.info(
            "Loading embeddings for %s: %s",
            ", ".join(source.experiment for source in sources),
            embeddings_zarr,
        )
        a = ad.read_zarr(embeddings_zarr)
        a.obs_names_make_unique()
        if "experiment" not in a.obs.columns:
            if len(sources) != 1:
                raise ValueError(
                    f"{embeddings_zarr}: pooled store needs obs['experiment'] "
                    "when it is referenced by multiple experiments"
                )
            a.obs["experiment"] = sources[0].experiment
        parts.append(a)
    adata = ad.concat(parts, join="outer") if len(parts) > 1 else parts[0]
    adata.obs_names_make_unique()

    fit = fit_pooled_witness_gmm(adata, config)
    marker_scores = fit.marker_scores
    p_adj_by_key = fit.adjusted_pvalues
    significant = fit.significant_conditions

    labels_dir = Path(config.output_dir) / "labels"
    plots_dir = labels_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # MMD-null diagnostic for every tested (marker, condition) — the significance gate.
    for marker, ms in marker_scores.items():
        for cond, mmd in ms.cond_mmd.items():
            p_adj = p_adj_by_key[(marker, cond)]
            plot_mmd_null(
                mmd.mmd2,
                mmd.null,
                mmd.p_value,
                p_adj,
                significant=cond in significant.get(marker, set()),
                marker=str(marker),
                condition=str(cond),
                output_path=plots_dir / f"mmd_null_{marker}_{cond}.png",
            )
        # MMD-vs-HPI kinetics (population divergence over time) when enabled.
        if ms.hpi_mmd:
            plot_mmd_vs_hpi(
                ms.hpi_mmd,
                config.mmd_pvalue_threshold,
                marker=str(marker),
                output_path=plots_dir / f"mmd_vs_hpi_{marker}.png",
            )

    # Pass 2: GMM-label each marker using the FDR-significant conditions.
    frames: list[pd.DataFrame] = []
    mmd_rows: list[dict] = []  # per (marker, condition) provenance sidecar
    for marker, ms in marker_scores.items():
        result = fit.marker_labels.get(marker)
        cond_gmm = result.cond_gmm if result else {}
        # Provenance row per tested (marker, condition): MMD gate + GMM summary.
        for cond, mmd in ms.cond_mmd.items():
            res = cond_gmm.get(cond)
            row = {
                "marker": str(marker),
                "condition": str(cond),
                "n_perturbed": int((ms.perturbed_mask & (ms.conditions == cond)).sum()),
                "mmd2": mmd.mmd2,
                "p_raw": mmd.p_value,
                "p_adjusted": p_adj_by_key[(marker, cond)],
                "mmd_significant": cond in significant.get(marker, set()),
                "gate": config.gate,
                "embedding_key": config.embedding_key or "X",
                "reference_sampling": config.reference_sampling,
                "max_reference_cells": config.max_reference_cells,
                "gmm_fit_population": config.gmm_fit_population,
                "gmm_covariance_type": config.gmm_covariance_type,
                "n_confident_positive": int((res.hard_label == 1).sum()) if res is not None else 0,
            }
            if isinstance(res, PercentileLabelResult):
                row.update(
                    {
                        "positive_fraction": float((res.hard_label == 1).mean()),
                        "negative_fraction": float((res.hard_label == 0).mean()),
                        "ambiguous_fraction": float((res.hard_label == -1).mean()),
                        "control_fp": res.control_fp,
                        "score_gate": res.gate,
                        "score_neg_gate": res.neg_gate,
                        "negative_quantile": config.negative_quantile,
                    }
                )
            elif res is not None:
                row.update(
                    {
                        "gmm_separated": bool(res.separated),
                        "gmm_fit_population": config.gmm_fit_population,
                        "gmm_covariance_type": config.gmm_covariance_type,
                        "n_gmm_fit": res.n_fit_samples,
                        "gmm_remodel_weight": float(res.gmm.weights_[res.remod_component]),
                        "gmm_bic": res.bic,
                        "gmm_aic": res.aic,
                        "gmm_delta_bic": res.bic_1comp - res.bic,
                    }
                )
            mmd_rows.append(row)
        # Gate diagnostic per condition: percentile gets its own plot (both score
        # distributions + the control-calibrated cut); GMM gets the two-mode plot.
        for cond, res in cond_gmm.items():
            if isinstance(res, PercentileLabelResult):
                plot_percentile_gate(
                    result.cond_scores[cond],
                    result.control_scores,
                    res,
                    marker=str(marker),
                    condition=str(cond),
                    pos_label=config.class_map["positive"],
                    neg_label=config.class_map["negative"],
                    output_path=plots_dir / f"percentile_gate_{marker}_{cond}.png",
                )
                continue
            plot_witness_gmm(
                result.cond_scores[cond],
                result.control_scores,
                res,
                config.gmm_pos_threshold,
                marker=str(marker),
                condition=str(cond),
                pos_label=config.class_map["positive"],
                neg_label=config.class_map["negative"],
                fit_population=(f"{config.gmm_fit_population}; covariance={config.gmm_covariance_type}"),
                output_path=plots_dir / f"witness_gmm_{marker}_{cond}.png",
            )
        if result is None:
            _logger.warning("Marker %r produced no labels (no significant + bimodal condition); skipping.", marker)
            continue
        frame = result.frame
        counts = frame[config.label_column].value_counts().to_dict()
        _logger.info("Marker %r: %d labeled cells %s", marker, len(frame), counts)
        plot_remodeling_vs_time(
            result.cond_time,
            result.cond_gmm,
            result.control_time,
            result.control_pos,
            config.class_map["positive"],
            marker=str(marker),
            output_path=plots_dir / f"remodeling_vs_time_{marker}.png",
            time_is_hpp="hours_post_perturbation" in ms.obs.columns,
        )
        frames.append(frame)

    # Disambiguate by marker: sibling single-marker configs often share a
    # label_column (e.g. three organelle markers → organelle_remodeling_state),
    # which would clobber a bare <label_column>.<fmt>. Prefix with the filtered
    # marker(s) so each config writes its own file.
    stem = f"{'_'.join(config.marker_filters)}_{config.label_column}" if config.marker_filters else config.label_column
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Population-level provenance sidecar always written — even when a marker
    # abstains (no bimodal + significant condition), so the MMD/GMM evidence for
    # the abstain decision is auditable.
    mmd_path = labels_dir / f"{stem}_mmd.csv"
    pd.DataFrame(mmd_rows).to_csv(mmd_path, index=False)
    _logger.info("Wrote MMD/GMM provenance to %s", mmd_path)

    if not frames:
        # Legitimate abstain (e.g. time-matched witness leaves a weak channel
        # unimodal): write no annotation, but do not crash — the diagnostics +
        # sidecar above record why.
        _logger.warning(
            "No markers produced confident labels (no significant + bimodal condition). "
            "Wrote diagnostics + %s but no annotation file.",
            mmd_path.name,
        )
        return None

    out = pd.concat(frames, ignore_index=True)
    output_path = labels_dir / f"{stem}.{config.annotation_format}"
    if output_path.suffix == ".parquet":
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)
    _logger.info("Wrote %d annotations (%s) to %s", len(out), config.label_column, output_path)
    return output_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the witness-GMM labels YAML config (top-level 'witness_gmm_labels' key).",
)
def main(config_path: Path) -> None:
    """Generate an annotation file from the MMD witness + per-condition GMM (Stage A)."""
    from dynaclr.evaluation.evaluate_config import WitnessGmmLabelsConfig

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = load_config(config_path)
    config = WitnessGmmLabelsConfig(**raw["witness_gmm_labels"])
    out = generate_witness_gmm_annotation(config)
    if out is None:
        click.echo("No confident labels produced (marker abstained); diagnostics written, no annotation file.")
    else:
        click.echo(f"Wrote witness-GMM annotation to {out}")


if __name__ == "__main__":
    main()
