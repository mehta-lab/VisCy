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
3. Per perturbed condition, fit a 2-component GMM on the scores; cells above the
   posterior threshold are confident positives. Negatives = all control-well
   cells (well identity). Near-noise markers (unimodal GMM) are skipped.
4. Map positive/negative to the config's class vocabulary and write the
   annotation file (CSV or parquet by extension).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import click
import numpy as np
import pandas as pd

from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.mmd import median_heuristic, mmd_permutation_test, subsample, witness_function
from viscy_utils.evaluation.witness_gmm import fit_gmm_labels

if TYPE_CHECKING:
    from dynaclr.evaluation.evaluate_config import WitnessGmmExperiment, WitnessGmmLabelsConfig

_logger = logging.getLogger(__name__)


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


def build_marker_annotation(
    adata: ad.AnnData,
    experiments: list[WitnessGmmExperiment],
    config: WitnessGmmLabelsConfig,
) -> pd.DataFrame | None:
    """Label one marker's cells via witness → per-condition GMM.

    Builds control/perturbed references from each experiment's wells/filters,
    scores every cell with the MMD witness, fits a 2-component GMM on each
    perturbed condition's scores, and assembles a per-cell annotation frame with
    the config's ``label_column`` set to the mapped class vocabulary. Negatives
    are all control-well cells (well identity); positives are perturbed cells
    clearing ``gmm_pos_threshold``; ambiguous perturbed cells are dropped.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings for a single marker, pooled across experiments. ``obs`` must
        carry ``experiment``, ``fov_name``, the annotation key, and
        ``config.condition_column``.
    experiments : list[WitnessGmmExperiment]
        Per-experiment reference specs (control/perturbed wells or filters).
    config : WitnessGmmLabelsConfig
        Labeling settings (threshold, bandwidth, class map, condition column).

    Returns
    -------
    pd.DataFrame or None
        Annotation frame with the join-key columns plus ``t`` and the
        ``label_column``. ``None`` when references are missing or the GMM is
        unimodal for every condition (near-noise marker → skipped).
    """
    obs = adata.obs
    rng = np.random.default_rng(config.random_seed)
    key_cols = _annotation_key_columns(obs)

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

    X_all = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X_ctrl = X_all[control_mask]
    Y_pert = X_all[perturbed_mask]
    if len(X_ctrl) == 0 or len(Y_pert) == 0:
        _logger.warning("No control/perturbed reference cells found; skipping marker.")
        return None

    X_ref = subsample(X_ctrl, config.max_reference_cells, rng)
    Y_ref = subsample(Y_pert, config.max_reference_cells, rng)
    bandwidth = config.bandwidth if config.bandwidth is not None else median_heuristic(X_ref, Y_ref)
    scores = witness_function(X_all, X_ref, Y_ref, bandwidth=bandwidth)

    pos_label = config.class_map["positive"]
    neg_label = config.class_map["negative"]

    # Negatives: all control-well cells (well identity, no gate).
    labels = np.full(len(obs), None, dtype=object)
    labels[control_mask] = neg_label

    # Positives: per perturbed condition, GMM-gate the scores.
    conditions = obs[config.condition_column].to_numpy()
    any_separated = False
    for cond in pd.unique(conditions[perturbed_mask]):
        cond_mask = perturbed_mask & (conditions == cond)
        if cond_mask.sum() < 5:
            continue
        # Significance gate: is this condition's cloud actually distinct from the
        # control reference? A non-significant MMD means the perturbation left no
        # detectable signature — skip rather than manufacture labels from noise.
        _mmd2, p_value, _null = mmd_permutation_test(
            X_ref,
            subsample(X_all[cond_mask], config.max_reference_cells, rng),
            n_permutations=config.mmd_n_permutations,
            bandwidth=bandwidth,
            seed=config.random_seed,
        )
        if p_value > config.mmd_pvalue_threshold:
            _logger.warning(
                "MMD not significant for condition %r (p=%.3g > %.3g); no positives labeled.",
                cond,
                p_value,
                config.mmd_pvalue_threshold,
            )
            continue
        res = fit_gmm_labels(scores[cond_mask], pos_threshold=config.gmm_pos_threshold, random_state=config.random_seed)
        if not res.separated:
            _logger.warning("GMM unimodal for condition %r (p=%.3g); no positives labeled.", cond, p_value)
            continue
        any_separated = True
        idx = np.flatnonzero(cond_mask)[res.hard_label == 1]
        labels[idx] = pos_label

    if not any_separated:
        return None

    keep = labels != None  # noqa: E711 — object-array null test
    frame = obs.loc[keep, key_cols].copy()
    if "t" in obs.columns and "t" not in frame.columns:
        frame["t"] = obs.loc[keep, "t"].to_numpy()
    frame[config.label_column] = labels[keep]
    # Normalize fov_name to match the annotation-loader convention.
    frame["fov_name"] = frame["fov_name"].astype(object).str.strip("/")
    return frame.reset_index(drop=True)


def generate_witness_gmm_annotation(config: WitnessGmmLabelsConfig) -> Path:
    """Run Stage A end to end and write the annotation file.

    Loads each experiment's embeddings zarr, pools per marker, labels via
    :func:`build_marker_annotation`, concatenates, and writes to
    ``config.output_path`` (CSV or parquet by extension).

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
    for exp in config.experiments:
        _logger.info("Loading embeddings for %s: %s", exp.experiment, exp.embeddings_zarr)
        a = ad.read_zarr(exp.embeddings_zarr)
        a.obs_names_make_unique()
        if "experiment" not in a.obs.columns:
            a.obs["experiment"] = exp.experiment
        parts.append(a)
    adata = ad.concat(parts, join="outer") if len(parts) > 1 else parts[0]
    adata.obs_names_make_unique()

    markers = config.marker_filters or list(pd.unique(adata.obs["marker"]))
    frames: list[pd.DataFrame] = []
    for marker in markers:
        sub = adata[adata.obs["marker"] == marker]
        if sub.n_obs == 0:
            _logger.warning("No cells for marker %r; skipping.", marker)
            continue
        frame = build_marker_annotation(sub.copy(), config.experiments, config)
        if frame is None:
            _logger.warning("Marker %r produced no labels (missing refs or unimodal GMM); skipping.", marker)
            continue
        counts = frame[config.label_column].value_counts().to_dict()
        _logger.info("Marker %r: %d labeled cells %s", marker, len(frame), counts)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No markers produced labels — check references, threshold, and condition_column.")

    out = pd.concat(frames, ignore_index=True)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    click.echo(f"Wrote witness-GMM annotation to {out}")


if __name__ == "__main__":
    main()
