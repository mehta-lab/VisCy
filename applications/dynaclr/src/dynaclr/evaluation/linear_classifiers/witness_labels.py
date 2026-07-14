"""MMD-witness weak labeling for linear classifiers.

Derives discrete per-cell pseudo-labels from the MMD witness score instead of
annotation CSVs. For a marker-filtered, cross-experiment pool of embeddings:

1. Build a control reference (X) and a perturbed reference (Y) from
   per-experiment control/perturbed wells.
2. Fit the empirical MMD witness on (X, Y) and score every cell — a signed
   scalar measuring how much the cell looks like control (positive) vs
   perturbed (negative).
3. Gate the scores into ``control`` / ``perturbed`` labels, dropping an
   ambiguous middle band as unlabeled (the analog of the annotation path's
   ``!= "unknown"`` filter).

The output is an AnnData whose ``obs[label_column]`` holds the pseudo-labels,
consumed by the same ``train_linear_classifier`` path as annotation labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from viscy_utils.evaluation.mmd import median_heuristic, witness_function

if TYPE_CHECKING:
    import anndata as ad

    from dynaclr.evaluation.evaluate_config import WitnessLabelSource, WitnessSettings


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


def build_witness_labels(
    adata: ad.AnnData,
    witness_labels: list[WitnessLabelSource],
    settings: WitnessSettings,
    random_seed: int = 42,
) -> ad.AnnData:
    """Weak-label a marker-filtered embedding pool via the MMD witness score.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings already filtered to a single marker. ``obs`` must carry
        ``experiment`` and ``fov_name``.
    witness_labels : list[WitnessLabelSource]
        Per-experiment control/perturbed well specs.
    settings : WitnessSettings
        Kernel + gating settings.
    random_seed : int
        Seed for reference subsampling. Default: 42.

    Returns
    -------
    ad.AnnData
        Subset of ``adata`` containing only the labeled (non-dead-zone) cells,
        with the gated pseudo-label written to ``obs[settings.label_column]``.
        Empty AnnData if no reference cells were found in either group.
    """
    obs = adata.obs
    rng = np.random.default_rng(random_seed)

    control_mask = np.zeros(len(obs), dtype=bool)
    perturbed_mask = np.zeros(len(obs), dtype=bool)
    for src in witness_labels:
        exp_mask = (obs["experiment"] == src.experiment).to_numpy(dtype=bool)
        if not exp_mask.any():
            continue
        fov = obs["fov_name"]
        control_mask |= exp_mask & _well_prefix_mask(fov, src.control_wells)
        perturbed_mask |= exp_mask & _well_prefix_mask(fov, src.perturbed_wells)

    X_all = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X_ctrl = X_all[control_mask]
    Y_pert = X_all[perturbed_mask]
    if len(X_ctrl) == 0 or len(Y_pert) == 0:
        import anndata as ad_

        return ad_.AnnData(
            X=np.empty((0, adata.n_vars), dtype=X_all.dtype),
            obs=obs.iloc[:0].copy(),
            var=adata.var.copy(),
        )

    X_ref = _subsample(X_ctrl, settings.max_reference_cells, rng)
    Y_ref = _subsample(Y_pert, settings.max_reference_cells, rng)

    bandwidth = settings.bandwidth if settings.bandwidth is not None else median_heuristic(X_ref, Y_ref)
    scores = witness_function(X_all, X_ref, Y_ref, bandwidth=bandwidth)

    return _gate_scores(adata, scores, settings)


def _subsample(X: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample rows of ``X`` to at most ``max_n`` (no-op if None/small)."""
    if max_n is None or len(X) <= max_n:
        return X
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx]


def _gate_scores(adata: ad.AnnData, scores: np.ndarray, settings: WitnessSettings) -> ad.AnnData:
    """Gate witness scores into pseudo-labels and return only the labeled subset.

    Cells with ``|score|`` at or below the ``dead_zone`` quantile of ``|score|``
    are dropped (ambiguous). Above the dead-zone, sign decides the class:
    positive → control, negative → perturbed.

    Parameters
    ----------
    adata : ad.AnnData
        Marker-filtered embeddings (same order as ``scores``).
    scores : np.ndarray
        Witness scores, shape (adata.n_obs,).
    settings : WitnessSettings
        Gating settings.

    Returns
    -------
    ad.AnnData
        Labeled subset with ``obs[settings.label_column]`` set.
    """
    if settings.dead_zone > 0.0:
        threshold = float(np.quantile(np.abs(scores), settings.dead_zone))
    else:
        threshold = 0.0

    labeled_mask = np.abs(scores) > threshold if threshold > 0.0 else np.ones(len(scores), dtype=bool)
    labels = np.where(scores > 0, settings.control_label, settings.perturbed_label)

    out = adata[labeled_mask].copy()
    out.obs[settings.label_column] = pd.Categorical(labels[labeled_mask])
    return out
