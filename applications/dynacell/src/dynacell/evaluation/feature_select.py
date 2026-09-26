"""Variance + correlation feature pruning for the CP regionprops track.

Vendored from pycytominer (BSD-3); no runtime dep.

The math implemented here is adapted from pycytominer
(https://github.com/cytomining/pycytominer, BSD-3-Clause).

Notes
-----
All functions operate on float64 numpy arrays directly — never DataFrames —
so they can be used inside tight evaluation loops without pandas overhead.
"""

from __future__ import annotations

import numpy as np

# Defaults shared with downstream consumers (e.g. the cp_selected_feature_mask
# JSON sidecar emitted by the evaluation pipeline, and the CP reference). Keep
# `select_gt_features`, `variance_threshold`, and `correlation_threshold`
# keyword defaults aligned
# with these constants so the sidecar cannot drift from the actual call.
DEFAULT_FREQ_CUT = 0.05
DEFAULT_UNIQUE_CUT = 0.01
DEFAULT_CORR_THRESHOLD = 0.9


def variance_threshold(
    X_pooled: np.ndarray,
    freq_cut: float = DEFAULT_FREQ_CUT,
    unique_cut: float = DEFAULT_UNIQUE_CUT,
) -> np.ndarray:
    """Drop near-constant columns.

    A column is dropped iff EITHER condition is True:

    - frequency-ratio test: ``count_of_2nd_most_common / count_of_most_common
      < freq_cut`` (a column whose values cluster on one dominant level is
      dropped). When the column has only one unique value, ``freq_ratio`` is
      defined as ``0.0`` and the column is dropped.
    - uniqueness test: ``n_unique_values / n_samples < unique_cut`` (a column
      with very few distinct values relative to row count is dropped).

    Parameters
    ----------
    X_pooled : np.ndarray
        Shape ``(n_samples, n_features)``, float64.
    freq_cut : float, optional
        Threshold for the frequency-ratio test. Defaults to ``0.05``.
    unique_cut : float, optional
        Threshold for the uniqueness test. Defaults to ``0.01``.

    Returns
    -------
    np.ndarray
        Boolean keep-mask of shape ``(n_features,)``. ``True`` entries are
        kept.

    Notes
    -----
    Iterates columns and calls :func:`numpy.unique` per column —
    ``O(n_features * n_samples log n_samples)``. Designed for the small
    (~15-22 column) CP regionprops matrix; do not feed deep-feature
    matrices (1024+ columns) through this without vectorizing first.
    """
    n_samples, n_features = X_pooled.shape
    keep = np.ones(n_features, dtype=bool)
    for j in range(n_features):
        col = X_pooled[:, j]
        _, counts = np.unique(col, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        if counts_sorted.size == 1:
            freq_ratio = 0.0
        else:
            freq_ratio = counts_sorted[1] / counts_sorted[0]
        uniqueness = counts_sorted.size / n_samples
        if freq_ratio < freq_cut or uniqueness < unique_cut:
            keep[j] = False
    return keep


def correlation_threshold(
    X_pooled: np.ndarray,
    threshold: float = DEFAULT_CORR_THRESHOLD,
    method: str = "pearson",
) -> np.ndarray:
    """Greedy iterative drop of correlated columns.

    Builds the absolute-Pearson-correlation matrix, walks every pair with
    ``|corr| > threshold`` in descending ``|corr|`` order, and for each
    surviving pair drops the column whose total sum of ``|corr|`` with the
    remaining columns is larger (ties broken by higher column index).
    NaN correlations (from zero-variance columns) are treated as ``0``.

    Parameters
    ----------
    X_pooled : np.ndarray
        Shape ``(n_samples, n_features)``, float64.
    threshold : float, optional
        Absolute-correlation threshold above which a pair is considered
        redundant. Defaults to ``0.9``.
    method : str, optional
        Only ``"pearson"`` is supported. Any other value raises
        ``ValueError``.

    Returns
    -------
    np.ndarray
        Boolean keep-mask of shape ``(n_features,)``.

    Raises
    ------
    ValueError
        If ``method`` is anything other than ``"pearson"``.
    """
    if method != "pearson":
        raise ValueError(f"Only method='pearson' is supported, got method={method!r}.")

    n_features = X_pooled.shape[1]
    if n_features == 0:
        return np.ones(0, dtype=bool)

    corr = np.corrcoef(X_pooled, rowvar=False)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))
    # corrcoef on a single column returns a 0-d array; promote to 2-d.
    corr = np.atleast_2d(corr)
    np.fill_diagonal(corr, 0.0)

    # Collect all (i, j) with i<j and corr[i,j] > threshold, descending by corr.
    iu, ju = np.triu_indices(n_features, k=1)
    mask = corr[iu, ju] > threshold
    pair_i = iu[mask]
    pair_j = ju[mask]
    pair_c = corr[pair_i, pair_j]
    order = np.argsort(-pair_c, kind="stable")
    pair_i = pair_i[order]
    pair_j = pair_j[order]

    keep = np.ones(n_features, dtype=bool)
    for i, j in zip(pair_i, pair_j, strict=False):
        if not keep[i] or not keep[j]:
            continue
        surviving = keep.copy()
        # Exclude self when summing each candidate's connectivity to survivors.
        surviving_i = surviving.copy()
        surviving_i[i] = False
        surviving_j = surviving.copy()
        surviving_j[j] = False
        sum_i = corr[i, surviving_i].sum()
        sum_j = corr[j, surviving_j].sum()
        # Use a tolerance: when the two sums are effectively equal (e.g. a lone
        # correlated pair with no other strong neighbors), fall through to the
        # higher-index tie-break.
        if np.isclose(sum_i, sum_j, rtol=1e-5, atol=1e-8):
            drop = max(i, j)
        elif sum_i > sum_j:
            drop = i
        else:
            drop = j
        keep[drop] = False
    return keep


def select_gt_features(
    gt: np.ndarray,
    freq_cut: float = DEFAULT_FREQ_CUT,
    unique_cut: float = DEFAULT_UNIQUE_CUT,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> np.ndarray:
    """Select features on ground-truth cells alone, never on predictions.

    Variance pruning first (cheap), then correlation pruning on the survivors.
    The fit set holds only GT rows, so the kept feature subset is a property of
    the target and cannot move with the model being evaluated (pooling a
    prediction into the fit is what used to give every model its own mask).
    Used to build the per-target CP reference
    (:mod:`dynacell.evaluation.cp_reference`).

    Parameters
    ----------
    gt : np.ndarray
        Shape ``(n_gt, n_features)``, float64. Rows pooled across every
        dataset the reference is fit on.
    freq_cut : float, optional
        Forwarded to :func:`variance_threshold`. Defaults to ``0.05``.
    unique_cut : float, optional
        Forwarded to :func:`variance_threshold`. Defaults to ``0.01``.
    corr_threshold : float, optional
        Forwarded to :func:`correlation_threshold`. Defaults to ``0.9``.

    Returns
    -------
    np.ndarray
        Boolean keep-mask of shape ``(n_features,)``.
    """
    mask_var = variance_threshold(gt, freq_cut=freq_cut, unique_cut=unique_cut)
    mask_corr_local = correlation_threshold(gt[:, mask_var], threshold=corr_threshold, method="pearson")
    keep_mask = np.zeros(gt.shape[1], dtype=bool)
    keep_mask[np.flatnonzero(mask_var)[mask_corr_local]] = True
    return keep_mask
