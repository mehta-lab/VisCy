"""Maximum Mean Discrepancy (MMD) computation for embedding zarrs.

Implements the unbiased RBF-kernel MMD² estimator with median-heuristic
bandwidth selection and an optional permutation test for p-values.

This module is the library backend used by the ``compute-mmd`` CLI command.
"""

import logging
from itertools import product
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_logger = logging.getLogger(__name__)


# ── low-level kernel helpers ──────────────────────────────────────────────────

def _sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances (n × m)."""
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _sq_dist(A, B))


def median_heuristic_gamma(
    X: np.ndarray,
    Y: np.ndarray,
    max_points: int = 2000,
    rng: Union[int, np.random.Generator] = 0,
) -> float:
    """Estimate RBF bandwidth via the median heuristic.

    Parameters
    ----------
    X, Y : np.ndarray
        Sample arrays.
    max_points : int
        Subsample size used for the distance computation.
    rng : int or Generator
        Random seed or generator.

    Returns
    -------
    float
        ``gamma = 1 / (2 * median_squared_distance)``.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    Z = np.vstack([X, Y])
    if Z.shape[0] > max_points:
        Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
    D2 = _sq_dist(Z, Z)
    tri = D2[np.triu_indices(D2.shape[0], k=1)]
    med = np.median(tri)
    return 1.0 / (2.0 * med + 1e-12)


def mmd2_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
) -> float:
    """Unbiased MMD² estimate with RBF kernel.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (m, d)
    gamma : float
        RBF kernel bandwidth parameter.

    Returns
    -------
    float
        Unbiased MMD².
    """
    n, m = X.shape[0], Y.shape[0]
    Kxx = _rbf_kernel(X, X, gamma); np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf_kernel(Y, Y, gamma); np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf_kernel(X, Y, gamma)
    return (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )


def mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
    n_perm: int = 1000,
    max_cells: int = 2000,
    rng: Union[int, np.random.Generator] = 0,
) -> tuple[float, Optional[float], float]:
    """Compute MMD² with optional permutation test p-value.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Embeddings for group A.
    Y : np.ndarray, shape (m, d)
        Embeddings for group B.
    gamma : float or None
        RBF gamma; estimated via median heuristic if None.
    n_perm : int
        Number of permutations.  Set to 0 to skip the test (p = None).
    max_cells : int
        Subsample each group to at most this many cells before computing.
    rng : int or Generator
        Random seed or generator.

    Returns
    -------
    (mmd2, p_value, gamma)
        ``p_value`` is None when ``n_perm == 0``.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Subsample
    if len(X) > max_cells:
        X = X[rng.choice(len(X), max_cells, replace=False)]
    if len(Y) > max_cells:
        Y = Y[rng.choice(len(Y), max_cells, replace=False)]

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    if gamma is None:
        gamma = median_heuristic_gamma(X, Y, rng=rng)

    observed = mmd2_unbiased(X, Y, gamma)

    if n_perm == 0:
        return observed, None, gamma

    Z = np.vstack([X, Y])
    n = len(X)
    perm_stats = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        idx = rng.permutation(len(Z))
        perm_stats[b] = mmd2_unbiased(Z[idx[:n]], Z[idx[n:]], gamma)

    # +1 smoothing (Phipson & Smyth 2010)
    p_value = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)
    return observed, p_value, gamma


# ── data helpers ──────────────────────────────────────────────────────────────

def _to_np(X) -> np.ndarray:
    return np.array(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def _apply_filter(obs: pd.DataFrame, filter_spec: Optional[dict]) -> np.ndarray:
    """Return boolean mask for obs rows matching filter_spec.

    If filter_spec is None, all rows are selected.
    Supported keys: ``startswith`` (str or list[str]) or ``equals`` (str),
    paired with ``column``.
    """
    if filter_spec is None:
        return np.ones(len(obs), dtype=bool)

    col = filter_spec["column"]
    values = obs[col].astype(str)

    if "startswith" in filter_spec:
        prefixes = filter_spec["startswith"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        mask = np.zeros(len(obs), dtype=bool)
        for p in prefixes:
            mask |= values.str.startswith(p).values
        return mask

    if "equals" in filter_spec:
        return (values == str(filter_spec["equals"])).values

    raise ValueError(
        "filter_spec must contain 'startswith' or 'equals'. "
        f"Got: {list(filter_spec.keys())}"
    )


# ── main computation ──────────────────────────────────────────────────────────

def compute_mmd(
    zarr_a: Union[str, Path],
    zarr_b: Optional[Union[str, Path]],
    filter_a: Optional[dict],
    filter_b: Optional[dict],
    group_by: Optional[list[str]],
    use_pca: bool,
    n_pca: int,
    n_perm: int,
    max_cells: int,
    random_seed: int,
) -> pd.DataFrame:
    """Compute MMD² between two groups of embeddings.

    Loads zarr(s), applies obs filters, optionally groups by obs columns,
    fits a shared PCA if requested, and computes MMD² (with optional
    permutation p-value) for every group combination.

    Parameters
    ----------
    zarr_a : str or Path
        Path to AnnData zarr for group A.
    zarr_b : str or Path or None
        Path to AnnData zarr for group B.  If None or same as zarr_a,
        the single zarr is loaded once and both filters are applied to it.
    filter_a : dict or None
        Obs filter for group A (see :func:`_apply_filter`).
    filter_b : dict or None
        Obs filter for group B.
    group_by : list[str] or None
        Obs column names to stratify comparisons by.  MMD is computed
        separately for each unique value combination found in **both** groups.
        Pass None or an empty list for a single overall comparison.
    use_pca : bool
        Fit shared PCA on the combined (filtered) embeddings before MMD.
    n_pca : int
        Number of PCA components.
    n_perm : int
        Permutation test iterations (0 = skip, p_value will be NaN).
    max_cells : int
        Maximum cells per group passed to the MMD kernel computation.
    random_seed : int
        Global random seed.

    Returns
    -------
    pd.DataFrame
        One row per group combination with columns:
        ``[*group_by, n_a, n_b, mmd2, p_value, gamma]``.
    """
    rng = np.random.default_rng(random_seed)

    # Load data
    same_zarr = zarr_b is None or str(zarr_b) == str(zarr_a)

    _logger.info("Loading zarr A: %s", zarr_a)
    adata_a = ad.read_zarr(zarr_a)
    adata_a.obs_names_make_unique()

    if same_zarr:
        adata_b = adata_a
        _logger.info("Using same zarr for group B")
    else:
        _logger.info("Loading zarr B: %s", zarr_b)
        adata_b = ad.read_zarr(zarr_b)
        adata_b.obs_names_make_unique()

    mask_a = _apply_filter(adata_a.obs, filter_a)
    mask_b = _apply_filter(adata_b.obs, filter_b)
    _logger.info(
        "Cells after filtering — A: %d / %d,  B: %d / %d",
        mask_a.sum(), len(adata_a),
        mask_b.sum(), len(adata_b),
    )

    X_a_full = _to_np(adata_a.X[mask_a])
    X_b_full = _to_np(adata_b.X[mask_b])
    obs_a = adata_a.obs[mask_a].reset_index(drop=True)
    obs_b = adata_b.obs[mask_b].reset_index(drop=True)

    # PCA
    if use_pca:
        _logger.info("Fitting shared PCA-%d on combined filtered cells ...", n_pca)
        sc  = StandardScaler()
        pca = PCA(n_components=n_pca, random_state=random_seed)
        combined = np.vstack([X_a_full, X_b_full])
        Z_all = pca.fit_transform(sc.fit_transform(combined))
        var_exp = pca.explained_variance_ratio_.sum() * 100
        _logger.info("PCA explained variance: %.1f%%", var_exp)
        X_a_full = Z_all[:len(X_a_full)].astype(np.float32)
        X_b_full = Z_all[len(X_a_full):].astype(np.float32)

    # Build group combinations
    if not group_by:
        groups = [{}]  # single overall comparison
    else:
        # Find unique values per column in group A, intersect with group B
        col_vals = []
        for col in group_by:
            vals_a = set(obs_a[col].astype(str).unique())
            vals_b = set(obs_b[col].astype(str).unique())
            common = sorted(vals_a & vals_b)
            if not common:
                _logger.warning(
                    "No common values for group_by column '%s' — skipping.", col
                )
                return pd.DataFrame()
            col_vals.append(common)
        groups = [
            dict(zip(group_by, combo))
            for combo in product(*col_vals)
        ]

    _logger.info(
        "Computing MMD for %d group combination(s) ...", len(groups)
    )

    rows = []
    for group in groups:
        # Build mask for this group
        ga = np.ones(len(obs_a), dtype=bool)
        gb = np.ones(len(obs_b), dtype=bool)
        for col, val in group.items():
            ga &= obs_a[col].astype(str) == val
            gb &= obs_b[col].astype(str) == val

        Xa = X_a_full[ga]
        Xb = X_b_full[gb]

        label = ", ".join(f"{k}={v}" for k, v in group.items()) or "overall"
        if len(Xa) < 5 or len(Xb) < 5:
            _logger.warning(
                "Skipping %s: too few cells (A=%d, B=%d)", label, len(Xa), len(Xb)
            )
            continue

        mmd2, p_val, gamma = mmd_permutation_test(
            Xa, Xb,
            gamma=None,
            n_perm=n_perm,
            max_cells=max_cells,
            rng=rng,
        )

        row = {**group, "n_a": len(Xa), "n_b": len(Xb),
               "mmd2": mmd2,
               "p_value": p_val if p_val is not None else float("nan"),
               "gamma": gamma}
        rows.append(row)
        _logger.info(
            "  %-40s  n_a=%5d  n_b=%5d  MMD²=%.5f  p=%.4f",
            label, len(Xa), len(Xb), mmd2,
            p_val if p_val is not None else float("nan"),
        )

    return pd.DataFrame(rows)
