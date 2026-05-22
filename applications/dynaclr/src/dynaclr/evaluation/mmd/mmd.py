"""Core MMD² computation between two pooled groups of cell embeddings.

Each group is defined by one or more (zarr_path, well_name, well_id) entries.
Cells are identified by the ``fov_name`` column in AnnData ``.obs``, which
has the format ``well_name/well_id/pos_id``. Filtering matches any row whose
``fov_name`` starts with ``well_name/well_id/``.

All wells within a group are concatenated into a single embedding matrix
before MMD² is computed.
"""

import logging
import warnings
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np

_logger = logging.getLogger(__name__)

_MIN_CELLS_WARN = 500
_MIN_CELLS_RECOMMEND = 1000


# ── kernel helpers ────────────────────────────────────────────────────────────

def _sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances (n × m)."""
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)


def _median_heuristic_gamma(
    X: np.ndarray,
    Y: np.ndarray,
    max_points: int = 2000,
    rng: np.random.Generator = None,
) -> float:
    """Estimate RBF bandwidth via the median heuristic.

    Parameters
    ----------
    X, Y : np.ndarray
        Sample arrays.
    max_points : int
        Subsample size used for the distance computation.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    float
        ``gamma = 1 / (2 * median_squared_distance)``.
    """
    Z = np.vstack([X, Y])
    if Z.shape[0] > max_points:
        Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
    D2 = _sq_dist(Z, Z)
    tri = D2[np.triu_indices(D2.shape[0], k=1)]
    med = np.median(tri)
    return 1.0 / (2.0 * med + 1e-12)


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _sq_dist(A, B))


def _mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD² estimate with RBF kernel."""
    n, m = len(X), len(Y)
    Kxx = _rbf_kernel(X, X, gamma)
    np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf_kernel(Y, Y, gamma)
    np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf_kernel(X, Y, gamma)
    return (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )


# ── data loading ──────────────────────────────────────────────────────────────

def _load_well_embeddings(
    zarr_path: Union[str, Path],
    well_name: str,
    well_id: Union[str, int],
) -> np.ndarray:
    """Load embeddings for one well from an AnnData zarr.

    Filters rows in ``.obs`` where ``fov_name`` starts with
    ``"{well_name}/{well_id}/"``.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the AnnData zarr store.
    well_name : str
        Well name component of the fov_name (e.g. ``"B03"``).
    well_id : str or int
        Well ID component of the fov_name (e.g. ``"1"``).

    Returns
    -------
    np.ndarray, shape (n_cells, n_features)
        Float32 embedding matrix for the selected cells.
    """
    prefix = f"{well_name}/{well_id}/"
    _logger.info("Loading %s — well prefix '%s'", zarr_path, prefix)

    adata = ad.read_zarr(zarr_path)
    adata.obs_names_make_unique()

    mask = adata.obs["fov_name"].astype(str).str.startswith(prefix).values
    n_total = len(adata)
    n_selected = mask.sum()

    if n_selected == 0:
        raise ValueError(
            f"No cells found for prefix '{prefix}' in {zarr_path}. "
            f"Available fov_name values (first 10): "
            f"{adata.obs['fov_name'].unique()[:10].tolist()}"
        )

    _logger.info("  %d / %d cells selected", n_selected, n_total)
    X = adata.X[mask]
    return np.array(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def _pool_group(wells: list[dict]) -> np.ndarray:
    """Load and concatenate embeddings for all wells in a group.

    Parameters
    ----------
    wells : list of dict
        Each dict must have keys ``zarr_path``, ``well_name``, ``well_id``.

    Returns
    -------
    np.ndarray, shape (n_cells_total, n_features)
    """
    parts = []
    for w in wells:
        emb = _load_well_embeddings(w["zarr_path"], w["well_name"], w["well_id"])
        parts.append(emb)
    return np.concatenate(parts, axis=0)


# ── MMD computation ───────────────────────────────────────────────────────────

def compute_mmd(
    group_a: list[dict],
    group_b: list[dict],
    n_perm: int = 1000,
    max_cells: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Compute MMD² between two pooled groups of cell embeddings.

    Parameters
    ----------
    group_a : list of dict
        Wells for group A. Each dict: ``zarr_path``, ``well_name``, ``well_id``.
    group_b : list of dict
        Wells for group B.
    n_perm : int, optional
        Number of permutations for the p-value estimate. Set to 0 to skip.
        By default 1000.
    max_cells : int, optional
        Maximum cells per group passed to the MMD kernel (subsampled if larger).
        By default 2000.
    random_seed : int, optional
        Random seed. By default 42.

    Returns
    -------
    dict with keys:
        ``n_a``, ``n_b``, ``mmd2``, ``p_value`` (None if n_perm=0), ``gamma``.
    """
    rng = np.random.default_rng(random_seed)

    _logger.info("Pooling group A (%d well(s)) ...", len(group_a))
    X = _pool_group(group_a)

    _logger.info("Pooling group B (%d well(s)) ...", len(group_b))
    Y = _pool_group(group_b)

    _logger.info("Group A: %d cells  |  Group B: %d cells", len(X), len(Y))

    # Sample size warnings
    for label, arr in [("A", X), ("B", Y)]:
        n = len(arr)
        if n < _MIN_CELLS_WARN:
            warnings.warn(
                f"Group {label} has only {n} cells. MMD estimates may be unreliable. "
                f"At least {_MIN_CELLS_WARN} cells are recommended; "
                f"{_MIN_CELLS_RECOMMEND} or more for best results.",
                UserWarning,
                stacklevel=2,
            )
        elif n < _MIN_CELLS_RECOMMEND:
            warnings.warn(
                f"Group {label} has {n} cells. For best results, "
                f"{_MIN_CELLS_RECOMMEND} or more cells are recommended.",
                UserWarning,
                stacklevel=2,
            )

    # Subsample for kernel computation
    Xs = X[rng.choice(len(X), min(len(X), max_cells), replace=False)].astype(np.float64)
    Ys = Y[rng.choice(len(Y), min(len(Y), max_cells), replace=False)].astype(np.float64)

    gamma = _median_heuristic_gamma(Xs, Ys, rng=rng)
    observed = _mmd2_unbiased(Xs, Ys, gamma)
    _logger.info("MMD² = %.6f  (gamma=%.6f)", observed, gamma)

    p_value = None
    if n_perm > 0:
        _logger.info("Running permutation test (%d permutations) ...", n_perm)
        Z = np.vstack([Xs, Ys])
        n = len(Xs)
        perm_stats = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            idx = rng.permutation(len(Z))
            perm_stats[b] = _mmd2_unbiased(Z[idx[:n]], Z[idx[n:]], gamma)
        # +1 smoothing (Phipson & Smyth 2010)
        p_value = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)
        _logger.info("p-value = %.4f", p_value)

    return {
        "n_a": len(X),
        "n_b": len(Y),
        "mmd2": observed,
        "p_value": p_value,
        "gamma": gamma,
    }
