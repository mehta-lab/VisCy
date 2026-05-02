"""Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel and permutation test."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def median_heuristic(X: NDArray, Y: NDArray, subsample: int = 1000) -> float:
    """Compute Gaussian RBF bandwidth via the median heuristic.

    Subsamples jointly from X and Y, computes all pairwise squared Euclidean
    distances, and returns the median. This is the standard bandwidth selection
    for MMD tests (Gretton et al., 2012).

    Parameters
    ----------
    X : NDArray
        Samples from distribution P, shape (n, d).
    Y : NDArray
        Samples from distribution Q, shape (m, d).
    subsample : int
        Max samples to draw from the joint (X, Y) pool for median computation.

    Returns
    -------
    float
        Bandwidth sigma^2 for the Gaussian RBF kernel.
    """
    rng = np.random.default_rng(0)
    pool = np.concatenate([X, Y], axis=0).astype(np.float32)
    if len(pool) > subsample:
        idx = rng.choice(len(pool), subsample, replace=False)
        pool = pool[idx]
    sq_dists = cdist(pool, pool, metric="sqeuclidean")
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    return float(np.median(upper)) + 1e-12


def gaussian_rbf_kernel(X: NDArray, Y: NDArray, bandwidth: float) -> NDArray:
    """Compute Gaussian RBF kernel matrix K(X, Y) in float32.

    K(x, y) = exp(-||x - y||^2 / (2 * bandwidth))

    Parameters
    ----------
    X : NDArray
        Shape (n, d).
    Y : NDArray
        Shape (m, d).
    bandwidth : float
        Kernel bandwidth (sigma^2). Must be > 0.

    Returns
    -------
    NDArray
        Kernel matrix, shape (n, m), float32.
    """
    sq_dists = cdist(X.astype(np.float32), Y.astype(np.float32), metric="sqeuclidean")
    return np.exp(-sq_dists / (2.0 * bandwidth), dtype=np.float32)


def compute_mmd_unbiased(X: NDArray, Y: NDArray, bandwidth: float | None = None) -> float:
    """Compute the unbiased quadratic-time MMD^2 estimator.

    MMD^2_u = (1/(n(n-1))) sum_{i!=j} k(x_i, x_j)
            + (1/(m(m-1))) sum_{i!=j} k(y_i, y_j)
            - (2/(nm)) sum_{i,j} k(x_i, y_j)

    Parameters
    ----------
    X : NDArray
        Samples from distribution P, shape (n, d).
    Y : NDArray
        Samples from distribution Q, shape (m, d).
    bandwidth : float or None
        Gaussian RBF bandwidth. None = median heuristic.

    Returns
    -------
    float
        Unbiased MMD^2 estimate.
    """
    if bandwidth is None:
        bandwidth = median_heuristic(X, Y)
    n = len(X)
    m = len(Y)
    K_XX = gaussian_rbf_kernel(X, X, bandwidth)
    K_YY = gaussian_rbf_kernel(Y, Y, bandwidth)
    K_XY = gaussian_rbf_kernel(X, Y, bandwidth)
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    mmd2 = K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2.0 * K_XY.mean()
    return float(mmd2)


_MMD_PERM_MAX_N = 20_000


def mmd_permutation_test(
    X: NDArray,
    Y: NDArray,
    n_permutations: int = 1000,
    bandwidth: float | None = None,
    seed: int = 42,
) -> tuple[float, float, NDArray]:
    """MMD^2 with vectorized permutation test for significance.

    Precomputes the pooled kernel matrix K_pool once, then all permutations
    are evaluated via vectorized row/column sums — no repeated cdist calls
    and no Python loop over individual permutations.

    Strategy: for each permutation p, MMD^2 = sum_X/n(n-1) + sum_Y/m(m-1) - 2*mean_XY
    where sum_X = sum of K_pool[ix,ix] off-diagonal = (K_pool[ix,:] * one_hot_X).sum().
    We represent each permutation as a binary label vector z in {0,1}^(n+m),
    then use K_pool @ z and K_pool @ (1-z) to get row sums in O(n_perm * N) ops.

    Parameters
    ----------
    X : NDArray
        Samples from distribution P, shape (n, d).
    Y : NDArray
        Samples from distribution Q, shape (m, d).
    n_permutations : int
        Number of permutations for the null distribution.
    bandwidth : float or None
        Gaussian RBF bandwidth. None = median heuristic (computed once).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mmd2 : float
        Observed MMD^2 (unbiased).
    p_value : float
        Permutation test p-value.
    null_distribution : NDArray
        Null MMD^2 values from permutations, shape (n_permutations,).
    """
    if bandwidth is None:
        bandwidth = median_heuristic(X, Y)
    n = len(X)
    m = len(Y)
    N = n + m
    # The pooled kernel matrix is (N, N) float32 — quadratic in N. Cap N
    # explicitly so callers see a clear error rather than an OOM when they
    # forget to subsample (50k => 10 GB; 100k => 40 GB).
    if N > _MMD_PERM_MAX_N:
        raise ValueError(
            f"mmd_permutation_test pooled kernel would be ({N}, {N}) float32 "
            f"≈ {(N * N * 4) / 1e9:.1f} GB. Subsample X and/or Y so that "
            f"len(X) + len(Y) <= {_MMD_PERM_MAX_N}."
        )
    pool = np.concatenate([X, Y], axis=0).astype(np.float32)
    # Compute full pooled kernel matrix once: (N, N) float32
    K = gaussian_rbf_kernel(pool, pool, bandwidth)
    np.fill_diagonal(K, 0.0)

    def _mmd2_from_labels(z: NDArray) -> NDArray:
        """Vectorized MMD^2 for a batch of label vectors.

        Parameters
        ----------
        z : NDArray
            Shape (n_perm, N), float32, 1 = assigned to X group.

        Returns
        -------
        NDArray
            MMD^2 values, shape (n_perm,).
        """
        nz = z.sum(axis=1)  # actual n per permutation (n_perm,)
        mz = N - nz  # actual m per permutation
        # Row sums of K restricted to X-group and Y-group
        # K @ z.T -> (N, n_perm), then z @ (K @ z.T) -> (n_perm, n_perm) diagonal = sum_XX
        KzT = K @ z.T  # (N, n_perm)
        sum_XX = (z * KzT.T).sum(axis=1)  # (n_perm,) — within-X kernel sums (diagonal zeroed)
        sum_YY = ((1 - z) * (K @ (1 - z).T).T).sum(axis=1)  # (n_perm,) — within-Y
        sum_XY = (z * (K @ (1 - z).T).T).sum(axis=1)  # (n_perm,) — cross
        kxx = sum_XX / (nz * (nz - 1))
        kyy = sum_YY / (mz * (mz - 1))
        kxy = sum_XY / (nz * mz)
        return kxx + kyy - 2.0 * kxy

    # Observed: original split (first n are X)
    z_obs = np.zeros((1, N), dtype=np.float32)
    z_obs[0, :n] = 1.0
    observed = float(_mmd2_from_labels(z_obs)[0])

    # Null: random permutations as binary label vectors
    rng = np.random.default_rng(seed)
    # Generate all permutation indices at once
    perms = np.stack([rng.permutation(N) for _ in range(n_permutations)])  # (n_perm, N)
    z_null = np.zeros((n_permutations, N), dtype=np.float32)
    row_idx = np.arange(n_permutations)[:, None]
    z_null[row_idx, perms[:, :n]] = 1.0

    null = _mmd2_from_labels(z_null)
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return observed, p_value, null
