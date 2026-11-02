"""Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel and permutation test.

GPU-accelerated via PyTorch: kernel matrices are built on the available device
(CUDA if present, otherwise CPU). The public API is device-agnostic — inputs and
outputs are NumPy arrays / Python floats — so callers do not need to manage
tensors or devices.
"""

import numpy as np
import torch
from numpy.typing import NDArray

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subsample(X: NDArray, max_n: int | None, rng: np.random.Generator) -> NDArray:
    """Randomly subsample rows of ``X`` to at most ``max_n`` (no-op if None/small)."""
    if max_n is None or len(X) <= max_n:
        return X
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx]


def _sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances via the (a-b)^2 = a^2 + b^2 - 2ab expansion.

    Parameters
    ----------
    A : torch.Tensor
        Shape (n, d).
    B : torch.Tensor
        Shape (m, d).

    Returns
    -------
    torch.Tensor
        Squared distances, shape (n, m). Clamped at 0 to absorb the small
        negative values the expansion can produce from floating-point error.
    """
    A2 = (A * A).sum(1, keepdim=True)
    B2 = (B * B).sum(1, keepdim=True).T
    return (A2 + B2 - 2.0 * (A @ B.T)).clamp_min_(0.0)


def _rbf_kernel(A: torch.Tensor, B: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gaussian RBF kernel matrix on the device of ``A``.

    K(a, b) = exp(-||a - b||^2 / (2 * bandwidth))

    Parameters
    ----------
    A : torch.Tensor
        Shape (n, d).
    B : torch.Tensor
        Shape (m, d).
    bandwidth : float
        Kernel bandwidth (sigma^2). Must be > 0.

    Returns
    -------
    torch.Tensor
        Kernel matrix, shape (n, m).
    """
    return torch.exp(-_sq_dists(A, B) / (2.0 * bandwidth))


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
    device = _DEVICE
    P = torch.from_numpy(pool).to(device)
    d2 = _sq_dists(P, P)
    n = d2.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    return float(d2[mask].median().item()) + 1e-12


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
    device = _DEVICE
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    n = len(X)
    m = len(Y)
    K_XX = _rbf_kernel(Xt, Xt, bandwidth)
    K_YY = _rbf_kernel(Yt, Yt, bandwidth)
    K_XY = _rbf_kernel(Xt, Yt, bandwidth)
    K_XX.fill_diagonal_(0.0)
    K_YY.fill_diagonal_(0.0)
    # Reduce in float64 so the estimate is symmetric in (X, Y) to machine
    # precision despite the float32 kernel.
    mmd2 = K_XX.double().sum() / (n * (n - 1)) + K_YY.double().sum() / (m * (m - 1)) - 2.0 * K_XY.double().mean()
    return float(mmd2.item())


_MMD_PERM_MAX_N = 20_000


def mmd_permutation_test(
    X: NDArray,
    Y: NDArray,
    n_permutations: int = 1000,
    bandwidth: float | None = None,
    seed: int = 42,
) -> tuple[float, float, NDArray]:
    """MMD^2 with vectorized permutation test for significance.

    Builds the pooled RBF kernel matrix K once on the available device, then
    evaluates the observed split and all permutations in a single batch of
    matrix multiplications — no per-permutation Python loop and no repeated
    distance computations.

    Strategy: represent each permutation as a binary label vector z in
    {0,1}^(n+m) (1 = assigned to X group), then for a batch of P permutations
    stacked as Z (P, N), the within-X / within-Y / cross kernel sums follow from
    K @ Z.T and K @ (1 - Z).T. With the kernel diagonal zeroed, this gives the
    unbiased MMD^2 for every permutation at once in O(P * N^2) GEMM ops.

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
    # explicitly so callers see a clear error rather than an OOM (host or GPU)
    # when they forget to subsample (50k => 10 GB; 100k => 40 GB).
    if N > _MMD_PERM_MAX_N:
        raise ValueError(
            f"mmd_permutation_test pooled kernel would be ({N}, {N}) float32 "
            f"≈ {(N * N * 4) / 1e9:.1f} GB. Subsample X and/or Y so that "
            f"len(X) + len(Y) <= {_MMD_PERM_MAX_N}."
        )
    device = _DEVICE
    pool = np.concatenate([X, Y], axis=0).astype(np.float32)
    P = torch.from_numpy(pool).to(device)
    # Compute full pooled kernel matrix once: (N, N)
    K = _rbf_kernel(P, P, bandwidth)
    K.fill_diagonal_(0.0)

    # Label matrix: row 0 = observed split (first n are X), rows 1: = random
    # permutations. Every row has exactly n ones, so group sizes are preserved.
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(N) for _ in range(n_permutations)])  # (P, N)
    labels = np.zeros((n_permutations + 1, N), dtype=np.float32)
    labels[0, :n] = 1.0
    row_idx = np.arange(n_permutations)[:, None]
    labels[1:][row_idx, perms[:, :n]] = 1.0
    z = torch.from_numpy(labels).to(device)  # (P+1, N)
    one_minus_z = 1.0 - z

    # Vectorized kernel sums for every split at once
    KzX = (K @ z.T).T  # (P+1, N) row sums restricted to X group
    KzY = (K @ one_minus_z.T).T  # (P+1, N) row sums restricted to Y group
    sum_XX = (z * KzX).sum(1)  # within-X (diagonal zeroed)
    sum_YY = (one_minus_z * KzY).sum(1)  # within-Y
    sum_XY = (z * KzY).sum(1)  # cross
    mmd2_all = sum_XX / (n * (n - 1)) + sum_YY / (m * (m - 1)) - 2.0 * sum_XY / (n * m)
    mmd2_all = mmd2_all.cpu().numpy()

    observed = float(mmd2_all[0])
    null = mmd2_all[1:]
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return observed, p_value, null


def witness_function(
    query: NDArray,
    X: NDArray,
    Y: NDArray,
    bandwidth: float | None = None,
    chunk_size: int = 500,
) -> NDArray:
    """Evaluate the empirical MMD witness function at query points.

    The witness function is the RKHS direction along which distributions P
    (sampled by ``X``) and Q (sampled by ``Y``) differ most. Evaluated at a
    point z with the Gaussian RBF kernel k:

        w(z) = (1/n) sum_i k(z, x_i) - (1/m) sum_j k(z, y_j)

    A positive score means z looks more like X (P); a negative score means it
    looks more like Y (Q). The squared MMD equals the difference in mean witness
    score between the two samples, so this is the per-point contribution to the
    MMD.

    Query points are processed in chunks so the intermediate kernel matrices
    stay bounded regardless of ``len(query)``.

    Parameters
    ----------
    query : NDArray
        Points at which to evaluate the witness function, shape (q, d).
    X : NDArray
        Samples from distribution P, shape (n, d). Positive scores lean toward X.
    Y : NDArray
        Samples from distribution Q, shape (m, d). Negative scores lean toward Y.
    bandwidth : float or None
        Gaussian RBF bandwidth (sigma^2). None = median heuristic on (X, Y).
    chunk_size : int
        Number of query points evaluated per batch.

    Returns
    -------
    NDArray
        Witness scores, shape (q,), float64.
    """
    if bandwidth is None:
        bandwidth = median_heuristic(X, Y)
    device = _DEVICE
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    q = np.asarray(query, dtype=np.float32)
    scores = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), chunk_size):
        sl = slice(start, start + chunk_size)
        Qc = torch.from_numpy(q[sl]).to(device)
        k_x = _rbf_kernel(Qc, Xt, bandwidth).mean(dim=1)
        k_y = _rbf_kernel(Qc, Yt, bandwidth).mean(dim=1)
        scores[sl] = (k_x - k_y).double().cpu().numpy()
    return scores
