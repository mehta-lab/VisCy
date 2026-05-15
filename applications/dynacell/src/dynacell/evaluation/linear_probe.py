"""FOV-stratified linear-probe diagnostics for per-cell embeddings."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


class MADScaler(BaseEstimator, TransformerMixin):
    """Median-absolute-deviation scaler.

    For each column j: x_scaled = (x - median_j) / (mad_j + eps), with
    mad_j = median(|x_j - median_j|). This is the user's specified
    "robustMAD" normalization — NOT sklearn's RobustScaler (which uses
    IQR = Q3 - Q1).
    """

    def fit(self, X, y=None):
        """Compute per-column median and MAD.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self : MADScaler
            Fitted scaler.
        """
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)
        return self

    def transform(self, X):
        """Apply MAD normalization using fitted statistics.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Scaled feature matrix with the same shape as ``X``.
        """
        return (X - self.median_) / (self.mad_ + 1e-12)


def indistinguishability(auroc: float) -> float:
    """Map an AUROC to an indistinguishability score in [0, 1].

    The score is ``1 - 2 * |AUROC - 0.5|``. Higher values indicate that
    the two classes are less distinguishable. AUROC of 0.5 (chance)
    maps to 1.0; AUROC of 0.0 or 1.0 (perfectly separable) maps to 0.0.

    Parameters
    ----------
    auroc : float
        Area under the ROC curve.

    Returns
    -------
    float
        Indistinguishability score in [0, 1].
    """
    return 1.0 - 2.0 * abs(auroc - 0.5)


def fov_stratified_auroc(
    X: np.ndarray,
    y: np.ndarray,
    fov_id: np.ndarray,
    n_splits: int = 5,
    rng_seed: int = 2020,
) -> dict:
    """FOV-stratified linear-probe AUROC.

    Pipeline = ``MADScaler() + LogisticRegression(max_iter=2000,
    class_weight='balanced', random_state=rng_seed)``. Fits inside CV
    so the scaler is trained only on the fold's training cells —
    no leakage from val FOV statistics into normalization.

    Splitting uses :class:`sklearn.model_selection.GroupKFold` with
    ``groups=fov_id``. Each fold's val set contains entire FOVs absent
    from train. When ``n_unique_fovs < n_splits``, falls back to
    ``GroupKFold(n_unique_fovs)``; when the result is fewer than 2
    folds, returns ``auroc_std=NaN``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_cells, n_features).
    y : np.ndarray
        Binary labels of shape (n_cells,) with values in {0, 1}.
    fov_id : np.ndarray
        FOV identifier of shape (n_cells,); any hashable dtype.
    n_splits : int, optional
        Requested number of CV folds, by default 5.
    rng_seed : int, optional
        Random state for the logistic regression, by default 2020.

    Returns
    -------
    dict
        Dictionary with keys ``auroc_mean``, ``auroc_std``, and
        ``n_folds``.
    """
    n_unique = len(np.unique(fov_id))
    effective_splits = min(n_splits, n_unique)

    if effective_splits < 2:
        warnings.warn(
            f"Only {n_unique} unique FOV(s); need >=2 to run GroupKFold. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "auroc_mean": float("nan"),
            "auroc_std": float("nan"),
            "n_folds": effective_splits,
        }

    pipeline = Pipeline(
        [
            ("scaler", MADScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=rng_seed,
                ),
            ),
        ]
    )

    splitter = GroupKFold(n_splits=effective_splits)
    aurocs: list[float] = []

    for train_idx, val_idx in splitter.split(X, y, groups=fov_id):
        y_val = y[val_idx]
        if len(np.unique(y_val)) < 2:
            warnings.warn(
                "Skipping fold with only one class in validation set.",
                UserWarning,
                stacklevel=2,
            )
            continue
        pipeline.fit(X[train_idx], y[train_idx])
        proba = pipeline.predict_proba(X[val_idx])[:, 1]
        aurocs.append(roc_auc_score(y_val, proba))

    if len(aurocs) == 0:
        return {
            "auroc_mean": float("nan"),
            "auroc_std": float("nan"),
            "n_folds": effective_splits,
        }

    auroc_mean = float(np.mean(aurocs))
    auroc_std = float(np.std(aurocs)) if len(aurocs) >= 2 else float("nan")

    return {
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "n_folds": effective_splits,
    }


def paired_auroc(
    x_a: np.ndarray,
    x_b: np.ndarray,
    fov_a: np.ndarray,
    fov_b: np.ndarray,
    n_splits: int = 5,
    rng_seed: int = 2020,
) -> dict:
    """FOV-stratified binary probe on two stacked cohorts.

    Builds ``X = vstack([x_a, x_b])``, ``y = [0…, 1…]``, and
    ``fov_id = concat([fov_a, fov_b])``, then delegates to
    :func:`fov_stratified_auroc`. Callers add their own column prefix
    when mapping the result back into a metrics row.

    Returns the same dict as :func:`fov_stratified_auroc`, or an
    all-NaN result with ``n_folds=0`` when either side is empty.
    """
    if x_a.size == 0 or x_b.size == 0:
        return {"auroc_mean": float("nan"), "auroc_std": float("nan"), "n_folds": 0}
    X = np.vstack([x_a, x_b])
    y = np.concatenate([np.zeros(len(x_a), dtype=np.int8), np.ones(len(x_b), dtype=np.int8)])
    fov_ids = np.concatenate([fov_a, fov_b])
    return fov_stratified_auroc(X, y, fov_ids, n_splits=n_splits, rng_seed=rng_seed)
