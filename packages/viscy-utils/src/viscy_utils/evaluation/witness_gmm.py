"""Gaussian-mixture gating of MMD-witness scores into confident pseudo-labels.

The MMD witness score (:func:`viscy_utils.evaluation.mmd.witness_function`) is a
scalar per cell measuring how far its embedding leans toward the perturbed
reference distribution. This module fits a two-component Gaussian mixture to
those 1-D scores and derives a confidence-gated label: the lower-mean component
is the remodeled/perturbed mode, and a cell is called positive when its
posterior for that mode clears a threshold.

This replaces the earlier hard sign + dead-zone gate. The GMM crossover is a
calibrated boundary (it adapts to the two modes' locations) rather than a
hardcoded sign-at-zero cut, which matters for the heavily imbalanced gated
classes the witness produces.

The single public function :func:`fit_gmm_labels` is pure NumPy/scikit-learn so
it can be reused from application code and standalone analysis scripts alike.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


@dataclass
class GmmLabelResult:
    """Result of gating 1-D witness scores with a two-component GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        The fitted two-component mixture.
    remod_component : int
        Index of the remodeled/perturbed component (the one with the more
        negative mean; witness scores are negative for perturbed-leaning cells).
    posterior : NDArray
        Per-cell posterior probability of the remodeled component, shape ``(n,)``.
    hard_label : NDArray
        Per-cell gated label, int8, shape ``(n,)``: ``1`` where
        ``posterior >= pos_threshold`` (confident positive), else ``-1``
        (ambiguous / negative — caller decides).
    converged : bool
        Whether the EM fit converged.
    separated : bool
        Whether the two component means are meaningfully distinct (``True`` when
        ``|mean_0 - mean_1| > eps * pooled_std``). ``False`` flags a near-noise
        marker whose scores are effectively unimodal — the caller should skip it
        rather than manufacture labels from a single mode.
    """

    gmm: GaussianMixture
    remod_component: int
    posterior: NDArray
    hard_label: NDArray
    converged: bool
    separated: bool


def fit_gmm_labels(
    scores_1d: NDArray,
    pos_threshold: float = 0.8,
    n_components: int = 2,
    n_init: int = 5,
    random_state: int = 42,
    separation_eps: float = 2.0,
) -> GmmLabelResult:
    """Fit a two-component GMM to witness scores and derive confident labels.

    The lower-mean component is taken as the remodeled/perturbed mode (witness
    scores lean negative toward the perturbed reference). A cell is labeled a
    confident positive (``1``) when its posterior for that mode is at least
    ``pos_threshold``; otherwise ``-1``. Deterministic under ``random_state``.

    Parameters
    ----------
    scores_1d : NDArray
        Witness scores, shape ``(n,)`` (reshaped to ``(n, 1)`` internally).
    pos_threshold : float, optional
        Posterior of the remodeled component at or above which a cell is a
        confident positive. By default 0.8.
    n_components : int, optional
        Number of mixture components. By default 2.
    n_init : int, optional
        Number of EM initializations (best kept). By default 5.
    random_state : int, optional
        Seed for the EM initialization. By default 42.
    separation_eps : float, optional
        A marker is considered ``separated`` when the gap between the two
        component means exceeds ``separation_eps`` times their mean component
        standard deviation (a separation-to-width ratio). By default 2.0, so the
        two Gaussians must stand roughly two standard deviations apart — a true
        bimodal fit, not one unimodal blob split in half. By default 2.0.

    Returns
    -------
    GmmLabelResult
        The fitted mixture, remodeled-component index, per-cell posterior and
        gated ``hard_label``, and the ``converged`` / ``separated`` flags.
    """
    scores_1d = np.asarray(scores_1d, dtype=np.float64).ravel()
    X = scores_1d.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=n_init)
    gmm.fit(X)

    means = gmm.means_.ravel()
    remod_component = int(np.argmin(means))
    posterior = gmm.predict_proba(X)[:, remod_component]

    hard_label = np.full(len(scores_1d), -1, dtype=np.int8)
    hard_label[posterior >= pos_threshold] = 1

    # Separation ratio: mean gap relative to the components' own spread. A true
    # bimodal fit has the two Gaussians standing well apart from each other
    # (ratio large); splitting one unimodal blob in half gives overlapping
    # components whose gap is comparable to their width (ratio ~1-2).
    component_std = float(np.sqrt(gmm.covariances_.ravel()).mean())
    mean_gap = float(means.max() - means.min())
    separated = bool(mean_gap > separation_eps * component_std)

    return GmmLabelResult(
        gmm=gmm,
        remod_component=remod_component,
        posterior=posterior,
        hard_label=hard_label,
        converged=bool(gmm.converged_),
        separated=separated,
    )
