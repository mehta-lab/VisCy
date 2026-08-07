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
    bic : float
        Bayesian information criterion of the fitted mixture (lower is better).
    aic : float
        Akaike information criterion of the fitted mixture (lower is better).
    bic_1comp : float
        BIC of a single-component (unimodal) fit to the same scores. A large
        ``bic_1comp - bic`` corroborates that two components are justified — an
        independent, likelihood-based cross-check on the ``separated`` heuristic.
        Note: at large ``n`` the likelihood term dominates the parameter penalty,
        so 2 components almost always wins; read the *gap*, not the sign.
    """

    gmm: GaussianMixture
    remod_component: int
    posterior: NDArray
    hard_label: NDArray
    converged: bool
    separated: bool
    bic: float
    aic: float
    bic_1comp: float


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

    # Likelihood-based model-selection scores. bic_1comp is a single-component fit
    # to the same data, so the caller can read the ΔBIC (bic_1comp - bic) as an
    # independent cross-check on the `separated` heuristic.
    gmm_1 = GaussianMixture(n_components=1, random_state=random_state, n_init=n_init).fit(X)

    return GmmLabelResult(
        gmm=gmm,
        remod_component=remod_component,
        posterior=posterior,
        hard_label=hard_label,
        converged=bool(gmm.converged_),
        separated=separated,
        bic=float(gmm.bic(X)),
        aic=float(gmm.aic(X)),
        bic_1comp=float(gmm_1.bic(X)),
    )


def _gaussian_pdf(x: NDArray, mu: float, sigma: float) -> NDArray:
    """1-D Gaussian density (sigma floored to avoid divide-by-zero)."""
    sigma = max(float(sigma), 1e-9)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


@dataclass
class ControlAnchoredResult:
    """Result of gating perturbed witness scores against a control-anchored baseline.

    Models the perturbed scores as a two-component mixture
    ``perturbed = pi * N(mu_c, sigma_c) + (1 - pi) * N(mu_r, sigma_r)`` where the
    baseline component is **frozen to the control distribution** and only the
    remodel component + mixing weight are fit. "Remodel" therefore means *excess
    over the control baseline*, not one of two modes discovered within the
    perturbed cells — the right frame when infection shifts the proportions of a
    shared state-space rather than creating a distinct new state.

    Parameters
    ----------
    mu_c, sigma_c : float
        Frozen baseline Gaussian, fit to the control witness scores.
    mu_r, sigma_r : float
        Fitted remodel Gaussian (the perturbed-enriched excess).
    pi_baseline : float
        Estimated fraction of perturbed cells still explained by the baseline
        component; ``1 - pi_baseline`` is the remodeled fraction. Weakly
        identified when the shift is small — treat as approximate.
    posterior : NDArray
        Per perturbed cell: P(remodel | score), shape ``(n_perturbed,)``.
    hard_label : NDArray
        Per perturbed cell, int8: ``1`` where ``posterior >= threshold`` (remodel),
        else ``-1``.
    threshold : float
        Posterior threshold, calibrated so the control false-positive rate equals
        ``control_fp_target``.
    control_fp : float
        Realized control false-positive rate at ``threshold`` (≈ the target).
    converged : bool
        Whether the constrained EM converged.
    """

    mu_c: float
    sigma_c: float
    mu_r: float
    sigma_r: float
    pi_baseline: float
    posterior: NDArray
    hard_label: NDArray
    threshold: float
    control_fp: float
    converged: bool


def fit_control_anchored_labels(
    perturbed_scores: NDArray,
    control_scores: NDArray,
    control_fp_target: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> ControlAnchoredResult:
    """Gate perturbed witness scores against a control-anchored baseline.

    The baseline Gaussian is frozen to the control scores' mean/std; the perturbed
    scores are then modeled as ``pi * baseline + (1 - pi) * remodel`` and only the
    remodel Gaussian + mixing weight ``pi`` are fit by a constrained EM (the
    baseline component's parameters never move). A perturbed cell's remodel
    posterior is thresholded, with the threshold **calibrated so the control
    false-positive rate equals ``control_fp_target``** — a controlled error rate
    rather than a fixed posterior cut. Unlike :func:`fit_gmm_labels`, this never
    abstains on a unimodal perturbed distribution: it measures the *excess* over
    baseline, so a subtle proportion shift still yields labels (with a small
    remodeled fraction).

    Parameters
    ----------
    perturbed_scores : NDArray
        Witness scores of the condition's perturbed cells, shape ``(n,)``.
    control_scores : NDArray
        Witness scores of the (time-matched) control-reference cells, shape ``(m,)``.
    control_fp_target : float, optional
        Target control false-positive rate; the posterior threshold is set so this
        fraction of control cells is called remodel. By default 0.05.
    max_iter : int, optional
        Max constrained-EM iterations. By default 200.
    tol : float, optional
        Log-likelihood convergence tolerance. By default 1e-6.

    Returns
    -------
    ControlAnchoredResult
    """
    p = np.asarray(perturbed_scores, dtype=np.float64).ravel()
    c = np.asarray(control_scores, dtype=np.float64).ravel()

    # Frozen baseline from control.
    mu_c = float(c.mean())
    sigma_c = float(c.std())

    # Initialize the remodel component on the side of the perturbed cloud away from
    # control (witness is negative = perturbed-leaning, so init below the mean).
    mu_r = float(p.mean() - p.std())
    sigma_r = float(p.std())
    pi = 0.5  # baseline weight

    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        base = pi * _gaussian_pdf(p, mu_c, sigma_c)
        rem = (1 - pi) * _gaussian_pdf(p, mu_r, sigma_r)
        total = base + rem + 1e-300
        resp_r = rem / total  # responsibility of the remodel component
        # M-step — update ONLY the remodel component and the mixing weight.
        nr = resp_r.sum()
        if nr > 1e-6:
            mu_r = float((resp_r * p).sum() / nr)
            sigma_r = float(np.sqrt((resp_r * (p - mu_r) ** 2).sum() / nr))
            sigma_r = max(sigma_r, 1e-6)
        pi = float(1.0 - nr / len(p))
        pi = min(max(pi, 1e-6), 1 - 1e-6)
        ll = float(np.log(total).sum())
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    def _posterior(x: NDArray) -> NDArray:
        base = pi * _gaussian_pdf(x, mu_c, sigma_c)
        rem = (1 - pi) * _gaussian_pdf(x, mu_r, sigma_r)
        return rem / (base + rem + 1e-300)

    post_p = _posterior(p)
    post_c = _posterior(c)
    # Calibrate the threshold so the control FP rate matches the target: the
    # (1 - target) quantile of the control posteriors.
    threshold = float(np.quantile(post_c, 1.0 - control_fp_target))
    control_fp = float((post_c >= threshold).mean())

    hard_label = np.full(len(p), -1, dtype=np.int8)
    hard_label[post_p >= threshold] = 1

    return ControlAnchoredResult(
        mu_c=mu_c,
        sigma_c=sigma_c,
        mu_r=mu_r,
        sigma_r=sigma_r,
        pi_baseline=pi,
        posterior=post_p,
        hard_label=hard_label,
        threshold=threshold,
        control_fp=control_fp,
        converged=converged,
    )
