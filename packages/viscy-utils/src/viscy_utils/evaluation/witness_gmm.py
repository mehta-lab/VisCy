"""Gaussian-mixture gating of MMD-witness scores into confident pseudo-labels.

The MMD witness score (:func:`viscy_utils.evaluation.mmd.witness_function`) is a
scalar per cell measuring how far its embedding leans toward the perturbed
reference distribution. This module turns those 1-D scores into gated labels.

Two gates are provided, and the choice depends on whether the perturbed
population is genuinely **bimodal** or merely **shifted**:

- :func:`fit_gmm_labels` — two-component GMM on the perturbed scores; positive
  when the remodel-mode posterior clears a threshold. Right when perturbation
  creates a distinct second state. Reports a ``separated`` flag so a unimodal
  blob split in half can be rejected rather than labeled.
- :func:`fit_percentile_labels` — gate at a quantile of the **control** scores,
  fixing the control false-positive rate by construction. Assumes nothing and
  fits nothing, so it neither requires bimodality nor invents a second component
  when there is none. Right when perturbation shifts a unimodal population, and
  its free parameter (an error rate) means the same thing across markers, plates
  and timepoints — a posterior cut does not.

Both are pure NumPy/scikit-learn so they can be reused from application code and
standalone analysis scripts alike.
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
    n_fit_samples : int
        Number of witness scores used to fit the mixture. This can differ from
        ``len(posterior)`` when the mixture is fit on a separate calibration
        population (for example, balanced control + perturbed scores) and then
        applied to the requested cells.
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
    n_fit_samples: int

    @property
    def component_stds(self) -> NDArray:
        """One standard deviation per component, including tied covariance."""
        if self.gmm.covariance_type == "tied":
            value = float(np.sqrt(self.gmm.covariances_[0, 0]))
            return np.full(self.gmm.n_components, value, dtype=np.float64)
        covariance = self.gmm.covariances_.reshape(self.gmm.n_components, -1)
        return np.sqrt(covariance[:, 0])


def fit_gmm_labels(
    scores_1d: NDArray,
    pos_threshold: float = 0.8,
    n_components: int = 2,
    n_init: int = 5,
    random_state: int = 42,
    separation_eps: float = 2.0,
    fit_scores_1d: NDArray | None = None,
    covariance_type: str = "full",
) -> GmmLabelResult:
    """Fit a two-component GMM to witness scores and derive confident labels.

    The lower-mean component is taken as the remodeled/perturbed mode (witness
    scores lean negative toward the perturbed reference). A cell is labeled a
    confident positive (``1``) when its posterior for that mode is at least
    ``pos_threshold``; otherwise ``-1``. Deterministic under ``random_state``.

    Parameters
    ----------
    scores_1d : NDArray
        Witness scores to label, shape ``(n,)`` (reshaped to ``(n, 1)``
        internally).
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
    fit_scores_1d : NDArray or None, optional
        Separate witness scores on which to fit the GMM. The fitted model is then
        applied to ``scores_1d``. ``None`` preserves the original behavior and
        fits on the cells being labeled. This separation supports either the
        literal joint control+perturbed calibration population or a balanced
        sensitivity sample without duplicating or reordering output cells.
        Default: ``None``.
    covariance_type : {"full", "tied"}, optional
        Component variance model. ``"full"`` preserves the original independent
        1-D variances. ``"tied"`` shares one variance across both components,
        which guarantees a single monotonic posterior transition in 1-D and avoids
        a broad component capturing both distribution tails. Default: ``"full"``.

    Returns
    -------
    GmmLabelResult
        The fitted mixture, remodeled-component index, per-cell posterior and
        gated ``hard_label``, and the ``converged`` / ``separated`` flags.
    """
    scores_1d = np.asarray(scores_1d, dtype=np.float64).ravel()
    fit_scores = scores_1d if fit_scores_1d is None else np.asarray(fit_scores_1d, dtype=np.float64).ravel()
    if covariance_type not in ("full", "tied"):
        raise ValueError(f"covariance_type must be 'full' or 'tied', got {covariance_type!r}")
    if scores_1d.size == 0:
        raise ValueError("scores_1d must contain at least one score")
    if fit_scores.size < n_components:
        raise ValueError(f"fit_scores_1d must contain at least {n_components} scores; got {fit_scores.size}")
    if not np.isfinite(scores_1d).all() or not np.isfinite(fit_scores).all():
        raise ValueError("witness scores must all be finite")
    X = scores_1d.reshape(-1, 1)
    X_fit = fit_scores.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    gmm.fit(X_fit)

    means = gmm.means_.ravel()
    remod_component = int(np.argmin(means))
    posterior = gmm.predict_proba(X)[:, remod_component]

    hard_label = np.full(len(scores_1d), -1, dtype=np.int8)
    hard_label[posterior >= pos_threshold] = 1

    # Separation ratio: mean gap relative to the components' own spread. A true
    # bimodal fit has the two Gaussians standing well apart from each other
    # (ratio large); splitting one unimodal blob in half gives overlapping
    # components whose gap is comparable to their width (ratio ~1-2).
    if covariance_type == "tied":
        component_std = float(np.sqrt(gmm.covariances_[0, 0]))
    else:
        component_std = float(np.sqrt(gmm.covariances_.ravel()).mean())
    mean_gap = float(means.max() - means.min())
    separated = bool(mean_gap > separation_eps * component_std)

    # Likelihood-based model-selection scores. bic_1comp is a single-component fit
    # to the same data, so the caller can read the ΔBIC (bic_1comp - bic) as an
    # independent cross-check on the `separated` heuristic.
    gmm_1 = GaussianMixture(
        n_components=1,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    ).fit(X_fit)

    return GmmLabelResult(
        gmm=gmm,
        remod_component=remod_component,
        posterior=posterior,
        hard_label=hard_label,
        converged=bool(gmm.converged_),
        separated=separated,
        bic=float(gmm.bic(X_fit)),
        aic=float(gmm.aic(X_fit)),
        bic_1comp=float(gmm_1.bic(X_fit)),
        n_fit_samples=int(fit_scores.size),
    )


@dataclass
class PercentileLabelResult:
    """Result of gating witness scores at a control-calibrated percentile.

    A cell is called positive when its witness score falls below ``gate``, the
    ``control_fp_target`` quantile of the **control** scores. The false-positive
    rate on controls is therefore fixed *by construction* rather than measured
    after the fact, and the free parameter is an interpretable error rate that
    means the same thing on every marker, plate and timepoint — unlike a
    posterior cut, whose operating point drifts with the fitted mixture.

    Unlike :func:`fit_gmm_labels` this makes **no distributional assumption and
    fits nothing**. It neither requires the perturbed scores to be bimodal nor
    invents a second component when they are not, which is the right frame when
    perturbation shifts a unimodal population rather than splitting it.

    Both cuts are quantiles of the **control** distribution, but they are not
    mirror images of each other, because the two tails are not comparable. The
    positive cut sits in the control's lower tail, where perturbed cells are
    dense. The negative cut sits at ``negative_quantile`` — low on the control
    distribution, marking where control mass actually begins — because the
    perturbed scores typically die out well before the control's *upper* tail; a
    mirrored ``1 - control_fp_target`` cut would label almost nothing (measured:
    39 of 29,652 cells on 04_14 SEC61B).

    Cells between the two cuts resemble neither reference confidently and are
    left unlabeled. That band matters: the perturbed histogram is continuous
    across the positive gate, so without a separate negative cut a cell just
    above the gate — indistinguishable from a positive — would become a confident
    negative on no evidence.

    Parameters
    ----------
    gate : float
        Positive cut: the ``control_fp_target`` quantile of control scores. Cells
        below it are positive (witness leans negative = perturbed).
    neg_gate : float
        Negative cut: the ``negative_quantile`` quantile of control scores. Cells
        above it have reached the range where controls actually live.
    posterior : NDArray
        Per cell, in ``[0, 1]``: how far past ``gate`` the score sits, normalized
        by the distance from ``gate`` to the most extreme score. This is a graded
        **ordering**, not a probability — it carries no distributional claim, and
        is intended as a per-sample confidence weight for downstream training.
        ``0`` for cells at or above the gate.
    hard_label : NDArray
        Per cell, int8: ``1`` below ``gate`` (positive), ``0`` above ``neg_gate``
        (negative), ``-1`` in between (ambiguous — caller drops these).
    control_fp : float
        Realized control false-positive rate at ``gate`` (≈ the target; exact
        equality is limited by ties and sample size).
    """

    gate: float
    neg_gate: float
    posterior: NDArray
    hard_label: NDArray
    control_fp: float


def fit_percentile_labels(
    scores: NDArray,
    control_scores: NDArray,
    control_fp_target: float = 0.02,
    negative_quantile: float = 0.10,
) -> PercentileLabelResult:
    """Gate witness scores against two control-calibrated percentiles.

    ``gate = quantile(control, control_fp_target)`` and
    ``neg_gate = quantile(control, negative_quantile)``. A cell is positive below
    ``gate``, negative above ``neg_gate``, and ambiguous in between.

    The two cuts are deliberately **not** mirror images. ``gate`` sits in the
    control's lower tail, where perturbed cells are dense, so it fixes the
    control false-positive rate. ``neg_gate`` sits low on the control
    distribution — where control mass begins — because perturbed scores usually
    die out before the control's upper tail, so a mirrored ``1 - target`` cut
    labels almost nothing (measured: 39 of 29,652 cells on 04_14 SEC61B).

    The middle band matters: the perturbed score distribution is usually
    continuous across ``gate`` with no valley there, so labeling everything above
    ``gate`` as negative would turn cells indistinguishable from positives into
    confident negatives. Those cells resemble neither reference and are excluded.

    The returned ``posterior`` grades each cell by how far beyond ``gate`` it
    sits, scaled to ``[0, 1]`` by the most extreme score present. It is a rank-like
    confidence with no distributional assumption behind it — deliberately not a
    probability, since under a pure distribution shift there is no second
    population for a probability to refer to.

    Parameters
    ----------
    scores : NDArray
        Witness scores to gate, shape ``(n,)``. Pass **all** cells (control and
        perturbed alike) to label them under one rule.
    control_scores : NDArray
        Witness scores of the control reference, shape ``(m,)``. Both cuts are
        quantiles of this distribution.
    control_fp_target : float, optional
        Fraction of control cells called positive, by construction. Default 0.02.
    negative_quantile : float, optional
        Control quantile above which a cell is called negative — where the control
        distribution begins. Must exceed ``control_fp_target``. Default 0.10.

    Returns
    -------
    PercentileLabelResult
    """
    if negative_quantile <= control_fp_target:
        raise ValueError(
            f"negative_quantile ({negative_quantile}) must exceed control_fp_target "
            f"({control_fp_target}); otherwise the positive and negative regions overlap."
        )
    s = np.asarray(scores, dtype=np.float64).ravel()
    c = np.asarray(control_scores, dtype=np.float64).ravel()

    gate = float(np.quantile(c, control_fp_target))
    neg_gate = float(np.quantile(c, negative_quantile))

    hard_label = np.full(len(s), -1, dtype=np.int8)  # ambiguous by default
    hard_label[s < gate] = 1
    hard_label[s > neg_gate] = 0

    # Graded confidence by distance past the gate, normalized by the furthest
    # score seen. Assumption-free: pure ordering, no density model. Cells at or
    # above the gate get 0.
    span = gate - float(s.min()) if s.size else 0.0
    if span > 0:
        posterior = np.clip((gate - s) / span, 0.0, 1.0)
    else:
        posterior = np.zeros_like(s)

    control_fp = float((c < gate).mean()) if c.size else 0.0

    return PercentileLabelResult(
        gate=gate,
        neg_gate=neg_gate,
        posterior=posterior,
        hard_label=hard_label,
        control_fp=control_fp,
    )
