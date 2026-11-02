"""Tests for GMM gating of witness scores (:mod:`viscy_utils.evaluation.witness_gmm`)."""

import numpy as np

from viscy_utils.evaluation.witness_gmm import fit_gmm_labels


def _bimodal_scores(n=500, seed=0):
    """Two well-separated modes: a negative (remodeled) mode and a positive one."""
    rng = np.random.default_rng(seed)
    remod = rng.normal(-3.0, 0.3, n)  # perturbed / remodeled — more negative
    unaff = rng.normal(3.0, 0.3, n)
    return np.concatenate([remod, unaff]), n


def test_fit_gmm_labels_two_modes():
    scores, n = _bimodal_scores()
    res = fit_gmm_labels(scores, pos_threshold=0.8)

    # The remodeled component is the lower-mean mode.
    means = res.gmm.means_.ravel()
    assert res.remod_component == int(np.argmin(means))
    assert res.separated
    assert res.converged

    # The negative-mode cells (first half) should be the confident positives.
    assert res.hard_label[:n].mean() > 0.9  # ~all labeled +1
    assert res.hard_label[n:].mean() < -0.9  # ~all labeled -1
    # Posterior of the remodeled mode is high for the negative cells, low for positive.
    assert res.posterior[:n].mean() > 0.9
    assert res.posterior[n:].mean() < 0.1


def test_fit_gmm_labels_unimodal_not_separated():
    rng = np.random.default_rng(1)
    scores = rng.normal(0.0, 1.0, 1000)  # single mode / noise
    res = fit_gmm_labels(scores)
    assert not res.separated


def test_fit_gmm_labels_deterministic():
    scores, _ = _bimodal_scores(seed=2)
    a = fit_gmm_labels(scores, random_state=7)
    b = fit_gmm_labels(scores, random_state=7)
    np.testing.assert_array_equal(a.hard_label, b.hard_label)
    np.testing.assert_allclose(a.posterior, b.posterior)


def test_fit_gmm_labels_threshold_strictness():
    scores, n = _bimodal_scores()
    lax = fit_gmm_labels(scores, pos_threshold=0.5)
    strict = fit_gmm_labels(scores, pos_threshold=0.99)
    # A stricter posterior bar yields no more confident positives than a lax one.
    assert (strict.hard_label == 1).sum() <= (lax.hard_label == 1).sum()


def test_fit_gmm_labels_can_fit_a_separate_pooled_population():
    """A control+perturbed calibration pool can anchor a missing control-like mode.

    The scored population contains only the negative perturbation mode. Fitting
    that population alone would ask a two-component GMM to split one cloud;
    fitting a balanced pool recovers the biological negative/control modes while
    still returning one posterior per requested perturbation cell.
    """
    rng = np.random.default_rng(11)
    control = rng.normal(1.0, 0.15, 600)
    perturbed = rng.normal(-1.0, 0.15, 80)
    pooled = np.concatenate([control[: len(perturbed)], perturbed])

    res = fit_gmm_labels(
        perturbed,
        fit_scores_1d=pooled,
        pos_threshold=0.8,
        random_state=4,
    )

    assert res.n_fit_samples == len(pooled)
    assert len(res.posterior) == len(perturbed)
    assert res.separated
    assert res.posterior.mean() > 0.99
    np.testing.assert_allclose(np.sort(res.gmm.means_.ravel()), [-1.0, 1.0], atol=0.08)


def test_tied_covariance_has_one_monotonic_posterior_transition():
    rng = np.random.default_rng(12)
    control = rng.normal(0.22, 0.03, 1000)
    perturbed = rng.normal(-0.10, 0.13, 1000)
    scores = np.linspace(-0.5, 0.5, 2000)
    result = fit_gmm_labels(
        scores,
        fit_scores_1d=np.concatenate([control, perturbed]),
        covariance_type="tied",
    )
    posterior = result.posterior
    assert np.all(np.diff(posterior) <= 1e-12)
    assert len(result.component_stds) == 2
    assert result.component_stds[0] == result.component_stds[1]


def test_percentile_gate_labels_a_pure_shift():
    """A unimodal SHIFTED perturbed population still gets labeled — the case that
    breaks the GMM (no second mode to find) and the reason this gate exists.

    Also pins the FP calibration: the gate is a quantile of the control scores, so
    the control false-positive rate matches the target by construction.
    """
    from viscy_utils.evaluation.witness_gmm import fit_gmm_labels, fit_percentile_labels

    rng = np.random.default_rng(2)
    control = rng.normal(0.0, 1.0, 4000)
    # One Gaussian, shifted — no subpopulation, no bimodality anywhere.
    perturbed = rng.normal(-1.0, 1.0, 4000)

    # The GMM has no honest split to find here and says so via `separated`.
    assert not fit_gmm_labels(perturbed).separated

    res = fit_percentile_labels(perturbed, control, control_fp_target=0.02)
    assert abs(res.control_fp - 0.02) < 0.005
    # A 1-sigma shift puts far more than the 2% control rate past the gate.
    assert (res.hard_label == 1).mean() > 0.10
    # Three-way: positives below `gate`, negatives above `neg_gate`, and a middle
    # band left unlabeled. Without the separate negative cut, cells just above the
    # positive gate — indistinguishable from positives — would become confident
    # negatives.
    assert res.gate < res.neg_gate
    assert ((perturbed < res.gate) == (res.hard_label == 1)).all()
    assert ((perturbed > res.neg_gate) == (res.hard_label == 0)).all()
    assert (res.hard_label == -1).any()


def test_percentile_posterior_is_graded_and_bounded():
    """The posterior is a bounded, monotone ordering (it feeds downstream CE
    weighting) and is 0 for every cell at or above the gate."""
    from viscy_utils.evaluation.witness_gmm import fit_percentile_labels

    rng = np.random.default_rng(3)
    control = rng.normal(0.0, 1.0, 2000)
    perturbed = rng.normal(-1.5, 1.0, 2000)
    res = fit_percentile_labels(perturbed, control, control_fp_target=0.02)

    assert res.posterior.min() >= 0.0 and res.posterior.max() <= 1.0
    # Non-positives carry no weight.
    assert (res.posterior[res.hard_label == -1] == 0.0).all()
    # Monotone in the score: the most extreme cell outranks a marginal one.
    order = np.argsort(perturbed)
    assert res.posterior[order[0]] >= res.posterior[order[len(order) // 2]]
