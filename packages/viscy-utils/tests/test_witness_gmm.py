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


def test_control_anchored_labels_calibrated_fp():
    """Control-anchored gate: labels the excess over baseline and holds the
    control false-positive rate near the target, without needing bimodality."""
    from viscy_utils.evaluation.witness_gmm import fit_control_anchored_labels

    rng = np.random.default_rng(0)
    control = rng.normal(0.0, 1.0, 4000)
    # Perturbed = mostly baseline + a shifted (remodel) minority (subtle, overlapping).
    perturbed = np.concatenate([rng.normal(0.0, 1.0, 3000), rng.normal(-2.0, 1.0, 1000)])
    res = fit_control_anchored_labels(perturbed, control, control_fp_target=0.05)
    # FP rate is calibrated to the target.
    assert abs(res.control_fp - 0.05) < 0.02
    # Some perturbed cells are labeled remodel; remodel fraction is positive and < 1.
    assert 0.0 < (1.0 - res.pi_baseline) < 1.0
    assert (res.hard_label == 1).any()
    # Remodel component sits below (more negative than) the control baseline.
    assert res.mu_r < res.mu_c


def test_control_anchored_no_shift_low_positives():
    """When perturbed == control (no real shift), few cells clear the calibrated
    threshold — the FP-calibrated cut keeps the positive rate near the target."""
    from viscy_utils.evaluation.witness_gmm import fit_control_anchored_labels

    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 1.0, 4000)
    perturbed = rng.normal(0.0, 1.0, 4000)  # identical distribution
    res = fit_control_anchored_labels(perturbed, control, control_fp_target=0.05)
    # No real excess → perturbed positive rate stays near the target FP (~5%).
    assert (res.hard_label == 1).mean() < 0.15
