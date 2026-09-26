"""Tests for :mod:`dynacell.evaluation.feature_metrics`.

Runs the real :func:`compute_feature_similarity` / :func:`compute_feature_similarity_pairwise`
end-to-end with tiny inputs to keep the suite fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynacell.evaluation.feature_metrics import (
    compute_feature_similarity,
    compute_feature_similarity_pairwise,
)


def test_identical_inputs_give_zero_distances() -> None:
    """Same array on both sides -> all distance metrics collapse to ~0, similarities to ~1."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 32)).astype(np.float32)

    result = compute_feature_similarity(
        x,
        x,
        "CP",
        kid_subsets=20,
        kid_subset_size=50,
        prc_bootstrap_subsets=20,
        mind_num_projections=200,
    )

    assert result["CP_FID"] == pytest.approx(0.0, abs=1e-4)
    # KID on identical pools is still nonzero from the bootstrap subset sampling and the
    # poly-kernel bias term; tolerate up to 0.1 in absolute value (shifted case >> 1).
    assert abs(result["CP_KID"]) < 0.1
    assert np.isfinite(result["CP_KID_std"])
    # PRC with bootstrap resampling-with-replacement on identical pools still drops
    # ~5-15% of unique rows per draw, so per-bootstrap precision/recall < 1.0; the
    # bootstrap mean over 20 draws lands around 0.9.
    assert result["CP_Precision"] > 0.85
    assert result["CP_Recall"] > 0.85
    assert result["CP_F1"] > 0.85
    assert result["CP_MIND"] < 1e-3
    assert result["CP_Median_Cosine_Similarity"] == pytest.approx(1.0)


def test_shifted_distributions_give_nonzero_metrics() -> None:
    """Mean-shift between pred and target -> nonzero FID/KID/MIND."""
    rng = np.random.default_rng(0)
    n, d = 200, 32
    target = rng.standard_normal((n, d)).astype(np.float32)
    pred = target + 1.0

    result = compute_feature_similarity(
        pred,
        target,
        "CP",
        kid_subsets=20,
        kid_subset_size=50,
        prc_bootstrap_subsets=20,
        mind_num_projections=200,
    )

    assert result["CP_FID"] > 1.0
    assert result["CP_KID"] > 0.0
    assert result["CP_KID_std"] > 0.0
    assert result["CP_MIND"] > 0.0


def test_prc_bootstrap_std_is_nonzero() -> None:
    """Bootstrap actually resamples -> Precision/Recall/F1 std > 0 on non-trivial inputs."""
    rng = np.random.default_rng(1)
    n, d = 200, 32
    target = rng.standard_normal((n, d)).astype(np.float32)
    pred = target + 0.5 * rng.standard_normal((n, d)).astype(np.float32)

    result = compute_feature_similarity(
        pred,
        target,
        "CP",
        kid_subsets=20,
        kid_subset_size=50,
        prc_bootstrap_subsets=20,
        mind_num_projections=200,
    )

    assert result["CP_Precision_std"] > 0.0
    assert result["CP_Recall_std"] > 0.0
    assert result["CP_F1_std"] > 0.0


def test_prc_asymmetric_high_precision_low_recall() -> None:
    """Pred covers only one of target's two clusters -> high precision, lower recall.

    Target has two well-separated Gaussian clusters; pred samples only the first.
    Generated samples land inside the real manifold (high precision) but the second
    real cluster has no nearby generated points (low recall). Guards against an
    accidental ``features_1``/``features_2`` swap in the wrapper, which would
    invert the gap. Row counts are matched so the in-tree median cosine helper
    (which requires aligned rows) does not crash.
    """
    rng = np.random.default_rng(0)
    cluster_a = rng.standard_normal((100, 8)).astype(np.float32)
    cluster_b = rng.standard_normal((100, 8)).astype(np.float32) + 5.0
    target = np.vstack([cluster_a, cluster_b])
    pred = rng.standard_normal((200, 8)).astype(np.float32)

    result = compute_feature_similarity(
        pred,
        target,
        "CP",
        prc_bootstrap_subsets=20,
        kid_subsets=20,
        kid_subset_size=50,
        mind_num_projections=100,
    )

    assert result["CP_Precision"] - result["CP_Recall"] > 0.05


def test_kid_small_cohort_returns_nan() -> None:
    """Effective KID subset < 16 -> KID mean/std NaN, other metrics remain finite."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 8)).astype(np.float32)

    result = compute_feature_similarity(
        x,
        x,
        "CP",
        kid_subsets=5,
        kid_subset_size=1000,
        prc_bootstrap_subsets=5,
        mind_num_projections=50,
    )

    assert np.isnan(result["CP_KID"])
    assert np.isnan(result["CP_KID_std"])
    assert np.isfinite(result["CP_FID"])
    assert np.isfinite(result["CP_MIND"])
    assert np.isfinite(result["CP_Precision"])
    assert np.isfinite(result["CP_Recall"])
    assert np.isfinite(result["CP_F1"])
    assert np.isfinite(result["CP_Median_Cosine_Similarity"])


def test_empty_arrays_return_all_nan() -> None:
    """Empty inputs -> all 11 metric keys present, all NaN."""
    empty = np.empty((0, 0), dtype=np.float32)
    result = compute_feature_similarity(empty, empty, "CP")

    expected_keys = {
        "CP_FID",
        "CP_KID",
        "CP_KID_std",
        "CP_Precision",
        "CP_Precision_std",
        "CP_Recall",
        "CP_Recall_std",
        "CP_F1",
        "CP_F1_std",
        "CP_MIND",
        "CP_Median_Cosine_Similarity",
    }
    assert set(result.keys()) == expected_keys
    assert np.isnan(np.array(list(result.values()))).all()


def test_feature_dim_mismatch_raises() -> None:
    """Mismatched feature dims -> ValueError."""
    pred = np.zeros((10, 8), dtype=np.float32)
    target = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="dim mismatch|Feature dim"):
        compute_feature_similarity(pred, target, "CP")


def test_pairwise_variant_returns_four_metrics() -> None:
    """Pairwise variant exposes only FID, KID (mean+std), and cosine."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 16)).astype(np.float32)

    result = compute_feature_similarity_pairwise(
        x,
        x,
        "CP",
        kid_subsets=20,
        kid_subset_size=20,
    )

    assert set(result.keys()) == {
        "CP_FID",
        "CP_KID",
        "CP_KID_std",
        "CP_Median_Cosine_Similarity",
    }
    assert result["CP_FID"] == pytest.approx(0.0, abs=1e-4)
    # KID on identical pools is still nonzero from the bootstrap subset sampling and the
    # poly-kernel bias term; tolerate the small-subset magnitude.
    assert abs(result["CP_KID"]) < 0.5
    assert result["CP_Median_Cosine_Similarity"] == pytest.approx(1.0)


def test_pairwise_empty_returns_all_nan() -> None:
    """Pairwise variant on empty inputs -> 4 NaN keys with the requested prefix."""
    empty = np.empty((0, 0), dtype=np.float32)
    result = compute_feature_similarity_pairwise(empty, empty, "DINOv3")

    assert set(result.keys()) == {
        "DINOv3_FID",
        "DINOv3_KID",
        "DINOv3_KID_std",
        "DINOv3_Median_Cosine_Similarity",
    }
    assert np.isnan(np.array(list(result.values()))).all()


def test_seed_reproducibility() -> None:
    """Same seed -> identical PRC means; different seed -> at least one PRC mean differs."""
    rng = np.random.default_rng(2)
    n, d = 200, 16
    target = rng.standard_normal((n, d)).astype(np.float32)
    pred = target + 0.5 * rng.standard_normal((n, d)).astype(np.float32)

    kwargs = dict(
        prefix="CP",
        kid_subsets=20,
        kid_subset_size=50,
        prc_bootstrap_subsets=20,
        mind_num_projections=200,
    )

    result_a = compute_feature_similarity(pred, target, rng_seed=2020, **kwargs)
    result_b = compute_feature_similarity(pred, target, rng_seed=2020, **kwargs)
    result_c = compute_feature_similarity(pred, target, rng_seed=4040, **kwargs)

    assert result_a["CP_Precision"] == result_b["CP_Precision"]
    assert result_a["CP_Recall"] == result_b["CP_Recall"]
    assert result_a["CP_F1"] == result_b["CP_F1"]

    assert (
        result_a["CP_Precision"] != result_c["CP_Precision"]
        or result_a["CP_Recall"] != result_c["CP_Recall"]
        or result_a["CP_F1"] != result_c["CP_F1"]
    )


_FID_KEYS = {"CP_FID"}
_PRC_KEYS = {"CP_Precision", "CP_Precision_std", "CP_Recall", "CP_Recall_std", "CP_F1", "CP_F1_std"}
_MIND_KEYS = {"CP_MIND"}


@pytest.mark.parametrize(
    ("flag", "dropped"),
    [("compute_fid", _FID_KEYS), ("compute_prc", _PRC_KEYS), ("compute_mind", _MIND_KEYS)],
)
def test_disabled_metric_is_omitted_and_others_unchanged(flag: str, dropped: set[str]) -> None:
    """A switched-off metric loses its keys; every remaining value is bit-identical."""
    rng = np.random.default_rng(3)
    target = rng.standard_normal((120, 16)).astype(np.float32)
    pred = target + 0.5 * rng.standard_normal((120, 16)).astype(np.float32)
    kwargs = dict(prefix="CP", kid_subsets=20, kid_subset_size=50, prc_bootstrap_subsets=10, mind_num_projections=100)

    full = compute_feature_similarity(pred, target, **kwargs)
    reduced = compute_feature_similarity(pred, target, **kwargs, **{flag: False})

    assert set(reduced) == set(full) - dropped
    # Remaining columns keep their relative order and exact values.
    assert list(reduced) == [k for k in full if k not in dropped]
    for key, value in reduced.items():
        assert value == full[key], key


def test_all_flags_on_preserves_column_order() -> None:
    """Defaults emit the historical 11-column order the saved CSVs carry."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 8)).astype(np.float32)
    result = compute_feature_similarity(
        x, x, "CP", kid_subsets=5, kid_subset_size=20, prc_bootstrap_subsets=5, mind_num_projections=20
    )
    expected = [
        "CP_FID",
        "CP_KID",
        "CP_KID_std",
        "CP_Precision",
        "CP_Precision_std",
        "CP_Recall",
        "CP_Recall_std",
        "CP_F1",
        "CP_F1_std",
        "CP_MIND",
        "CP_Median_Cosine_Similarity",
    ]
    assert list(result) == expected
    assert list(compute_feature_similarity(np.empty((0, 0)), np.empty((0, 0)), "CP")) == expected


def test_empty_input_honours_flags() -> None:
    """The all-NaN path carries exactly the enabled keys, matching a populated call."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((60, 8)).astype(np.float32)
    flags = dict(compute_fid=False, compute_prc=False, compute_mind=False)
    empty = compute_feature_similarity(np.empty((0, 0)), np.empty((0, 0)), "CP", **flags)
    populated = compute_feature_similarity(x, x, "CP", kid_subsets=5, kid_subset_size=20, **flags)

    assert list(empty) == list(populated) == ["CP_KID", "CP_KID_std", "CP_Median_Cosine_Similarity"]
    assert np.isnan(np.array(list(empty.values()))).all()


def test_pairwise_fid_flag() -> None:
    """``compute_fid=False`` drops the per-timepoint FID; KID/cosine are bit-identical."""
    rng = np.random.default_rng(6)
    target = rng.standard_normal((50, 16)).astype(np.float32)
    pred = target + 0.3 * rng.standard_normal((50, 16)).astype(np.float32)

    full = compute_feature_similarity_pairwise(pred, target, "CP", kid_subsets=20, kid_subset_size=20)
    reduced = compute_feature_similarity_pairwise(
        pred, target, "CP", kid_subsets=20, kid_subset_size=20, compute_fid=False
    )
    empty = compute_feature_similarity_pairwise(np.empty((0, 0)), np.empty((0, 0)), "CP", compute_fid=False)

    assert list(reduced) == ["CP_KID", "CP_KID_std", "CP_Median_Cosine_Similarity"]
    assert list(empty) == list(reduced)
    for key, value in reduced.items():
        assert value == full[key], key
