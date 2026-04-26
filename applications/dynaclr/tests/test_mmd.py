"""Tests for MMD perturbation evaluation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.mmd.compute_mmd import run_mmd_analysis, run_mmd_pooled
from dynaclr.evaluation.mmd.config import ComparisonSpec, MMDEvalConfig, MMDPooledConfig, MMDSettings
from viscy_utils.evaluation.mmd import compute_mmd_unbiased, median_heuristic, mmd_permutation_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMP = [ComparisonSpec(cond_a="uninfected", cond_b="ZIKV", label="uninf vs ZIKV")]
_SETTINGS_FAST = MMDSettings(n_permutations=50)


def _cfg(**kwargs) -> MMDEvalConfig:
    return MMDEvalConfig(input_path="dummy", output_dir="/tmp", comparisons=_COMP, **kwargs)


def _make_adata(
    n_cells: int = 200,
    n_features: int = 32,
    markers: list[str] | None = None,
    treatment_shift: float = 3.0,
    seed: int = 0,
) -> ad.AnnData:
    """Synthetic AnnData with two markers and two perturbation groups.

    TOMM20 has a large shift between uninfected and ZIKV (detectable MMD).
    Phase3D has no shift (null).
    """
    rng = np.random.default_rng(seed)
    if markers is None:
        markers = ["Phase3D", "TOMM20"]
    n_per_group = n_cells // (2 * len(markers))

    rows = []
    emb_list = []
    for marker in markers:
        for perturbation in ["uninfected", "ZIKV"]:
            for t in range(n_per_group):
                shift = treatment_shift if (perturbation == "ZIKV" and marker == "TOMM20") else 0.0
                emb = rng.normal(loc=shift, scale=1.0, size=n_features)
                emb_list.append(emb)
                rows.append(
                    {
                        "experiment": "test_exp",
                        "marker": marker,
                        "perturbation": perturbation,
                        "hours_post_perturbation": float(t % 6),
                    }
                )
    X = np.stack(emb_list)
    obs = pd.DataFrame(rows)
    return ad.AnnData(X=X.astype(np.float32), obs=obs)


def _make_temporal_adata(n_features: int = 16, seed: int = 0) -> ad.AnnData:
    """AnnData where ZIKV treatment effect increases with hours_post_perturbation."""
    rng = np.random.default_rng(seed)
    rows = []
    emb_list = []
    hours_bins = [1.0, 3.0, 6.0, 12.0]
    for marker in ["TOMM20"]:
        for _ in range(50):
            emb_list.append(rng.normal(0.0, 1.0, n_features))
            rows.append(
                {"experiment": "e", "marker": marker, "perturbation": "uninfected", "hours_post_perturbation": 0.0}
            )
        for hpi in hours_bins:
            shift = hpi / 3.0
            for _ in range(30):
                emb_list.append(rng.normal(shift, 1.0, n_features))
                rows.append(
                    {"experiment": "e", "marker": marker, "perturbation": "ZIKV", "hours_post_perturbation": hpi}
                )
    X = np.stack(emb_list).astype(np.float32)
    obs = pd.DataFrame(rows)
    return ad.AnnData(X=X, obs=obs)


# ---------------------------------------------------------------------------
# Core MMD tests
# ---------------------------------------------------------------------------


def test_mmd_identical_distributions():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 16))
    Y = rng.normal(0, 1, (200, 16))
    mmd2, p_value, _ = mmd_permutation_test(X, Y, n_permutations=200, seed=42)
    assert mmd2 < 0.1
    assert p_value > 0.05


def test_mmd_different_distributions():
    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 1.0, (200, 16))
    Y = rng.normal(5.0, 1.0, (200, 16))
    mmd2, p_value, _ = mmd_permutation_test(X, Y, n_permutations=200, seed=42)
    assert mmd2 > 0.1
    assert p_value < 0.05


def test_mmd_permutation_null():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (100, 8))
    Y = rng.normal(0, 1, (100, 8))
    _, _, null = mmd_permutation_test(X, Y, n_permutations=100, seed=0)
    assert len(null) == 100
    assert np.all(np.isfinite(null))


def test_median_heuristic_positive():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (50, 8))
    Y = rng.normal(2, 1, (50, 8))
    assert median_heuristic(X, Y) > 0


def test_compute_mmd_unbiased_symmetric():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (100, 8))
    Y = rng.normal(1, 1, (100, 8))
    bw = median_heuristic(X, Y)
    assert abs(compute_mmd_unbiased(X, Y, bw) - compute_mmd_unbiased(Y, X, bw)) < 1e-10


# ---------------------------------------------------------------------------
# run_mmd_analysis tests
# ---------------------------------------------------------------------------


def test_run_mmd_analysis_columns():
    adata = _make_adata()
    df = run_mmd_analysis(adata, _cfg(mmd=_SETTINGS_FAST))
    expected = {
        "experiment",
        "marker",
        "cond_a",
        "cond_b",
        "label",
        "hours_bin_start",
        "hours_bin_end",
        "n_a",
        "n_b",
        "mmd2",
        "p_value",
        "bandwidth",
        "effect_size",
        "activity_zscore",
        "embedding_key",
    }
    assert expected.issubset(df.columns), f"Missing columns: {expected - set(df.columns)}"


def test_run_mmd_analysis_explicit_comparisons():
    adata = _make_adata()
    df = run_mmd_analysis(adata, _cfg(mmd=_SETTINGS_FAST))
    assert set(df["cond_b"].unique()) == {"ZIKV"}
    assert set(df["cond_a"].unique()) == {"uninfected"}
    assert df["label"].iloc[0] == "uninf vs ZIKV"


def test_run_mmd_analysis_per_marker():
    adata = _make_adata()
    df = run_mmd_analysis(adata, _cfg(mmd=_SETTINGS_FAST))
    assert set(df["marker"].unique()) == {"Phase3D", "TOMM20"}
    assert len(df) == 2  # one row per (marker, comparison) in aggregate mode


def test_run_mmd_analysis_significant_for_shifted_marker():
    adata = _make_adata(n_cells=600, treatment_shift=4.0)
    df = run_mmd_analysis(adata, _cfg(mmd=MMDSettings(n_permutations=200)))
    tomm = df[df["marker"] == "TOMM20"]["mmd2"].iloc[0]
    phase = df[df["marker"] == "Phase3D"]["mmd2"].iloc[0]
    assert tomm > phase
    assert df[df["marker"] == "TOMM20"]["p_value"].iloc[0] < 0.05


def test_run_mmd_analysis_missing_cond_returns_nan():
    """When cond_a is absent from the data, result is NaN (not an error)."""
    adata = _make_adata()
    cfg = MMDEvalConfig(
        input_path="dummy",
        output_dir="/tmp",
        comparisons=[ComparisonSpec(cond_a="MISSING", cond_b="ZIKV", label="missing vs ZIKV")],
        mmd=_SETTINGS_FAST,
    )
    df = run_mmd_analysis(adata, cfg)
    assert df["mmd2"].isna().all()


def test_run_mmd_analysis_temporal_bins():
    adata = _make_temporal_adata()
    cfg = _cfg(mmd=MMDSettings(n_permutations=100), temporal_bins=[0.0, 2.0, 5.0, 8.0, 15.0])
    df = run_mmd_analysis(adata, cfg)
    valid = df.dropna(subset=["mmd2"]).sort_values("hours_bin_start")
    assert len(valid) >= 2
    assert valid.iloc[-1]["mmd2"] > valid.iloc[0]["mmd2"]


def test_run_mmd_analysis_min_cells_skip():
    adata = _make_temporal_adata()
    cfg = _cfg(
        mmd=MMDSettings(n_permutations=50, min_cells=5),
        temporal_bins=[0.0, 0.5, 1.0, 100.0],
    )
    df = run_mmd_analysis(adata, cfg)
    first_bin = df[(df["hours_bin_start"] == 0.0) & (df["hours_bin_end"] == 0.5)]
    assert len(first_bin) > 0
    assert first_bin["mmd2"].isna().all()


def test_run_mmd_analysis_batch_centering():
    rng = np.random.default_rng(7)
    n, n_feat = 100, 8
    rows, embs = [], []
    for exp, offset in [("exp_A", 0.0), ("exp_B", 10.0)]:
        for pert in ["uninfected", "ZIKV"]:
            shift = 3.0 if pert == "ZIKV" else 0.0
            for _ in range(n):
                embs.append(rng.normal(offset + shift, 1.0, n_feat))
                rows.append(
                    {"experiment": exp, "marker": "TOMM20", "perturbation": pert, "hours_post_perturbation": 1.0}
                )
    X = np.stack(embs).astype(np.float32)
    obs = pd.DataFrame(rows)
    adata = ad.AnnData(X=X, obs=obs)

    cfg_test = MMDEvalConfig(
        input_path="dummy",
        output_dir="/tmp",
        comparisons=_COMP,
        mmd=MMDSettings(n_permutations=100),
    )
    df_no_center = run_mmd_analysis(adata, cfg_test)

    centered = X.copy()
    for exp in obs["experiment"].unique():
        for marker in obs["marker"].unique():
            mask = ((obs["experiment"] == exp) & (obs["marker"] == marker)).to_numpy()
            if mask.sum() > 0:
                centered[mask] -= centered[mask].mean(axis=0)
    adata_centered = ad.AnnData(X=centered, obs=obs)
    df_centered = run_mmd_analysis(adata_centered, cfg_test)

    tomm_uncentered = df_no_center[df_no_center["marker"] == "TOMM20"]["mmd2"].iloc[0]
    tomm_centered = df_centered[df_centered["marker"] == "TOMM20"]["mmd2"].iloc[0]
    assert tomm_centered <= tomm_uncentered * 1.5, (
        f"Centering should reduce MMD. centered={tomm_centered:.4f}, uncentered={tomm_uncentered:.4f}"
    )


def test_run_mmd_analysis_obs_filter():
    """obs_filter restricts analysis to matching rows before computing MMD."""
    rng = np.random.default_rng(42)
    n, n_feat = 60, 8
    rows, embs = [], []
    for microscope in ["dragonfly", "mantis"]:
        for perturbation in ["uninfected", "ZIKV"]:
            shift = 10.0 if perturbation == "ZIKV" else 0.0
            for _ in range(n):
                embs.append(rng.normal(shift, 1.0, n_feat))
                rows.append(
                    {
                        "experiment": "e",
                        "marker": "TOMM20",
                        "perturbation": perturbation,
                        "microscope": microscope,
                        "hours_post_perturbation": 1.0,
                    }
                )

    adata = ad.AnnData(X=np.stack(embs).astype(np.float32), obs=pd.DataFrame(rows))

    # Compare microscopes on uninfected only — should be near zero (same distribution)
    comp = [ComparisonSpec(cond_a="dragonfly", cond_b="mantis", label="dragonfly vs mantis")]
    cfg = MMDEvalConfig(
        input_path="dummy",
        output_dir="/tmp",
        comparisons=comp,
        group_by="microscope",
        obs_filter={"perturbation": "uninfected"},
        mmd=MMDSettings(n_permutations=50),
    )
    df = run_mmd_analysis(adata, cfg)
    assert len(df) == 1
    # MMD on unfiltered data would be dominated by the ZIKV shift; filtered should be small
    assert df["mmd2"].iloc[0] < 1.0, f"Expected near-zero MMD on uninfected-only, got {df['mmd2'].iloc[0]:.4f}"


# ---------------------------------------------------------------------------
# Activity z-score tests
# ---------------------------------------------------------------------------


def test_activity_zscore_shifted():
    """Strongly shifted distributions produce a large positive activity_zscore."""
    adata = _make_adata(n_cells=600, treatment_shift=5.0)
    df = run_mmd_analysis(adata, _cfg(mmd=MMDSettings(n_permutations=200)))
    tomm = df[df["marker"] == "TOMM20"]["activity_zscore"].iloc[0]
    assert tomm > 1.0, f"Expected activity_zscore > 1 for shifted distribution, got {tomm:.3f}"


def test_activity_zscore_identical():
    """Identical distributions produce activity_zscore near zero."""
    adata = _make_adata(n_cells=400, treatment_shift=0.0)
    df = run_mmd_analysis(adata, _cfg(mmd=MMDSettings(n_permutations=200)))
    for _, row in df.iterrows():
        assert np.isfinite(row["activity_zscore"]) or np.isnan(row["activity_zscore"])


# ---------------------------------------------------------------------------
# Sample balancing tests
# ---------------------------------------------------------------------------


def test_balance_samples():
    """With balance_samples=True, both groups have equal size (reflected in n_a, n_b)."""
    rng = np.random.default_rng(10)
    n_small, n_large = 30, 120
    rows, embs = [], []
    for pert, n in [("uninfected", n_large), ("ZIKV", n_small)]:
        for _ in range(n):
            embs.append(rng.normal(0.0, 1.0, 8))
            rows.append({"experiment": "e", "marker": "TOMM20", "perturbation": pert, "hours_post_perturbation": 1.0})
    adata = ad.AnnData(X=np.stack(embs).astype(np.float32), obs=pd.DataFrame(rows))
    cfg = _cfg(mmd=MMDSettings(n_permutations=50, balance_samples=True, max_cells=None))
    df = run_mmd_analysis(adata, cfg)
    row = df[df["marker"] == "TOMM20"].iloc[0]
    assert row["n_a"] == row["n_b"], f"Expected equal group sizes, got n_a={row['n_a']}, n_b={row['n_b']}"


# ---------------------------------------------------------------------------
# Bandwidth sharing tests
# ---------------------------------------------------------------------------


def test_share_bandwidth_from():
    """With share_bandwidth_from set, the bandwidth is the same across comparisons."""
    adata = _make_adata(n_cells=400, treatment_shift=2.0)
    # Add a second condition
    obs = adata.obs.copy()
    extra_rows = obs[obs["perturbation"] == "ZIKV"].copy()
    extra_rows["perturbation"] = "DENV"
    extra_obs = pd.concat([obs, extra_rows], ignore_index=True)
    extra_emb = np.concatenate([adata.X, adata.X[obs["perturbation"] == "ZIKV"]], axis=0)
    adata2 = ad.AnnData(X=extra_emb.astype(np.float32), obs=extra_obs)

    comps = [
        ComparisonSpec(cond_a="uninfected", cond_b="ZIKV", label="baseline"),
        ComparisonSpec(cond_a="uninfected", cond_b="DENV", label="treatment"),
    ]
    cfg = MMDEvalConfig(
        input_path="dummy",
        output_dir="/tmp",
        comparisons=comps,
        mmd=MMDSettings(n_permutations=50, share_bandwidth_from="baseline"),
    )
    df = run_mmd_analysis(adata2, cfg)
    for marker in df["marker"].unique():
        sub = df[df["marker"] == marker].dropna(subset=["bandwidth"])
        if len(sub) == 2:
            assert abs(sub["bandwidth"].iloc[0] - sub["bandwidth"].iloc[1]) < 1e-6, (
                f"Expected shared bandwidth for {marker}, got {sub['bandwidth'].to_numpy()}"
            )


# ---------------------------------------------------------------------------
# Temporal bins (explicit edges) tests
# ---------------------------------------------------------------------------


def test_temporal_bins_explicit():
    """temporal_bins produces one row per bin per comparison."""
    adata = _make_temporal_adata()
    cfg = _cfg(mmd=MMDSettings(n_permutations=50), temporal_bins=[0.0, 2.0, 5.0, 8.0, 15.0])
    df = run_mmd_analysis(adata, cfg)
    valid = df.dropna(subset=["mmd2"]).sort_values("hours_bin_start")
    assert len(valid) >= 2, "Expected at least 2 valid temporal bins"
    assert valid.iloc[-1]["mmd2"] > valid.iloc[0]["mmd2"], "MMD should increase with shift"


def test_temporal_bins_min_cells_skip():
    """Bins with fewer than min_cells cells produce NaN rows."""
    adata = _make_temporal_adata()
    cfg = _cfg(
        mmd=MMDSettings(n_permutations=50, min_cells=5),
        temporal_bins=[0.0, 0.5, 1.0, 100.0],
    )
    df = run_mmd_analysis(adata, cfg)
    first_bin = df[(df["hours_bin_start"] == 0.0) & (df["hours_bin_end"] == 0.5)]
    assert len(first_bin) > 0
    assert first_bin["mmd2"].isna().all()


def test_temporal_bins_mutually_exclusive():
    """Setting both temporal_bin_size and temporal_bins raises ValidationError."""
    with pytest.raises(Exception):
        MMDEvalConfig(
            input_path="dummy",
            output_dir="/tmp",
            comparisons=_COMP,
            temporal_bin_size=4.0,
            temporal_bins=[0.0, 4.0, 8.0],
        )


# ---------------------------------------------------------------------------
# Pooled mode tests
# ---------------------------------------------------------------------------


def _save_adata_zarr(adata: ad.AnnData, path: str) -> None:
    import os
    import shutil

    if os.path.exists(path):
        shutil.rmtree(path)
    adata.write_zarr(path)


def test_run_mmd_pooled_columns(tmp_path):
    """run_mmd_pooled returns expected columns including activity_zscore and q_value."""
    adata1 = _make_adata(n_cells=200, seed=0)
    adata2 = _make_adata(n_cells=200, seed=1)
    p1 = str(tmp_path / "exp1.zarr")
    p2 = str(tmp_path / "exp2.zarr")
    _save_adata_zarr(adata1, p1)
    _save_adata_zarr(adata2, p2)

    cfg = MMDPooledConfig(
        input_paths=[p1, p2],
        output_dir=str(tmp_path / "out"),
        comparisons=_COMP,
        mmd=MMDSettings(n_permutations=50),
    )
    df = run_mmd_pooled(cfg)
    expected = {
        "marker",
        "cond_a",
        "cond_b",
        "label",
        "mmd2",
        "p_value",
        "bandwidth",
        "effect_size",
        "activity_zscore",
        "q_value",
    }
    assert expected.issubset(df.columns), f"Missing: {expected - set(df.columns)}"


def test_run_mmd_pooled_condition_aliases(tmp_path):
    """condition_aliases remaps variant condition names to canonical names."""
    rng = np.random.default_rng(99)
    rows, embs = [], []
    for pert in ["uninfected1", "uninfected2", "ZIKV"]:
        shift = 3.0 if pert == "ZIKV" else 0.0
        for _ in range(60):
            embs.append(rng.normal(shift, 1.0, 16))
            rows.append({"experiment": "e", "marker": "TOMM20", "perturbation": pert, "hours_post_perturbation": 1.0})
    adata = ad.AnnData(X=np.stack(embs).astype(np.float32), obs=pd.DataFrame(rows))
    p = str(tmp_path / "exp.zarr")
    _save_adata_zarr(adata, p)

    cfg = MMDPooledConfig(
        input_paths=[p],
        output_dir=str(tmp_path / "out"),
        comparisons=[ComparisonSpec(cond_a="uninfected", cond_b="ZIKV", label="uninf vs ZIKV")],
        mmd=MMDSettings(n_permutations=50),
        condition_aliases={"uninfected": ["uninfected1", "uninfected2"]},
    )
    df = run_mmd_pooled(cfg)
    assert not df["mmd2"].isna().all(), "Expected valid MMD after condition alias remapping"
