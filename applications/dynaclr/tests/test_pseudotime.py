"""Tests for pseudotime evaluation modules (alignment, signals, metrics, plotting)."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pseudotime.alignment import (
    align_tracks,
    assign_t_perturb,
    filter_tracks,
    identify_lineages,
)
from pseudotime.metrics import (
    aggregate_population,
    compute_track_timing,
    find_half_max_time,
    find_onset_time,
    find_peak_metrics,
    run_statistical_tests,
)
from pseudotime.plotting import (
    plot_cell_heatmap,
    plot_onset_comparison,
    plot_response_curves,
    plot_timing_distributions,
)
from pseudotime.signals import (
    extract_annotation_signal,
    extract_embedding_distance,
    extract_prediction_signal,
)


# ── Shared Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def tracking_df():
    """Synthetic tracking DataFrame with 3 FOVs.

    C/2/000: 3 tracks (root=0, children=1,2), 10 timepoints, infected at t=5
    C/2/001: 1 orphan track (id=3), 10 timepoints, infected at t=7
    B/1/000: 2 control tracks (id=0,1), 10 timepoints, no infection
    """
    rows = []
    for track_id, parent in [(0, -1), (1, 0), (2, 0)]:
        for t in range(10):
            rows.append(
                {
                    "fov_name": "C/2/000",
                    "track_id": track_id,
                    "parent_track_id": parent,
                    "t": t,
                    "infection_state": "infected" if t >= 5 else "uninfected",
                    "organelle_state": "remodel" if t >= 5 else "noremodel",
                }
            )
    for t in range(10):
        rows.append(
            {
                "fov_name": "C/2/001",
                "track_id": 3,
                "parent_track_id": -1,
                "t": t,
                "infection_state": "infected" if t >= 7 else "uninfected",
                "organelle_state": "remodel" if t >= 7 else "noremodel",
            }
        )
    for track_id in [0, 1]:
        for t in range(10):
            rows.append(
                {
                    "fov_name": "B/1/000",
                    "track_id": track_id,
                    "parent_track_id": -1,
                    "t": t,
                    "infection_state": "uninfected",
                    "organelle_state": "noremodel",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_adata(tracking_df):
    """AnnData keyed by (fov_name, track_id, t) with classifier predictions."""
    rng = np.random.default_rng(42)
    n = len(tracking_df)
    X = rng.standard_normal((n, 16)).astype(np.float32)

    obs = tracking_df[["fov_name", "track_id", "t"]].copy().reset_index(drop=True)
    predicted = tracking_df["organelle_state"].values.copy()
    obs["predicted_organelle_state"] = predicted

    adata = ad.AnnData(X=X, obs=obs)

    classes = ["noremodel", "remodel"]
    proba = np.zeros((n, 2), dtype=np.float32)
    for i, state in enumerate(predicted):
        proba[i] = [0.15, 0.85] if state == "remodel" else [0.85, 0.15]
    adata.obsm["predicted_organelle_state_proba"] = proba
    adata.uns["predicted_organelle_state_classes"] = classes

    return adata


@pytest.fixture
def aligned_df(tracking_df):
    """Aligned DataFrame for infected FOVs with t_relative_minutes."""
    infected = tracking_df[tracking_df["fov_name"].str.startswith("C/2")].copy()
    infected.loc[infected["fov_name"] == "C/2/000", "t_perturb"] = 5
    infected.loc[infected["fov_name"] == "C/2/001", "t_perturb"] = 7
    infected["t_perturb"] = infected["t_perturb"].astype(int)
    infected["t_relative_minutes"] = (
        (infected["t"] - infected["t_perturb"]) * 30.0
    )
    return infected.reset_index(drop=True)


# ── TestAlignment ────────────────────────────────────────────────────


class TestAlignment:
    def test_identify_lineages_groups_parent_child(self, tracking_df):
        fov_df = tracking_df[tracking_df["fov_name"] == "C/2/000"]
        lineages = identify_lineages(fov_df)
        assert len(lineages) == 1
        fov, track_ids = lineages[0]
        assert fov == "C/2/000"
        assert 0 in track_ids
        assert len(track_ids) == 2

    def test_identify_lineages_both_branches(self, tracking_df):
        fov_df = tracking_df[tracking_df["fov_name"] == "C/2/000"]
        lineages = identify_lineages(fov_df, return_both_branches=True)
        assert len(lineages) == 2
        branches = [set(ids) for _, ids in lineages]
        assert {0, 1} in branches
        assert {0, 2} in branches

    def test_filter_tracks_by_fov(self, tracking_df):
        filtered = filter_tracks(tracking_df, fov_pattern="C/2")
        assert set(filtered["fov_name"].unique()) == {"C/2/000", "C/2/001"}

    def test_filter_tracks_by_min_timepoints(self, tracking_df):
        filtered = filter_tracks(tracking_df, min_timepoints=11)
        assert len(filtered) == 0

    def test_assign_t_perturb_lineage_aware(self, tracking_df):
        fov_df = tracking_df[tracking_df["fov_name"] == "C/2/000"].copy()
        result = assign_t_perturb(
            fov_df, frame_interval_minutes=30.0, min_track_timepoints=1
        )
        t_perturbs = result.groupby("track_id")["t_perturb"].first()
        assert t_perturbs.nunique() == 1
        assert t_perturbs.iloc[0] == 5

    def test_assign_t_perturb_orphan(self, tracking_df):
        fov_df = tracking_df[tracking_df["fov_name"] == "C/2/001"].copy()
        result = assign_t_perturb(
            fov_df, frame_interval_minutes=30.0, min_track_timepoints=1
        )
        assert result["t_perturb"].iloc[0] == 7

    def test_align_tracks_convenience(self, tracking_df):
        result = align_tracks(
            tracking_df,
            frame_interval_minutes=30.0,
            fov_pattern="C/2",
            min_track_timepoints=1,
        )
        assert "t_perturb" in result.columns
        assert "t_relative_minutes" in result.columns
        assert all(result["fov_name"].str.startswith("C/2"))


# ── TestSignals ──────────────────────────────────────────────────────


class TestSignals:
    def test_annotation_signal_binary(self, aligned_df):
        result = extract_annotation_signal(aligned_df)
        remodel = aligned_df["organelle_state"] == "remodel"
        assert (result.loc[remodel, "signal"] == 1.0).all()
        assert (result.loc[~remodel, "signal"] == 0.0).all()

    def test_prediction_signal_binary(self, synthetic_adata, aligned_df):
        result = extract_prediction_signal(
            synthetic_adata, aligned_df, task="organelle_state"
        )
        assert "signal" in result.columns
        remodel = aligned_df["organelle_state"] == "remodel"
        assert (result.loc[remodel, "signal"] == 1.0).all()
        assert (result.loc[~remodel, "signal"] == 0.0).all()

    def test_prediction_signal_probability(self, synthetic_adata, aligned_df):
        result = extract_prediction_signal(
            synthetic_adata,
            aligned_df,
            task="organelle_state",
            use_probability=True,
        )
        assert "signal" in result.columns
        remodel = aligned_df["organelle_state"] == "remodel"
        assert result.loc[remodel, "signal"].mean() > 0.7
        assert result.loc[~remodel, "signal"].mean() < 0.3

    def test_embedding_distance_per_track(self, synthetic_adata, aligned_df):
        result = extract_embedding_distance(
            synthetic_adata,
            aligned_df,
            baseline_method="per_track",
            baseline_window_minutes=(-180, -60),
        )
        assert "signal" in result.columns
        valid = result["signal"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_embedding_distance_control_well(self, synthetic_adata, aligned_df):
        result = extract_embedding_distance(
            synthetic_adata,
            aligned_df,
            baseline_method="control_well",
            control_fov_pattern="B/1",
        )
        assert "signal" in result.columns
        valid = result["signal"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()


# ── TestMetrics ──────────────────────────────────────────────────────


class TestMetrics:
    def test_aggregate_population_fraction(self, aligned_df):
        df = extract_annotation_signal(aligned_df)
        time_bins = np.arange(-180, 181, 30)
        pop = aggregate_population(
            df, time_bins, signal_type="fraction", min_cells_per_bin=1
        )
        assert "fraction" in pop.columns
        assert "ci_lower" in pop.columns
        assert "ci_upper" in pop.columns
        pre = pop[pop["time_minutes"] < 0]
        assert (pre["fraction"].dropna() == 0.0).all()

    def test_aggregate_population_continuous(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "t_relative_minutes": np.linspace(-300, 300, n),
                "signal": np.concatenate(
                    [rng.normal(0.1, 0.05, 50), rng.normal(0.5, 0.1, 50)]
                ),
            }
        )
        time_bins = np.arange(-300, 301, 60)
        pop = aggregate_population(
            df, time_bins, signal_type="continuous", min_cells_per_bin=1
        )
        assert "mean" in pop.columns
        assert "median" in pop.columns
        assert "q25" in pop.columns
        assert "q75" in pop.columns

    def test_find_onset_time_detected(self):
        rows = []
        for t in range(-600, 901, 30):
            frac = 0.8 if t >= 120 else 0.0
            rows.append({"time_minutes": t, "fraction": frac, "n_cells": 20})
        pop_df = pd.DataFrame(rows)
        onset, threshold, bl_mean, bl_std = find_onset_time(pop_df)
        assert onset is not None
        assert onset == 120

    def test_find_onset_time_not_detected(self):
        rows = [
            {"time_minutes": t, "fraction": 0.0, "n_cells": 20}
            for t in range(-600, 901, 30)
        ]
        pop_df = pd.DataFrame(rows)
        onset, threshold, bl_mean, bl_std = find_onset_time(pop_df)
        assert onset is None

    def test_find_half_max_time(self):
        rows = []
        for t in range(-300, 601, 30):
            if t < 0:
                frac = 0.0
            else:
                frac = min(1.0, t / 300.0)
            rows.append({"time_minutes": t, "fraction": frac, "n_cells": 20})
        pop_df = pd.DataFrame(rows)
        t50 = find_half_max_time(pop_df)
        assert not np.isnan(t50)
        assert 0 < t50 < 300

    def test_find_peak_metrics(self):
        rows = []
        for t in range(-300, 601, 30):
            if t < 0:
                frac = 0.0
            elif t <= 150:
                frac = t / 150.0 * 0.8
            elif t <= 300:
                frac = 0.8 - (t - 150) / 150.0 * 0.8
            else:
                frac = 0.0
            rows.append({"time_minutes": t, "fraction": frac, "n_cells": 20})
        pop_df = pd.DataFrame(rows)
        metrics = find_peak_metrics(pop_df)
        assert not np.isnan(metrics["T_peak_minutes"])
        assert metrics["peak_amplitude"] > 0
        assert metrics["auc"] > 0

    def test_compute_track_timing_fraction(self, aligned_df):
        df = extract_annotation_signal(aligned_df)
        timing = compute_track_timing(df)
        assert "onset_minutes" in timing.columns
        assert "total_positive_minutes" in timing.columns
        assert len(timing) > 0
        assert (timing["onset_minutes"] >= 0).all()

    def test_run_statistical_tests(self, aligned_df):
        df_a = extract_annotation_signal(aligned_df)
        df_a["organelle"] = "SEC61"
        df_b = df_a.copy()
        df_b["organelle"] = "TOMM20"

        organelle_results = {
            "SEC61": {"combined_df": df_a},
            "TOMM20": {"combined_df": df_b},
        }
        timing_a = compute_track_timing(df_a)
        timing_a["organelle"] = "SEC61"
        timing_b = compute_track_timing(df_b)
        timing_b["organelle"] = "TOMM20"
        track_timing = pd.concat([timing_a, timing_b], ignore_index=True)

        stats = run_statistical_tests(organelle_results, track_timing)
        assert isinstance(stats, pd.DataFrame)
        assert "Test" in stats.columns
        assert "p_value" in stats.columns
        assert len(stats) > 0


# ── TestPlotting ─────────────────────────────────────────────────────


class TestPlotting:
    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")

    def test_plot_response_curves_saves_files(self, aligned_df, tmp_path):
        df = extract_annotation_signal(aligned_df)
        time_bins = np.arange(-180, 181, 30)
        pop = aggregate_population(
            df, time_bins, signal_type="fraction", min_cells_per_bin=1
        )
        curves = {"SEC61": pop}
        configs = {"SEC61": {"label": "SEC61", "color": "blue"}}
        fig = plot_response_curves(curves, configs, tmp_path)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "response_curves.pdf").exists()
        assert (tmp_path / "response_curves.png").exists()

    def test_plot_cell_heatmap_returns_figure(self, aligned_df):
        df = extract_annotation_signal(aligned_df)
        time_bins = np.arange(-180, 181, 30)
        fig = plot_cell_heatmap(df, time_bins, organelle_label="SEC61")
        assert isinstance(fig, plt.Figure)

    def test_plot_timing_distributions_saves_files(self, aligned_df, tmp_path):
        df = extract_annotation_signal(aligned_df)
        df["organelle"] = "SEC61"
        timing = compute_track_timing(df)
        timing["organelle"] = "SEC61"
        configs = {"SEC61": {"label": "SEC61", "color": "blue"}}
        fig = plot_timing_distributions(timing, configs, tmp_path)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "timing_distributions.pdf").exists()
        assert (tmp_path / "timing_distributions.png").exists()

    def test_plot_onset_comparison_saves_files(self, tmp_path):
        timing_metrics = pd.DataFrame(
            {
                "organelle": ["SEC61", "TOMM20"],
                "T_onset_minutes": [60.0, 120.0],
                "T_50_minutes": [180.0, 240.0],
                "T_peak_minutes": [300.0, 360.0],
            }
        )
        fig = plot_onset_comparison(timing_metrics, tmp_path)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "onset_comparison.pdf").exists()
        assert (tmp_path / "onset_comparison.png").exists()
