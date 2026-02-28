"""Population aggregation, timing detection, and statistical tests.

Provides functions to aggregate per-cell signals into population-level
response curves, detect timing metrics (onset, T50, peak), compute
per-track timing statistics, and run statistical comparisons.

Ported from:
- .ed_planning/tmp/scripts/annotation_remodling.py (fraction aggregation, onset, stats)
- .ed_planning/tmp/scripts/multi_organelle_remodeling.py (continuous aggregation, T50, peak)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

_logger = logging.getLogger(__name__)


def aggregate_population(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    signal_col: str = "signal",
    signal_type: Literal["fraction", "continuous"] = "fraction",
    ci_alpha: float = 0.05,
    min_cells_per_bin: int = 5,
) -> pd.DataFrame:
    """Bin cells by t_relative_minutes and aggregate signal per bin.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with t_relative_minutes and signal columns.
    time_bins : np.ndarray
        Bin edges in minutes (e.g., np.arange(-600, 901, 30)).
    signal_col : str
        Column containing the signal values.
    signal_type : {"fraction", "continuous"}
        - "fraction": binary signal, computes fraction + Wilson CI.
        - "continuous": numeric signal, computes mean/median/IQR.
    ci_alpha : float
        Significance level for confidence intervals.
    min_cells_per_bin : int
        Minimum cells for a bin to be included (fewer → NaN values).

    Returns
    -------
    pd.DataFrame
        For "fraction": columns time_minutes, fraction, ci_lower, ci_upper,
        n_cells, n_positive.
        For "continuous": columns time_minutes, mean, median, std, q25, q75,
        n_cells.
    """
    valid = df.dropna(subset=[signal_col]).copy()
    valid["time_bin"] = pd.cut(
        valid["t_relative_minutes"],
        bins=time_bins,
        labels=time_bins[:-1],
        right=False,
    )
    valid["time_bin"] = valid["time_bin"].astype(float)

    results = []
    for bin_start in time_bins[:-1]:
        bin_data = valid[valid["time_bin"] == bin_start]
        n_total = len(bin_data)

        if signal_type == "fraction":
            n_positive = int(bin_data[signal_col].sum()) if n_total > 0 else 0
            if n_total == 0:
                results.append(
                    {
                        "time_minutes": bin_start,
                        "fraction": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "n_cells": 0,
                        "n_positive": 0,
                    }
                )
            else:
                fraction = n_positive / n_total
                ci_low, ci_high = proportion_confint(
                    n_positive, n_total, alpha=ci_alpha, method="wilson"
                )
                results.append(
                    {
                        "time_minutes": bin_start,
                        "fraction": fraction,
                        "ci_lower": ci_low,
                        "ci_upper": ci_high,
                        "n_cells": n_total,
                        "n_positive": n_positive,
                    }
                )
        else:  # continuous
            if n_total == 0:
                results.append(
                    {
                        "time_minutes": bin_start,
                        "mean": np.nan,
                        "median": np.nan,
                        "std": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "n_cells": 0,
                    }
                )
            else:
                vals = bin_data[signal_col].values
                results.append(
                    {
                        "time_minutes": bin_start,
                        "mean": np.mean(vals),
                        "median": np.median(vals),
                        "std": np.std(vals),
                        "q25": np.percentile(vals, 25),
                        "q75": np.percentile(vals, 75),
                        "n_cells": n_total,
                    }
                )

    return pd.DataFrame(results)


def find_onset_time(
    population_df: pd.DataFrame,
    baseline_window: tuple[float, float] = (-600, -120),
    sigma_threshold: float = 2.0,
    min_cells_per_bin: int = 5,
    signal_col: str | None = None,
) -> tuple[float | None, float, float, float]:
    """Find the first post-infection bin where signal exceeds baseline + N*sigma.

    Parameters
    ----------
    population_df : pd.DataFrame
        Output of aggregate_population.
    baseline_window : tuple[float, float]
        (min_minutes, max_minutes) for baseline calculation.
    sigma_threshold : float
        Number of standard deviations above baseline for onset.
    min_cells_per_bin : int
        Minimum cells per bin to consider valid.
    signal_col : str or None
        Signal column name. If None, auto-detects ("fraction" or "mean").

    Returns
    -------
    tuple of (onset_minutes, threshold, baseline_mean, baseline_std)
        onset_minutes is None if onset is not detected.
    """
    if signal_col is None:
        signal_col = "fraction" if "fraction" in population_df.columns else "mean"

    baseline = population_df[
        (population_df["time_minutes"] >= baseline_window[0])
        & (population_df["time_minutes"] < baseline_window[1])
        & (population_df["n_cells"] >= min_cells_per_bin)
    ]

    if len(baseline) < 3:
        return None, np.nan, np.nan, np.nan

    mean_bl = baseline[signal_col].mean()
    std_bl = baseline[signal_col].std()
    threshold = mean_bl + sigma_threshold * std_bl

    post_infection = population_df[
        (population_df["time_minutes"] >= 0)
        & (population_df["n_cells"] >= min_cells_per_bin)
    ]
    onset_rows = post_infection[post_infection[signal_col] > threshold]

    if len(onset_rows) > 0:
        return onset_rows["time_minutes"].iloc[0], threshold, mean_bl, std_bl
    return None, threshold, mean_bl, std_bl


def find_half_max_time(
    population_df: pd.DataFrame,
    signal_col: str | None = None,
) -> float:
    """Find T50: time when signal reaches half of max response.

    Parameters
    ----------
    population_df : pd.DataFrame
        Output of aggregate_population.
    signal_col : str or None
        Signal column name. If None, auto-detects ("fraction" or "mean").

    Returns
    -------
    float
        T50 in minutes, or NaN if not found.
    """
    if signal_col is None:
        signal_col = "fraction" if "fraction" in population_df.columns else "mean"

    post_infection = population_df[population_df["time_minutes"] >= 0]
    if len(post_infection) == 0 or post_infection[signal_col].isna().all():
        return np.nan

    max_val = post_infection[signal_col].max()
    baseline_data = population_df[population_df["time_minutes"] < -60]
    baseline_mean = (
        baseline_data[signal_col].mean() if len(baseline_data) > 0 else 0.0
    )

    half_max = baseline_mean + (max_val - baseline_mean) / 2

    exceeds = post_infection[signal_col] > half_max
    if exceeds.any():
        t50_idx = post_infection[exceeds].index[0]
        return population_df.loc[t50_idx, "time_minutes"]
    return np.nan


def find_peak_metrics(
    population_df: pd.DataFrame,
    signal_col: str | None = None,
) -> dict[str, float]:
    """Extract peak-related metrics for pulsatile dynamics.

    Parameters
    ----------
    population_df : pd.DataFrame
        Output of aggregate_population.
    signal_col : str or None
        Signal column name. If None, auto-detects ("fraction" or "mean").

    Returns
    -------
    dict with keys: T_peak_minutes, peak_amplitude, T_return_minutes,
        pulse_duration_minutes, auc.
    """
    if signal_col is None:
        signal_col = "fraction" if "fraction" in population_df.columns else "mean"

    nan_result = {
        "T_peak_minutes": np.nan,
        "peak_amplitude": np.nan,
        "T_return_minutes": np.nan,
        "pulse_duration_minutes": np.nan,
        "auc": np.nan,
    }

    post_infection = population_df[population_df["time_minutes"] >= 0].copy()
    baseline_data = population_df[population_df["time_minutes"] < -60]

    if len(post_infection) == 0 or post_infection[signal_col].isna().all():
        return nan_result

    baseline_mean = (
        baseline_data[signal_col].mean() if len(baseline_data) > 0 else 0.0
    )
    baseline_std = baseline_data[signal_col].std() if len(baseline_data) > 0 else 0.0

    # Peak
    peak_idx = post_infection[signal_col].idxmax()
    t_peak = population_df.loc[peak_idx, "time_minutes"]
    peak_amplitude = population_df.loc[peak_idx, signal_col] - baseline_mean

    # Return to baseline (within 1 sigma)
    return_threshold = baseline_mean + 1 * baseline_std
    after_peak = post_infection[post_infection["time_minutes"] > t_peak]
    returns = after_peak[after_peak[signal_col] < return_threshold]

    t_return = np.nan
    if len(returns) > 0:
        return_idx = returns.index[0]
        t_return = population_df.loc[return_idx, "time_minutes"]

    # Pulse duration
    onset_result = find_onset_time(population_df, signal_col=signal_col)
    t_onset = onset_result[0]
    pulse_duration = np.nan
    if t_onset is not None and not np.isnan(t_return):
        pulse_duration = t_return - t_onset

    # AUC (area under curve from baseline)
    valid_mask = post_infection[signal_col].notna()
    if valid_mask.sum() > 1:
        times = post_infection.loc[valid_mask, "time_minutes"].values
        values = post_infection.loc[valid_mask, signal_col].values - baseline_mean
        auc = float(np.trapezoid(values, times))
    else:
        auc = np.nan

    return {
        "T_peak_minutes": t_peak,
        "peak_amplitude": peak_amplitude,
        "T_return_minutes": t_return,
        "pulse_duration_minutes": pulse_duration,
        "auc": auc,
    }


def compute_track_timing(
    df: pd.DataFrame,
    signal_col: str = "signal",
    signal_type: Literal["fraction", "continuous"] = "fraction",
    positive_value: float = 1.0,
) -> pd.DataFrame:
    """Compute per-track onset, duration, and span of positive signal.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with signal, t_relative_minutes, fov_name, track_id columns.
        Should also have "experiment" and "organelle" columns if available.
    signal_col : str
        Column containing signal values.
    signal_type : {"fraction", "continuous"}
        If "fraction", positive frames are where signal == positive_value.
        If "continuous", onset is the first frame where signal exceeds the
        track's pre-infection mean + 2*std.
    positive_value : float
        Threshold for binary positive detection (used for "fraction" mode).

    Returns
    -------
    pd.DataFrame
        Columns: organelle, fov_name, track_id, experiment, onset_minutes,
        total_positive_minutes, span_minutes, n_positive_frames, n_total_frames.
    """
    valid = df.dropna(subset=[signal_col]).copy()

    group_cols = ["fov_name", "track_id"]
    extra_cols = []
    for col in ["experiment", "organelle"]:
        if col in valid.columns:
            group_cols.append(col)
            extra_cols.append(col)

    rows = []
    for keys, track_df in valid.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fov_name = keys[0]
        track_id = keys[1]
        extra = {col: keys[i + 2] for i, col in enumerate(extra_cols)}

        if signal_type == "fraction":
            positive_frames = track_df[track_df[signal_col] == positive_value]
        else:
            # For continuous signals, define positive as exceeding
            # pre-infection baseline + 2*std
            pre = track_df[track_df["t_relative_minutes"] < 0]
            if len(pre) >= 2:
                threshold = pre[signal_col].mean() + 2 * pre[signal_col].std()
            else:
                threshold = track_df[signal_col].median()
            positive_frames = track_df[track_df[signal_col] > threshold]

        if len(positive_frames) == 0:
            continue

        first_t_rel = positive_frames["t_relative_minutes"].min()
        last_t_rel = positive_frames["t_relative_minutes"].max()

        # Estimate frame interval
        frame_interval = track_df["t_relative_minutes"].diff().dropna()
        interval = frame_interval.mode().iloc[0] if len(frame_interval) > 0 else 30.0

        total_positive_minutes = len(positive_frames) * interval
        span_minutes = last_t_rel - first_t_rel + interval

        row = {
            "fov_name": fov_name,
            "track_id": track_id,
            "onset_minutes": first_t_rel,
            "total_positive_minutes": total_positive_minutes,
            "span_minutes": span_minutes,
            "n_positive_frames": len(positive_frames),
            "n_total_frames": len(track_df),
            **extra,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def run_statistical_tests(
    organelle_results: dict[str, dict],
    track_timing_df: pd.DataFrame,
    control_results: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Run statistical tests comparing organelle remodeling dynamics.

    Tests performed:
    1. Fisher's exact: remodeling vs infection (if control data available)
    2. Mann-Whitney U: onset timing between organelle pairs
    3. Mann-Whitney U: duration between organelle pairs
    4. Fisher's exact: pre vs post-infection per organelle

    Parameters
    ----------
    organelle_results : dict[str, dict]
        Per-organelle results. Each value must have "combined_df" with
        columns: organelle_state (or signal), t_relative_minutes.
    track_timing_df : pd.DataFrame
        Output of compute_track_timing with "organelle" column.
    control_results : dict[str, dict] or None
        Per-organelle control data with keys: n_total, n_remodel, fraction.

    Returns
    -------
    pd.DataFrame
        Columns: Test, Method, Statistic, p_value, N1, N2.
    """
    stat_rows = []
    organelle_names = list(organelle_results.keys())

    # Test 1: Remodeling vs infection (Fisher's exact)
    if control_results:
        for org in organelle_names:
            if org not in control_results:
                continue
            combined = organelle_results[org].get("combined_df")
            if combined is None:
                continue

            # Determine signal column
            if "organelle_state" in combined.columns:
                annotated = combined.dropna(subset=["organelle_state"])
                n_inf_pos = (annotated["organelle_state"] == "remodel").sum()
                n_inf_neg = (annotated["organelle_state"] == "noremodel").sum()
            elif "signal" in combined.columns:
                annotated = combined.dropna(subset=["signal"])
                n_inf_pos = int(annotated["signal"].sum())
                n_inf_neg = len(annotated) - n_inf_pos
            else:
                continue

            ctrl = control_results[org]
            n_ctrl_pos = ctrl["n_remodel"]
            n_ctrl_neg = ctrl["n_total"] - n_ctrl_pos

            table = [[n_inf_pos, n_inf_neg], [n_ctrl_pos, n_ctrl_neg]]
            odds_ratio, p_val = fisher_exact(table, alternative="greater")

            stat_rows.append(
                {
                    "Test": f"Remodeling vs infection ({org})",
                    "Method": "Fisher's exact (one-sided)",
                    "Statistic": f"OR={odds_ratio:.1f}",
                    "p_value": p_val,
                    "N1": n_inf_pos + n_inf_neg,
                    "N2": n_ctrl_pos + n_ctrl_neg,
                }
            )

    # Tests 2 & 3: Pairwise onset and duration comparisons
    for i in range(len(organelle_names)):
        for j in range(i + 1, len(organelle_names)):
            org_a, org_b = organelle_names[i], organelle_names[j]

            onset_a = track_timing_df[track_timing_df["organelle"] == org_a][
                "onset_minutes"
            ]
            onset_b = track_timing_df[track_timing_df["organelle"] == org_b][
                "onset_minutes"
            ]

            if len(onset_a) > 0 and len(onset_b) > 0:
                u_stat, p_val = mannwhitneyu(
                    onset_a, onset_b, alternative="two-sided"
                )
                stat_rows.append(
                    {
                        "Test": f"Onset timing {org_a} vs {org_b}",
                        "Method": "Mann-Whitney U (two-sided)",
                        "Statistic": f"U={u_stat:.0f}",
                        "p_value": p_val,
                        "N1": len(onset_a),
                        "N2": len(onset_b),
                    }
                )

            dur_a = track_timing_df[track_timing_df["organelle"] == org_a][
                "span_minutes"
            ]
            dur_b = track_timing_df[track_timing_df["organelle"] == org_b][
                "span_minutes"
            ]

            if len(dur_a) > 0 and len(dur_b) > 0:
                u_stat, p_val = mannwhitneyu(
                    dur_a, dur_b, alternative="two-sided"
                )
                stat_rows.append(
                    {
                        "Test": f"Duration {org_a} vs {org_b}",
                        "Method": "Mann-Whitney U (two-sided)",
                        "Statistic": f"U={u_stat:.0f}",
                        "p_value": p_val,
                        "N1": len(dur_a),
                        "N2": len(dur_b),
                    }
                )

    # Test 4: Pre vs post-infection per organelle (Fisher's exact)
    for org in organelle_names:
        combined = organelle_results[org].get("combined_df")
        if combined is None:
            continue

        if "organelle_state" in combined.columns:
            annotated = combined.dropna(subset=["organelle_state"])
            pre = annotated[annotated["t_relative_minutes"] < 0]
            post = annotated[annotated["t_relative_minutes"] >= 0]
            pre_pos = (pre["organelle_state"] == "remodel").sum()
            pre_neg = (pre["organelle_state"] == "noremodel").sum()
            post_pos = (post["organelle_state"] == "remodel").sum()
            post_neg = (post["organelle_state"] == "noremodel").sum()
        elif "signal" in combined.columns:
            annotated = combined.dropna(subset=["signal"])
            pre = annotated[annotated["t_relative_minutes"] < 0]
            post = annotated[annotated["t_relative_minutes"] >= 0]
            pre_pos = int(pre["signal"].sum())
            pre_neg = len(pre) - pre_pos
            post_pos = int(post["signal"].sum())
            post_neg = len(post) - post_pos
        else:
            continue

        if (pre_pos + pre_neg) == 0 or (post_pos + post_neg) == 0:
            continue

        table = [[post_pos, post_neg], [pre_pos, pre_neg]]
        odds_ratio, p_val = fisher_exact(table, alternative="greater")

        stat_rows.append(
            {
                "Test": f"Pre vs post infection ({org})",
                "Method": "Fisher's exact (one-sided)",
                "Statistic": f"OR={odds_ratio:.1f}",
                "p_value": p_val,
                "N1": post_pos + post_neg,
                "N2": pre_pos + pre_neg,
            }
        )

    return pd.DataFrame(stat_rows)
