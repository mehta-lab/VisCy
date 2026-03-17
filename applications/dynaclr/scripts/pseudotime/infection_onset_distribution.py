# %%
"""
Infection onset timing distribution and phenotype binning.

Measures the absolute time from experiment start to first infection
(T_perturbation) per track, then bins cells by early/mid/late infection
to compare downstream phenotype responses (death, remodeling).

Supports both annotation-based and prediction-based infection timing.

Usage: Run as a Jupyter-compatible script (# %% cell markers).
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# %%
# ===========================================================================
# Configuration
# ===========================================================================

ANNOTATIONS_ROOT = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")
EMBEDDINGS_ROOT = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")

# All experiments start at 3 HPI (hours post-infection).
# t=0 in the data corresponds to 3 HPI, so absolute HPI = t_minutes/60 + T_OFFSET_HPI.
T_OFFSET_HPI = 3.0

EXPERIMENTS = {
    "G3BP1 (Stress Granule)": {
        "datasets": [
            {
                "annotations_path": ANNOTATIONS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "embeddings_path": EMBEDDINGS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 30,
                "label": "2025_07_24 ZIKV",
            },
            {
                "annotations_path": ANNOTATIONS_ROOT
                / "2025_01_24_A549_G3BP1_DENV"
                / "2025_01_24_A549_G3BP1_DENV_combined_annotations.csv",
                "embeddings_path": EMBEDDINGS_ROOT
                / "2025_01_24_A549_G3BP1_DENV"
                / "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3",
                "fov_pattern": "C/2",
                "frame_interval_minutes": 10,
                "label": "2025_01_24 DENV",
            },
        ],
        "remodel_task": "organelle_state_g3bp1",
        "remodel_ann_col": "organelle_state",
        "remodel_positive": "remodel",
    },
    "SEC61B (ER)": {
        "datasets": [
            {
                "annotations_path": ANNOTATIONS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv",
                "embeddings_path": EMBEDDINGS_ROOT
                / "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
                / "4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3",
                "fov_pattern": "A/2",
                "frame_interval_minutes": 30,
                "label": "2025_07_24 ZIKV",
            },
        ],
        "remodel_task": "organelle_state_sec61b",
        "remodel_ann_col": "organelle_state",
        "remodel_positive": "remodel",
    },
}

MIN_TRACK_TIMEPOINTS = 10

# Smoothing: require N consecutive frames of a state before calling it a true event.
# Set to 1 to disable (raw first-frame detection).
MIN_CONSECUTIVE_FRAMES = 3

# Binning strategy: terciles by default, or custom edges
N_BINS = 3

RESULTS_DIR = Path(__file__).parent / "results" / "infection_onset_distribution"

SAVE_FIGURES = False

# %%
# ===========================================================================
# Step 1: Helper — extract per-track events from annotations
# ===========================================================================


def extract_annotation_events(
    ann_df: pd.DataFrame,
    fov_pattern: str,
    frame_interval: float,
    remodel_col: str = "organelle_state",
    remodel_positive: str = "remodel",
) -> pd.DataFrame:
    """Extract per-track first-event timings from annotation CSV."""
    filtered = ann_df[ann_df["fov_name"].astype(str).str.startswith(fov_pattern)].copy()
    has_division = "cell_division_state" in filtered.columns
    rows = []
    for (fov, tid), g in filtered.groupby(["fov_name", "track_id"]):
        if len(g) < MIN_TRACK_TIMEPOINTS:
            continue
        t_start, t_end = g["t"].min(), g["t"].max()
        inf = g[g["infection_state"] == "infected"]
        dead = g[g["cell_death_state"] == "dead"]
        remodel = g[g[remodel_col] == remodel_positive]

        t_infection = inf["t"].min() if len(inf) > 0 else None
        t_death = dead["t"].min() if len(dead) > 0 else None
        t_remodel = remodel["t"].min() if len(remodel) > 0 else None

        t_division = None
        if has_division:
            mitosis = g[g["cell_division_state"] == "mitosis"]
            t_division = mitosis["t"].min() if len(mitosis) > 0 else None

        rows.append(
            {
                "fov_name": fov,
                "track_id": tid,
                "source": "annotation",
                "t_track_start": t_start * frame_interval,
                "t_track_end": t_end * frame_interval,
                "track_duration_min": (t_end - t_start) * frame_interval,
                "t_infection_min": (t_infection * frame_interval if t_infection is not None else None),
                "t_death_min": (t_death * frame_interval if t_death is not None else None),
                "t_remodel_min": (t_remodel * frame_interval if t_remodel is not None else None),
                "t_division_min": (t_division * frame_interval if t_division is not None else None),
                "ever_infected": t_infection is not None,
                "ever_dead": t_death is not None,
                "ever_remodeled": t_remodel is not None,
                "ever_divided": t_division is not None,
            }
        )
    return pd.DataFrame(rows)


# %%
# ===========================================================================
# Step 2: Helper — extract per-track events from predictions
# ===========================================================================


def _first_consecutive_event(
    sorted_t: np.ndarray,
    is_positive: np.ndarray,
    min_consecutive: int,
) -> float | None:
    """Return the t value where min_consecutive consecutive positive frames first occur."""
    if min_consecutive <= 1:
        positives = sorted_t[is_positive]
        return float(positives[0]) if len(positives) > 0 else None

    run = 0
    for i, pos in enumerate(is_positive):
        if pos:
            run += 1
            if run >= min_consecutive:
                return float(sorted_t[i - min_consecutive + 1])
        else:
            run = 0
    return None


def extract_prediction_events(
    embeddings_path: Path,
    fov_pattern: str,
    frame_interval: float,
    remodel_task: str = "organelle_state_g3bp1",
    remodel_positive: str = "remodel",
) -> pd.DataFrame:
    """Extract per-track first-event timings from sensor + organelle + phase zarrs."""
    sensor = ad.read_zarr(embeddings_path / "timeaware_sensor_160patch_104ckpt.zarr")
    organelle = ad.read_zarr(embeddings_path / "timeaware_organelle_160patch_104ckpt.zarr")
    phase = ad.read_zarr(embeddings_path / "timeaware_phase_160patch_104ckpt.zarr")

    sensor_obs = sensor.obs[sensor.obs["fov_name"].astype(str).str.startswith(fov_pattern)].copy()
    organelle_obs = organelle.obs[organelle.obs["fov_name"].astype(str).str.startswith(fov_pattern)].copy()
    phase_obs = phase.obs[phase.obs["fov_name"].astype(str).str.startswith(fov_pattern)].copy()

    merge_keys = ["fov_name", "track_id", "t"]
    pred_remodel_col = f"predicted_{remodel_task}"

    # Check if phase has division predictions
    has_division = "predicted_cell_division_state" in phase_obs.columns

    merged = sensor_obs[merge_keys + ["predicted_infection_state", "predicted_cell_death_state"]].merge(
        organelle_obs[merge_keys + [pred_remodel_col]],
        on=merge_keys,
        how="inner",
    )
    if has_division:
        merged = merged.merge(
            phase_obs[merge_keys + ["predicted_cell_division_state"]],
            on=merge_keys,
            how="inner",
        )

    rows = []
    for (fov, tid), g in merged.groupby(["fov_name", "track_id"]):
        if len(g) < MIN_TRACK_TIMEPOINTS:
            continue
        g = g.sort_values("t")
        t_start, t_end = g["t"].min(), g["t"].max()

        sorted_t = g["t"].to_numpy()
        t_infection = _first_consecutive_event(
            sorted_t,
            (g["predicted_infection_state"] == "infected").values,
            MIN_CONSECUTIVE_FRAMES,
        )
        t_death = _first_consecutive_event(
            sorted_t,
            (g["predicted_cell_death_state"] == "dead").values,
            MIN_CONSECUTIVE_FRAMES,
        )
        t_remodel = _first_consecutive_event(
            sorted_t,
            (g[pred_remodel_col] == remodel_positive).values,
            MIN_CONSECUTIVE_FRAMES,
        )
        t_division = None
        if has_division:
            t_division = _first_consecutive_event(
                sorted_t,
                (g["predicted_cell_division_state"] == "mitosis").values,
                MIN_CONSECUTIVE_FRAMES,
            )

        rows.append(
            {
                "fov_name": fov,
                "track_id": tid,
                "source": "prediction",
                "t_track_start": t_start * frame_interval,
                "t_track_end": t_end * frame_interval,
                "track_duration_min": (t_end - t_start) * frame_interval,
                "t_infection_min": (t_infection * frame_interval if t_infection is not None else None),
                "t_death_min": (t_death * frame_interval if t_death is not None else None),
                "t_remodel_min": (t_remodel * frame_interval if t_remodel is not None else None),
                "t_division_min": (t_division * frame_interval if t_division is not None else None),
                "ever_infected": t_infection is not None,
                "ever_dead": t_death is not None,
                "ever_remodeled": t_remodel is not None,
                "ever_divided": t_division is not None,
            }
        )
    return pd.DataFrame(rows)


# %%
# ===========================================================================
# Step 3: Process all experiments (multiple datasets per organelle)
# ===========================================================================

all_results = {}

for exp_name, cfg in EXPERIMENTS.items():
    print(f"\n{'=' * 60}")
    print(f"  {exp_name}")
    print(f"{'=' * 60}")

    all_ann_events = []
    all_pred_events = []

    for ds in cfg["datasets"]:
        print(f"\n  Dataset: {ds['label']}")

        ann_df = pd.read_csv(ds["annotations_path"])
        ann_ev = extract_annotation_events(
            ann_df,
            fov_pattern=ds["fov_pattern"],
            frame_interval=ds["frame_interval_minutes"],
            remodel_col=cfg["remodel_ann_col"],
            remodel_positive=cfg["remodel_positive"],
        )
        ann_ev["dataset"] = ds["label"]
        all_ann_events.append(ann_ev)
        print(f"    Annotation: {len(ann_ev)} tracks, {ann_ev['ever_infected'].sum()} infected")

        pred_ev = extract_prediction_events(
            embeddings_path=ds["embeddings_path"],
            fov_pattern=ds["fov_pattern"],
            frame_interval=ds["frame_interval_minutes"],
            remodel_task=cfg["remodel_task"],
            remodel_positive=cfg["remodel_positive"],
        )
        pred_ev["dataset"] = ds["label"]
        all_pred_events.append(pred_ev)
        print(f"    Prediction: {len(pred_ev)} tracks, {pred_ev['ever_infected'].sum()} infected")

    ann_events_df = pd.concat(all_ann_events, ignore_index=True)
    pred_events_df = pd.concat(all_pred_events, ignore_index=True)

    # Convert to HPI (hours post-inoculation)
    for df in [ann_events_df, pred_events_df]:
        df["t_infection_hpi"] = df["t_infection_min"] / 60 + T_OFFSET_HPI
        df["t_death_hpi"] = df["t_death_min"] / 60 + T_OFFSET_HPI
        df["t_remodel_hpi"] = df["t_remodel_min"] / 60 + T_OFFSET_HPI
        df["t_division_hpi"] = df["t_division_min"] / 60 + T_OFFSET_HPI

    print(f"\n  Combined annotation: {len(ann_events_df)} tracks, {ann_events_df['ever_infected'].sum()} infected")
    print(f"  Combined prediction: {len(pred_events_df)} tracks, {pred_events_df['ever_infected'].sum()} infected")

    all_results[exp_name] = {
        "cfg": cfg,
        "ann_events_df": ann_events_df,
        "pred_events_df": pred_events_df,
    }

# %%
# ===========================================================================
# Step 4: Bin infected tracks by infection onset time
# ===========================================================================


def bin_and_analyze(events_df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Bin infected tracks by T_infection terciles and summarize phenotypes."""
    infected = events_df[events_df["ever_infected"]].copy()
    if len(infected) < N_BINS:
        print(f"  Too few infected tracks ({len(infected)}) for {N_BINS} bins")
        return infected

    # Tercile binning — labels in HPI (hours post-inoculation)
    _, bin_edges = pd.qcut(infected["t_infection_hpi"], q=N_BINS, retbins=True)
    bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f} HPI" for i in range(len(bin_edges) - 1)]
    infected["infection_bin"] = pd.qcut(
        infected["t_infection_hpi"],
        q=N_BINS,
        labels=bin_labels,
    )

    print(f"\n## {source_label}: Translocation onset bins")
    print(f"  Bin edges (HPI): {[f'{e:.1f}' for e in bin_edges]}")
    print(f"  Labels: {bin_labels}")

    has_division = "ever_divided" in infected.columns

    for bin_label in bin_labels:
        subset = infected[infected["infection_bin"] == bin_label]
        n = len(subset)
        n_dead = subset["ever_dead"].sum()
        n_remodel = subset["ever_remodeled"].sum()

        print(
            f"\n  **{bin_label}** (n={n}, T_inf range: "
            f"{subset['t_infection_min'].min():.0f}-{subset['t_infection_min'].max():.0f} min)"
        )
        print(f"    Death rate: {n_dead}/{n} = {n_dead / max(n, 1):.1%}")
        print(f"    Remodel rate: {n_remodel}/{n} = {n_remodel / max(n, 1):.1%}")

        if has_division:
            n_divided = subset["ever_divided"].sum()
            print(f"    Division rate: {n_divided}/{n} = {n_divided / max(n, 1):.1%}")

        # Time from infection to death/remodel for those that have it
        both_dead = subset[subset["ever_dead"]].copy()
        if len(both_dead) > 0:
            dt = both_dead["t_death_min"] - both_dead["t_infection_min"]
            print(
                f"    Translocation→Death: median={dt.median():.0f} min, mean={dt.mean():.0f} min (n={len(both_dead)})"
            )

        both_remodel = subset[subset["ever_remodeled"]].copy()
        if len(both_remodel) > 0:
            dt = both_remodel["t_remodel_min"] - both_remodel["t_infection_min"]
            print(
                f"    Translocation→Remodel: median={dt.median():.0f} min,"
                f" mean={dt.mean():.0f} min (n={len(both_remodel)})"
            )

        if has_division:
            both_divided = subset[subset["ever_divided"]].copy()
            if len(both_divided) > 0:
                dt = both_divided["t_division_min"] - both_divided["t_infection_min"]
                print(
                    f"    Translocation→Division: median={dt.median():.0f} min,"
                    f" mean={dt.mean():.0f} min (n={len(both_divided)})"
                )

    # Kruskal-Wallis across bins for infection→death, infection→remodel, infection→division
    event_tests = [
        ("Translocation→Death", "t_death_min"),
        ("Translocation→Remodel", "t_remodel_min"),
    ]
    if has_division:
        event_tests.append(("Translocation→Division", "t_division_min"))
    for event_label, event_col in event_tests:
        infected_with_event = infected.dropna(subset=[event_col]).copy()
        infected_with_event["delta"] = infected_with_event[event_col] - infected_with_event["t_infection_min"]
        groups = [g["delta"].to_numpy() for _, g in infected_with_event.groupby("infection_bin") if len(g) >= 2]
        if len(groups) >= 2:
            h_stat, h_p = stats.kruskal(*groups)
            print(f"\n  Kruskal-Wallis ({event_label} across bins): H={h_stat:.2f}, p={h_p:.4g}")

    return infected


for exp_name, res in all_results.items():
    ann_binned = bin_and_analyze(res["ann_events_df"], f"{exp_name} (Annotation)")
    pred_binned = bin_and_analyze(res["pred_events_df"], f"{exp_name} (Prediction)")
    res["ann_binned"] = ann_binned
    res["pred_binned"] = pred_binned

# %%
# ===========================================================================
# Step 5: Plots — per experiment: onset distribution + response histograms
# ===========================================================================

if SAVE_FIGURES:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _plot_kde_by_bin(ax, binned_df, event_col, delta_label):
    """Plot KDE curves of response time per infection bin."""
    if "infection_bin" not in binned_df.columns:
        return
    categories = binned_df["infection_bin"].cat.categories
    for i, bin_label in enumerate(categories):
        subset = binned_df[binned_df["infection_bin"] == bin_label]
        dt = (subset[event_col] - subset["t_infection_min"]).dropna()
        if len(dt) >= 3:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(dt, bw_method="scott")
            x_grid = np.linspace(dt.min() - 30, dt.max() + 30, 200)
            ax.plot(x_grid, kde(x_grid), color=BIN_COLORS[i % len(BIN_COLORS)], linewidth=2)
            ax.fill_between(
                x_grid,
                kde(x_grid),
                alpha=0.15,
                color=BIN_COLORS[i % len(BIN_COLORS)],
                label=f"{bin_label} (n={len(dt)})",
            )
        elif len(dt) > 0:
            ax.axvline(
                dt.median(),
                color=BIN_COLORS[i % len(BIN_COLORS)],
                linestyle="--",
                label=f"{bin_label} (n={len(dt)})",
            )
    ax.legend(fontsize=8)
    ax.set_xlabel(f"{delta_label} (min)")
    ax.set_ylabel("Density")


for exp_name, res in all_results.items():
    ann_infected = res["ann_events_df"][res["ann_events_df"]["ever_infected"]]
    pred_infected = res["pred_events_df"][res["pred_events_df"]["ever_infected"]]
    ann_binned = res["ann_binned"]
    pred_binned = res["pred_binned"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(exp_name, fontsize=14, fontweight="bold")

    # --- Row 1: Annotation-based ---
    ax = axes[0, 0]
    if len(ann_infected) > 0:
        ax.hist(
            ann_infected["t_infection_hpi"],
            bins=20,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="white",
        )
    ax.set_xlabel("T_infection (HPI)")
    ax.set_ylabel("Number of tracks")
    ax.set_title("A. Annotation: infection onset")

    for ax, (delta_label, event_col, panel) in zip(
        [axes[0, 1], axes[0, 2], axes[0, 3]],
        [
            ("Translocation → Death", "t_death_min", "B"),
            ("Translocation → Remodel", "t_remodel_min", "C"),
            ("Translocation → Division", "t_division_min", "D"),
        ],
    ):
        _plot_kde_by_bin(ax, ann_binned, event_col, delta_label)
        ax.set_title(f"{panel}. Annotation: {delta_label}")

    # --- Row 2: Prediction-based ---
    ax = axes[1, 0]
    if len(pred_infected) > 0:
        ax.hist(
            pred_infected["t_infection_hpi"],
            bins=30,
            alpha=0.7,
            color="#ff7f0e",
            edgecolor="white",
        )
    ax.set_xlabel("T_infection (HPI)")
    ax.set_ylabel("Number of tracks")
    ax.set_title("E. Prediction: infection onset")

    for ax, (delta_label, event_col, panel) in zip(
        [axes[1, 1], axes[1, 2], axes[1, 3]],
        [
            ("Translocation → Death", "t_death_min", "F"),
            ("Translocation → Remodel", "t_remodel_min", "G"),
            ("Translocation → Division", "t_division_min", "H"),
        ],
    ):
        _plot_kde_by_bin(ax, pred_binned, event_col, delta_label)
        ax.set_title(f"{panel}. Prediction: {delta_label}")

    plt.tight_layout()
    if SAVE_FIGURES:
        prefix = exp_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(RESULTS_DIR / f"{prefix}_onset_binning.png", dpi=150, bbox_inches="tight")
        fig.savefig(RESULTS_DIR / f"{prefix}_onset_binning.pdf", bbox_inches="tight")
    plt.show()

# %%
# ===========================================================================
# Step 7: Response time comparison — are elapsed times the same across bins?
# ===========================================================================


def plot_response_time_comparison(
    binned_df: pd.DataFrame,
    source_label: str,
    output_dir: Path,
) -> None:
    """Boxplot + swarm of response times per infection bin with pairwise tests."""
    if "infection_bin" not in binned_df.columns:
        return

    # Compute deltas
    binned_df = binned_df.copy()
    binned_df["infection_to_death"] = binned_df["t_death_min"] - binned_df["t_infection_min"]
    binned_df["infection_to_remodel"] = binned_df["t_remodel_min"] - binned_df["t_infection_min"]
    has_division = "t_division_min" in binned_df.columns
    if has_division:
        binned_df["infection_to_division"] = binned_df["t_division_min"] - binned_df["t_infection_min"]

    n_panels = 4 if has_division else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    bin_categories = list(binned_df["infection_bin"].cat.categories)

    # --- Response time boxplots ---
    boxplot_items = [
        ("infection_to_death", "Translocation → Death (min)", "Death"),
        ("infection_to_remodel", "Translocation → Remodel (min)", "Remodel"),
    ]
    if has_division:
        boxplot_items.append(("infection_to_division", "Translocation → Division (min)", "Division"))
    for ax, (delta_col, ylabel, title_suffix) in zip(
        axes[: len(boxplot_items)],
        boxplot_items,
    ):
        plot_data = []
        positions = []
        tick_labels = []
        bin_names = []
        for i, bin_label in enumerate(bin_categories):
            vals = binned_df.loc[binned_df["infection_bin"] == bin_label, delta_col].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                positions.append(i)
                tick_labels.append(f"{bin_label}\n(n={len(vals)})")
                bin_names.append(bin_label)

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{source_label}: {title_suffix}")
            continue

        bp = ax.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.5)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for patch, color in zip(bp["boxes"], colors[: len(plot_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points
        for pos, vals in zip(positions, plot_data):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.4, s=12, color="black", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{source_label}: {title_suffix} response time")
        ax.set_xlabel("Translocation onset bin")

        # Pairwise Mann-Whitney U tests
        test_results = []
        for i in range(len(plot_data)):
            for j in range(i + 1, len(plot_data)):
                if len(plot_data[i]) >= 3 and len(plot_data[j]) >= 3:
                    u_stat, u_p = stats.mannwhitneyu(plot_data[i], plot_data[j], alternative="two-sided")
                    test_results.append(f"{bin_names[i]} vs {bin_names[j]}: p={u_p:.4g}")

        if test_results:
            test_text = "\n".join(test_results)
            ax.text(
                0.98,
                0.98,
                test_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

    # --- Phenotype rates per bin ---
    ax = axes[-1]
    rates = []
    for bin_label in bin_categories:
        subset = binned_df[binned_df["infection_bin"] == bin_label]
        n = len(subset)
        row_dict = {
            "bin": bin_label,
            "death_rate": subset["ever_dead"].sum() / max(n, 1),
            "remodel_rate": subset["ever_remodeled"].sum() / max(n, 1),
            "n": n,
        }
        if has_division:
            row_dict["division_rate"] = subset["ever_divided"].sum() / max(n, 1)
        rates.append(row_dict)
    rates_df = pd.DataFrame(rates)

    x = np.arange(len(bin_categories))
    n_bars = 3 if has_division else 2
    width = 0.8 / n_bars
    ax.bar(
        x - width,
        rates_df["death_rate"],
        width,
        label="Death rate",
        color="#d62728",
        alpha=0.7,
    )
    ax.bar(
        x,
        rates_df["remodel_rate"],
        width,
        label="Remodel rate",
        color="#1f77b4",
        alpha=0.7,
    )
    if has_division:
        ax.bar(
            x + width,
            rates_df["division_rate"],
            width,
            label="Division rate",
            color="#2ca02c",
            alpha=0.7,
        )
    for i, row in rates_df.iterrows():
        max_rate = max(row["death_rate"], row["remodel_rate"])
        if has_division:
            max_rate = max(max_rate, row["division_rate"])
        ax.text(
            i,
            max_rate + 0.02,
            f"n={row['n']}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_categories, rotation=15, ha="right")
    ax.set_ylabel("Fraction of tracks")
    ax.set_title(f"{source_label}: phenotype rates by bin")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    if SAVE_FIGURES:
        prefix = source_label.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"{prefix}_response_time_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        fig.savefig(output_dir / f"{prefix}_response_time_comparison.pdf", bbox_inches="tight")
    plt.show()

    # Print summary table
    print(f"\n## {source_label}: Response time summary (median min)")
    summary_rows = []
    for bin_label in bin_categories:
        subset = binned_df[binned_df["infection_bin"] == bin_label]
        death_dt = subset["infection_to_death"].dropna()
        remodel_dt = subset["infection_to_remodel"].dropna()
        row_dict = {
            "bin": bin_label,
            "n_tracks": len(subset),
            "transloc→death median": (f"{death_dt.median():.0f}" if len(death_dt) > 0 else "—"),
            "transloc→death n": len(death_dt),
            "transloc→remodel median": (f"{remodel_dt.median():.0f}" if len(remodel_dt) > 0 else "—"),
            "transloc→remodel n": len(remodel_dt),
        }
        if has_division:
            division_dt = subset["infection_to_division"].dropna()
            row_dict["transloc→division median"] = f"{division_dt.median():.0f}" if len(division_dt) > 0 else "—"
            row_dict["transloc→division n"] = len(division_dt)
        summary_rows.append(row_dict)
    print(pd.DataFrame(summary_rows).to_string(index=False))


for exp_name, res in all_results.items():
    plot_response_time_comparison(res["pred_binned"], f"{exp_name} (Prediction)", RESULTS_DIR)
    plot_response_time_comparison(res["ann_binned"], f"{exp_name} (Annotation)", RESULTS_DIR)

# %%
# ===========================================================================
# Step 7a: Continuous scatter — HPI vs response time (no binning)
# ===========================================================================


def plot_hpi_vs_response(
    events_df: pd.DataFrame,
    source_label: str,
    output_dir: Path,
) -> None:
    """Scatter plot of translocation onset (HPI) vs response time with regression."""
    infected = events_df[events_df["ever_infected"]].copy()
    if len(infected) < 5:
        print(f"  {source_label}: too few infected tracks ({len(infected)}) for scatter")
        return

    infected["infection_to_death"] = infected["t_death_min"] - infected["t_infection_min"]
    infected["infection_to_remodel"] = infected["t_remodel_min"] - infected["t_infection_min"]

    response_items = [
        ("infection_to_death", "Transloc → Death (min)"),
        ("infection_to_remodel", "Transloc → Remodel (min)"),
    ]
    has_division = "t_division_min" in infected.columns
    if has_division:
        infected["infection_to_division"] = infected["t_division_min"] - infected["t_infection_min"]
        response_items.append(("infection_to_division", "Transloc → Division (min)"))

    n_panels = len(response_items)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"{source_label}: T_translocation vs response time",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (delta_col, xlabel) in zip(axes, response_items):
        valid = infected.dropna(subset=[delta_col])
        x = valid[delta_col].to_numpy()
        y = valid["t_infection_hpi"].to_numpy()

        if len(x) < 3:
            ax.text(
                0.5,
                0.5,
                f"n={len(x)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("T_translocation (HPI)")
            continue

        # Color by division status if available
        if has_division and "ever_divided" in valid.columns:
            divided_mask = valid["ever_divided"].to_numpy()
            ax.scatter(
                x[~divided_mask],
                y[~divided_mask],
                alpha=0.5,
                s=20,
                color="#1f77b4",
                label="No division",
                zorder=2,
            )
            ax.scatter(
                x[divided_mask],
                y[divided_mask],
                alpha=0.7,
                s=30,
                color="#2ca02c",
                marker="^",
                label="Divided",
                zorder=3,
            )
            ax.legend(fontsize=8)
        else:
            ax.scatter(x, y, alpha=0.5, s=20, color="#1f77b4", zorder=2)

        ax.text(
            0.03,
            0.97,
            f"n={len(x)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("T_translocation (HPI)")

    plt.tight_layout()
    if SAVE_FIGURES:
        prefix = source_label.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"{prefix}_hpi_vs_response.png",
            dpi=150,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"{prefix}_hpi_vs_response.pdf",
            bbox_inches="tight",
        )
    plt.show()


for exp_name, res in all_results.items():
    plot_hpi_vs_response(res["pred_events_df"], f"{exp_name} (Prediction)", RESULTS_DIR)
    plot_hpi_vs_response(res["ann_events_df"], f"{exp_name} (Annotation)", RESULTS_DIR)

# %%
# ===========================================================================
# Step 7b: Division confound analysis — do divided cells respond faster?
# ===========================================================================


def plot_division_confound(
    binned_df: pd.DataFrame,
    source_label: str,
    output_dir: Path,
) -> None:
    """Compare response times between divided and non-divided cells.

    Tests whether cells that underwent mitosis have shorter
    translocation→death or translocation→remodel times, which would
    indicate division is a confound for the observed phenotype timing.
    """
    if "ever_divided" not in binned_df.columns:
        return
    if "infection_bin" not in binned_df.columns:
        return

    binned_df = binned_df.copy()
    binned_df["infection_to_death"] = binned_df["t_death_min"] - binned_df["t_infection_min"]
    binned_df["infection_to_remodel"] = binned_df["t_remodel_min"] - binned_df["t_infection_min"]
    binned_df["division_label"] = binned_df["ever_divided"].map({True: "Divided", False: "No division"})

    bin_categories = list(binned_df["infection_bin"].cat.categories)
    response_cols = [
        ("infection_to_death", "Transloc → Death (min)"),
        ("infection_to_remodel", "Transloc → Remodel (min)"),
    ]

    # --- Figure 1: Boxplots stratified by division within each bin ---
    fig, axes = plt.subplots(
        len(response_cols),
        len(bin_categories),
        figsize=(6 * len(bin_categories), 5 * len(response_cols)),
        squeeze=False,
    )
    fig.suptitle(
        f"{source_label}: Response times — Divided vs Not divided",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, (delta_col, ylabel) in enumerate(response_cols):
        for col_idx, bin_label in enumerate(bin_categories):
            ax = axes[row_idx, col_idx]
            subset = binned_df[binned_df["infection_bin"] == bin_label].dropna(subset=[delta_col])
            divided = subset[subset["ever_divided"]][delta_col]
            not_divided = subset[~subset["ever_divided"]][delta_col]

            plot_data = []
            labels = []
            colors_box = []
            if len(not_divided) > 0:
                plot_data.append(not_divided.values)
                labels.append(f"No div\n(n={len(not_divided)})")
                colors_box.append("#1f77b4")
            if len(divided) > 0:
                plot_data.append(divided.values)
                labels.append(f"Divided\n(n={len(divided)})")
                colors_box.append("#2ca02c")

            if len(plot_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                bp = ax.boxplot(
                    plot_data,
                    patch_artist=True,
                    widths=0.5,
                )
                for patch, c in zip(bp["boxes"], colors_box):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
                for pos, vals in enumerate(plot_data, 1):
                    jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
                    ax.scatter(
                        pos + jitter,
                        vals,
                        alpha=0.4,
                        s=12,
                        color="black",
                        zorder=3,
                    )
                ax.set_xticklabels(labels)

                # Mann-Whitney if both groups have enough data
                if len(divided) >= 3 and len(not_divided) >= 3:
                    _, p = stats.mannwhitneyu(not_divided, divided, alternative="two-sided")
                    ax.set_title(f"{bin_label}\np={p:.4g}", fontsize=10)
                else:
                    ax.set_title(bin_label, fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(ylabel)

    plt.tight_layout()
    if SAVE_FIGURES:
        prefix = source_label.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"{prefix}_division_confound.png",
            dpi=150,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"{prefix}_division_confound.pdf",
            bbox_inches="tight",
        )
    plt.show()

    # --- Figure 2: Was division before or after translocation? ---
    infected_divided = binned_df[binned_df["ever_divided"]].dropna(subset=["t_division_min"])
    if len(infected_divided) > 0:
        infected_divided = infected_divided.copy()
        infected_divided["division_relative_to_transloc"] = (
            infected_divided["t_division_min"] - infected_divided["t_infection_min"]
        )
        n_before = (infected_divided["division_relative_to_transloc"] < 0).sum()
        n_after = (infected_divided["division_relative_to_transloc"] >= 0).sum()
        median_dt = infected_divided["division_relative_to_transloc"].median()

        print(f"\n## {source_label}: Division timing relative to translocation")
        print(f"  Divided before translocation: {n_before}/{len(infected_divided)}")
        print(f"  Divided after translocation: {n_after}/{len(infected_divided)}")
        print(f"  Median division–translocation gap: {median_dt:.0f} min")

        # Per-bin breakdown
        for bin_label in bin_categories:
            sub = infected_divided[infected_divided["infection_bin"] == bin_label]
            if len(sub) > 0:
                n_b = (sub["division_relative_to_transloc"] < 0).sum()
                n_a = (sub["division_relative_to_transloc"] >= 0).sum()
                print(
                    f"    {bin_label}: {n_b} before, {n_a} after transloc "
                    f"(median gap: {sub['division_relative_to_transloc'].median():.0f} min)"
                )

    # --- Summary: overall Mann-Whitney (pooled across bins) ---
    print(f"\n## {source_label}: Pooled divided vs not-divided response times")
    for delta_col, label in response_cols:
        valid = binned_df.dropna(subset=[delta_col])
        div_vals = valid[valid["ever_divided"]][delta_col]
        nodiv_vals = valid[~valid["ever_divided"]][delta_col]
        if len(div_vals) >= 3 and len(nodiv_vals) >= 3:
            _, p = stats.mannwhitneyu(nodiv_vals, div_vals, alternative="two-sided")
            print(
                f"  {label}: no-div median={nodiv_vals.median():.0f} min (n={len(nodiv_vals)}), "
                f"div median={div_vals.median():.0f} min (n={len(div_vals)}), "
                f"p={p:.4g}"
            )
        else:
            print(f"  {label}: no-div n={len(nodiv_vals)}, div n={len(div_vals)} — too few for test")


for exp_name, res in all_results.items():
    plot_division_confound(res["pred_binned"], f"{exp_name} (Prediction)", RESULTS_DIR)
    plot_division_confound(res["ann_binned"], f"{exp_name} (Annotation)", RESULTS_DIR)

# %%
# ===========================================================================
# Step 8: Save CSVs
# ===========================================================================

if SAVE_FIGURES:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for exp_name, res in all_results.items():
        prefix = exp_name.replace(" ", "_").replace("(", "").replace(")", "")
        res["ann_events_df"].to_csv(RESULTS_DIR / f"{prefix}_annotation_events.csv", index=False)
        res["pred_events_df"].to_csv(RESULTS_DIR / f"{prefix}_prediction_events.csv", index=False)

        if "infection_bin" in res["ann_binned"].columns:
            res["ann_binned"].to_csv(RESULTS_DIR / f"{prefix}_annotation_binned.csv", index=False)
        if "infection_bin" in res["pred_binned"].columns:
            res["pred_binned"].to_csv(RESULTS_DIR / f"{prefix}_prediction_binned.csv", index=False)

    print(f"\nAll results saved to {RESULTS_DIR}")

# %%
