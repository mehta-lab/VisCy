# %%
"""
Multi-channel correlation: infection, death, and organelle remodeling.

Uses classifier predictions from different channels to ask:
- Do cells that get infected earlier also die faster?
- Is faster death correlated with faster organelle remodeling?

Pipeline:
1. Load sensor zarr → T_perturb (infection onset), T_death (cell death onset)
2. Load organelle zarr → T_remodel (organelle remodeling onset)
3. Merge per-track event timings
4. Correlate and visualize

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

DATASET_ROOT = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics"
    "/2025_01_24_A549_G3BP1_DENV/4-phenotyping/predictions"
    "/DynaCLR-2D-BagOfChannels-timeaware/v3"
)

SENSOR_ZARR = DATASET_ROOT / "timeaware_sensor_160patch_104ckpt.zarr"
ORGANELLE_ZARR = DATASET_ROOT / "timeaware_organelle_160patch_104ckpt.zarr"

FOV_PATTERN = "C/2"  # infected wells
FRAME_INTERVAL_MINUTES = 10
MIN_TRACK_TIMEPOINTS = 3

RESULTS_DIR = Path(__file__).parent / "results" / "infection_death_remodeling"

# %%
# ===========================================================================
# Step 1: Load data and filter to infected wells
# ===========================================================================

sensor = ad.read_zarr(SENSOR_ZARR)
organelle = ad.read_zarr(ORGANELLE_ZARR)

print(f"Sensor: {sensor.shape[0]:,} cells")
print(f"Organelle: {organelle.shape[0]:,} cells")

# Filter to infected FOVs
sensor_obs = sensor.obs[sensor.obs["fov_name"].astype(str).str.startswith(FOV_PATTERN)].copy()
organelle_obs = organelle.obs[organelle.obs["fov_name"].astype(str).str.startswith(FOV_PATTERN)].copy()

print(f"\nAfter FOV filter ({FOV_PATTERN}):")
print(f"  Sensor: {len(sensor_obs):,} cells")
print(f"  Organelle: {len(organelle_obs):,} cells")

# %%
# ===========================================================================
# Step 2: Build per-cell merged dataframe
# ===========================================================================

merge_keys = ["fov_name", "track_id", "t"]

sensor_cols = merge_keys + [
    "predicted_infection_state",
    "predicted_cell_death_state",
]
organelle_cols = merge_keys + [
    "predicted_organelle_state_g3bp1",
]

merged = sensor_obs[sensor_cols].merge(
    organelle_obs[organelle_cols],
    on=merge_keys,
    how="inner",
)

merged["t_minutes"] = merged["t"] * FRAME_INTERVAL_MINUTES

print(f"\nMerged: {len(merged):,} cells across {merged.groupby(['fov_name', 'track_id']).ngroups} tracks")
print(f"  Infection: {merged['predicted_infection_state'].value_counts().to_dict()}")
print(f"  Death: {merged['predicted_cell_death_state'].value_counts().to_dict()}")
print(f"  Remodel: {merged['predicted_organelle_state_g3bp1'].value_counts().to_dict()}")

# %%
# ===========================================================================
# Step 3: Compute per-track event timings
# ===========================================================================


def find_first_event(group: pd.DataFrame, col: str, value: str) -> float | None:
    """Return t_minutes of the first frame matching value, or None."""
    hits = group.loc[group[col] == value, "t_minutes"]
    if len(hits) > 0:
        return hits.min()
    return None


track_events = []
for (fov, tid), group in merged.groupby(["fov_name", "track_id"]):
    group = group.sort_values("t")
    n_frames = len(group)
    if n_frames < MIN_TRACK_TIMEPOINTS:
        continue

    t_start = group["t_minutes"].min()
    t_end = group["t_minutes"].max()
    track_duration = t_end - t_start

    t_infection = find_first_event(group, "predicted_infection_state", "infected")
    t_death = find_first_event(group, "predicted_cell_death_state", "dead")
    t_remodel = find_first_event(group, "predicted_organelle_state_g3bp1", "remodel")

    # Was cell ever infected, dead, remodeled?
    ever_infected = t_infection is not None
    ever_dead = t_death is not None
    ever_remodeled = t_remodel is not None

    # Time from infection to death / remodeling
    infection_to_death = (t_death - t_infection) if (ever_infected and ever_dead) else None
    infection_to_remodel = (t_remodel - t_infection) if (ever_infected and ever_remodeled) else None
    remodel_to_death = (t_death - t_remodel) if (ever_remodeled and ever_dead) else None

    track_events.append(
        {
            "fov_name": fov,
            "track_id": tid,
            "n_frames": n_frames,
            "track_duration_min": track_duration,
            "t_infection_min": t_infection,
            "t_death_min": t_death,
            "t_remodel_min": t_remodel,
            "ever_infected": ever_infected,
            "ever_dead": ever_dead,
            "ever_remodeled": ever_remodeled,
            "infection_to_death_min": infection_to_death,
            "infection_to_remodel_min": infection_to_remodel,
            "remodel_to_death_min": remodel_to_death,
        }
    )

events_df = pd.DataFrame(track_events)

print(f"\n## Track Event Summary ({len(events_df)} tracks)")
print(f"  Ever infected: {events_df['ever_infected'].sum()}")
print(f"  Ever dead: {events_df['ever_dead'].sum()}")
print(f"  Ever remodeled: {events_df['ever_remodeled'].sum()}")
print(f"  Infected & dead: {(events_df['ever_infected'] & events_df['ever_dead']).sum()}")
print(f"  Infected & remodeled: {(events_df['ever_infected'] & events_df['ever_remodeled']).sum()}")
print(f"  All three: {(events_df['ever_infected'] & events_df['ever_dead'] & events_df['ever_remodeled']).sum()}")

# %%
# ===========================================================================
# Step 4: Descriptive statistics
# ===========================================================================

infected_tracks = events_df[events_df["ever_infected"]].copy()

print("\n## Timing distributions (infected tracks only)")
for col_label, col in [
    ("Infection → Death", "infection_to_death_min"),
    ("Infection → Remodel", "infection_to_remodel_min"),
    ("Remodel → Death", "remodel_to_death_min"),
]:
    valid = infected_tracks[col].dropna()
    if len(valid) > 0:
        print(f"\n  **{col_label}** (n={len(valid)})")
        print(f"    median: {valid.median():.0f} min, mean: {valid.mean():.0f} min, std: {valid.std():.0f} min")
        print(f"    range: [{valid.min():.0f}, {valid.max():.0f}] min")

# Compare death rates: infected vs uninfected
infected_dead = events_df["ever_infected"] & events_df["ever_dead"]
uninfected_dead = ~events_df["ever_infected"] & events_df["ever_dead"]
n_infected = events_df["ever_infected"].sum()
n_uninfected = (~events_df["ever_infected"]).sum()

print("\n## Death rates")
print(f"  Infected tracks: {infected_dead.sum()}/{n_infected} = {infected_dead.sum() / max(n_infected, 1):.1%}")
print(
    f"  Uninfected tracks: {uninfected_dead.sum()}/{n_uninfected} = {uninfected_dead.sum() / max(n_uninfected, 1):.1%}"
)

if n_infected > 0 and n_uninfected > 0:
    table = np.array(
        [
            [infected_dead.sum(), n_infected - infected_dead.sum()],
            [uninfected_dead.sum(), n_uninfected - uninfected_dead.sum()],
        ]
    )
    chi2, p_val, _, _ = stats.chi2_contingency(table)
    print(f"  Chi-squared: {chi2:.2f}, p={p_val:.4g}")

# %%
# ===========================================================================
# Step 5: Correlation — infection_to_death vs infection_to_remodel
# ===========================================================================

both = infected_tracks.dropna(subset=["infection_to_death_min", "infection_to_remodel_min"]).copy()

print(f"\n## Correlation: Infection→Death vs Infection→Remodel (n={len(both)})")

if len(both) >= 5:
    r_pearson, p_pearson = stats.pearsonr(both["infection_to_remodel_min"], both["infection_to_death_min"])
    r_spearman, p_spearman = stats.spearmanr(both["infection_to_remodel_min"], both["infection_to_death_min"])
    print(f"  Pearson r={r_pearson:.3f}, p={p_pearson:.4g}")
    print(f"  Spearman rho={r_spearman:.3f}, p={p_spearman:.4g}")

    # Bin tracks into early/late remodelers (median split)
    median_remodel = both["infection_to_remodel_min"].median()
    both["remodel_speed"] = np.where(
        both["infection_to_remodel_min"] <= median_remodel, "early_remodel", "late_remodel"
    )

    for label, subdf in both.groupby("remodel_speed"):
        death_times = subdf["infection_to_death_min"]
        print(
            f"\n  {label} (n={len(subdf)}): death at median {death_times.median():.0f} min,"
            f" mean {death_times.mean():.0f} min"
        )

    early = both.loc[both["remodel_speed"] == "early_remodel", "infection_to_death_min"]
    late = both.loc[both["remodel_speed"] == "late_remodel", "infection_to_death_min"]
    if len(early) >= 3 and len(late) >= 3:
        u_stat, u_p = stats.mannwhitneyu(early, late, alternative="two-sided")
        print(f"\n  Mann-Whitney U test (early vs late remodelers death time): U={u_stat:.0f}, p={u_p:.4g}")

# %%
# ===========================================================================
# Step 6: Plots
# ===========================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Panel A: Scatter of infection→remodel vs infection→death ---
ax = axes[0, 0]
if len(both) >= 5:
    ax.scatter(
        both["infection_to_remodel_min"],
        both["infection_to_death_min"],
        alpha=0.4,
        s=15,
        edgecolors="none",
    )
    # Regression line
    slope, intercept, _, _, _ = stats.linregress(both["infection_to_remodel_min"], both["infection_to_death_min"])
    x_fit = np.linspace(both["infection_to_remodel_min"].min(), both["infection_to_remodel_min"].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r--", label=f"r={r_pearson:.2f}, p={p_pearson:.2g}")
    ax.legend()
ax.set_xlabel("Infection → Remodel (min)")
ax.set_ylabel("Infection → Death (min)")
ax.set_title("A. Remodeling vs Death timing")

# --- Panel B: Distribution of infection→death for infected vs all tracks ---
ax = axes[0, 1]
infected_death_times = infected_tracks["infection_to_death_min"].dropna()
if len(infected_death_times) > 0:
    ax.hist(infected_death_times, bins=30, alpha=0.7, color="#d62728", edgecolor="white")
ax.set_xlabel("Infection → Death (min)")
ax.set_ylabel("Number of tracks")
ax.set_title("B. Time from infection to death")

# --- Panel C: Death rate comparison ---
ax = axes[1, 0]
categories = ["Infected", "Uninfected"]
dead_counts = [infected_dead.sum(), uninfected_dead.sum()]
alive_counts = [n_infected - infected_dead.sum(), n_uninfected - uninfected_dead.sum()]
x = np.arange(len(categories))
width = 0.35
ax.bar(x - width / 2, dead_counts, width, label="Dead", color="#d62728")
ax.bar(x + width / 2, alive_counts, width, label="Alive", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Number of tracks")
ax.set_title("C. Death rates by infection status")
ax.legend()

# --- Panel D: Boxplot of death timing by remodel speed ---
ax = axes[1, 1]
if len(both) >= 5:
    early_vals = both.loc[both["remodel_speed"] == "early_remodel", "infection_to_death_min"].to_numpy()
    late_vals = both.loc[both["remodel_speed"] == "late_remodel", "infection_to_death_min"].to_numpy()
    bp = ax.boxplot(
        [early_vals, late_vals],
        labels=["Early remodelers", "Late remodelers"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][1].set_facecolor("#ff7f0e")
    ax.set_ylabel("Infection → Death (min)")
    ax.set_title("D. Death timing by remodel speed")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "infection_death_remodeling.png", dpi=150, bbox_inches="tight")
fig.savefig(RESULTS_DIR / "infection_death_remodeling.pdf", bbox_inches="tight")
plt.show()
print(f"Saved to {RESULTS_DIR}")

# %%
# ===========================================================================
# Step 7: Timeline heatmap — per-track state over time
# ===========================================================================

# Show a sample of infected tracks with all 3 states over time
infected_tids = infected_tracks.sort_values("t_infection_min").head(50)
sample_keys = set(zip(infected_tids["fov_name"], infected_tids["track_id"]))

sample = merged[merged.apply(lambda r: (r["fov_name"], r["track_id"]) in sample_keys, axis=1)].copy()

if len(sample) > 0:
    # Align to infection time
    sample = sample.merge(
        infected_tids[["fov_name", "track_id", "t_infection_min"]],
        on=["fov_name", "track_id"],
    )
    sample["t_rel"] = sample["t_minutes"] - sample["t_infection_min"]

    # Encode states as numeric for heatmap
    sample["infection_num"] = (sample["predicted_infection_state"] == "infected").astype(int)
    sample["death_num"] = (sample["predicted_cell_death_state"] == "dead").astype(int)
    sample["remodel_num"] = (sample["predicted_organelle_state_g3bp1"] == "remodel").astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    time_bins = np.arange(sample["t_rel"].min(), sample["t_rel"].max() + FRAME_INTERVAL_MINUTES, FRAME_INTERVAL_MINUTES)

    track_labels = []
    for i, ((fov, tid), _) in enumerate(infected_tids.iterrows()):
        track_labels.append(f"{fov}:{tid}")

    for ax, (title, col) in zip(
        axes,
        [
            ("Infection", "infection_num"),
            ("Death", "death_num"),
            ("Remodeling", "remodel_num"),
        ],
    ):
        # Pivot: rows=tracks, cols=time bins
        track_list = list(zip(infected_tids["fov_name"], infected_tids["track_id"]))
        matrix = np.full((len(track_list), len(time_bins) - 1), np.nan)

        for i, (fov, tid) in enumerate(track_list):
            track_data = sample[(sample["fov_name"] == fov) & (sample["track_id"] == tid)]
            for _, row in track_data.iterrows():
                bin_idx = np.searchsorted(time_bins, row["t_rel"]) - 1
                if 0 <= bin_idx < matrix.shape[1]:
                    matrix[i, bin_idx] = row[col]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel("Time relative to infection (min)")
        ax.set_title(title)

        # Set x tick labels
        n_ticks = min(10, len(time_bins))
        tick_positions = np.linspace(0, len(time_bins) - 2, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{time_bins[t]:.0f}" for t in tick_positions], rotation=45)

    axes[0].set_ylabel("Tracks (sorted by infection time)")
    plt.colorbar(im, ax=axes[-1], label="State (0=no, 1=yes)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "track_timeline_heatmap.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "track_timeline_heatmap.pdf", bbox_inches="tight")
    plt.show()

# %%
# ===========================================================================
# Step 8: Save results
# ===========================================================================

events_df.to_csv(RESULTS_DIR / "track_events.csv", index=False)
if len(both) > 0:
    both.to_csv(RESULTS_DIR / "infected_remodeled_dead_tracks.csv", index=False)

print(f"\nAll results saved to {RESULTS_DIR}")

# %%
