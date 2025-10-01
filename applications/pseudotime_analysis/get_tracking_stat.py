# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_zarr

from viscy.representation.pseudotime import (
    CytoDtw,
)

# %%
logger = logging.getLogger("viscy")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


features_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_06_26_A549_G3BP1_ZIKV/4-phenotyping/predictions/anndata_predictions/phase_160patch_104ckpt_ver3max.zarr"
# %%
# Load AnnData directly
adata = read_zarr(features_path)
print("Loaded AnnData with shape:", adata.shape)
print("Available columns:", adata.obs.columns.tolist())

# Instantiate the CytoDtw object with AnnData
cytodtw = CytoDtw(adata)
feature_df = cytodtw.adata.obs

min_timepoints = 0
filtered_lineages = cytodtw.get_lineages(min_timepoints)

fov_stats = cytodtw.get_track_statistics(filtered_lineages, per_fov=True)
logger.info("\n=== Confluence Table Format ===")
logger.info(
    "| FOV Name | Lineages | Total Tracks | Tracks/Lineage (mean ± std) | Total Timepoints/Lineage (mean ± std) | Timepoints/Track (mean ± std) |"
)
logger.info(
    "|----------|----------|--------------|------------------------------|---------------------------------------|-------------------------------|"
)
for _, row in fov_stats.iterrows():
    logger.info(
        f"| {row['fov_name']} | {row['n_lineages']} | {row['total_tracks']} | "
        f"{row['mean_tracks_per_lineage']:.2f} ± {row['std_tracks_per_lineage']:.2f} | "
        f"{row['mean_total_timepoints']:.2f} ± {row['std_total_timepoints']:.2f} | "
        f"{row['mean_timepoints_per_track']:.2f} ± {row['std_timepoints_per_track']:.2f} |"
    )

logger.info("\n=== Global Statistics (All FOVs) ===")
min_t = adata.obs["t"].min()
max_t = adata.obs["t"].max()
n_timepoints = max_t - min_t + 1
global_lineages = fov_stats["n_lineages"].sum()
global_tracks = fov_stats["total_tracks"].sum()
logger.info(f"Total Timepoints: ({n_timepoints})")
logger.info(f"Total lineages: {global_lineages}")
logger.info(f"Total tracks: {global_tracks}")
logger.info(
    f"Tracks per lineage (global): {fov_stats['mean_tracks_per_lineage'].mean():.2f} ± {fov_stats['mean_tracks_per_lineage'].std():.2f}"
)
logger.info(
    f"Total timepoints per lineage (global): {fov_stats['mean_total_timepoints'].mean():.2f} ± {fov_stats['mean_total_timepoints'].std():.2f}"
)
logger.info(
    f"Timepoints per track (global): {fov_stats['mean_timepoints_per_track'].mean():.2f} ± {fov_stats['mean_timepoints_per_track'].std():.2f}"
)

track_stats = cytodtw.get_track_statistics(filtered_lineages, per_fov=False)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(
    track_stats["total_timepoints"],
    bins=30,
    color="#1f77b4",
    alpha=0.7,
    edgecolor="black",
)
axes[0, 0].axvline(
    track_stats["total_timepoints"].mean(),
    color="#ff7f0e",
    linestyle="--",
    linewidth=2,
    label=f'Mean: {track_stats["total_timepoints"].mean():.1f}',
)
axes[0, 0].set_xlabel("Total Timepoints per Lineage")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Distribution of Total Timepoints per Lineage")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(
    track_stats["n_tracks"],
    bins=range(1, int(track_stats["n_tracks"].max()) + 2),
    color="#1f77b4",
    alpha=0.7,
    edgecolor="black",
)
axes[0, 1].axvline(
    track_stats["n_tracks"].mean(),
    color="#ff7f0e",
    linestyle="--",
    linewidth=2,
    label=f'Mean: {track_stats["n_tracks"].mean():.2f}',
)
axes[0, 1].set_xlabel("Number of Tracks per Lineage")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Distribution of Tracks per Lineage")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(
    track_stats["mean_timepoints_per_track"],
    bins=30,
    color="#1f77b4",
    alpha=0.7,
    edgecolor="black",
)
axes[1, 0].axvline(
    track_stats["mean_timepoints_per_track"].mean(),
    color="#ff7f0e",
    linestyle="--",
    linewidth=2,
    label=f'Mean: {track_stats["mean_timepoints_per_track"].mean():.1f}',
)
axes[1, 0].set_xlabel("Mean Timepoints per Track")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Distribution of Mean Timepoints per Track")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(
    track_stats["n_tracks"],
    track_stats["total_timepoints"],
    alpha=0.6,
    s=50,
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.5,
)
axes[1, 1].set_xlabel("Number of Tracks")
axes[1, 1].set_ylabel("Total Timepoints")
axes[1, 1].set_title("Tracks vs Total Timepoints")
axes[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
