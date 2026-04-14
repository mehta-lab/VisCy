"""Minimal exploration of Zuben's gut cell classifier parquet with DynaCLR dataloader.

Parquet: /hpc/projects/jacobo_group/zuben/proj/gutCellClassifier/data/dynaclr_cell_index.parquet

Key findings:
- Flat schema: one row per (cell, t, channel). Compatible with MultiExperimentDataModule.
- NOT timelapse: all t=0, no temporal positives. Use positive_cell_source="self" (SimCLR).
- 25 experiments (AAY6/7/8 × day 0/1/2 × gut1-6), 4 channels, 6 perturbation stages.
- Missing: hours_post_perturbation (not needed for self-positive mode).

Usage::

   cd /home/eduardo.hirata/repos/viscy
   uv run python applications/dynaclr/scripts/dataloader_inspection/explore_gut_parquet.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # Gut Cell Parquet Explorer

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

# %% [markdown]
# ## 1. Parquet Summary

# %%
PARQUET_PATH = "/hpc/projects/jacobo_group/zuben/proj/gutCellClassifier/data/dynaclr_cell_index.parquet"
OUTPUT_DIR = Path("applications/dynaclr/scripts/dataloader_inspection/output/gut_parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

print(f"Experiments ({df['experiment'].nunique()}): {sorted(df['experiment'].unique())}\n")
print(f"Channels: {df['channel_name'].unique().tolist()}")
print(f"Perturbations: {sorted(df['perturbation'].unique())}")
print(f"t values: {sorted(df['t'].unique())}  <- all 0, not timelapse")
print(f"z range: {df['z'].min()} - {df['z'].max()}")

# %%
# Per-experiment cell counts and stage breakdown
print("\n## Per-experiment cell counts (unique cells × 4 channels = rows)")
for exp, g in df.groupby("experiment"):
    n_cells = g["cell_id"].nunique()
    stages = g["perturbation"].value_counts().to_dict()
    print(f"  {exp}: {n_cells} cells  |  stages={stages}")

# %% [markdown]
# ## 2. Sample random patches from zarr
#
# Direct zarr read bypasses the iohub channel_names issue.
# Array shape: (T, C, Z, Y, X) = (1, 4, ~98, H, W)
# Channel order: nuclear, septate, brush_border, SuH

CHANNEL_NAMES = ["nuclear", "septate", "brush_border", "SuH"]
PATCH_SIZE = 128  # pixels around cell center
N_SAMPLES_PER_CHANNEL = 4
N_STAGES = 3  # show first N stages


def read_patch(row: pd.Series, channel_idx: int, patch: int = PATCH_SIZE) -> np.ndarray | None:
    """Read a 2D patch around the cell center from zarr."""
    store = zarr.open(row["store_path"], mode="r")
    pos_path = f"{row['well']}/{row['fov']}"
    arr = store[pos_path]["0"]  # (T, C, Z, Y, X)
    z = int(row["z"])
    y = int(row["y"])
    x = int(row["x"])
    H, W = arr.shape[3], arr.shape[4]
    half = patch // 2
    y0, y1 = max(0, y - half), min(H, y + half)
    x0, x1 = max(0, x - half), min(W, x + half)
    t = int(row["t"])
    return arr[t, channel_idx, z, y0:y1, x0:x1]


# %% [markdown]
# ## 3. Grid: channels × perturbation stages

# %%
stages = sorted(df["perturbation"].unique())[:N_STAGES]
n_cols = N_SAMPLES_PER_CHANNEL
n_rows = len(CHANNEL_NAMES) * len(stages)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)
fig.suptitle("Gut cell patches: rows=channel×stage, cols=random samples", fontsize=10)

row_idx = 0
for stage in stages:
    stage_df = df[df["perturbation"] == stage]
    for ch_i, ch_name in enumerate(CHANNEL_NAMES):
        ch_df = stage_df[stage_df["channel_name"] == ch_name]
        sampled = ch_df.sample(min(N_SAMPLES_PER_CHANNEL, len(ch_df)), random_state=42)
        ax_row = axes[row_idx]
        for col_i, (_, row) in enumerate(sampled.iterrows()):
            patch = read_patch(row, ch_i)
            ax = ax_row[col_i]
            vmin, vmax = np.percentile(patch, [1, 99])
            ax.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(f"{ch_name}\n{stage}", fontsize=7)
        row_idx += 1

plt.tight_layout()
save_path = OUTPUT_DIR / "patches_channel_by_stage.png"
fig.savefig(save_path, dpi=120, bbox_inches="tight")
print(f"Saved: {save_path}")

# %% [markdown]
# ## 4. Stage distribution per experiment

# %%
fig, ax = plt.subplots(figsize=(14, 4))
pivot = (
    df.drop_duplicates(["cell_id", "perturbation"]).groupby(["experiment", "perturbation"]).size().unstack(fill_value=0)  # noqa: PD010
)
pivot.plot.bar(ax=ax, stacked=True, colormap="tab10")
ax.set_title("Cell counts by experiment and stage")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="stage", bbox_to_anchor=(1, 1))
plt.tight_layout()
save_path = OUTPUT_DIR / "stage_distribution.png"
fig.savefig(save_path, dpi=120, bbox_inches="tight")
print(f"Saved: {save_path}")

# %% [markdown]
# ## 5. Channel distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.drop_duplicates(["cell_id", "channel_name"])["channel_name"].value_counts().plot.bar(ax=axes[0], color="steelblue")
axes[0].set_title("Cells per channel")
axes[0].tick_params(axis="x", rotation=30)

df.drop_duplicates(["cell_id", "perturbation"])["perturbation"].value_counts().plot.bar(ax=axes[1], color="coral")
axes[1].set_title("Cells per stage")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
save_path = OUTPUT_DIR / "distributions.png"
fig.savefig(save_path, dpi=120, bbox_inches="tight")
print(f"Saved: {save_path}")

# %% [markdown]
# ## 6. DynaCLR DataModule (self-positive / SimCLR)
#
# Not timelapse (t=0 only) so use positive_cell_source="self" —
# augmentation creates two views of the same cell.

# %%
from dynaclr.data.datamodule import MultiExperimentDataModule

Z_WINDOW = 1
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_WORKERS = 4
N_BATCHES = 2

print("Building DataModule (self-positive, marker-grouped)...")
dm = MultiExperimentDataModule(
    cell_index_path=PARQUET_PATH,
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    channel_dropout_prob=0.0,
    positive_cell_source="self",
    channels_per_sample=1,
    batch_group_by=["marker"],
    stratify_by="perturbation",
)
dm.setup("fit")
print("Done.\n")

va = dm.train_dataset.index.valid_anchors
print(f"Valid anchors: {len(va):,}")
print(f"Channels: {va['marker'].value_counts().to_dict()}")
print(f"Perturbations: {va['perturbation'].value_counts().to_dict()}")


# %%
def plot_batch(batch: dict, batch_idx: int, title: str, save_path: Path | None = None) -> None:
    """Grid of anchor images annotated with channel + perturbation."""
    anchor = batch["anchor"].numpy()
    meta = batch["anchor_meta"]
    n = len(meta)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.8), squeeze=False)
    channels_in_batch = {m.get("marker", "?") for m in meta}
    perts_in_batch = {m.get("perturbation", "?") for m in meta}
    fig.suptitle(
        f"{title} — Batch {batch_idx}\nchannel={channels_in_batch}  |  stages={perts_in_batch}",
        fontsize=9,
    )
    for i, (ax, m) in enumerate(zip(axes[0], meta)):
        img = anchor[i]
        if img.ndim == 4:
            img = img[0, img.shape[1] // 2]
        elif img.ndim == 3:
            img = img[0]
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{m.get('marker', '?')}\n{m.get('perturbation', '?')}", fontsize=6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")


dl = dm.train_dataloader()
for i, batch in enumerate(dl):
    if i >= N_BATCHES:
        break
    meta = batch["anchor_meta"]
    print(f"Batch {i}: {len(meta)} samples  marker={{{meta[0].get('marker')}}}  anchor shape={batch['anchor'].shape}")
    plot_batch(
        batch, i, "Gut: marker-grouped, perturbation-stratified", save_path=OUTPUT_DIR / f"dataloader_batch_{i}.png"
    )

# %%
plt.show()
