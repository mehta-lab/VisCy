"""Visual inspection of MultiExperimentDataModule dataloader output.

Jupyter-like notebook (use ``# %%`` cells in VS Code or JupyterLab).
Covers all sampling configurations:

1. Classic triplet (anchor + positive from same lineage)
2. Experiment-aware vs experiment-mixed batches
3. Condition-balanced vs proportional sampling
4. Temporal enrichment (focal HPI concentration)
5. Leaky experiment mixing

Run as a script or step through cells interactively::

    python applications/dynaclr/scripts/dataloader_inspection/inspect_dataloader.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # MultiExperimentDataModule — Dataloader Inspection
#
# This notebook walks through **every sampling mode** of the DynaCLR training
# pipeline to visually verify that the dataloader produces correct cell patches
# under each configuration.
#
# Each batch dict contains:
# - `anchor` / `positive`: `Tensor (B, C, Z, Y, X)`
# - `anchor_meta` / `positive_meta`: `list[dict]` with per-sample metadata
#   (experiment, condition, fov_name, global_track_id, t, hours_post_perturbation, lineage_id)

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# %% [markdown]
# ## Configuration
#
# Edit these paths and parameters for your setup.

# %%
COLLECTION_PATH = (
    "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/collections/A549_ZIKV_multiorganelle.yml"
)
CELL_INDEX_PATH = None  # optional pre-built parquet for faster startup

Z_WINDOW = 30
YX_PATCH_SIZE = (384, 384)
FINAL_YX_PATCH_SIZE = (160, 160)
VAL_EXPERIMENTS: list[str] = []
TAU_RANGE = (0.5, 2.0)
TAU_DECAY_RATE = 2.0
BATCH_SIZE = 8
NUM_WORKERS = 1
N_BATCHES = 3  # batches to pull per scenario
N_SHOW = min(BATCH_SIZE, 6)  # samples per batch to visualize

# %% [markdown]
# ## Helper functions

# %%
from dynaclr.data.datamodule import MultiExperimentDataModule


def build_datamodule() -> MultiExperimentDataModule:
    """Build and setup a MultiExperimentDataModule once (expensive step)."""
    dm = MultiExperimentDataModule(
        collection_path=COLLECTION_PATH,
        z_window=Z_WINDOW,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        val_experiments=VAL_EXPERIMENTS,
        tau_range=TAU_RANGE,
        tau_decay_rate=TAU_DECAY_RATE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        channel_dropout_channels=[1],
        channel_dropout_prob=0.0,  # disabled for inspection
        cell_index_path=CELL_INDEX_PATH,
    )
    dm.setup("fit")
    return dm


def configure_sampling(
    dm: MultiExperimentDataModule,
    experiment_aware: bool = True,
    stratify_by: str | list[str] | None = "condition",
    leaky: float = 0.0,
    temporal_enrichment: bool = False,
    temporal_window_hours: float = 2.0,
    temporal_global_fraction: float = 0.3,
) -> MultiExperimentDataModule:
    """Reconfigure sampling parameters without re-running setup."""
    dm.experiment_aware = experiment_aware
    dm.stratify_by = stratify_by
    dm.leaky = leaky
    dm.temporal_enrichment = temporal_enrichment
    dm.temporal_window_hours = temporal_window_hours
    dm.temporal_global_fraction = temporal_global_fraction
    return dm


def pull_batches(dm: MultiExperimentDataModule, n_batches: int = N_BATCHES) -> list[dict]:
    """Pull n_batches from the train dataloader."""
    dl = dm.train_dataloader()
    batches = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batches.append(batch)
    return batches


def print_batch_meta(batches: list[dict]) -> None:
    """Print per-sample metadata and tensor stats for each batch."""
    for i, batch in enumerate(batches):
        anchor = batch["anchor"]
        positive = batch.get("positive")
        anchor_meta = batch["anchor_meta"]
        positive_meta = batch.get("positive_meta")

        print(f"Batch {i}: anchor {tuple(anchor.shape)}")
        print(f"  anchor  range=[{anchor.min():.3f}, {anchor.max():.3f}]  mean={anchor.mean():.3f}")
        if positive is not None:
            identical = sum(torch.allclose(anchor[j], positive[j]) for j in range(anchor.shape[0]))
            print(f"  positive range=[{positive.min():.3f}, {positive.max():.3f}]  mean={positive.mean():.3f}")
            print(f"  identical anchor-positive pairs: {identical}/{anchor.shape[0]}")

        print()
        for si, am in enumerate(anchor_meta):
            pm = positive_meta[si] if positive_meta is not None else {}
            delta_t = pm.get("t", "?") - am["t"] if positive_meta is not None else "N/A"
            print(
                f"  sample {si}: "
                f"exp={am['experiment']!s:.40s}  cond={am['condition']:<12s}  "
                f"fov={am['fov_name']:<10s}  track={am['global_track_id']!s:.30s}  "
                f"t={am['t']}  hpi={am['hours_post_perturbation']:.1f}  "
                f"lineage={am['lineage_id']!s:.30s}  "
                f"pos_t={pm.get('t', 'N/A')}  delta_t={delta_t}"
            )
        print()


def _short_exp_name(name: str, max_len: int = 20) -> str:
    """Shorten experiment names like '2025_07_22_A549_SEC61_...' to '07_22_A549_SEC61...'."""
    # Drop the year prefix (e.g. "2025_") if present
    parts = name.split("_", 2)
    if len(parts) >= 3 and len(parts[0]) == 4 and parts[0].isdigit():
        short = "_".join(parts[1:])
    else:
        short = name
    if len(short) > max_len:
        short = short[:max_len] + "..."
    return short


def plot_batch_composition(batches: list[dict], title: str) -> None:
    """Bar charts showing experiment and condition composition per batch from batch metadata."""
    n = len(batches)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    fig.suptitle(title, fontsize=14, y=1.01)

    for bi, batch in enumerate(batches):
        meta_df = pd.DataFrame(batch["anchor_meta"])

        # Experiment distribution
        ax = axes[0, bi]
        exp_counts = meta_df["experiment"].value_counts()
        short_labels = [_short_exp_name(name) for name in exp_counts.index]
        bars = ax.barh(short_labels, exp_counts.values, color="steelblue")
        ax.set_title(f"Batch {bi} — experiments", fontsize=10)
        ax.set_xlabel("count")
        for bar, count in zip(bars, exp_counts.values):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
            )

        # Condition distribution
        ax = axes[1, bi]
        cond_counts = meta_df["condition"].value_counts()
        bars = ax.barh(list(cond_counts.index), cond_counts.values, color="coral")
        ax.set_title(f"Batch {bi} — conditions", fontsize=10)
        ax.set_xlabel("count")
        for bar, count in zip(bars, cond_counts.values):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
            )

    plt.tight_layout()


def plot_batch_hpi(batches: list[dict], title: str, hpi_range: tuple[float, float] | None = None) -> None:
    """Histogram of hours_post_perturbation per batch from batch metadata."""
    n = len(batches)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5), squeeze=False)
    fig.suptitle(title, fontsize=14)

    # Compute global HPI range across all batches for consistent axes
    if hpi_range is None:
        all_hpi = np.concatenate([np.array([m["hours_post_perturbation"] for m in b["anchor_meta"]]) for b in batches])
        hpi_range = (float(all_hpi.min()), float(all_hpi.max()))

    for bi, batch in enumerate(batches):
        ax = axes[0, bi]
        hpi = np.array([m["hours_post_perturbation"] for m in batch["anchor_meta"]])
        ax.hist(hpi, bins=20, range=hpi_range, color="mediumpurple", edgecolor="white")
        ax.set_title(f"Batch {bi}", fontsize=10)
        ax.set_xlabel("hours post perturbation")
        ax.set_ylabel("count")
        mean_hpi = hpi.mean()
        ax.axvline(
            mean_hpi,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={mean_hpi:.1f}",
        )
        ax.legend(fontsize=8)

    plt.tight_layout()


def plot_anchor_positive_grid(batches: list[dict], title: str, n_show: int = N_SHOW) -> None:
    """Plot anchor vs positive mid-Z slices per channel, with metadata labels per sample."""
    n_channels = batches[0]["anchor"].shape[1]
    for bi, batch in enumerate(batches):
        anchor = batch["anchor"].numpy()  # (B, C, Z, Y, X)
        positive = batch.get("positive")
        positive = positive.numpy() if positive is not None else None
        anchor_meta = batch["anchor_meta"]
        positive_meta = batch.get("positive_meta")
        mid_z = anchor.shape[2] // 2

        n_rows = n_channels * (2 if positive is not None else 1)
        fig, axes = plt.subplots(n_rows, n_show, figsize=(3 * n_show, 3 * n_rows), squeeze=False)
        fig.suptitle(f"{title} — Batch {bi}, mid-Z (z={mid_z})", fontsize=14)

        for si in range(n_show):
            am = anchor_meta[si]
            col_label = f"s{si} | {am['condition']}\nt={am['t']} hpi={am['hours_post_perturbation']:.1f}"
            if positive_meta is not None:
                pm = positive_meta[si]
                pos_label = f"t={pm['t']} hpi={pm['hours_post_perturbation']:.1f}"

            for ch in range(n_channels):
                ax = axes[ch, si]
                ax.imshow(anchor[si, ch, mid_z], cmap="gray")
                if si == 0:
                    ax.set_ylabel(f"anchor ch{ch}", fontsize=9)
                if ch == 0:
                    ax.set_title(col_label, fontsize=7)
                ax.axis("off")

                if positive is not None:
                    ax = axes[n_channels + ch, si]
                    ax.imshow(positive[si, ch, mid_z], cmap="gray")
                    if si == 0:
                        ax.set_ylabel(f"positive ch{ch}", fontsize=9)
                    if ch == 0:
                        ax.set_title(pos_label, fontsize=7)
                    ax.axis("off")

        plt.tight_layout()


def plot_z_montage(batches: list[dict], title: str = "Z-stack montage") -> None:
    """Plot Z-stack montage for the first sample of the first batch."""
    anchor0 = batches[0]["anchor"][0].numpy()  # (C, Z, Y, X)
    am = batches[0]["anchor_meta"][0]
    n_channels = anchor0.shape[0]
    n_z = anchor0.shape[1]
    n_z_show = min(n_z, 10)
    z_indices = np.linspace(0, n_z - 1, n_z_show, dtype=int)

    fig, axes = plt.subplots(n_channels, n_z_show, figsize=(2.5 * n_z_show, 2.5 * n_channels), squeeze=False)
    fig.suptitle(
        f"{title} — {am['experiment']} | {am['condition']} | fov={am['fov_name']} | t={am['t']}",
        fontsize=12,
    )
    for ch in range(n_channels):
        for zi_col, zi in enumerate(z_indices):
            ax = axes[ch, zi_col]
            ax.imshow(anchor0[ch, zi], cmap="gray")
            if ch == 0:
                ax.set_title(f"z={zi}", fontsize=8)
            ax.axis("off")
        axes[ch, 0].set_ylabel(f"ch{ch}", fontsize=9)
    plt.tight_layout()


# %% [markdown]
# ## Build datamodule (one-time setup)
#
# The expensive step: opens zarr stores, reads tracking CSVs, reconstructs
# lineages, computes valid anchors. Done **once** and reused across all
# scenarios — only the sampler configuration changes.

# %%
dm = build_datamodule()

# %% [markdown]
# ---
# ## 1. Classic Triplet — Anchor + Temporal Positive
#
# The baseline mode: `experiment_aware=True`, `stratify_by="condition"`.
# Each batch draws from a single experiment with balanced conditions.
# The positive is the same cell (same `lineage_id`) at a future timepoint
# `t + tau`, where `tau` is sampled with exponential decay favoring small offsets.

# %%
print("=" * 70)
print("SCENARIO 1: Classic triplet (experiment_aware + stratify_by='condition')")
print("=" * 70)

dm_classic = configure_sampling(dm, experiment_aware=True, stratify_by="condition")

ds = dm_classic.train_dataset
idx = ds.index
print()
print(idx.summary())
print()

va = idx.valid_anchors
for exp_name in va["experiment"].unique():
    exp_df = va[va["experiment"] == exp_name]
    conds = exp_df["condition"].value_counts().to_dict()
    cond_str = ", ".join(f"{k}={v}" for k, v in sorted(conds.items()))
    print(
        f"  {exp_name}: {len(exp_df)} anchors, "
        f"{exp_df['fov_name'].nunique()} fovs, "
        f"{exp_df['global_track_id'].nunique()} tracks, "
        f"t=[{exp_df['t'].min()}, {exp_df['t'].max()}], "
        f"conditions: {cond_str}"
    )
print()

# %% [markdown]
# ### 1a. Batch metadata — single experiment, balanced conditions
#
# Each batch should contain samples from **one experiment only** with
# roughly equal counts of each condition (e.g. ~50% infected, ~50% uninfected).
# The per-sample metadata shows experiment, condition, FOV, track, timepoint,
# HPI, and the delta_t between anchor and positive.

# %%
batches_classic = pull_batches(dm_classic)
print_batch_meta(batches_classic)

# %%
plot_batch_composition(batches_classic, "Scenario 1: Classic (experiment-aware + condition-balanced)")

# %% [markdown]
# ### 1b. Anchor vs positive patches
#
# The positive is the same cell at a different timepoint. Visual similarity
# (same cell morphology, shifted in time) confirms correct lineage-aware sampling.
# Column titles show condition, timepoint, and HPI for each sample.

# %%
plot_anchor_positive_grid(batches_classic, "Scenario 1: Classic triplet")

# %%
plot_z_montage(batches_classic, "Scenario 1: Classic triplet — Z-stack")

# %% [markdown]
# ---
# ## 2. Experiment-Mixed (experiment_aware=False)
#
# Batches draw from the **global pool** of all experiments.
# A single batch can contain cells from different experiments.
# Condition balancing still operates globally.

# %%
print("=" * 70)
print("SCENARIO 2: Experiment-mixed (experiment_aware=False)")
print("=" * 70)

dm_mixed = configure_sampling(dm, experiment_aware=False, stratify_by="condition")

# %% [markdown]
# ### 2a. Batch metadata — mixed experiments, globally balanced conditions
#
# Batches should show **multiple experiments** represented.
# Conditions should still be roughly balanced across all experiments.

# %%
batches_mixed = pull_batches(dm_mixed)
print_batch_meta(batches_mixed)

# %%
plot_batch_composition(batches_mixed, "Scenario 2: Experiment-mixed + condition-balanced")

# %%
plot_anchor_positive_grid(batches_mixed, "Scenario 2: Experiment-mixed")

# %% [markdown]
# ---
# ## 3. No Condition Balancing (stratify_by=None)
#
# Sampling is proportional to the natural distribution of conditions.
# If one condition has 10x more cells, it will dominate the batch.

# %%
print("=" * 70)
print("SCENARIO 3: Experiment-aware, NO stratification (stratify_by=None)")
print("=" * 70)

dm_no_bal = configure_sampling(dm, experiment_aware=True, stratify_by=None)

# %% [markdown]
# ### 3a. Batch metadata — proportional conditions
#
# Conditions should reflect the **natural ratio** in each experiment.
# Compare to Scenario 1 to see the effect of balancing.

# %%
batches_no_bal = pull_batches(dm_no_bal)
print_batch_meta(batches_no_bal)

# %%
plot_batch_composition(batches_no_bal, "Scenario 3: Experiment-aware, NO stratification")

# %%
plot_anchor_positive_grid(batches_no_bal, "Scenario 3: No stratification")

# %% [markdown]
# ---
# ## 4. Temporal Enrichment
#
# Concentrates each batch around a randomly chosen focal HPI
# (hours post perturbation). 70% of the batch comes from cells within
# `temporal_window_hours` of the focal HPI, 30% from all timepoints.
#
# This creates harder in-batch negatives: cells at similar disease stages
# that are NOT the same lineage.
#
# **Note**: temporal enrichment takes priority over condition balancing
# in the sampling cascade.

# %%
print("=" * 70)
print("SCENARIO 4: Temporal enrichment")
print("=" * 70)

dm_temporal = configure_sampling(
    dm,
    experiment_aware=True,
    stratify_by=None,
    temporal_enrichment=True,
    temporal_window_hours=2.0,
    temporal_global_fraction=0.3,
)

# %% [markdown]
# ### 4a. Batch metadata and HPI distribution
#
# Each batch should show a **concentration** around one focal HPI value,
# with a tail from the 30% global fraction. Compare to Scenario 1
# where HPI is not controlled.

# %%
batches_temporal = pull_batches(dm_temporal, n_batches=6)
print_batch_meta(batches_temporal)

# %%
# Global HPI range for consistent axes across scenarios
global_hpi_range = (
    float(va["hours_post_perturbation"].min()),
    float(va["hours_post_perturbation"].max()),
)
plot_batch_hpi(
    batches_temporal,
    "Scenario 4: Temporal enrichment — HPI distribution",
    hpi_range=global_hpi_range,
)

# %%
plot_batch_composition(batches_temporal, "Scenario 4: Temporal enrichment — composition")

# %% [markdown]
# ### 4b. Compare to non-enriched HPI distribution

# %%
plot_batch_hpi(
    batches_classic,
    "Scenario 1 (reference): Classic — HPI distribution (no enrichment)",
    hpi_range=global_hpi_range,
)

# %%
plot_anchor_positive_grid(batches_temporal[:N_BATCHES], "Scenario 4: Temporal enrichment")

# %% [markdown]
# ---
# ## 5. Leaky Experiment Mixing
#
# When `experiment_aware=True` and `leaky > 0`, a fraction of the batch
# is drawn from **other experiments**. This adds cross-experiment diversity
# while keeping batches mostly experiment-pure.
#
# With `leaky=0.3`, 30% of the batch comes from other experiments.

# %%
print("=" * 70)
print("SCENARIO 5: Leaky experiment mixing (leaky=0.3)")
print("=" * 70)

dm_leaky = configure_sampling(dm, experiment_aware=True, stratify_by="condition", leaky=0.3)

# %% [markdown]
# ### 5a. Batch metadata — mostly one experiment with cross-experiment leak
#
# Each batch should be dominated by one experiment (~70%) with
# a minority from other experiments (~30%).

# %%
batches_leaky = pull_batches(dm_leaky)
print_batch_meta(batches_leaky)

# %%
plot_batch_composition(batches_leaky, "Scenario 5: Leaky mixing (30%)")

# %%
plot_anchor_positive_grid(batches_leaky, "Scenario 5: Leaky experiment mixing")

# %% [markdown]
# ---
# ## 6. Multi-Column Stratification (condition + organelle)
#
# Balances batches by the cross-product of condition AND organelle.
# With 2 conditions and 3 organelles, each batch has ~equal representation
# of all 6 (condition, organelle) combinations.
#
# Requires `experiment_aware=False` to mix organelles within a batch
# (since each experiment entry maps to one organelle).

# %%
print("=" * 70)
print("SCENARIO 6: Multi-column stratification (condition + organelle)")
print("=" * 70)

dm_multi_strat = configure_sampling(dm, experiment_aware=False, stratify_by=["condition", "organelle"])

# %%
batches_multi_strat = pull_batches(dm_multi_strat)
print_batch_meta(batches_multi_strat)

# %%
plot_batch_composition(batches_multi_strat, "Scenario 6: stratify_by=[condition, organelle]")

# %% [markdown]
# ---
# ## 7. Fully Random (no experiment-awareness, no stratification)
#
# Baseline: purely random sampling from the global pool.
# Batch composition reflects the natural distribution of experiments
# and conditions proportionally to their sample counts.

# %%
print("=" * 70)
print("SCENARIO 7: Fully random (no experiment-awareness, no stratification)")
print("=" * 70)

dm_random = configure_sampling(dm, experiment_aware=False, stratify_by=None)

# %%
batches_random = pull_batches(dm_random)
print_batch_meta(batches_random)

# %%
plot_batch_composition(batches_random, "Scenario 6: Fully random")

# %%
plot_batch_hpi(
    batches_random,
    "Scenario 6: Fully random — HPI distribution",
    hpi_range=global_hpi_range,
)

# %%
plot_anchor_positive_grid(batches_random, "Scenario 6: Fully random")

# %% [markdown]
# ---
# ## 8. Bag of Channels (bag_of_channels=True)
#
# Each sample reads **one randomly selected source channel** instead of all.
# Output shape is `(B, 1, Z, Y, X)`. This is the "bag of channels" contrastive
# learning approach where the model learns features consistent across all
# channel types (phase, GFP, mCherry, etc.).

# %%
print("=" * 70)
print("SCENARIO 8: Bag of channels (single channel per sample)")
print("=" * 70)

dm_bag = MultiExperimentDataModule(
    collection_path=COLLECTION_PATH,
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    val_experiments=VAL_EXPERIMENTS,
    tau_range=TAU_RANGE,
    tau_decay_rate=TAU_DECAY_RATE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    bag_of_channels=True,
    channel_dropout_channels=[],
    channel_dropout_prob=0.0,
    cell_index_path=CELL_INDEX_PATH,
)
dm_bag.setup("fit")

print(f"Channel names (transforms): {dm_bag._channel_names}")
print(f"Num source channels in registry: {dm_bag.train_dataset.index.registry.num_source_channels}")
print(f"bag_of_channels: {dm_bag.train_dataset.bag_of_channels}")
print()

# %%
batches_bag = pull_batches(dm_bag)
print_batch_meta(batches_bag)

# %% [markdown]
# ### 8a. Verify single-channel output shape
#
# Each sample should have shape `(1, Z, Y, X)` instead of `(C, Z, Y, X)`.

# %%
for bi, batch in enumerate(batches_bag):
    anchor_shape = tuple(batch["anchor"].shape)
    positive_shape = tuple(batch["positive"].shape) if "positive" in batch else None
    print(f"Batch {bi}: anchor={anchor_shape}, positive={positive_shape}")
    assert anchor_shape[1] == 1, f"Expected 1 channel, got {anchor_shape[1]}"
print("\nAll batches have single-channel output.")

# %%
plot_anchor_positive_grid(batches_bag, "Scenario 8: Bag of channels (1 channel per sample)")

# %%
plot_batch_composition(batches_bag, "Scenario 8: Bag of channels — composition")

# %% [markdown]
# ---
# ## 9. Transforms — what `on_after_batch_transfer` does
#
# During training, Lightning calls `on_after_batch_transfer` which applies:
# 1. Normalizations (if any)
# 2. Augmentations (if any)
# 3. Final center crop from `yx_patch_size` -> `final_yx_patch_size`
# 4. ChannelDropout on anchor and positive (skipped when bag_of_channels)
#
# The raw batches above skip this because there's no Trainer.
# Here we apply the transforms manually to see the effect.

# %%
from viscy_data._utils import _transform_channel_wise

batch_raw = batches_classic[0]
anchor_raw = batch_raw["anchor"]
positive_raw = batch_raw.get("positive")
n_channels = anchor_raw.shape[1]
channel_names = dm_classic._channel_names

# Build the same transform pipeline the datamodule uses
transform = dm_classic._augmentation_transform

anchor_transformed = _transform_channel_wise(
    transform=transform,
    channel_names=channel_names,
    patch=anchor_raw,
    norm_meta=None,
)
positive_transformed = (
    _transform_channel_wise(
        transform=transform,
        channel_names=channel_names,
        patch=positive_raw,
        norm_meta=None,
    )
    if positive_raw is not None
    else None
)

# Apply channel dropout
anchor_dropout = dm_classic.channel_dropout(anchor_transformed)
positive_dropout = dm_classic.channel_dropout(positive_transformed) if positive_transformed is not None else None

print(f"Raw anchor shape:         {tuple(anchor_raw.shape)}")
print(f"Transformed anchor shape: {tuple(anchor_transformed.shape)}")
print(f"After dropout shape:      {tuple(anchor_dropout.shape)}")

# %% [markdown]
# ### 9a. Raw vs transformed vs dropout — side by side
#
# Left: raw patch (384x384). Middle: after crop (160x160). Right: after channel dropout.

# %%
mid_z = anchor_raw.shape[2] // 2
am = batch_raw["anchor_meta"][0]
sample_title = f"{am['experiment']} | {am['condition']} | t={am['t']}"

fig, axes = plt.subplots(n_channels, 3, figsize=(10, 4 * n_channels), squeeze=False)
fig.suptitle(f"Transforms pipeline — sample 0\n{sample_title}", fontsize=12)

stage_labels = [
    "Raw (384x384)",
    f"Cropped ({FINAL_YX_PATCH_SIZE[0]}x{FINAL_YX_PATCH_SIZE[1]})",
    "After ChannelDropout",
]
stage_tensors = [anchor_raw[0], anchor_transformed[0], anchor_dropout[0]]

for ch in range(n_channels):
    for col, (label, tensor) in enumerate(zip(stage_labels, stage_tensors)):
        ax = axes[ch, col]
        z_idx = tensor.shape[1] // 2
        ax.imshow(tensor[ch, z_idx].numpy(), cmap="gray")
        if ch == 0:
            ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel(f"ch{ch}", fontsize=9)
        ax.axis("off")

plt.tight_layout()

# %% [markdown]
# ---
# ## 10. Profiling — where is the dataloader slowest?
#
# Profiles `setup()`, sampler iteration, and `__getitems__` (I/O) separately.

# %%
import time


def profile_setup(n_runs: int = 3) -> None:
    """Time datamodule setup (index building, zarr traversal)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        build_datamodule()
        times.append(time.perf_counter() - t0)
    print(f"setup(): {np.mean(times):.2f}s +/- {np.std(times):.2f}s (n={n_runs})")


def profile_sampler(dm: MultiExperimentDataModule, n_batches: int = 50) -> None:
    """Time sampler batch generation (no I/O)."""
    from viscy_data.sampler import FlexibleBatchSampler

    sampler = FlexibleBatchSampler(
        valid_anchors=dm.train_dataset.index.valid_anchors,
        batch_size=BATCH_SIZE,
        experiment_aware=dm.experiment_aware,
        leaky=dm.leaky,
        stratify_by=dm.stratify_by,
        temporal_enrichment=dm.temporal_enrichment,
        temporal_window_hours=dm.temporal_window_hours,
        temporal_global_fraction=dm.temporal_global_fraction,
        seed=dm.seed,
    )
    t0 = time.perf_counter()
    for i, _ in enumerate(sampler):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    print(f"sampler ({n_batches} batches): {elapsed:.4f}s  ({elapsed / n_batches * 1000:.2f} ms/batch)")


def profile_getitems(dm: MultiExperimentDataModule, n_batches: int = 10) -> None:
    """Time __getitems__ (tensorstore I/O + positive sampling)."""
    ds = dm.train_dataset
    va = ds.index.valid_anchors
    rng = np.random.default_rng(42)

    io_times = []
    for _ in range(n_batches):
        indices = rng.choice(len(va), size=BATCH_SIZE, replace=False).tolist()
        t0 = time.perf_counter()
        ds.__getitems__(indices)
        io_times.append(time.perf_counter() - t0)

    print(
        f"__getitems__ ({n_batches} batches of {BATCH_SIZE}): "
        f"{np.mean(io_times):.3f}s +/- {np.std(io_times):.3f}s per batch  "
        f"({np.mean(io_times) / BATCH_SIZE * 1000:.1f} ms/sample)"
    )


def profile_dataloader(dm: MultiExperimentDataModule, n_batches: int = 10) -> None:
    """Time end-to-end dataloader iteration (sampler + I/O + collation)."""
    dl = dm.train_dataloader()
    # Warm up tensorstore caches
    for i, _ in enumerate(dl):
        if i >= 1:
            break

    t0 = time.perf_counter()
    for i, _ in enumerate(dl):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    print(f"dataloader ({n_batches} batches): {elapsed:.2f}s  ({elapsed / n_batches * 1000:.1f} ms/batch)")


# %%
print("=" * 70)
print("PROFILING")
print("=" * 70)
print()

profile_setup(n_runs=2)
print()

configure_sampling(dm, experiment_aware=True, stratify_by="condition")
profile_sampler(dm, n_batches=100)
print()

profile_getitems(dm, n_batches=10)
print()

profile_dataloader(dm, n_batches=10)

# %% [markdown]
# ---
# ## Summary
#
# | Scenario | experiment_aware | stratify_by | temporal_enrichment | leaky | Expected behavior |
# |----------|------------------|--------------------------|---------------------|-------|-------------------|
# | 1. Classic | True | condition | False | 0.0 | Single experiment per batch, equal conditions |
# | 2. Experiment-mixed | False | condition | False | 0.0 | All experiments mixed, globally balanced conditions |
# | 3. No stratification | True | None | False | 0.0 | Single experiment, proportional conditions |
# | 4. Temporal enrichment | True | None | True | 0.0 | HPP-concentrated batches |
# | 5. Leaky mixing | True | condition | False | 0.3 | ~70% primary experiment, ~30% from others |
# | 6. Multi-column strat | False | [condition, organelle] | False | 0.0 | Equal (condition, organelle) groups |
# | 7. Fully random | False | None | False | 0.0 | Natural distribution of everything |
# | 8. Bag of channels | True | condition | False | 0.0 | Single random channel per sample (B,1,Z,Y,X) |
#
# ### Pipeline stages
#
# | Stage | What happens |
# |-------|-------------|
# | `setup()` | Build ExperimentRegistry, open zarrs, build tracks DataFrame, compute valid_anchors |
# | Sampler | Pick experiment -> condition balance -> temporal enrich -> emit index list |
# | `__getitems__` | Tensorstore I/O: slice patches for anchor + sample & slice positive |
# | `on_after_batch_transfer` | Normalizations -> augmentations -> center crop -> channel dropout |

# %%
plt.show()

# %%
