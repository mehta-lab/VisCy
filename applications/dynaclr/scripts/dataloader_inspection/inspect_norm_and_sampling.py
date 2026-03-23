"""Inspect normalization and sampling for DynaCLR-2D-BagOfChannels-v3.

Showcases all three ``channels_per_sample`` modes:

1. **Random** (``channels_per_sample=1``): one random channel per sample.
2. **Fixed** (``channels_per_sample=["labelfree", "reporter_gfp"]``): specific channels.
3. **All** (``channels_per_sample=None``): all source channels.

For each mode, draws 4 batches of 8 samples, applies normalization
(no augmentation), and prints anchor_meta + positive_meta alongside
per-sample pixel statistics.

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/inspect_norm_and_sampling.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # Channel Selection & Normalization Inspector
#
# Demonstrates the three `channels_per_sample` modes side by side.

# %%
from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_data._utils import BatchedCenterSpatialCropd, _transform_channel_wise
from viscy_transforms import NormalizeSampled

# %% [markdown]
# ## Configuration

# %%
CELL_INDEX_PATH = (
    "/hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/DynaCLR-2D-BagOfChannels-v3.parquet"
)
Z_WINDOW = 1
YX_PATCH_SIZE = (192, 192)
FINAL_YX_PATCH_SIZE = (160, 160)
BATCH_SIZE = 8
NUM_WORKERS = 1
N_BATCHES = 4


# %% [markdown]
# ## Helpers


# %%
def build_dm(
    channels_per_sample=1,
    normalizations=None,
    stratify_by=None,
    batch_group_by=None,
    val_experiments=None,
):
    """Build a DataModule with the given config (exercises the real setup path)."""
    if normalizations is None:
        normalizations = [
            NormalizeSampled(
                keys=["channel_0"],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            ),
        ]
    if stratify_by is None:
        stratify_by = ["condition", "marker"]
    dm = MultiExperimentDataModule(
        collection_path=None,
        cell_index_path=CELL_INDEX_PATH,
        z_window=Z_WINDOW,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        channel_dropout_prob=0.0,
        positive_cell_source="lookup",
        positive_match_columns=["lineage_id"],
        tau_range=(0.5, 2.0),
        tau_decay_rate=2.0,
        channels_per_sample=channels_per_sample,
        batch_group_by=batch_group_by,
        stratify_by=stratify_by,
        val_experiments=val_experiments if val_experiments is not None else [],
        normalizations=normalizations,
        seed=42,
    )
    dm.setup("fit")
    return dm


def reconfigure_dm(dm, channels_per_sample, normalizations):
    """Swap channel mode and normalizations without rebuilding the index."""
    from dynaclr.data.dataset import MultiExperimentTripletDataset

    dm.channels_per_sample = channels_per_sample
    dm.normalizations = normalizations
    if channels_per_sample is None:
        dm._channel_names = dm.train_dataset.index.registry.source_channel_labels
    elif isinstance(channels_per_sample, int):
        dm._channel_names = [f"channel_{i}" for i in range(channels_per_sample)]
    else:
        dm._channel_names = list(channels_per_sample)

    train_index = dm.train_dataset.index
    dm.train_dataset = MultiExperimentTripletDataset(
        index=train_index,
        fit=True,
        tau_range_hours=dm.tau_range,
        tau_decay_rate=dm.tau_decay_rate,
        cache_pool_bytes=dm.cache_pool_bytes,
        channels_per_sample=channels_per_sample,
        positive_cell_source=dm.positive_cell_source,
        positive_match_columns=dm.positive_match_columns,
        positive_channel_source=dm.positive_channel_source,
        label_columns=dm.label_columns,
        cross_scope_fraction=dm.cross_scope_fraction,
        hpi_window=dm.hpi_window,
    )
    return dm


def print_index_summary(dm):
    """Print a summary of the training index."""
    va = dm.train_dataset.index.valid_anchors
    print(f"  Anchors: {len(va):,}  |  Experiments: {va['experiment'].nunique()}")
    print(f"  Source channel labels: {dm.train_dataset.index.registry.source_channel_labels}")
    print(f"  Channel names for transforms: {dm._channel_names}")
    print(f"  channels_per_sample: {dm.channels_per_sample}")
    print()


def inspect_batches(dm, title):
    """Draw batches, print metadata and pixel stats, plot anchor+positive pairs."""
    channel_names = dm._channel_names
    norm_transform = Compose(
        dm.normalizations
        + [
            BatchedCenterSpatialCropd(
                keys=channel_names,
                roi_size=(Z_WINDOW, FINAL_YX_PATCH_SIZE[0], FINAL_YX_PATCH_SIZE[1]),
            ),
        ]
    )

    dl = dm.train_dataloader()
    batches = []
    for batch in dl:
        batches.append(batch)
        if len(batches) >= N_BATCHES:
            break

    for bi, batch in enumerate(batches):
        print(f"\n{'=' * 80}")
        print(f"{title} — BATCH {bi}")
        print(f"{'=' * 80}")

        anchor_raw = batch["anchor"]
        positive_raw = batch["positive"]
        anchor_meta = batch["anchor_meta"]
        positive_meta = batch["positive_meta"]
        anchor_norm_meta = batch.get("anchor_norm_meta")
        positive_norm_meta = batch.get("positive_norm_meta")

        anchor_normed = _transform_channel_wise(
            transform=norm_transform,
            channel_names=channel_names,
            patch=anchor_raw,
            norm_meta=anchor_norm_meta,
        )
        positive_normed = _transform_channel_wise(
            transform=norm_transform,
            channel_names=channel_names,
            patch=positive_raw,
            norm_meta=positive_norm_meta,
        )

        n = len(anchor_meta)
        n_channels = anchor_raw.shape[1]

        experiments = Counter(m.get("experiment", "?") for m in anchor_meta)
        conditions = Counter(m.get("condition", "?") for m in anchor_meta)
        print(f"  Shape: anchor={list(anchor_raw.shape)}, positive={list(positive_raw.shape)}")
        print(f"  Experiments: {dict(experiments)}")
        print(f"  Conditions: {dict(conditions)}")

        for i in range(n):
            am = anchor_meta[i]
            pm = positive_meta[i]

            a_raw = anchor_raw[i].float()
            a_norm = anchor_normed[i].float()
            p_raw = positive_raw[i].float()
            p_norm = positive_normed[i].float()

            same_lineage = am.get("lineage_id") == pm.get("lineage_id")
            a_t = am.get("t", "?")
            p_t = pm.get("t", "?")
            delta_t = p_t - a_t if isinstance(a_t, (int, float)) and isinstance(p_t, (int, float)) else "?"

            print(f"\n  Sample {i}:")
            print(f"    anchor_meta:   {am}")
            print(f"    positive_meta: {pm}")
            print(f"    same_lineage={same_lineage}, anchor_frame={a_t}, positive_frame={p_t}, delta_frames={delta_t}")
            for c in range(n_channels):
                ch_label = channel_names[c] if c < len(channel_names) else f"ch{c}"
                print(
                    f"    {ch_label:20s} anchor raw  → mean={a_raw[c].mean():.4f}, "
                    f"std={a_raw[c].std():.4f}, min={a_raw[c].min():.4f}, max={a_raw[c].max():.4f}"
                )
                print(
                    f"    {ch_label:20s} anchor norm → mean={a_norm[c].mean():.4f}, "
                    f"std={a_norm[c].std():.4f}, min={a_norm[c].min():.4f}, max={a_norm[c].max():.4f}"
                )
                print(
                    f"    {ch_label:20s} pos    raw  → mean={p_raw[c].mean():.4f}, "
                    f"std={p_raw[c].std():.4f}, min={p_raw[c].min():.4f}, max={p_raw[c].max():.4f}"
                )
                print(
                    f"    {ch_label:20s} pos    norm → mean={p_norm[c].mean():.4f}, "
                    f"std={p_norm[c].std():.4f}, min={p_norm[c].min():.4f}, max={p_norm[c].max():.4f}"
                )

            if anchor_norm_meta is not None:
                nm = anchor_norm_meta[i]
                if nm is not None:
                    for ch_name, levels in nm.items():
                        for level_name, stats in levels.items():
                            stat_vals = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in stats.items()}
                            print(f"    norm_meta[{ch_name}][{level_name}] = {stat_vals}")

        # Plot: for each channel, 2 rows (anchor normed, positive normed) × n columns (samples)
        n_plot_rows = n_channels * 2
        fig, axes = plt.subplots(n_plot_rows, n, figsize=(n * 2.2, n_plot_rows * 2.0), squeeze=False)
        fig.suptitle(f"{title} — Batch {bi}", fontsize=10)
        for i in range(n):
            am = anchor_meta[i]
            pm = positive_meta[i]
            a_t = am.get("t", "?")
            p_t = pm.get("t", "?")
            for c in range(n_channels):
                ch_label = channel_names[c] if c < len(channel_names) else f"ch{c}"
                row_anchor = c * 2
                row_positive = c * 2 + 1

                img_anchor = anchor_normed[i, c, 0].float().numpy()
                vmin_a, vmax_a = np.percentile(img_anchor, [1, 99])
                ax = axes[row_anchor, i]
                ax.imshow(img_anchor, cmap="gray", vmin=vmin_a, vmax=vmax_a)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel(f"{ch_label}\nanchor", fontsize=7)
                if c == 0:
                    ax.set_title(
                        f"{am.get('experiment', '?')[:12]}\n{am.get('condition', '?')}\nframe={a_t}",
                        fontsize=6,
                    )

                img_positive = positive_normed[i, c, 0].float().numpy()
                vmin_p, vmax_p = np.percentile(img_positive, [1, 99])
                ax = axes[row_positive, i]
                ax.imshow(img_positive, cmap="gray", vmin=vmin_p, vmax=vmax_p)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel(f"{ch_label}\npositive", fontsize=7)
                if c == 0:
                    dt = p_t - a_t if isinstance(a_t, (int, float)) and isinstance(p_t, (int, float)) else "?"
                    ax.set_title(f"frame={p_t}, df={dt}", fontsize=6)

        plt.tight_layout()


# %% [markdown]
# ---
# ## Mode 1: `channels_per_sample=1` (random single channel)
#
# Each sample reads 1 random channel. Transform key: `channel_0`.
# This is the "bag of channels" recipe.
# The base DM is built here (one-time index construction).

# %%
print("\n" + "#" * 80)
print("# MODE 1: channels_per_sample=1 (random single channel)")
print("#" * 80)
dm = build_dm()
print_index_summary(dm)
inspect_batches(dm, "Mode 1: random (1 channel)")

# %% [markdown]
# ---
# ## Mode 1.1: `channels_per_sample=1` with `timepoint_statistics`
#
# Same as Mode 1 but normalizes using per-timepoint stats instead of
# per-FOV stats. This tests that `timepoint_statistics` are correctly
# pre-resolved in the dataset.

# %%
print("\n" + "#" * 80)
print("# MODE 1.1: channels_per_sample=1 (timepoint_statistics)")
print("#" * 80)
reconfigure_dm(
    dm,
    channels_per_sample=1,
    normalizations=[
        NormalizeSampled(
            keys=["channel_0"],
            level="timepoint_statistics",
            subtrahend="mean",
            divisor="std",
        ),
    ],
)
print_index_summary(dm)
inspect_batches(dm, "Mode 1.1: random (1 ch, timepoint norm)")

# %% [markdown]
# ---
# ## Mode 1.2: Same marker + channel per batch
#
# `batch_group_by=["marker", "source_channel_label"]` groups by both
# marker and channel. Each batch is homogeneous: e.g., all TOMM20 GFP,
# or all SEC61B labelfree. `stratify_by=["condition"]` balances
# infected/uninfected within each (marker, channel) group.

# %%
print("\n" + "#" * 80)
print("# MODE 1.2: batch_group_by=[marker, source_channel_label]")
print("#" * 80)
dm = build_dm(batch_group_by=["marker", "source_channel_label"], stratify_by=["condition"])
print_index_summary(dm)
inspect_batches(dm, "Mode 1.2: same marker + channel per batch")

# %% [markdown]
# ---
# ## Mode 2: `channels_per_sample=["labelfree", "reporter_gfp"]` (fixed channels)
#
# Each sample reads exactly labelfree + reporter_gfp. Transform keys
# match the label names.

# %%
print("\n" + "#" * 80)
print("# MODE 2: channels_per_sample=['labelfree', 'reporter_gfp'] (fixed)")
print("#" * 80)
reconfigure_dm(
    dm,
    channels_per_sample=["labelfree", "reporter_gfp"],
    normalizations=[
        NormalizeSampled(
            keys=["labelfree", "reporter_gfp"],
            level="fov_statistics",
            subtrahend="mean",
            divisor="std",
        ),
    ],
)
print_index_summary(dm)
inspect_batches(dm, "Mode 2: fixed [labelfree, reporter_gfp]")

# %% [markdown]
# ---
# ## Mode 3: `channels_per_sample=None` (all channels)
#
# Each sample reads all 3 source channels. Transform keys are the
# source channel labels from the registry.

# %%
print("\n" + "#" * 80)
print("# MODE 3: channels_per_sample=None (all channels)")
print("#" * 80)
reconfigure_dm(
    dm,
    channels_per_sample=None,
    normalizations=[
        NormalizeSampled(
            keys=["labelfree", "reporter_gfp", "reporter_mcherry"],
            level="fov_statistics",
            subtrahend="mean",
            divisor="std",
        ),
    ],
)
print_index_summary(dm)
inspect_batches(dm, "Mode 3: all channels")

# %%
plt.show()

# %%
