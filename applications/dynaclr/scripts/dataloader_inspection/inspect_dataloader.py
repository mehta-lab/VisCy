"""DynaCLR dataloader demo — contrastive sampling for biological discovery.

Jupyter-like notebook (use ``# %%`` cells in VS Code or JupyterLab).

Each scenario demonstrates a different contrastive training configuration,
grounded in a specific biological question. The expensive index-building
step (zarr traversal, lineage reconstruction) runs **once** — scenarios
only reconfigure the lightweight sampler.

Run as a script or step through cells interactively::

    python applications/dynaclr/scripts/dataloader_inspection/inspect_dataloader.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # Contrastive Sampling for Multichannel Microscopy
#
# ## Entry points
#
# 1. **Collection YAML** → `ExperimentRegistry.from_collection()`
# 2. **Cell index parquet** → `ExperimentRegistry.from_cell_index()`
#
# ## Batch dict
#
# ```python
# {
#     "anchor":       Tensor(B, C, Z, Y, X),
#     "positive":     Tensor(B, C, Z, Y, X),
#     "anchor_meta":  list[dict],
#     "positive_meta": list[dict],
# }
# ```

# %%
from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dynaclr.data.datamodule import MultiExperimentDataModule

# %% [markdown]
# ## Configuration
#
# Set ONE of `COLLECTION_PATH` or `CELL_INDEX_PATH`.
# When both are set, `collection_path` takes precedence.

# %%
COLLECTION_PATH = None
CELL_INDEX_PATH = (
    "/hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/DynaCLR-2D-BagOfChannels-v3.parquet"
)

Z_WINDOW = 1  # 2D max-projected data
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)
BATCH_SIZE = 16
NUM_WORKERS = 1
N_BATCHES = 3


# %% [markdown]
# ## Helpers


# %%
def pull_batches(dm: MultiExperimentDataModule, n: int = N_BATCHES) -> list[dict]:
    """Pull n batches from the train dataloader."""
    dl = dm.train_dataloader()
    batches = []
    for batch in dl:
        batches.append(batch)
        if len(batches) >= n:
            break
    return batches


def summarize_index(dm: MultiExperimentDataModule) -> None:
    """Print a compact summary of the train index."""
    va = dm.train_dataset.index.valid_anchors
    print(f"  Total anchors: {len(va):,}")
    print(f"  Experiments:   {va['experiment'].nunique()}")
    print(f"  FOVs:          {va['fov_name'].nunique()}")
    for exp, g in va.groupby("experiment"):
        conds = g["condition"].value_counts().to_dict()
        cond_str = ", ".join(f"{k}={v}" for k, v in sorted(conds.items()))
        marker_col = g["marker"] if "marker" in g.columns else pd.Series(dtype=str)
        markers = [str(m) for m in marker_col.dropna().unique()]
        marker_str = f"  marker={','.join(markers)}" if markers else ""
        print(f"    {exp}: {len(g):,} anchors, {cond_str}{marker_str}")
    print()


def reconfigure(
    dm: MultiExperimentDataModule,
    batch_group_by: str | None = "experiment",
    stratify_by: str | list[str] | None = "condition",
    leaky: float = 0.0,
    temporal_enrichment: bool = False,
    temporal_window_hours: float = 2.0,
    temporal_global_fraction: float = 0.3,
) -> MultiExperimentDataModule:
    """Swap sampler params on an existing DataModule (no re-setup)."""
    dm.batch_group_by = batch_group_by
    dm.stratify_by = stratify_by
    dm.leaky = leaky
    dm.temporal_enrichment = temporal_enrichment
    dm.temporal_window_hours = temporal_window_hours
    dm.temporal_global_fraction = temporal_global_fraction
    return dm


def print_batch_meta(batches: list[dict]) -> None:
    """Print batch-level summary: composition and positive pairing."""
    for i, batch in enumerate(batches):
        anchor_meta = batch["anchor_meta"]
        positive_meta = batch.get("positive_meta", [{}] * len(anchor_meta))
        n = len(anchor_meta)

        experiments = Counter(m.get("experiment", "?") for m in anchor_meta)
        conditions = Counter(m.get("condition", "?") for m in anchor_meta)
        markers = Counter(m.get("marker", "?") for m in anchor_meta if m.get("marker"))

        print(f"Batch {i} ({n} samples):")
        print(f"  Experiments: {dict(experiments)}")
        print(f"  Conditions:  {dict(conditions)}")
        if markers:
            print(f"  Markers:     {dict(markers)}")

        if batch.get("positive") is not None:
            same_lineage = sum(
                1 for am, pm in zip(anchor_meta, positive_meta) if am.get("lineage_id") == pm.get("lineage_id")
            )
            identical = sum(torch.allclose(batch["anchor"][j], batch["positive"][j]) for j in range(n))
            print(f"  Same-lineage positives: {same_lineage}/{n}")
            print(f"  Identical patches:      {identical}/{n}")
        print()


def plot_composition(batches: list[dict], title: str, keys: list[str] | None = None) -> None:
    """Bar charts showing metadata distribution per batch."""
    if keys is None:
        keys = ["experiment", "condition"]
    n_batches = len(batches)
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, n_batches, figsize=(4 * n_batches, 3 * n_keys), squeeze=False)
    fig.suptitle(title, fontsize=13, y=1.01)

    for bi, batch in enumerate(batches):
        meta_df = pd.DataFrame(batch["anchor_meta"])
        for ki, key in enumerate(keys):
            ax = axes[ki, bi]
            if key not in meta_df.columns:
                ax.set_visible(False)
                continue
            counts = meta_df[key].value_counts()
            labels = [str(v)[:25] for v in counts.index]
            ax.barh(labels, counts.values, color=f"C{ki}")
            ax.set_title(f"Batch {bi} — {key}", fontsize=9)
            ax.set_xlabel("count")
    plt.tight_layout()


def plot_pairs(batches: list[dict], title: str, n_show: int = 6) -> None:
    """Plot anchor vs positive mid-Z slices."""
    batch = batches[0]
    anchor = batch["anchor"].numpy()
    positive = batch.get("positive")
    positive = positive.numpy() if positive is not None else None
    anchor_meta = batch["anchor_meta"]
    positive_meta = batch.get("positive_meta")
    mid_z = anchor.shape[2] // 2
    n_show = min(n_show, anchor.shape[0])

    n_rows = 2 if positive is not None else 1
    fig, axes = plt.subplots(n_rows, n_show, figsize=(2.5 * n_show, 2.5 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=12)

    for si in range(n_show):
        am = anchor_meta[si]
        label = f"{am.get('condition', '?')}\nt={am.get('t', '?')}"
        ax = axes[0, si]
        ax.imshow(anchor[si, 0, mid_z], cmap="gray")
        ax.set_title(label, fontsize=7)
        ax.axis("off")

        if positive is not None:
            pm = positive_meta[si] if positive_meta else {}
            pos_label = f"t={pm.get('t', '?')}"
            ax = axes[1, si]
            ax.imshow(positive[si, 0, mid_z], cmap="gray")
            ax.set_title(pos_label, fontsize=7)
            ax.axis("off")

    axes[0, 0].set_ylabel("anchor", fontsize=9)
    if n_rows > 1:
        axes[1, 0].set_ylabel("positive", fontsize=9)
    plt.tight_layout()


# %% [markdown]
# ---
# ## Build DataModule (one-time setup)
#
# This is the expensive step: opens zarr stores, reads tracking CSVs,
# reconstructs lineages, computes valid anchors. Done **once** and reused
# across all sampling scenarios.

# %%
print("Building DataModule (this may take a few minutes on first run)...")
print(f"  collection_path: {COLLECTION_PATH}")
print(f"  cell_index_path: {CELL_INDEX_PATH}")

dm = MultiExperimentDataModule(
    collection_path=COLLECTION_PATH,
    cell_index_path=CELL_INDEX_PATH,
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    channel_dropout_prob=0.0,
    # Temporal defaults — will be reconfigured per scenario
    positive_cell_source="lookup",
    positive_match_columns=["lineage_id"],
    tau_range=(0.5, 2.0),
    tau_decay_rate=2.0,
)
dm.setup("fit")
print("Done.")
summarize_index(dm)

# %% [markdown]
# ---
# ## Scenario 1: Self-Positive (SimCLR baseline)
#
# **Biological question**: Can the model learn augmentation-invariant features
# without any temporal or perturbation signal?
#
# Anchor IS the positive. Augmentations create the two views.
# Works with any dataset — no lineage or gene matching needed.
#
# **Note**: This requires `positive_cell_source="self"` which changes the
# dataset, so we build a separate DataModule (cheap — reuses the same
# zarr stores).

# %%
print("=" * 70)
print("SCENARIO 1: Self-positive (SimCLR)")
print("=" * 70)

dm_self = MultiExperimentDataModule(
    collection_path=COLLECTION_PATH,
    cell_index_path=CELL_INDEX_PATH,
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    channel_dropout_prob=0.0,
    positive_cell_source="self",
    batch_group_by=None,
    stratify_by="condition",
    channels_per_sample=1,
)
dm_self.setup("fit")

batches_self = pull_batches(dm_self)
print_batch_meta(batches_self)

# %%
plot_composition(batches_self, "SimCLR: self-positive, condition-balanced")
plot_pairs(batches_self, "SimCLR: anchor = positive (same patch)")

# %% [markdown]
# ---
# ## Scenario 2: Temporal Positive — Learn Dynamics
#
# **Biological question**: What cellular features are preserved across time
# in the same lineage? The positive is the same cell (or daughter) at a
# different timepoint `t + tau`.
#
# `batch_group_by="experiment"` keeps batches within one experiment.
# `stratify_by="condition"` balances infected vs uninfected.

# %%
print("=" * 70)
print("SCENARIO 2: Temporal positive — same lineage, different timepoint")
print("=" * 70)

reconfigure(dm, batch_group_by="experiment", stratify_by="condition")

batches_temporal = pull_batches(dm)
print_batch_meta(batches_temporal)

# %%
plot_composition(batches_temporal, "Temporal: same lineage, balanced conditions")
plot_pairs(batches_temporal, "Temporal: anchor at t, positive at t+tau")

# %% [markdown]
# ---
# ## Scenario 3: Temporal Enrichment — Focal HPI Concentration
#
# **Biological question**: At what HPI do cells undergo the most dramatic
# morphological changes? Concentrating batches around a focal HPI creates
# harder in-batch negatives — cells at similar disease stages that are
# NOT the same lineage.
#
# 70% of each batch comes from within `temporal_window_hours` of a
# randomly chosen focal HPI, 30% from all timepoints.

# %%
print("=" * 70)
print("SCENARIO 3: Temporal enrichment — focal HPI concentration")
print("=" * 70)

reconfigure(
    dm,
    batch_group_by="experiment",
    stratify_by=None,
    temporal_enrichment=True,
    temporal_window_hours=2.0,
    temporal_global_fraction=0.3,
)

batches_enriched = pull_batches(dm, n=6)
print_batch_meta(batches_enriched)

# %%
fig, axes = plt.subplots(1, len(batches_enriched), figsize=(4 * len(batches_enriched), 3), squeeze=False)
fig.suptitle("Temporal enrichment: HPI distribution per batch", fontsize=12)
for bi, batch in enumerate(batches_enriched):
    hpi = np.array([m["hours_post_perturbation"] for m in batch["anchor_meta"]])
    ax = axes[0, bi]
    ax.hist(hpi, bins=15, color="mediumpurple", edgecolor="white")
    ax.axvline(hpi.mean(), color="red", linestyle="--", label=f"mean={hpi.mean():.1f}")
    ax.set_xlabel("HPI")
    ax.set_title(f"Batch {bi}", fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout()

# %% [markdown]
# ---
# ## Scenario 4: Cross-Organelle Invariance
#
# **Biological question**: Do infection-state features generalize across
# organelle markers? If SEC61B (ER) and TOMM20 (mitochondria) show similar
# embedding trajectories, the representation captures a systemic response
# rather than organelle-specific artifacts.
#
# `batch_group_by=None` mixes markers in each batch.
# `stratify_by=["condition", "marker"]` ensures balanced representation.

# %%
print("=" * 70)
print("SCENARIO 4: Cross-marker — stratify by condition + marker")
print("=" * 70)

reconfigure(dm, batch_group_by=None, stratify_by=["condition", "marker"])

batches_cross_marker = pull_batches(dm)
print_batch_meta(batches_cross_marker)

# %%
plot_composition(
    batches_cross_marker,
    "Cross-marker: balanced (condition, marker)",
    keys=["condition", "marker", "experiment"],
)

# %% [markdown]
# ---
# ## Scenario 5: Leaky Experiment Mixing — Batch Effect Awareness
#
# **Biological question**: How much do batch effects confound the embedding?
# Mixing 30% of each batch from other experiments exposes the model to
# cross-batch variation as in-batch negatives.
#
# `batch_group_by="experiment"` + `leaky=0.3` → 70% primary experiment,
# 30% from other experiments.

# %%
print("=" * 70)
print("SCENARIO 5: Leaky mixing — 30% cross-experiment")
print("=" * 70)

reconfigure(dm, batch_group_by="experiment", stratify_by="condition", leaky=0.3)

batches_leaky = pull_batches(dm)
print_batch_meta(batches_leaky)

# %%
plot_composition(batches_leaky, "Leaky mixing: 70% primary + 30% cross-experiment")

# %% [markdown]
# ---
# ## Scenario 6: Bag of Channels — Channel-Invariant Features
#
# **Biological question**: Can the model learn features consistent across
# imaging modalities (phase, GFP, mCherry)? Each sample sees one randomly
# selected channel — output is `(B, 1, Z, Y, X)`.
#
# **Note**: `channels_per_sample=1` changes the dataset, so we build a
# separate DataModule.

# %%
print("=" * 70)
print("SCENARIO 6: Bag of channels — single random channel per sample")
print("=" * 70)

dm_bag = MultiExperimentDataModule(
    collection_path=COLLECTION_PATH,
    cell_index_path=CELL_INDEX_PATH,
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    channel_dropout_prob=0.0,
    positive_cell_source="lookup",
    positive_match_columns=["lineage_id"],
    batch_group_by="experiment",
    stratify_by="condition",
    channels_per_sample=1,
    tau_range=(0.5, 2.0),
)
dm_bag.setup("fit")

batches_bag = pull_batches(dm_bag)
print_batch_meta(batches_bag)

# %%
for bi, batch in enumerate(batches_bag):
    shape = tuple(batch["anchor"].shape)
    assert shape[1] == 1, f"Expected 1 channel, got {shape[1]}"
    print(f"Batch {bi}: anchor shape = {shape}")
print("All batches have single-channel output (B, 1, Z, Y, X).")

# %%
plot_pairs(batches_bag, "Bag of channels: one random channel per sample")

# %% [markdown]
# ---
# ## Summary
#
# | Scenario | positive_cell_source | positive_match_columns | batch_group_by | stratify_by | Biological question |
# |---|---|---|---|---|---|
# | 1. SimCLR | self | — | null | condition | Augmentation-invariant features |
# | 2. Temporal | lookup | lineage_id | experiment | condition | What persists across time in a lineage? |
# | 3. Temporal enriched | lookup | lineage_id | experiment | None | Finer temporal distinctions at key HPI |
# | 4. Cross-marker | lookup | lineage_id | null | [condition, marker] | Systemic vs marker-specific response |
# | 5. Leaky mixing | lookup | lineage_id | experiment | condition | Disentangle biology from batch effects |
# | 6. Bag of channels | lookup | lineage_id | experiment | condition | Channel-invariant representations |
#
# ### Performance notes
#
# - Scenarios 2–5 share one DataModule (sampler-only reconfiguration, no re-setup)
# - Scenarios 1 and 6 need separate DataModules (`positive_cell_source` / `channels_per_sample` change the dataset)
# - Use `cell_index_path` to skip zarr traversal on subsequent runs

# %%
