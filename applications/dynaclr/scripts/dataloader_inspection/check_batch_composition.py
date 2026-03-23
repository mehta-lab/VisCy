"""Visual verification of batch composition for DynaCLR contrastive training.

Jupyter-like notebook (use ``# %%`` cells in VS Code or JupyterLab).

For each sampling configuration, draws N_BATCHES batches and creates a
figure per batch showing anchor/positive pairs side by side with metadata
annotations. Checkmarks verify that the sampling contract holds:

- Same-lineage positives (temporal mode)
- Same-condition batches (stratified)
- Same-experiment batches (experiment-aware)
- Single-channel output (bag-of-channels)

Usage::

    python applications/dynaclr/scripts/dataloader_inspection/check_batch_composition.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # Batch Composition Checker
#
# Visual QC for contrastive batch sampling. Each figure = one batch.
# Columns = anchor/positive pairs. Annotations show metadata + pass/fail.

# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule

# %% [markdown]
# ## Configuration

# %%
COLLECTION_PATH = (
    "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/collections/DynaCLR-2D-BagOfChannels-v3.yml"
)
CELL_INDEX_PATH = None

Z_WINDOW = 1
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)
BATCH_SIZE = 10
NUM_WORKERS = 1
N_BATCHES = 3
OUTPUT_DIR = None  # set to a Path to save figures, e.g. Path("/tmp/dynaclr_batch_check")


# %% [markdown]
# ## Helpers


# %%
def _img_2d(tensor_5d: np.ndarray, sample_idx: int) -> np.ndarray:
    """Extract a 2D image from (B, C, Z, Y, X) tensor."""
    img = tensor_5d[sample_idx]
    if img.ndim == 4:
        img = img[0, img.shape[1] // 2]
    elif img.ndim == 3:
        img = img[0]
    return img


def plot_batch_pairs(
    batch: dict,
    batch_idx: int,
    title: str,
    checks: dict[str, callable] | None = None,
    save_path: Path | None = None,
) -> None:
    """One figure per batch: 2 rows (anchor/positive) x N samples.

    Parameters
    ----------
    batch : dict
        Batch dict with anchor, positive, anchor_meta, positive_meta.
    batch_idx : int
        Batch number (for figure title).
    title : str
        Figure title prefix.
    checks : dict[str, callable] or None
        Named checks: {label: fn(anchor_meta, positive_meta) -> bool}.
        Each check is annotated per sample.
    save_path : Path or None
        If set, save the figure to this path.
    """
    anchor = batch["anchor"].numpy()
    positive = batch.get("positive")
    positive = positive.numpy() if positive is not None else None
    anchor_meta = batch["anchor_meta"]
    positive_meta = batch.get("positive_meta", [{}] * len(anchor_meta))
    n = len(anchor_meta)

    has_positive = positive is not None
    n_rows = 2 if has_positive else 1
    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2.2, n_rows * 2.8), squeeze=False)

    # Batch-level summary in title
    experiments = Counter(m.get("experiment", "?") for m in anchor_meta)
    conditions = Counter(m.get("condition", "?") for m in anchor_meta)
    exp_str = ", ".join(f"{k[:20]}={v}" for k, v in experiments.most_common(3))
    cond_str = ", ".join(f"{k}={v}" for k, v in conditions.most_common(5))
    fig.suptitle(
        f"{title} — Batch {batch_idx}\nexp: {exp_str}  |  cond: {cond_str}",
        fontsize=9,
    )

    for i in range(n):
        am = anchor_meta[i]
        pm = positive_meta[i] if i < len(positive_meta) else {}

        # Anchor image
        img_a = _img_2d(anchor, i)
        vmin, vmax = np.percentile(img_a, [1, 99])
        ax = axes[0, i]
        ax.imshow(img_a, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        # Build annotation lines
        lines = [f"{am.get('condition', '?')}", f"t={am.get('t', '?')}"]
        if am.get("marker"):
            lines.append(f"{am['marker']}")

        # Run checks
        if checks:
            for label, fn in checks.items():
                passed = fn(am, pm)
                mark = "\u2713" if passed else "\u2717"
                lines.append(f"{label}{mark}")

        ax.set_title("\n".join(lines), fontsize=6, linespacing=1.2)

        # Positive image
        if has_positive:
            img_p = _img_2d(positive, i)
            vmin_p, vmax_p = np.percentile(img_p, [1, 99])
            ax = axes[1, i]
            ax.imshow(img_p, cmap="gray", vmin=vmin_p, vmax=vmax_p)
            ax.set_xticks([])
            ax.set_yticks([])

            pos_lines = [f"{pm.get('condition', '?')}", f"t={pm.get('t', '?')}"]
            if pm.get("marker"):
                pos_lines.append(f"{pm['marker']}")
            ax.set_title("\n".join(pos_lines), fontsize=6, linespacing=1.2)

    axes[0, 0].set_ylabel("anchor", fontsize=8)
    if has_positive:
        axes[1, 0].set_ylabel("positive", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")


def run_scenario(
    dm: MultiExperimentDataModule,
    name: str,
    checks: dict[str, callable] | None = None,
) -> list[dict]:
    """Pull batches, print summary, plot pairs with checks."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    dl = dm.train_dataloader()
    batches = [batch for i, batch in enumerate(dl) if i < N_BATCHES]

    for bi, batch in enumerate(batches):
        meta = batch["anchor_meta"]
        n = len(meta)
        exps = Counter(m.get("experiment", "?") for m in meta)
        conds = Counter(m.get("condition", "?") for m in meta)
        print(f"  Batch {bi}: {n} samples, experiments={dict(exps)}, conditions={dict(conds)}")

        plot_batch_pairs(
            batch,
            bi,
            name,
            checks=checks,
            save_path=OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_batch{bi}.png" if OUTPUT_DIR else None,
        )

    return batches


# %% [markdown]
# ---
# ## Build DataModule (one-time)

# %%
print("Building DataModule...")
dm = MultiExperimentDataModule(
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
    tau_range=(0.5, 2.0),
    tau_decay_rate=2.0,
    channels_per_sample=1,
)
dm.setup("fit")
print("Done.\n")

va = dm.train_dataset.index.valid_anchors
print(f"Anchors: {len(va):,}  |  Experiments: {va['experiment'].nunique()}  |  FOVs: {va['fov_name'].nunique()}")
for exp, g in va.groupby("experiment"):
    conds = g["condition"].value_counts().to_dict()
    print(f"  {exp}: {len(g):,} anchors, {conds}")
print()

# %% [markdown]
# ---
# ## 1. Temporal + Experiment-Aware + Condition-Balanced
#
# **Checks**: each batch from one experiment, balanced conditions,
# positive is same lineage at different t.

# %%
dm.batch_group_by = "experiment"
dm.stratify_by = "condition"
dm.leaky = 0.0
dm.temporal_enrichment = False

run_scenario(
    dm,
    "1 Temporal experiment-aware",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
        "diff_t": lambda am, pm: am.get("t") != pm.get("t"),
    },
)

# %% [markdown]
# ---
# ## 2. Temporal + Cross-Organelle
#
# **Checks**: batch mixes organelles, balanced (condition, organelle).

# %%
dm.batch_group_by = None
dm.stratify_by = ["condition", "organelle"]

run_scenario(
    dm,
    "2 Cross-organelle",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
    },
)

# %% [markdown]
# ---
# ## 3. Temporal Enrichment
#
# **Checks**: HPI concentrated around a focal point per batch.

# %%
dm.batch_group_by = "experiment"
dm.stratify_by = None
dm.temporal_enrichment = True
dm.temporal_window_hours = 2.0
dm.temporal_global_fraction = 0.3

batches_enriched = run_scenario(
    dm,
    "3 Temporal enrichment",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
    },
)

# HPI histogram
fig, axes = plt.subplots(1, len(batches_enriched), figsize=(4 * len(batches_enriched), 3), squeeze=False)
fig.suptitle("Temporal enrichment: HPI per batch", fontsize=11)
for bi, batch in enumerate(batches_enriched):
    hpi = [m["hours_post_perturbation"] for m in batch["anchor_meta"]]
    ax = axes[0, bi]
    ax.hist(hpi, bins=12, color="mediumpurple", edgecolor="white")
    ax.axvline(np.mean(hpi), color="red", linestyle="--", label=f"mean={np.mean(hpi):.1f}")
    ax.set_xlabel("HPI")
    ax.set_title(f"Batch {bi}", fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Leaky Experiment Mixing
#
# **Checks**: batch dominated by one experiment (~70%), rest from others.

# %%
dm.batch_group_by = "experiment"
dm.stratify_by = "condition"
dm.temporal_enrichment = False
dm.leaky = 0.3

run_scenario(
    dm,
    "4 Leaky 30pct",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
    },
)

# %% [markdown]
# ---
# ## 5. Self-Positive (SimCLR)
#
# **Checks**: anchor and positive are identical patches.

# %%
dm_simclr = MultiExperimentDataModule(
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
dm_simclr.setup("fit")

run_scenario(
    dm_simclr,
    "5 SimCLR self-positive",
    checks={
        "same_patch": lambda am, pm: am.get("global_track_id") == pm.get("global_track_id"),
    },
)

# %% [markdown]
# ---
# ## Summary
#
# All figures saved to:

# %%
plt.show()

# %%
