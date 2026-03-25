"""Visual verification of batch composition for DynaCLR contrastive training.

Jupyter-like notebook (use ``# %%`` cells in VS Code or JupyterLab).

For each sampling configuration, draws N_BATCHES batches and creates a
figure per batch showing anchor/positive pairs side by side with metadata
annotations. Checkmarks verify that the sampling contract holds:

- Same-lineage positives (temporal mode)
- Same-marker batches (bag-of-channels)
- Same-experiment batches (experiment-aware)
- Perturbation-balanced batches (stratified)

Usage::

    python applications/dynaclr/scripts/dataloader_inspection/check_batch_composition.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # Batch Composition Checker
#
# Visual QC for contrastive batch sampling with flat parquet.
# Each figure = one batch. Columns = anchor/positive pairs.
# Annotations show metadata + pass/fail checks.

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
# --- EDIT THESE ---
CELL_INDEX_PATH = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/example_flat.parquet"
COLLECTION_PATH = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/example_cell_index.yaml"

Z_WINDOW = 1
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)
BATCH_SIZE = 8
NUM_WORKERS = 4
N_BATCHES = 3
OUTPUT_DIR = Path("applications/dynaclr/scripts/dataloader_inspection/results/batch_composition")


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
    """One figure per batch: 2 rows (anchor/positive) x N samples."""
    anchor = batch["anchor"].numpy()
    positive = batch.get("positive")
    positive = positive.numpy() if positive is not None else None
    anchor_meta = batch["anchor_meta"]
    positive_meta = batch.get("positive_meta", [{}] * len(anchor_meta))
    n = len(anchor_meta)

    has_positive = positive is not None
    n_rows = 2 if has_positive else 1
    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2.2, n_rows * 2.8), squeeze=False)

    experiments = Counter(m.get("experiment", "?") for m in anchor_meta)
    perturbations = Counter(m.get("perturbation", "?") for m in anchor_meta)
    exp_str = ", ".join(f"{k[:20]}={v}" for k, v in experiments.most_common(3))
    pert_str = ", ".join(f"{k}={v}" for k, v in perturbations.most_common(5))
    fig.suptitle(f"{title} — Batch {batch_idx}\nexp: {exp_str}  |  pert: {pert_str}", fontsize=9)

    for i in range(n):
        am = anchor_meta[i]
        pm = positive_meta[i] if i < len(positive_meta) else {}

        img_a = _img_2d(anchor, i)
        vmin, vmax = np.percentile(img_a, [1, 99])
        ax = axes[0, i]
        ax.imshow(img_a, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        lines = [f"{am.get('perturbation', '?')}", f"t={am.get('t', '?')}"]
        if checks:
            for label, fn in checks.items():
                passed = fn(am, pm)
                mark = "\u2713" if passed else "\u2717"
                lines.append(f"{label}{mark}")
        ax.set_title("\n".join(lines), fontsize=6, linespacing=1.2)

        if has_positive:
            img_p = _img_2d(positive, i)
            vmin_p, vmax_p = np.percentile(img_p, [1, 99])
            ax = axes[1, i]
            ax.imshow(img_p, cmap="gray", vmin=vmin_p, vmax=vmax_p)
            ax.set_xticks([])
            ax.set_yticks([])
            pos_lines = [f"{pm.get('perturbation', '?')}", f"t={pm.get('t', '?')}"]
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
    batches = []
    for i, batch in enumerate(dl):
        if i >= N_BATCHES:
            break
        batches.append(batch)

    for bi, batch in enumerate(batches):
        meta = batch["anchor_meta"]
        n = len(meta)
        exps = Counter(m.get("experiment", "?") for m in meta)
        perts = Counter(m.get("perturbation", "?") for m in meta)
        markers = Counter(m.get("marker", "?") for m in meta)
        print(
            f"  Batch {bi}: {n} samples, markers={dict(markers)}, experiments={dict(exps)}, perturbations={dict(perts)}"
        )

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
# ## Build DataModule
#
# Uses a flat parquet (one row per cell x timepoint x channel).
# The parquet has per-row `channel_name`, `marker`, `perturbation`.

# %%
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    batch_group_by=["marker"],
    stratify_by="perturbation",
)
dm.setup("fit")
print("Done.\n")

va = dm.train_dataset.index.valid_anchors
print(f"Anchors: {len(va):,}  |  Experiments: {va['experiment'].nunique()}  |  FOVs: {va['fov_name'].nunique()}")
for exp, g in va.groupby("experiment"):
    markers = g["marker"].value_counts().to_dict() if "marker" in g.columns else {}
    perts = g["perturbation"].value_counts().to_dict()
    print(f"  {exp}: {len(g):,} anchors, markers={markers}, perturbations={perts}")
print()

# %% [markdown]
# ---
# ## Marker distribution
#
# Check if markers are balanced. If Phase3D dominates, set
# ``group_weights`` to flatten sampling:
#
# ```python
# dm.group_weights = {"Phase3D": 1, "SEC61B": 1, "G3BP1": 1, "pAL17": 1}
# ```
#
# ### batch_group_by / stratify_by
#
# Any column in ``valid_anchors`` (i.e. the flat parquet) can be used.
# Common choices:
#
# ### batch_group_by examples
#
# | batch_group_by              | Effect                          |
# |-----------------------------|--------------------------------------|
# | ``["marker"]``              | One marker per batch (bag-of-channels) |
# | ``["experiment"]``          | One experiment per batch             |
# | ``["experiment", "marker"]``| One (experiment, marker) per batch   |
# | ``None``                    | No grouping, draw from all           |
#
# ### stratify_by examples
#
# | stratify_by                     | Effect                           |
# |---------------------------------|---------------------------------------|
# | ``"perturbation"``              | Balance infected/uninfected (default) |
# | ``["perturbation", "marker"]``  | Balance both within batch             |
# | ``"experiment"``                | Balance experiments within batch      |
# | ``None``                        | No balancing, pure random             |

# %%
marker_counts = va["marker"].value_counts()
print("Marker distribution in valid_anchors:")
for marker, count in marker_counts.items():
    pct = 100 * count / len(va)
    print(f"  {marker}: {count:,} ({pct:.1f}%)")
print()

# %% [markdown]
# ---
# ## 1. Bag-of-Channels: Marker-Grouped + Perturbation-Balanced
#
# **The primary bag-of-channels configuration.**
# Each batch has one marker (Phase3D, SEC61B, G3BP1, or pAL17).
# Perturbation balanced within. Positive is same lineage at different t.

# %%
dm.batch_group_by = ["marker"]
dm.stratify_by = "perturbation"
dm.leaky = 0.0
dm.temporal_enrichment = False

run_scenario(
    dm,
    "1 Bag-of-channels marker-grouped",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
        "diff_t": lambda am, pm: am.get("t") != pm.get("t"),
    },
)

# %% [markdown]
# ---
# ## 2. Experiment-Grouped + Perturbation-Balanced

# %%
dm.batch_group_by = ["experiment"]
dm.stratify_by = "perturbation"

run_scenario(
    dm,
    "2 Experiment-grouped",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
        "diff_t": lambda am, pm: am.get("t") != pm.get("t"),
    },
)

# %% [markdown]
# ---
# ## 3. (Experiment, Marker)-Grouped
#
# Most fine-grained: one (experiment, marker) combo per batch.

# %%
dm.batch_group_by = ["experiment", "marker"]
dm.stratify_by = "perturbation"

run_scenario(
    dm,
    "3 Experiment+marker grouped",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
        "diff_t": lambda am, pm: am.get("t") != pm.get("t"),
    },
)

# %% [markdown]
# ---
# ## 4. Temporal Enrichment

# %%
dm.batch_group_by = ["marker"]
dm.stratify_by = None
dm.temporal_enrichment = True
dm.temporal_window_hours = 2.0
dm.temporal_global_fraction = 0.3

batches_enriched = run_scenario(
    dm,
    "4 Temporal enrichment",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
    },
)

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
# ## 5. Leaky Marker Mixing (30%)

# %%
dm.batch_group_by = ["marker"]
dm.stratify_by = "perturbation"
dm.temporal_enrichment = False
dm.leaky = 0.3

run_scenario(
    dm,
    "5 Leaky 30pct",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
    },
)

# %% [markdown]
# ---
# ## 6. Self-Positive (SimCLR)

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
    batch_group_by=["marker"],
    stratify_by="perturbation",
    channels_per_sample=1,
)
dm_simclr.setup("fit")

run_scenario(
    dm_simclr,
    "6 SimCLR self-positive",
    checks={
        "same_patch": lambda am, pm: am.get("global_track_id") == pm.get("global_track_id"),
    },
)

# %% [markdown]
# ---
# ## 7. Multi-Column Stratification: Perturbation + Marker
#
# ``stratify_by=["perturbation", "marker"]`` creates composite strata
# (e.g. ``"ZIKV|Phase3D"``, ``"uninfected|SEC61B"``). Each stratum gets
# an equal share of the batch, balancing both perturbation and marker
# within each batch.

# %%
dm.batch_group_by = None
dm.stratify_by = ["perturbation", "marker"]
dm.leaky = 0.0
dm.temporal_enrichment = False

batches_multi = run_scenario(
    dm,
    "7 Stratify perturbation+marker",
    checks={
        "lineage": lambda am, pm: am.get("lineage_id") == pm.get("lineage_id"),
        "diff_t": lambda am, pm: am.get("t") != pm.get("t"),
    },
)

# Print strata distribution per batch
for bi, batch in enumerate(batches_multi):
    meta = batch["anchor_meta"]
    strata = Counter(f"{m.get('perturbation', '?')}|{m.get('marker', '?')}" for m in meta)
    print(f"  Batch {bi} strata: {dict(strata)}")

# %% [markdown]
# ---
# ## 8 & 9. Normalization Comparison
#
# Compare ``fov_statistics`` (per-FOV mean/std, same stats for all
# timepoints) vs ``timepoint_statistics`` (per-FOV *per-timepoint*
# mean/std, adapts to intensity drift over time).

# %%
from viscy_transforms import NormalizeSampled


def run_normalization_scenario(name: str, level: str) -> None:
    dm_n = MultiExperimentDataModule(
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
        batch_group_by=["marker"],
        stratify_by="perturbation",
        normalizations=[
            NormalizeSampled(
                keys=["channel_0"],
                level=level,
                subtrahend="mean",
                divisor="std",
            ),
        ],
    )
    dm_n.setup("fit")

    dl_n = dm_n.train_dataloader()
    print(f"\n{'=' * 60}")
    print(f"{name} (level={level})")
    print(f"{'=' * 60}")

    for i, batch in enumerate(dl_n):
        if i >= N_BATCHES:
            break

        meta = batch["anchor_meta"]
        markers = Counter(m.get("marker", "?") for m in meta)
        perts = Counter(m.get("perturbation", "?") for m in meta)

        raw_anchor = batch["anchor"].numpy()
        raw_mean = raw_anchor.mean()
        raw_std = raw_anchor.std()

        batch_norm = dm_n.on_after_batch_transfer(batch, dataloader_idx=0)
        norm_anchor = batch_norm["anchor"].numpy()
        norm_mean = norm_anchor.mean()
        norm_std = norm_anchor.std()

        print(
            f"  Batch {i}: markers={dict(markers)}, perturbations={dict(perts)}"
            f"\n    raw:  mean={raw_mean:.2f}  std={raw_std:.2f}"
            f"\n    norm: mean={norm_mean:.4f}  std={norm_std:.4f}"
        )


# %%
run_normalization_scenario("8 fov_statistics normalization", "fov_statistics")

# %%
run_normalization_scenario("9 timepoint_statistics normalization", "timepoint_statistics")

# %% [markdown]
# ---
# ## Summary

# %%
plt.show()

# %%
