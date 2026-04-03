"""Dataloader demo: visualize raw, normalized, and augmented batches.

Jupyter-style notebook (use ``# %%`` cells in VS Code or JupyterLab).

Shows what the DynaCLR model actually receives as input. For each batch:

- **Row 0 (anchor raw)**: raw patches from zarr (no transforms).
- **Row 1 (anchor aug)**: after normalization + augmentation + crop
  (exactly what the model sees during training).
- **Row 2 (positive raw)**: positive pair raw patches.
- **Row 3 (positive aug)**: positive after transforms.

Each column annotation shows experiment, marker, perturbation, timepoint,
and lineage/temporal checks. Batch composition is summarized in the title.

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/dataloader_demo.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # DynaCLR Dataloader Demo
#
# Visualize anchor/positive pairs with normalization and augmentation.
# All parameters are inline — edit and re-run cells.
#
# ## Augmentation pipeline
#
# The augmentation order matters. The pipeline is:
#
# 1. **Normalize** on full extraction patch ``(45, 256, 256)``
# 2. **Affine** (rotate/scale/shear) on ``(45, 256, 256)``
# 3. **RandSpatialCrop** to ``(40, 228, 228)`` — random Z for focus
#    invariance + random YX for translation augmentation
# 4. **Flip, contrast, scale, smooth, noise** on ``(40, 228, 228)``
# 5. **CenterCrop** to ``(32, 160, 160)`` — auto-appended by datamodule,
#    removes rotation zero-fill artifacts at the edges

# %%
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_transforms import (
    BatchedRandAdjustContrastd,
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedRandGaussianNoised,
    BatchedRandGaussianSmoothd,
    BatchedRandScaleIntensityd,
    BatchedRandSpatialCropd,
    NormalizeSampled,
)

# %% [markdown]
# ## Configuration
#
# Everything is inline — edit and re-run.

# %%
# --- Data source ---
CELL_INDEX_PATH = (
    "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/DynaCLR-3D-BagOfChannels-v2.parquet"
)

# --- Patch extraction ---
Z_WINDOW = 32
Z_EXTRACTION_WINDOW = 45
Z_FOCUS_OFFSET = 0.3
YX_PATCH_SIZE = (256, 256)
FINAL_YX_PATCH_SIZE = (160, 160)

# --- Channel mode ---
# 1 = bag-of-channels (one random channel per sample, key="channel_0")
# None = all channels; ["Phase3D", "GFP"] = fixed list
CHANNELS_PER_SAMPLE = 1
CHANNEL_NAMES = ["channel_0"]

# --- Positive pair sampling ---
POSITIVE_CELL_SOURCE = "lookup"
POSITIVE_MATCH_COLUMNS = ["lineage_id"]
TAU_RANGE = (0.5, 2.0)
TAU_DECAY_RATE = 2.0

# --- Batch sampling ---
BATCH_SIZE = 10
BATCH_GROUP_BY = None
STRATIFY_BY = ["perturbation"]
SEED = 42

# --- Pixel size normalization ---
REFERENCE_PIXEL_SIZE_XY_UM = 0.1494
REFERENCE_PIXEL_SIZE_Z_UM = 0.174
FOCUS_CHANNEL = "Phase3D"

# --- Normalization ---
NORMALIZATIONS = [
    NormalizeSampled(
        keys=CHANNEL_NAMES,
        level="timepoint_statistics",
        subtrahend="mean",
        divisor="std",
    ),
]

# --- Augmentations ---
# The RandSpatialCrop goes after the affine to trim rotation artifacts
# and provide random Z + XY translation. The datamodule auto-appends
# a CenterCrop to [Z_WINDOW, 160, 160] at the end.
AUGMENTATIONS = [
    BatchedRandAffined(
        keys=CHANNEL_NAMES,
        prob=1,
        scale_range=[[0.9, 1.1], [0.9, 1.1], [0.9, 1.1]],
        rotate_range=[3.14, 0.0, 0.0],
        shear_range=[0.05, 0.05, 0.0, 0.05, 0.0, 0.05],
    ),
    BatchedRandSpatialCropd(
        keys=CHANNEL_NAMES,
        roi_size=[40, 228, 228],
    ),
    BatchedRandFlipd(keys=CHANNEL_NAMES, spatial_axes=[1, 2], prob=0.5),
    BatchedRandAdjustContrastd(keys=CHANNEL_NAMES, prob=0.5, gamma=(0.6, 1.6)),
    BatchedRandScaleIntensityd(keys=CHANNEL_NAMES, prob=0.5, factors=0.5),
    BatchedRandGaussianSmoothd(
        keys=CHANNEL_NAMES,
        prob=1,
        sigma_x=[0.25, 0.50],
        sigma_y=[0.25, 0.50],
        sigma_z=[0.0, 0.2],
    ),
    BatchedRandGaussianNoised(keys=CHANNEL_NAMES, prob=1, mean=0.0, std=0.1),
]

# --- Display ---
N_BATCHES = 4
N_SHOW = 10
NUM_WORKERS = 1
SHOW_AUGMENTED = True
OUTPUT_DIR = Path("applications/dynaclr/scripts/dataloader_inspection/results/dataloader_demo")


# %% [markdown]
# ## Helpers


# %%
def _img_2d(tensor_5d: np.ndarray, sample_idx: int) -> np.ndarray:
    """Extract a 2D slice from (B, C, Z, Y, X) for display."""
    img = tensor_5d[sample_idx]
    if img.ndim == 4:
        img = img[0, img.shape[1] // 2]
    elif img.ndim == 3:
        img = img[0]
    return img


def plot_batch(
    raw_batch: dict,
    aug_batch: dict | None,
    batch_idx: int,
    n_show: int,
    show_augmented: bool = True,
    save_path: Path | None = None,
) -> None:
    """Plot one batch: raw and augmented anchor/positive pairs."""
    anchor_raw = raw_batch["anchor"].numpy()
    positive_raw = raw_batch.get("positive")
    has_positive = positive_raw is not None
    if has_positive:
        positive_raw = positive_raw.numpy()

    anchor_meta = raw_batch["anchor_meta"]
    positive_meta = raw_batch.get("positive_meta", [{}] * len(anchor_meta))
    n = min(n_show, len(anchor_meta))

    row_labels = ["anchor (raw)"]
    if show_augmented and aug_batch is not None:
        row_labels.append("anchor (aug)")
    if has_positive:
        row_labels.append("positive (raw)")
        if show_augmented and aug_batch is not None:
            row_labels.append("positive (aug)")
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2.0, n_rows * 2.4), squeeze=False)

    markers = Counter(m.get("marker", "?") for m in anchor_meta[:n])
    perts = Counter(m.get("perturbation", "?") for m in anchor_meta[:n])
    m_str = " ".join(f"{k}={v}" for k, v in markers.most_common(5))
    p_str = " ".join(f"{k}={v}" for k, v in perts.most_common(5))
    fig.suptitle(
        f"Batch {batch_idx}  |  markers: {m_str}  |  pert: {p_str}",
        fontsize=9,
        fontweight="bold",
    )

    anchor_aug = aug_batch["anchor"].numpy() if (show_augmented and aug_batch) else None
    positive_aug = None
    if has_positive and show_augmented and aug_batch:
        pa = aug_batch.get("positive")
        positive_aug = pa.numpy() if pa is not None else None

    for i in range(n):
        am = anchor_meta[i]
        pm = positive_meta[i] if i < len(positive_meta) else {}

        row = 0
        img = _img_2d(anchor_raw, i)
        vmin, vmax = np.percentile(img, [1, 99])
        axes[row, i].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, i].set_xticks([])
        axes[row, i].set_yticks([])
        lines = [
            f"{am.get('experiment', '?')[:25]}",
            f"fov={am.get('fov_name', '?')}",
            f"track={am.get('global_track_id', '?')[-15:]}",
            f"marker={am.get('marker', '?')}",
            f"pert={am.get('perturbation', '?')}",
            f"t={am.get('t', '?')}",
        ]
        if has_positive:
            lin_ok = am.get("lineage_id") == pm.get("lineage_id")
            dt_ok = am.get("t") != pm.get("t")
            lines.append(f"lineage={'✓' if lin_ok else '✗'}  Δt={'✓' if dt_ok else '✗'}")
        axes[row, i].set_title("\n".join(lines), fontsize=5, linespacing=1.1)

        if anchor_aug is not None:
            row += 1
            img_a = _img_2d(anchor_aug, i)
            vmin_a, vmax_a = np.percentile(img_a, [1, 99])
            axes[row, i].imshow(img_a, cmap="gray", vmin=vmin_a, vmax=vmax_a)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
            axes[row, i].set_title(f"μ={img_a.mean():.2f} σ={img_a.std():.2f}", fontsize=5)

        if has_positive:
            row += 1
            img_p = _img_2d(positive_raw, i)
            vmin_p, vmax_p = np.percentile(img_p, [1, 99])
            axes[row, i].imshow(img_p, cmap="gray", vmin=vmin_p, vmax=vmax_p)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
            pos_lines = [
                f"fov={pm.get('fov_name', '?')}",
                f"track={pm.get('global_track_id', '?')[-15:]}",
                f"pert={pm.get('perturbation', '?')}  t={pm.get('t', '?')}",
            ]
            axes[row, i].set_title("\n".join(pos_lines), fontsize=5, linespacing=1.1)

            if positive_aug is not None:
                row += 1
                img_pa = _img_2d(positive_aug, i)
                vmin_pa, vmax_pa = np.percentile(img_pa, [1, 99])
                axes[row, i].imshow(img_pa, cmap="gray", vmin=vmin_pa, vmax=vmax_pa)
                axes[row, i].set_xticks([])
                axes[row, i].set_yticks([])
                axes[row, i].set_title(f"μ={img_pa.mean():.2f} σ={img_pa.std():.2f}", fontsize=5)

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=7, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    # plt.close(fig)


# %% [markdown]
# ## Build DataModule
#
# Passes normalizations + augmentations directly to the DataModule.
# ``on_after_batch_transfer`` applies: normalizations → augmentations
# (including RandSpatialCrop) → auto-appended CenterCrop to final size.

# %%
dm = MultiExperimentDataModule(
    cell_index_path=CELL_INDEX_PATH,
    z_window=Z_WINDOW,
    z_extraction_window=Z_EXTRACTION_WINDOW,
    z_focus_offset=Z_FOCUS_OFFSET,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    channels_per_sample=CHANNELS_PER_SAMPLE,
    positive_cell_source=POSITIVE_CELL_SOURCE,
    positive_match_columns=POSITIVE_MATCH_COLUMNS,
    tau_range=TAU_RANGE,
    tau_decay_rate=TAU_DECAY_RATE,
    batch_size=BATCH_SIZE,
    batch_group_by=BATCH_GROUP_BY,
    stratify_by=STRATIFY_BY,
    num_workers=NUM_WORKERS,
    seed=SEED,
    focus_channel=FOCUS_CHANNEL,
    reference_pixel_size_xy_um=REFERENCE_PIXEL_SIZE_XY_UM,
    reference_pixel_size_z_um=REFERENCE_PIXEL_SIZE_Z_UM,
    channel_dropout_prob=0.0,
    normalizations=NORMALIZATIONS,
    augmentations=AUGMENTATIONS,
)
dm.setup("fit")
print("DataModule ready.\n")

va = dm.train_dataset.index.valid_anchors
print(f"Anchors: {len(va):,}  |  Experiments: {va['experiment'].nunique()}")
for exp, g in va.groupby("experiment"):
    markers = g["marker"].value_counts().to_dict() if "marker" in g.columns else {}
    perts = g["perturbation"].value_counts().to_dict()
    print(f"  {exp}: {len(g):,} anchors, markers={markers}, perturbations={perts}")

# %% [markdown]
# ## Draw batches
#
# The dataloader returns raw patches ``(B, C, 45, 256, 256)`` (no transforms).
# ``dm.on_after_batch_transfer`` applies the full pipeline:
#
# 1. Normalize ``(45, 256, 256)``
# 2. Affine ``(45, 256, 256)``
# 3. RandSpatialCrop ``(40, 228, 228)``
# 4. Flip / contrast / noise ``(40, 228, 228)``
# 5. CenterCrop ``(32, 160, 160)`` (auto-appended)
#
# We deepcopy each batch so we can show raw vs augmented side by side.

# %%
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dl = dm.train_dataloader()
dl_iter = iter(dl)

for batch_idx in range(N_BATCHES):
    print(f"\n--- Batch {batch_idx} ---")
    batch = next(dl_iter)

    meta = batch["anchor_meta"]
    n = len(meta)
    markers = Counter(m.get("marker", "?") for m in meta)
    perts = Counter(m.get("perturbation", "?") for m in meta)
    print(f"  {n} samples, markers={dict(markers)}, perturbations={dict(perts)}")

    raw_batch = copy.deepcopy(batch)
    aug_batch = dm.on_after_batch_transfer(batch, dataloader_idx=0) if SHOW_AUGMENTED else None

    save_path = OUTPUT_DIR / f"batch_{batch_idx}.png" if OUTPUT_DIR else None
    plot_batch(
        raw_batch=raw_batch,
        aug_batch=aug_batch,
        batch_idx=batch_idx,
        n_show=N_SHOW,
        show_augmented=SHOW_AUGMENTED,
        save_path=save_path,
    )

# %%
print("\nDone.")

# %% [markdown]
# ## Re-run additional batches
#
# Edit ``batch_idx`` and re-run this cell to inspect more batches
# without restarting the dataloader iterator.

# %%
batch_idx = 9
batch = next(dl_iter)

meta = batch["anchor_meta"]
n = len(meta)
markers = Counter(m.get("marker", "?") for m in meta)
perts = Counter(m.get("perturbation", "?") for m in meta)
print(f"  {n} samples, markers={dict(markers)}, perturbations={dict(perts)}")

raw_batch = copy.deepcopy(batch)
aug_batch = dm.on_after_batch_transfer(batch, dataloader_idx=0) if SHOW_AUGMENTED else None

save_path = OUTPUT_DIR / f"batch_{batch_idx}.png" if OUTPUT_DIR else None
plot_batch(
    raw_batch=raw_batch,
    aug_batch=aug_batch,
    batch_idx=batch_idx,
    n_show=N_SHOW,
    show_augmented=SHOW_AUGMENTED,
    save_path=save_path,
)

# %%
