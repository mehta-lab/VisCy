"""2D MIP augmentation demo — inspect and verify the pipeline.

Jupyter-style notebook (use ``# %%`` cells in VS Code or JupyterLab).

Shows what the 2D MIP model receives as input and verifies:

- **Row 0 (anchor raw)**: center z-slice of the 20-slice raw extraction patch.
- **Row 1 (anchor aug)**: after normalize → affine → RandSpatialCrop(10) → MIP/center-slice → CenterCrop(160,160).

Column annotations show marker, perturbation, and the z-reduction strategy
applied (MIP for fluorescence, center-slice for label-free).

Pipeline:
  extract (20, 192, 192) → normalize → affine → RandSpatialCrop(10, 192, 192)
  → flip/contrast/noise → ZReduction (MIP or center-slice) → CenterCrop(1, 160, 160)

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/test_2d_mip_augmentation.py
"""

# ruff: noqa: E402, D103

# %% [markdown]
# # 2D MIP Augmentation Demo
#
# Verify the z-reduction strategy per marker and visualize raw vs augmented.
#
# ## Pipeline
#
# 1. **Extract** 20 z-slices around focus
# 2. **Normalize** (subtract mean, divide std)
# 3. **Affine** (rotate/scale/shear)
# 4. **RandSpatialCrop** to (10, 192, 192) — random Z for focus invariance
# 5. **Flip, contrast, scale, smooth, noise**
# 6. **ZReduction**: MIP for fluorescence, center-slice for label-free
# 7. **CenterCrop** to (1, 160, 160) — auto-appended by datamodule

# %%
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_data._utils import _transform_channel_wise
from viscy_data.channel_utils import parse_channel_name
from viscy_transforms import (
    BatchedChannelWiseZReductiond,
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

# %%
CELL_INDEX_PATH = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/test_2d_mip_mixed.parquet"

Z_WINDOW = 1
Z_EXTRACTION_WINDOW = 20
Z_FOCUS_OFFSET = 0.5
YX_PATCH_SIZE = (192, 192)
FINAL_YX_PATCH_SIZE = (160, 160)
CHANNEL_NAMES = ["channel_0"]

BATCH_SIZE = 16
N_BATCHES = 4
N_SHOW = 10
NUM_WORKERS = 4
OUTPUT_DIR = Path("/home/eduardo.hirata/repos/viscy/applications/dynaclr/scripts/dataloader_inspection/results")

# %% [markdown]
# ## Build DataModule

# %%
normalizations = [
    NormalizeSampled(
        keys=CHANNEL_NAMES,
        level="timepoint_statistics",
        subtrahend="mean",
        divisor="std",
    )
]
augmentations = [
    BatchedRandAffined(
        keys=CHANNEL_NAMES,
        prob=0.8,
        scale_range=[[0.8, 1.3], [0.8, 1.3], [0.8, 1.3]],
        rotate_range=[3.14, 0.0, 0.0],
        shear_range=[0.05, 0.05, 0.0, 0.05, 0.0, 0.05],
    ),
    BatchedRandFlipd(keys=CHANNEL_NAMES, spatial_axes=[1, 2], prob=0.5),
    BatchedRandAdjustContrastd(keys=CHANNEL_NAMES, prob=0.5, gamma=(0.6, 1.6)),
    BatchedRandScaleIntensityd(keys=CHANNEL_NAMES, prob=0.5, factors=0.5),
    BatchedRandGaussianSmoothd(
        keys=CHANNEL_NAMES,
        prob=0.5,
        sigma_x=[0.25, 0.50],
        sigma_y=[0.25, 0.50],
        sigma_z=[0.0, 0.0],
    ),
    BatchedRandGaussianNoised(keys=CHANNEL_NAMES, prob=0.5, mean=0.0, std=0.1),
    # Random Z crop: select 10 of 20 extracted slices for Z-invariance.
    BatchedRandSpatialCropd(keys=CHANNEL_NAMES, roi_size=[10, 192, 192]),
    # Z-reduction: MIP for fluorescence, center-slice for label-free.
    BatchedChannelWiseZReductiond(keys=CHANNEL_NAMES, allow_missing_keys=True),
]

dm = MultiExperimentDataModule(
    cell_index_path=CELL_INDEX_PATH,
    z_window=Z_WINDOW,
    z_extraction_window=Z_EXTRACTION_WINDOW,
    z_focus_offset=Z_FOCUS_OFFSET,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    channels_per_sample=1,
    positive_cell_source="lookup",
    positive_match_columns=["lineage_id"],
    tau_range=(0.5, 2.0),
    tau_decay_rate=2.0,
    stratify_by=["perturbation", "marker"],
    split_ratio=0.8,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=42,
    focus_channel="Phase3D",
    reference_pixel_size_xy_um=0.1494,
    channel_dropout_prob=0.0,
    normalizations=normalizations,
    augmentations=augmentations,
)
dm.setup("fit")

va = dm.train_dataset.index.valid_anchors
print(f"Anchors: {len(va):,}  |  Experiments: {va['experiment'].nunique()}")
for exp, g in va.groupby("experiment"):
    markers = g["marker"].value_counts().to_dict() if "marker" in g.columns else {}
    print(f"  {exp}: {len(g):,} anchors  markers={markers}")


# %% [markdown]
# ## Helpers


# %%
def _apply_augmentations(batch: dict) -> torch.Tensor:
    """Apply the full augmentation pipeline to a raw batch, return (B,C,1,H,W)."""
    norm_meta = batch.get("anchor_norm_meta")
    is_labelfree = torch.tensor(
        [parse_channel_name(m.get("marker", ""))["channel_type"] == "labelfree" for m in batch["anchor_meta"]],
        dtype=torch.bool,
    )
    return _transform_channel_wise(
        transform=dm._augmentation_transform,
        channel_names=dm._channel_names,
        patch=batch["anchor"],
        norm_meta=norm_meta,
        extra={"_is_labelfree": is_labelfree},
    )


def _img2d_raw(tensor: np.ndarray, sample_idx: int) -> np.ndarray:
    """Center z-slice from raw (B, C, Z, Y, X) for display."""
    vol = tensor[sample_idx, 0]  # (Z, Y, X)
    return vol[vol.shape[0] // 2]


def _img2d_aug(tensor: np.ndarray, sample_idx: int) -> np.ndarray:
    """2D image from augmented (B, C, 1, Y, X)."""
    return tensor[sample_idx, 0, 0]


def _strategy(marker: str) -> str:
    ct = parse_channel_name(marker)["channel_type"]
    return "center-slice" if ct == "labelfree" else "MIP"


def plot_batch(
    raw_batch: dict,
    aug_patch: torch.Tensor,
    batch_idx: int,
    n_show: int = N_SHOW,
    save_path: Path | None = None,
) -> None:
    anchor_raw = raw_batch["anchor"].numpy()
    anchor_aug = aug_patch.numpy()
    meta = raw_batch.get("anchor_meta", [])
    n = min(n_show, len(meta))

    markers = Counter(m.get("marker", "?") for m in meta[:n])
    perts = Counter(m.get("perturbation", "?") for m in meta[:n])
    m_str = " ".join(f"{k}={v}" for k, v in markers.most_common(5))
    p_str = " ".join(f"{k}={v}" for k, v in perts.most_common(5))

    fig, axes = plt.subplots(2, n, figsize=(n * 2.0, 2 * 2.4), squeeze=False)
    fig.suptitle(
        f"Batch {batch_idx}  |  markers: {m_str}  |  pert: {p_str}\n"
        f"raw z-depth={anchor_raw.shape[2]}  aug z-depth={anchor_aug.shape[2]}",
        fontsize=8,
        fontweight="bold",
    )

    for i in range(n):
        am = meta[i] if i < len(meta) else {}
        marker = am.get("marker", "?")
        strategy = _strategy(marker)

        # Row 0: raw center z-slice
        img_raw = _img2d_raw(anchor_raw, i)
        vmin, vmax = np.percentile(img_raw, [1, 99])
        axes[0, i].imshow(img_raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_title(
            "\n".join(
                [
                    f"{am.get('experiment', '?')[:20]}",
                    f"marker={marker}",
                    f"pert={am.get('perturbation', '?')}",
                    f"t={am.get('t', '?')}",
                    f"z_reduction={strategy}",
                ]
            ),
            fontsize=5,
            linespacing=1.1,
        )

        # Row 1: augmented (post ZReduction)
        img_aug = _img2d_aug(anchor_aug, i)
        vmin_a, vmax_a = np.percentile(img_aug, [1, 99])
        axes[1, i].imshow(img_aug, cmap="gray", vmin=vmin_a, vmax=vmax_a)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].set_title(f"μ={img_aug.mean():.2f} σ={img_aug.std():.2f}", fontsize=5)

    axes[0, 0].set_ylabel("raw (center z)", fontsize=7, fontweight="bold")
    axes[1, 0].set_ylabel("aug (MIP/center)", fontsize=7, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()


def check_batch(batch_idx: int, raw_batch: dict, aug_patch: torch.Tensor) -> None:
    """Assert shape and z-reduction correctness, print summary."""
    meta = raw_batch.get("anchor_meta", [])

    assert aug_patch.shape[2] == 1, f"Batch {batch_idx}: z should be 1, got {aug_patch.shape}"
    assert aug_patch.shape[3] == FINAL_YX_PATCH_SIZE[0], f"Y should be {FINAL_YX_PATCH_SIZE[0]}"
    assert aug_patch.shape[4] == FINAL_YX_PATCH_SIZE[1], f"X should be {FINAL_YX_PATCH_SIZE[1]}"
    print(f"  [PASS] shape: {tuple(aug_patch.shape)}")

    n_lf, n_fl = 0, 0
    for i, m in enumerate(meta):
        marker = m.get("marker", "")
        ct = parse_channel_name(marker)["channel_type"]
        assert not torch.all(aug_patch[i] == 0), f"Sample {i} ({marker}) is all zeros"
        if ct == "labelfree":
            n_lf += 1
        else:
            n_fl += 1

    raw_z = raw_batch["anchor"].shape[2]
    print(f"  [PASS] label-free (center-slice)={n_lf}  fluorescence (MIP)={n_fl}  raw_z={raw_z}")
    print(f"  [INFO] markers: {dict(Counter(m.get('marker', '?') for m in meta))}")


# %% [markdown]
# ## Draw batches

# %%
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dl = dm.train_dataloader()
dl_iter = iter(dl)

for batch_idx in range(N_BATCHES):
    print(f"\n--- Batch {batch_idx} ---")
    batch = next(dl_iter)
    raw_batch = copy.deepcopy(batch)
    aug_patch = _apply_augmentations(batch)
    check_batch(batch_idx, raw_batch, aug_patch)
    save_path = OUTPUT_DIR / f"batch_{batch_idx}.png" if OUTPUT_DIR else None
    plot_batch(raw_batch, aug_patch, batch_idx, save_path=save_path)

# %%
print("\nDone.")

# %% [markdown]
# ## Re-run additional batches
#
# Edit ``batch_idx`` and re-run this cell to inspect more batches
# without restarting the dataloader iterator.

# %%
batch_idx = N_BATCHES
batch = next(dl_iter)
raw_batch = copy.deepcopy(batch)
aug_patch = _apply_augmentations(batch)
check_batch(batch_idx, raw_batch, aug_patch)
plot_batch(raw_batch, aug_patch, batch_idx)

# %%
