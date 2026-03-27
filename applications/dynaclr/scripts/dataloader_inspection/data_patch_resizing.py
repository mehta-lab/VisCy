"""End-to-end proof that DynaCLR pixel-size normalization works.

Creates a temporary parquet with modified pixel sizes, feeds it through the
real ``MultiExperimentDataModule`` dataloader, and plots the output patches.

The Mantis experiment (0.1494 um/px) is the reference. The Dragonfly experiment
natively has 0.3953 um/px — we test with both the real value and an artificial
override to show the dataloader responds correctly.

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/data_patch_resizing.py
"""

# ruff: noqa: D103

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynaclr.data.datamodule import MultiExperimentDataModule
from viscy_transforms._crop import BatchedCenterSpatialCrop

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]

CELL_INDEX_PATH = _ROOT / "applications/dynaclr/configs/cell_index/dragonfly_mantis_demo.parquet"
OUTPUT_DIR = _ROOT / "applications/dynaclr/scripts/dataloader_inspection/output"
OUTPUT_PATH = OUTPUT_DIR / "data_patch_resizing.png"

Z_WINDOW = 1
YX_PATCH_SIZE = (200, 200)
FINAL_YX_PATCH_SIZE = (160, 160)
REFERENCE_PIXEL_SIZE_XY_UM = 0.1494
REFERENCE_PIXEL_SIZE_Z_UM = 0.2878
CHANNEL_NAME = "Phase3D"

DRAGONFLY_EXP = "2024_08_14_ZIKV_pal17_48h"
MANTIS_EXP = "2025_07_24_A549_SEC61B_ZIKV"

# Pixel sizes to test for Dragonfly (real + artificial overrides)
DRAGONFLY_PIXEL_SIZES = {
    "real (0.3953)": 0.3953,
    "override (0.1494)": 0.1494,  # same as reference — should be no-op
    "override (0.7)": 0.7,  # even coarser — should crop fewer pixels
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tmp_parquet(pixel_size_xy: float, pixel_size_z: float = REFERENCE_PIXEL_SIZE_Z_UM) -> str:
    """Write a temp parquet with Dragonfly pixel sizes overridden."""
    df = pd.read_parquet(CELL_INDEX_PATH)
    mask = df["experiment"] == DRAGONFLY_EXP
    df.loc[mask, "pixel_size_xy_um"] = pixel_size_xy
    df.loc[mask, "pixel_size_z_um"] = pixel_size_z
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    df.to_parquet(tmp.name)
    return tmp.name


def draw_one_sample(parquet_path: str) -> dict:
    """Build a datamodule, draw one batch, return first anchor patch + metadata."""
    dm = MultiExperimentDataModule(
        collection_path=None,
        cell_index_path=parquet_path,
        z_window=Z_WINDOW,
        yx_patch_size=YX_PATCH_SIZE,
        final_yx_patch_size=FINAL_YX_PATCH_SIZE,
        batch_size=8,
        num_workers=0,
        channels_per_sample=[CHANNEL_NAME],
        reference_pixel_size_xy_um=REFERENCE_PIXEL_SIZE_XY_UM,
        reference_pixel_size_z_um=REFERENCE_PIXEL_SIZE_Z_UM,
        positive_cell_source="self",
        tau_range=(0.0, 100.0),
        stratify_by=None,
    )
    dm.setup("fit")

    registry = dm.train_dataset.index.registry
    scale_factors = {e.name: registry.scale_factors[e.name] for e in registry.experiments}

    # Draw batches until we get one from each experiment
    loader = dm.train_dataloader()
    per_exp: dict[str, dict] = {}
    needed = {e.name for e in registry.experiments}

    for batch in loader:
        anchor = batch["anchor"]
        meta = batch["anchor_meta"]
        for i in range(anchor.shape[0]):
            exp_name = meta[i]["experiment"]
            if exp_name not in per_exp:
                per_exp[exp_name] = {
                    "patch": anchor[i],
                    "meta": meta[i],
                    "scale": scale_factors[exp_name],
                }
        if per_exp.keys() >= needed:
            break

    return per_exp


# ---------------------------------------------------------------------------
# Run the dataloader for each Dragonfly pixel size configuration
# ---------------------------------------------------------------------------

center_crop = BatchedCenterSpatialCrop(roi_size=(Z_WINDOW, FINAL_YX_PATCH_SIZE[0], FINAL_YX_PATCH_SIZE[1]))

all_results = {}
for label, px_size in DRAGONFLY_PIXEL_SIZES.items():
    print(f"\n--- Dragonfly pixel_size_xy_um = {px_size} ({label}) ---")
    tmp_path = make_tmp_parquet(px_size)
    per_exp = draw_one_sample(tmp_path)

    for exp_name, data in per_exp.items():
        scale = data["scale"]
        patch = data["patch"]  # (C, Z, Y, X) at yx_patch_size
        final = center_crop(patch[None])[0]
        key = f"{exp_name}\n{label}" if exp_name == DRAGONFLY_EXP else exp_name
        if exp_name == MANTIS_EXP and label != "real (0.3953)":
            continue  # Mantis is unchanged, only show once
        print(f"  {exp_name}: scale_yx={scale[1]:.3f}, patch={tuple(patch.shape)}")
        all_results[key] = {
            "patch_2d": patch[0, 0].numpy(),
            "final_2d": final[0, 0].numpy(),
            "scale": scale,
            "pixel_size_label": label if exp_name == DRAGONFLY_EXP else "reference",
        }

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

n = len(all_results)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
if n == 1:
    axes = axes[:, None]


def add_scalebar(ax, pixel_size_um, patch_px, bar_um=5.0):
    bar_px = bar_um / pixel_size_um
    y = patch_px - 12
    x0 = patch_px - bar_px - 8
    ax.plot(
        [x0, x0 + bar_px],
        [y, y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
    )
    ax.text(
        x0 + bar_px / 2,
        y - 8,
        f"{bar_um:.0f} um",
        color="white",
        fontsize=9,
        ha="center",
        fontweight="bold",
    )


for col, (key, r) in enumerate(all_results.items()):
    scale = r["scale"]

    # Row 0: Dataloader output (yx_patch_size, after _rescale_patch)
    ax = axes[0, col]
    patch = r["patch_2d"]
    vmin, vmax = np.percentile(patch, (1, 99))
    ax.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_XY_UM, YX_PATCH_SIZE[0])
    ax.set_title(
        f"{key}\nscale_yx=({scale[1]:.3f}, {scale[2]:.3f})\nDataloader: {YX_PATCH_SIZE[0]}x{YX_PATCH_SIZE[1]} px",
        fontsize=9,
        fontweight="bold",
    )
    ax.axis("off")

    # Row 1: After center crop = MODEL INPUT
    ax = axes[1, col]
    final = r["final_2d"]
    ax.imshow(final, cmap="gray", vmin=vmin, vmax=vmax)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_XY_UM, FINAL_YX_PATCH_SIZE[0])
    phys = FINAL_YX_PATCH_SIZE[0] * REFERENCE_PIXEL_SIZE_XY_UM
    ax.set_title(
        f"Model input: {FINAL_YX_PATCH_SIZE[0]}x{FINAL_YX_PATCH_SIZE[1]} px | {phys:.1f} um",
        fontsize=9,
    )
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2ecc71")
        spine.set_linewidth(3)

row_labels = [
    "Dataloader output\n(after _rescale_patch)",
    "Model input\n(after center crop)",
]
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].annotate(
        label,
        xy=(-0.12, 0.5),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )

fig.suptitle(
    f"Pixel-size normalization proof: reference={REFERENCE_PIXEL_SIZE_XY_UM} um/px\n"
    f"Same Dragonfly data with different declared pixel sizes -> different scale factors",
    fontsize=12,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.93])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PATH}")
plt.close(fig)
