"""End-to-end proof that the DynaCLR dataloader delivers uniform patches to the model.

Simulates the full pipeline for 40x and 60x microscopes with different pixel sizes:

1. ``_slice_patch``: crop ``round(yx_patch_size/2 * scale)`` native pixels
2. ``_rescale_patch``: rescale to ``yx_patch_size`` via nearest-exact interpolation
3. ``BatchedCenterSpatialCropd``: center-crop to ``final_yx_patch_size`` (what the model sees)

The plot proves that the model receives identically-sized patches covering the
same physical field of view regardless of native microscope resolution.

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/data_patch_resizing.py
"""

# ruff: noqa: D103

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from iohub.ngff import open_ome_zarr

# ---------------------------------------------------------------------------
# Configuration — edit these to match your training config
# ---------------------------------------------------------------------------

CELL_INDEX_PATH = Path("/home/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/example_flat.parquet")
OUTPUT_PATH = Path(
    "/home/eduardo.hirata/repos/viscy/applications/dynaclr/scripts/dataloader_inspection/output/data_patch_resizing.png"
)

YX_PATCH_SIZE = 200  # yx_patch_size: dataset crops and rescales to this
FINAL_YX_PATCH_SIZE = 160  # final_yx_patch_size: center crop (model input)

# Reference pixel size — choose the finest (highest resolution) microscope
REFERENCE_PIXEL_SIZE_UM = 0.108  # 60x objective

# Simulated microscopes: (label, pixel_size_um)
MICROSCOPES = [
    ("60x (0.108 um/px)", 0.108),
    ("40x (0.1494 um/px)", 0.1494),
    ("20x (0.2028 um/px)", 0.2028),
]

CHANNEL_INDEX = 0  # zarr channel to visualize (0 = Phase3D)
Z_SLICE = 0  # single z slice for 2D visualization


# ---------------------------------------------------------------------------
# Core logic — mirrors dynaclr.data.dataset._rescale_patch
# ---------------------------------------------------------------------------


def rescale_2d(patch: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """Rescale a (Y, X) tensor to target size using nearest-exact interpolation."""
    if patch.shape[0] == target[0] and patch.shape[1] == target[1]:
        return patch
    return F.interpolate(
        patch.float()[None, None],
        size=target,
        mode="nearest-exact",
    ).squeeze()


def center_crop_2d(patch: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a (Y, X) array to (size, size)."""
    h, w = patch.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return patch[y0 : y0 + size, x0 : x0 + size]


def compute_scale(reference_um: float, native_um: float) -> float:
    return reference_um / native_um


def crop_native(image: np.ndarray, yc: int, xc: int, scale: float, patch_size: int) -> np.ndarray:
    """Crop native pixels around (yc, xc), mirroring _slice_patch."""
    half = round((patch_size // 2) * scale)
    h, w = image.shape
    y0, y1 = max(0, yc - half), min(h, yc + half)
    x0, x1 = max(0, xc - half), min(w, xc + half)
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_parquet(CELL_INDEX_PATH)
df_phase = df[df["channel_name"] == "Phase3D"].copy()
store_path = df_phase["store_path"].iloc[0]

with open_ome_zarr(store_path, mode="r") as plate:
    first_pos = next(iter(plate.positions()))[1]
    fov_h, fov_w = first_pos["0"].shape[3], first_pos["0"].shape[4]

# Safe margin: half of largest possible native crop + buffer
max_scale = max(compute_scale(REFERENCE_PIXEL_SIZE_UM, ps) for _, ps in MICROSCOPES)
margin = round((YX_PATCH_SIZE // 2) * max_scale) + 20
safe = df_phase[
    (df_phase["y"] > margin)
    & (df_phase["y"] < fov_h - margin)
    & (df_phase["x"] > margin)
    & (df_phase["x"] < fov_w - margin)
]
row = safe.iloc[len(safe) // 2]
yc, xc, t = int(row["y"]), int(row["x"]), int(row["t"])
fov, well = row["fov"], row["well"]

print(f"Store: {store_path}")
print(f"FOV: {fov}, well: {well}, t: {t}, centroid: ({yc}, {xc})")
print(f"Reference pixel size: {REFERENCE_PIXEL_SIZE_UM} um")
print(f"yx_patch_size: {YX_PATCH_SIZE}, final_yx_patch_size: {FINAL_YX_PATCH_SIZE}")

with open_ome_zarr(store_path, mode="r") as plate:
    image = np.array(plate[f"{well}/{fov}"]["0"][t, CHANNEL_INDEX, Z_SLICE])

print(f"FOV shape: {image.shape}")

# ---------------------------------------------------------------------------
# Run pipeline for each microscope and collect results
# ---------------------------------------------------------------------------

results = []
for label, native_um in MICROSCOPES:
    scale = compute_scale(REFERENCE_PIXEL_SIZE_UM, native_um)

    # Step 1: _slice_patch — crop scaled number of native pixels
    native = crop_native(image, yc, xc, scale, YX_PATCH_SIZE)

    # Step 2: _rescale_patch — resize to yx_patch_size
    rescaled = rescale_2d(torch.from_numpy(native.copy()), (YX_PATCH_SIZE, YX_PATCH_SIZE)).numpy()

    # Step 3: BatchedCenterSpatialCropd — center crop to final_yx_patch_size
    final = center_crop_2d(rescaled, FINAL_YX_PATCH_SIZE)

    native_h, native_w = native.shape
    phys_crop = native_h * native_um
    phys_final = FINAL_YX_PATCH_SIZE * REFERENCE_PIXEL_SIZE_UM

    results.append(
        {
            "label": label,
            "native_um": native_um,
            "scale": scale,
            "native": native,
            "rescaled": rescaled,
            "final": final,
            "native_shape": (native_h, native_w),
            "phys_crop_um": phys_crop,
            "phys_final_um": phys_final,
        }
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

n = len(MICROSCOPES)
fig, axes = plt.subplots(4, n, figsize=(5.5 * n, 18), gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})


def add_scalebar(ax, pixel_size_um, patch_px, bar_um=5.0):
    bar_px = bar_um / pixel_size_um
    y = patch_px - 12
    x0 = patch_px - bar_px - 8
    ax.plot([x0, x0 + bar_px], [y, y], color="white", linewidth=3, solid_capstyle="butt")
    ax.text(
        x0 + bar_px / 2,
        y - 8,
        f"{bar_um:.0f} um",
        color="white",
        fontsize=9,
        ha="center",
        fontweight="bold",
    )


for col, r in enumerate(results):
    vmin, vmax = np.percentile(image, (1, 99))

    # --- Row 0: FOV context with crop box ---
    ax = axes[0, col]
    ctx = 300
    y0c, y1c = max(0, yc - ctx), min(image.shape[0], yc + ctx)
    x0c, x1c = max(0, xc - ctx), min(image.shape[1], xc + ctx)
    context = image[y0c:y1c, x0c:x1c]
    ax.imshow(context, cmap="gray", vmin=vmin, vmax=vmax)

    half = round((YX_PATCH_SIZE // 2) * r["scale"])
    box_y = (yc - half) - y0c
    box_x = (xc - half) - x0c
    rect = plt.Rectangle(
        (box_x, box_y),
        2 * half,
        2 * half,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(f"{r['label']}\nscale = {r['scale']:.3f}", fontsize=11, fontweight="bold")
    ax.axis("off")

    # --- Row 1: Raw native crop (different sizes per microscope) ---
    ax = axes[1, col]
    nc = r["native"]
    ax.imshow(nc, cmap="gray", vmin=vmin, vmax=vmax)
    nh, nw = r["native_shape"]
    ax.set_title(f"{nh}x{nw} native px\n{r['phys_crop_um']:.1f} um FOV", fontsize=10)
    ax.axis("off")

    # --- Row 2: After _rescale_patch → yx_patch_size ---
    ax = axes[2, col]
    ax.imshow(r["rescaled"], cmap="gray", vmin=vmin, vmax=vmax)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_UM, YX_PATCH_SIZE)
    ax.set_title(
        f"{YX_PATCH_SIZE}x{YX_PATCH_SIZE} px (rescaled)\n{YX_PATCH_SIZE * REFERENCE_PIXEL_SIZE_UM:.1f} um FOV",
        fontsize=10,
    )
    ax.axis("off")

    # --- Row 3: After center crop → final_yx_patch_size (MODEL INPUT) ---
    ax = axes[3, col]
    ax.imshow(r["final"], cmap="gray", vmin=vmin, vmax=vmax)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_UM, FINAL_YX_PATCH_SIZE)
    ax.set_title(
        f"{FINAL_YX_PATCH_SIZE}x{FINAL_YX_PATCH_SIZE} px (model input)\n"
        f"{r['phys_final_um']:.1f} um FOV | {REFERENCE_PIXEL_SIZE_UM} um/px",
        fontsize=10,
    )
    ax.axis("off")

    # Green border on row 3 to highlight "this is what the model sees"
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2ecc71")
        spine.set_linewidth(3)

# Row labels
row_labels = [
    "FOV + crop region",
    "1. Native crop\n(different sizes)",
    "2. Rescale to\nyx_patch_size",
    "3. Center crop to\nfinal_yx_patch_size\n(MODEL INPUT)",
]
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].annotate(
        label,
        xy=(-0.18, 0.5),
        xycoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )

fig.suptitle(
    f"End-to-end pixel-size normalization: reference={REFERENCE_PIXEL_SIZE_UM} um/px\n"
    f"yx_patch_size={YX_PATCH_SIZE} -> final_yx_patch_size={FINAL_YX_PATCH_SIZE} "
    f"(all microscopes -> same {FINAL_YX_PATCH_SIZE}x{FINAL_YX_PATCH_SIZE} px at {REFERENCE_PIXEL_SIZE_UM} um/px)",
    fontsize=12,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.94])
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PATH}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n## Pipeline: yx_patch_size={YX_PATCH_SIZE}, final_yx_patch_size={FINAL_YX_PATCH_SIZE}")
print(f"## Reference pixel size: {REFERENCE_PIXEL_SIZE_UM} um/px\n")
cols = [
    "Microscope",
    "Native (um/px)",
    "Scale",
    "Native crop (px)",
    "Phys FOV (um)",
    "Rescaled (px)",
    "Model input (px)",
    "Model FOV (um)",
]
header = "| " + " | ".join(cols) + " |"
sep = "|" + "|".join("-" * (len(c) + 2) for c in cols) + "|"
print(header)
print(sep)
for r in results:
    nh, nw = r["native_shape"]
    print(
        f"| {r['label']:>20s} | {r['native_um']:.4f}         | {r['scale']:.3f} "
        f"| {nh}x{nw}            | {r['phys_crop_um']:.1f}         "
        f"| {YX_PATCH_SIZE}x{YX_PATCH_SIZE}         | {FINAL_YX_PATCH_SIZE}x{FINAL_YX_PATCH_SIZE}            "
        f"| {r['phys_final_um']:.1f}          |"
    )
