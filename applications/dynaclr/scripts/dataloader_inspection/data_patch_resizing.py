"""End-to-end proof that DynaCLR pixel-size normalization works.

Builds the datamodule once to get sample metadata (cell coordinates),
then reads native zarr crops at different pixel-size-derived scales
and rescales them to show how the pipeline normalizes physical extent.

Row 0: Raw FOV with bounding boxes for each pixel-size variant.
Row 1: Native zarr crop → _rescale_patch → center crop = model input (160×160).

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/data_patch_resizing.py
"""

# %%
# ruff: noqa: D103

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from iohub.ngff.nodes import open_ome_zarr

from dynaclr.data.datamodule import MultiExperimentDataModule
from dynaclr.data.dataset import _rescale_patch
from viscy_transforms._crop import BatchedCenterSpatialCrop

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]

CELL_INDEX_PATH = _ROOT / "applications/dynaclr/configs/cell_index/example_mantis_dragonfly.parquet"
OUTPUT_DIR = _ROOT / "applications/dynaclr/scripts/dataloader_inspection/output"
OUTPUT_PATH = OUTPUT_DIR / "data_patch_resizing.png"

Z_WINDOW = 1
YX_PATCH_SIZE = (200, 200)
FINAL_YX_PATCH_SIZE = (160, 160)
REFERENCE_PIXEL_SIZE_XY_UM = 0.1494
CHANNEL_NAME = "Phase3D"

DRAGONFLY_EXP = "2024_08_14_ZIKV_pal17_48h"
MANTIS_EXP = "2025_07_24_A549_SEC61_ZIKV"

# Pixel sizes to visualize for Dragonfly
DRAGONFLY_PIXEL_SIZES = {
    "real (0.206)": 0.206,
    "same as ref (0.1494)": 0.1494,
    "coarser (0.7)": 0.7,
}

BBOX_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
INCLUDE_WELLS = ["A/2", "0/4"]

# ---------------------------------------------------------------------------
# Step 1: Build datamodule once to get sample metadata
# ---------------------------------------------------------------------------

print("Building datamodule...")
dm = MultiExperimentDataModule(
    cell_index_path=str(CELL_INDEX_PATH),
    z_window=Z_WINDOW,
    yx_patch_size=YX_PATCH_SIZE,
    final_yx_patch_size=FINAL_YX_PATCH_SIZE,
    batch_size=8,
    num_workers=0,
    channels_per_sample=[CHANNEL_NAME],
    reference_pixel_size_xy_um=REFERENCE_PIXEL_SIZE_XY_UM,
    reference_pixel_size_z_um=None,
    positive_cell_source="self",
    tau_range=(0.0, 100.0),
    stratify_by=None,
    include_wells=INCLUDE_WELLS,
)
dm.setup("fit")

registry = dm.train_dataset.index.registry

print("Drawing samples for metadata...")
loader = dm.train_dataloader()
per_exp: dict[str, dict] = {}
needed = {e.name for e in registry.experiments}

MAX_BATCHES = 200
for batch_idx, batch in enumerate(loader):
    anchor = batch["anchor"]
    meta = batch["anchor_meta"]
    for i in range(len(meta)):
        exp_name = meta[i]["experiment"]
        if exp_name not in per_exp:
            per_exp[exp_name] = {"meta": meta[i], "patch": anchor[i]}
    if per_exp.keys() >= needed:
        break
    if batch_idx >= MAX_BATCHES:
        print(f"  WARNING: only found experiments {set(per_exp.keys())} after {MAX_BATCHES} batches")
        break

for exp_name, d in per_exp.items():
    m = d["meta"]
    print(f"  {exp_name}: fov={m['fov_name']}, t={m['t']}, y={m['y_clamp']}, x={m['x_clamp']}")


# ---------------------------------------------------------------------------
# Step 2: Read raw FOV slices and native crops from zarr
# ---------------------------------------------------------------------------


def read_fov_and_crop(
    meta: dict,
    pixel_size_xy: float,
    z_focus: int,
    channel_name: str = CHANNEL_NAME,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read the focus Z-slice FOV and a native crop at the given pixel size.

    Returns
    -------
    fov : np.ndarray
        Full FOV 2D image at the focus Z-slice.
    crop : np.ndarray
        Native crop at the scale implied by pixel_size_xy.
    y_half, x_half : int
        Half-widths of the native crop in pixels.
    """
    store_path = meta["store_path"]
    fov_name = meta["fov_name"]
    t = int(meta["t"])
    y_center = int(meta["y_clamp"])
    x_center = int(meta["x_clamp"])

    scale_yx = REFERENCE_PIXEL_SIZE_XY_UM / pixel_size_xy
    y_half = round((YX_PATCH_SIZE[0] // 2) * scale_yx)
    x_half = round((YX_PATCH_SIZE[1] // 2) * scale_yx)

    fov_path = f"{store_path}/{fov_name}"
    with open_ome_zarr(fov_path, mode="r") as pos:
        ch_idx = list(pos.channel_names).index(channel_name)
        _, _, _, img_h, img_w = pos.data.shape

        fov = pos.data.oindex[t, ch_idx, z_focus, :, :]

        y0 = max(0, y_center - y_half)
        y1 = min(img_h, y_center + y_half)
        x0 = max(0, x_center - x_half)
        x1 = min(img_w, x_center + x_half)
        crop = pos.data.oindex[t, ch_idx, z_focus, y0:y1, x0:x1]

    return fov, crop, y_half, x_half


center_crop = BatchedCenterSpatialCrop(roi_size=(Z_WINDOW, FINAL_YX_PATCH_SIZE[0], FINAL_YX_PATCH_SIZE[1]))

z_focuses = {}
for e in registry.experiments:
    zr = registry.z_ranges[e.name]
    z_focuses[e.name] = (zr[0] + zr[1]) // 2
    print(f"  {e.name}: z_range={zr}, z_focus={z_focuses[e.name]}")

print("Reading zarr crops...")

results: list[dict] = []

# Mantis (reference — scale ≈ 1.0)
m_meta = per_exp[MANTIS_EXP]["meta"]
m_fov, m_crop, m_yh, m_xh = read_fov_and_crop(m_meta, REFERENCE_PIXEL_SIZE_XY_UM, z_focuses[MANTIS_EXP])
m_tensor = torch.from_numpy(m_crop).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
m_rescaled = _rescale_patch(m_tensor, (1.0, 1.0, 1.0), (Z_WINDOW, YX_PATCH_SIZE[0], YX_PATCH_SIZE[1]))
m_final = center_crop(m_rescaled[None])[0]
m_dl_patch = per_exp[MANTIS_EXP]["patch"]
m_dl_final = center_crop(m_dl_patch[None])[0]
results.append(
    {
        "label": f"{MANTIS_EXP}\nreference ({REFERENCE_PIXEL_SIZE_XY_UM} µm/px)",
        "exp": MANTIS_EXP,
        "fov": m_fov,
        "native_crop": m_crop,
        "final_2d": m_final[0, 0].numpy(),
        "dl_final_2d": m_dl_final[0, 0].numpy(),
        "scale_yx": 1.0,
        "pixel_size": REFERENCE_PIXEL_SIZE_XY_UM,
        "y_half": m_yh,
        "x_half": m_xh,
        "meta": m_meta,
    }
)

# Dragonfly — one entry per pixel-size variant
d_meta = per_exp[DRAGONFLY_EXP]["meta"]
d_dl_patch = per_exp[DRAGONFLY_EXP]["patch"]
d_dl_final = center_crop(d_dl_patch[None])[0]
d_fov = None

for i, (label, px_size) in enumerate(DRAGONFLY_PIXEL_SIZES.items()):
    fov, crop, y_half, x_half = read_fov_and_crop(d_meta, px_size, z_focuses[DRAGONFLY_EXP])
    if d_fov is None:
        d_fov = fov

    scale_yx = REFERENCE_PIXEL_SIZE_XY_UM / px_size
    scale = (1.0, scale_yx, scale_yx)
    target = (Z_WINDOW, YX_PATCH_SIZE[0], YX_PATCH_SIZE[1])

    crop_tensor = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)
    rescaled = _rescale_patch(crop_tensor, scale, target)
    final = center_crop(rescaled[None])[0]

    print(f"  {label}: scale_yx={scale_yx:.3f}, native_crop={crop.shape}, rescaled={tuple(rescaled.shape)}")

    results.append(
        {
            "label": f"{DRAGONFLY_EXP}\n{label}",
            "exp": DRAGONFLY_EXP,
            "fov": d_fov,
            "native_crop": crop,
            "final_2d": final[0, 0].numpy(),
            "dl_final_2d": d_dl_final[0, 0].numpy(),
            "scale_yx": scale_yx,
            "pixel_size": px_size,
            "y_half": y_half,
            "x_half": x_half,
            "meta": d_meta,
        }
    )

# ---------------------------------------------------------------------------
# Step 3: Plot
# ---------------------------------------------------------------------------


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
        f"{bar_um:.0f} µm",
        color="white",
        fontsize=9,
        ha="center",
        fontweight="bold",
    )


def add_bbox(ax, y_center, x_center, y_half, x_half, color, label, img_shape):
    y0 = max(0, y_center - y_half)
    x0 = max(0, x_center - x_half)
    h = min(y_center + y_half, img_shape[0]) - y0
    w = min(x_center + x_half, img_shape[1]) - x0
    rect = mpatches.Rectangle(
        (x0, y0),
        w,
        h,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        linestyle="-",
        label=label,
    )
    ax.add_patch(rect)


n = len(results)
fig, axes = plt.subplots(3, n, figsize=(5 * n, 14))
if n == 1:
    axes = axes[:, None]

for col, r in enumerate(results):
    meta = r["meta"]
    exp_name = r["exp"]
    y_center = int(meta["y_clamp"])
    x_center = int(meta["x_clamp"])

    # Row 0: Raw FOV with bounding box
    ax = axes[0, col]
    fov = r["fov"]
    vmin_raw, vmax_raw = np.percentile(fov, (1, 99))
    ax.imshow(fov, cmap="gray", vmin=vmin_raw, vmax=vmax_raw)

    if exp_name == DRAGONFLY_EXP:
        for i, (lbl, px_size) in enumerate(DRAGONFLY_PIXEL_SIZES.items()):
            s = REFERENCE_PIXEL_SIZE_XY_UM / px_size
            yh = round((YX_PATCH_SIZE[0] // 2) * s)
            xh = round((YX_PATCH_SIZE[1] // 2) * s)
            add_bbox(ax, y_center, x_center, yh, xh, BBOX_COLORS[i], lbl, fov.shape)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    else:
        add_bbox(
            ax,
            y_center,
            x_center,
            r["y_half"],
            r["x_half"],
            BBOX_COLORS[0],
            "reference",
            fov.shape,
        )

    ax.set_title(f"{r['label']}\nRaw FOV (mid-Z)", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Row 1: Model input (native crop → rescale → center crop)
    ax = axes[1, col]
    final = r["final_2d"]
    vmin, vmax = np.percentile(final, (1, 99))
    ax.imshow(final, cmap="gray", vmin=vmin, vmax=vmax)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_XY_UM, FINAL_YX_PATCH_SIZE[0])
    phys = FINAL_YX_PATCH_SIZE[0] * REFERENCE_PIXEL_SIZE_XY_UM
    ax.set_title(
        f"Model input: {FINAL_YX_PATCH_SIZE[0]}×{FINAL_YX_PATCH_SIZE[1]} px | {phys:.1f} µm\n"
        f"native crop: {r['native_crop'].shape} → scale_yx={r['scale_yx']:.3f}",
        fontsize=9,
    )
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2ecc71")
        spine.set_linewidth(3)

    # Row 2: Actual dataloader output (for comparison with "real" variant)
    ax = axes[2, col]
    dl_final = r["dl_final_2d"]
    vmin_dl, vmax_dl = np.percentile(dl_final, (1, 99))
    ax.imshow(dl_final, cmap="gray", vmin=vmin_dl, vmax=vmax_dl)
    add_scalebar(ax, REFERENCE_PIXEL_SIZE_XY_UM, FINAL_YX_PATCH_SIZE[0])
    ax.set_title(
        f"Dataloader output: {FINAL_YX_PATCH_SIZE[0]}×{FINAL_YX_PATCH_SIZE[1]} px\n"
        f"(same for all variants — real pixel size)",
        fontsize=9,
    )
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#e67e22")
        spine.set_linewidth(3)

row_labels = [
    "Raw FOV + crop region",
    "Expected\n(native crop → rescale → crop)",
    "Dataloader output\n(real pixel size)",
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
    f"Pixel-size normalization: reference={REFERENCE_PIXEL_SIZE_XY_UM} µm/px\n"
    f"Different pixel sizes → different native crops"
    f" → same {FINAL_YX_PATCH_SIZE[0]}×{FINAL_YX_PATCH_SIZE[1]} model input",
    fontsize=12,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.93])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PATH}")
plt.close(fig)

# %%
