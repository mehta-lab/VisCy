"""Inspect TripletDataModule with focus-centered z_range + on-the-fly Z-reduction.

Loads a 3D OME-Zarr + tracking, extracts a focus-centered Z window per FOV
(from ``focus_slice`` zattrs), collapses it to a single slice via ``z_reduction``
(so a 3D dataset feeds a 2D model), and visualizes a couple of batches.

Run cell-by-cell in an interactive window (VS Code / Jupyter) or top-to-bottom.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from iohub import open_ome_zarr

from viscy_data import TripletDataModule
from viscy_transforms import BatchedRandAffined, NormalizeSampled

# --- edit these ---------------------------------------------------------------
DATASET_DIR = "/hpc/projects/organelle_phenotyping/datasets/2026_04_08_A549_G3BP1_ZIKV"
DATA_PATH = f"{DATASET_DIR}/2026_04_08_A549_G3BP1_ZIKV.zarr"
TRACKS_PATH = f"{DATASET_DIR}/tracking.zarr"
SOURCE_CHANNEL = ["Phase3D", "raw GFP EX488 EM525-45"]  # label-free + fluorescence
FOCUS_CHANNEL = "Phase3D"  # channel whose focus plane centers the Z window
Z_EXTRACTION_WINDOW = 15  # Z slices extracted, centered on each FOV's focus plane
Z_FOCUS_OFFSET = 0.5  # fraction of the window below the focus plane
Z_REDUCTION = "mip"  # "mip" (fluorescence) / center-slice (label-free), or None
YX_PATCH = (256, 256)  # extracted == final (no YX rescale; set reference_pixel_size to rescale)
BATCH_SIZE = 8
N_BATCHES = 2
OUTPUT_DIR = "/home/eduardo.hirata/repos/viscy/applications/dynaclr/scripts/dataloader_inspection/output"
OUTPUT_PNG = f"{OUTPUT_DIR}/triplet_zprojection.png"
# ------------------------------------------------------------------------------

# %% [markdown]
# Peek at the store: channels, shape, and the per-FOV focus plane the window centers on.

# %%
with open_ome_zarr(DATA_PATH) as plate:
    print("channels:", plate.channel_names)
    name, pos = next(iter(plate.positions()))
    print("first FOV:", name, "| TCZYX:", pos["0"].shape)
    focus = pos.zattrs.get("focus_slice", {}).get(FOCUS_CHANNEL, {})
    print("per-FOV z_focus_mean:", focus.get("fov_statistics", {}).get("z_focus_mean"))

# %% [markdown]
# Build the datamodule. ``z_extraction_window`` (not ``z_range``) makes each FOV's
# window center on its own focus plane; ``z_reduction`` collapses Z to 1.

# %%
dm = TripletDataModule(
    data_path=DATA_PATH,
    tracks_path=TRACKS_PATH,
    source_channel=SOURCE_CHANNEL,
    z_extraction_window=Z_EXTRACTION_WINDOW,
    z_focus_offset=Z_FOCUS_OFFSET,
    focus_channel=FOCUS_CHANNEL,
    initial_yx_patch_size=YX_PATCH,
    final_yx_patch_size=YX_PATCH,
    z_reduction=Z_REDUCTION,
    batch_size=BATCH_SIZE,
    num_workers=0,
    normalizations=[
        NormalizeSampled(
            keys=SOURCE_CHANNEL,
            level="fov_statistics",
            subtrahend="mean",
            divisor="std",
        )
    ],
    augmentations=[
        # Random affine so anchor and positive (a clone of the anchor when
        # time_interval="any") diverge — applied per-key with fresh random
        # params, on the 3D stack before z_reduction. prob=1.0 = always fire.
        BatchedRandAffined(
            keys=SOURCE_CHANNEL,
            prob=1.0,
            # rotate_range is (Z, Y, X) radians: the Z entry is the in-plane (XY)
            # rotation. Rotating about Y/X would tumble the stack out of plane and
            # collapse to a strip after MIP, so keep those at 0.
            rotate_range=(3.14159, 0.0, 0.0),  # full in-plane rotation
            scale_range=(0.8, 1.2),
            translate_range=(0.0, 0.1, 0.1),  # up to 10% YX shift, no Z shift
        )
    ],
)
dm.setup(stage="fit")

# Per-FOV focus-centered windows resolved at setup (one slice per FOV).
resolved = dm.train_dataset.z_range
print(f"resolved {len(resolved)} per-FOV z-windows; e.g.:")
for fov, z_slice in list(resolved.items())[:5]:
    print(f"  {fov}: {z_slice}")

# %% [markdown]
# Pull a couple of batches and check shapes. After z_reduction the Z axis is 1.

# %%
batches = []
for i, batch in enumerate(dm.train_dataloader()):
    dm.on_after_batch_transfer(batch, 0)  # normalize + Z-reduce on the batch
    print(f"batch {i}: anchor {tuple(batch['anchor'].shape)} (expect Z=1)")
    batches.append(batch)
    if i + 1 >= N_BATCHES:
        break

# %% [markdown]
# Visualize anchor / positive / negative for the first few cells of each batch.
# Each panel is one channel of the Z-reduced (B, C, 1, Y, X) patch.

# %%
keys = [k for k in ("anchor", "positive", "negative") if k in batches[0]]
n_cells = min(4, BATCH_SIZE)
n_rows = N_BATCHES * n_cells
n_cols = len(keys) * len(SOURCE_CHANNEL)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), squeeze=False)

for bi, batch in enumerate(batches):
    for ci in range(n_cells):
        row = bi * n_cells + ci
        col = 0
        for key in keys:
            patch = batch[key][ci]  # (C, 1, Y, X)
            for ch_idx, ch in enumerate(SOURCE_CHANNEL):
                img = patch[ch_idx, 0].cpu().numpy()  # (Y, X)
                lo, hi = (img.min(), img.max()) if img.max() > img.min() else (0.0, 1.0)
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
                ax.set_title(f"b{bi} c{ci}\n{key}/{ch.split()[0]}", fontsize=7)
                ax.axis("off")
                col += 1

fig.suptitle(f"Triplet patches — z_extraction_window={Z_EXTRACTION_WINDOW}, z_reduction={Z_REDUCTION}", fontsize=10)
fig.tight_layout()
Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
print("saved:", OUTPUT_PNG)
