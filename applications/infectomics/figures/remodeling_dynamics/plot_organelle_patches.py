"""
Grid of cell image patches: organelle (green) and sensor (magenta) channels.

Rows = organelle types (G3BP1, SEC61B, TOMM20)
Cols = infection conditions (mock, ZIKV, DENV)

For each entry the script:
  - Opens the data zarr and tracking zarr
  - Extracts GFP (organelle) and RFP (sensor) patches centred on a tracked cell
  - Builds an RGB composite: Red+Blue = sensor (magenta), Green = organelle
  - Normalises each channel independently

Input: zarr files referenced in patch_list.
Output: organelle_patches_grid.png  (PNG for quality, not SVG)
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import zarr
import matplotlib.pyplot as plt
from skimage.measure import regionprops

# %% paths
data_root = Path("/path/to/organelle_dynamics/")   # update to actual experiment root
output_png = Path("organelle_patches_grid.png")

# %% patch configuration
patch_size = 200

# Each dict: organelle, infection, data_path (relative to data_root),
# track_path (relative to data_root), FOV, track_id (int), time (int)
patch_list = [
    # G3BP1
    {
        "organelle": "G3BP1",
        "infection": "mock",
        "data_path": "G3BP1/mock/data.zarr",
        "track_path": "G3BP1/mock/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "G3BP1",
        "infection": "ZIKV",
        "data_path": "G3BP1/ZIKV/data.zarr",
        "track_path": "G3BP1/ZIKV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "G3BP1",
        "infection": "DENV",
        "data_path": "G3BP1/DENV/data.zarr",
        "track_path": "G3BP1/DENV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    # SEC61B
    {
        "organelle": "SEC61B",
        "infection": "mock",
        "data_path": "SEC61B/mock/data.zarr",
        "track_path": "SEC61B/mock/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "SEC61B",
        "infection": "ZIKV",
        "data_path": "SEC61B/ZIKV/data.zarr",
        "track_path": "SEC61B/ZIKV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "SEC61B",
        "infection": "DENV",
        "data_path": "SEC61B/DENV/data.zarr",
        "track_path": "SEC61B/DENV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    # TOMM20
    {
        "organelle": "TOMM20",
        "infection": "mock",
        "data_path": "TOMM20/mock/data.zarr",
        "track_path": "TOMM20/mock/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "TOMM20",
        "infection": "ZIKV",
        "data_path": "TOMM20/ZIKV/data.zarr",
        "track_path": "TOMM20/ZIKV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
    {
        "organelle": "TOMM20",
        "infection": "DENV",
        "data_path": "TOMM20/DENV/data.zarr",
        "track_path": "TOMM20/DENV/tracks.zarr",
        "FOV": "B/3/001001",
        "track_id": 243,
        "time": 44,
    },
]

# Derive grid dimensions from patch_list
ORGANELLES = list(dict.fromkeys(p["organelle"] for p in patch_list))
INFECTIONS = list(dict.fromkeys(p["infection"] for p in patch_list))
N_ROWS = len(ORGANELLES)
N_COLS = len(INFECTIONS)


# %% patch extraction function
def get_patch(
    data: np.ndarray,
    seg_mask: np.ndarray,
    track_id: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract GFP (ch0) and RFP (ch1) patches centred on *track_id* in *seg_mask*.

    Parameters
    ----------
    data      : shape (C, Y, X) — at least 2 channels
    seg_mask  : shape (Y, X)    — integer label image
    track_id  : int             — label value for the target cell
    patch_size: int             — side length of the square patch in pixels

    Returns
    -------
    (gfp_patch, rfp_patch) each of shape (patch_size, patch_size)
    """
    cell_mask = seg_mask == track_id
    props = regionprops(cell_mask.astype(int))
    if not props:
        raise ValueError(f"track_id {track_id} not found in segmentation mask.")
    y_centroid, x_centroid = props[0].centroid

    half = patch_size // 2
    x_start = int(x_centroid - half)
    x_end = int(x_centroid + half)
    y_start = int(y_centroid - half)
    y_end = int(y_centroid + half)

    gfp_patch = data[0, y_start:y_end, x_start:x_end]
    rfp_patch = data[1, y_start:y_end, x_start:x_end]
    return gfp_patch, rfp_patch


# %% normalisation helper
def normalise(arr: np.ndarray) -> np.ndarray:
    """Stretch array to [0, 1], handling flat arrays safely."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr.astype(float) - lo) / (hi - lo)


# %% build figure grid
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3, N_ROWS * 3))
# Ensure axes is always 2-D
if N_ROWS == 1:
    axes = axes[np.newaxis, :]
if N_COLS == 1:
    axes = axes[:, np.newaxis]

# Column and row titles
for col_idx, infection in enumerate(INFECTIONS):
    axes[0][col_idx].set_title(infection, fontsize=11)

for row_idx, organelle in enumerate(ORGANELLES):
    axes[row_idx][0].set_ylabel(organelle, fontsize=11, rotation=90, labelpad=5)

# %% iterate patch_list and fill grid
for entry in patch_list:
    organelle = entry["organelle"]
    infection = entry["infection"]
    row_idx = ORGANELLES.index(organelle)
    col_idx = INFECTIONS.index(infection)
    ax = axes[row_idx][col_idx]

    data_path = data_root / entry["data_path"]
    track_path = data_root / entry["track_path"]
    fov = entry["FOV"]
    track_id = int(entry["track_id"])
    time_idx = int(entry["time"])

    try:
        data_store = zarr.open(str(data_path), mode="r")
        track_store = zarr.open(str(track_path), mode="r")

        # Navigate FOV path: split e.g. "B/3/001001"
        fov_parts = fov.split("/")
        data_fov = data_store
        track_fov = track_store
        for part in fov_parts:
            data_fov = data_fov[part]
            track_fov = track_fov[part]

        # data shape expected: (T, C, Y, X) or (C, Y, X)
        data_arr = np.array(data_fov)
        track_arr = np.array(track_fov)

        if data_arr.ndim == 4:
            # (T, C, Y, X) — select timepoint
            frame_data = data_arr[time_idx]      # (C, Y, X)
        else:
            frame_data = data_arr               # (C, Y, X)

        if track_arr.ndim == 3:
            seg_mask = track_arr[time_idx]       # (Y, X)
        else:
            seg_mask = track_arr

        gfp, rfp = get_patch(frame_data, seg_mask, track_id, patch_size)

        # Build RGB: green = GFP (organelle), red+blue = RFP (sensor → magenta)
        gfp_norm = normalise(gfp)
        rfp_norm = normalise(rfp)

        # Pad to exact patch_size if near border
        def pad_to(arr, size):
            h, w = arr.shape
            if h < size or w < size:
                padded = np.zeros((size, size), dtype=arr.dtype)
                padded[:h, :w] = arr
                return padded
            return arr[:size, :size]

        gfp_norm = pad_to(gfp_norm, patch_size)
        rfp_norm = pad_to(rfp_norm, patch_size)

        rgb = np.zeros((patch_size, patch_size, 3), dtype=float)
        rgb[:, :, 0] = rfp_norm   # Red   → sensor
        rgb[:, :, 1] = gfp_norm   # Green → organelle
        rgb[:, :, 2] = rfp_norm   # Blue  → sensor  (together = magenta)

        ax.imshow(rgb, interpolation="nearest")

    except Exception as exc:
        ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                fontsize=6, transform=ax.transAxes, color="red", wrap=True)

    ax.axis("off")

plt.tight_layout()
fig.savefig(output_png, format="png", dpi=150, bbox_inches="tight")
print(f"Saved: {output_png}")
plt.close(fig)
