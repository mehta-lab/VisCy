"""
Generate per-frame PNG images for embedding + cell image videos.

Each frame shows:
  Left  — 3 rows × 1 col: phase / organelle / sensor cell patches
  Right — PHATE scatter: background grey, current timepoint coloured by
          remodeling_status, highlighted tracked cell with larger black-edged marker.

Note: viscy.representation.evaluation.dataset_of_tracks may not be available as
anndata.  If the viscy API changes, replace the zarr loading block with direct
zarr indexing (which is already implemented below).

Input:
  embedding_csv   – organelle_embeddings_with_predictions.csv
  data_zarr_path  – data.zarr
  tracks_zarr_path – tracks.zarr
  output_dir      – video_frames/
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.stats_utils import get_significance_stars, add_significance_bar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import zarr

# %% paths and parameters
embedding_csv = Path("organelle_embeddings_with_predictions.csv")
data_zarr_path = Path("data.zarr")
tracks_zarr_path = Path("tracks.zarr")
output_dir = Path("video_frames/")

# %% video configuration
source_channels = [
    "Phase3D",
    "raw GFP EX488 EM525-45",
    "raw mCherry EX561 EM600-37",
]
CHANNEL_LABELS = ["Phase", "Organelle (GFP)", "Sensor (RFP)"]

z_slice = 0              # z-index for 3-D volumes
patch_size = 160         # pixels
time_interval_hours = 10 / 60   # 10 minutes per frame
start_hpi = 4.5          # hours post infection at first frame

# Tracks to visualise: {fov_name: [track_id, ...]}
viz_tracks = {
    "B/3/001001": [48, 50],
}

# Remodeling colour map for PHATE scatter
REMODELING_PALETTE = {
    "control": "green",
    "remodeled": "red",
}
BACKGROUND_COLOR = "0.4"    # grey for non-current timepoints
HIGHLIGHT_SIZE = 80
BACKGROUND_SIZE = 5
CURRENT_TP_SIZE = 15

output_dir.mkdir(parents=True, exist_ok=True)

# %% load embedding CSV
emb_df = pd.read_csv(embedding_csv)
print(f"Embeddings: {len(emb_df)} rows")
all_timepoints = sorted(emb_df["t"].unique())


# %% helper: normalise image for display
def normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr.astype(float) - lo) / (hi - lo)


# %% helper: extract patch around a tracked cell
def get_patch(
    data: np.ndarray,
    seg_mask: np.ndarray,
    track_id: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (phase, gfp, rfp) patches centred on track_id.

    data     : shape (C, [Z,] Y, X)
    seg_mask : shape (Y, X)
    """
    cell_mask = seg_mask == track_id
    props = regionprops(cell_mask.astype(int))
    if not props:
        blank = np.zeros((patch_size, patch_size), dtype=float)
        return blank, blank, blank

    y_c, x_c = props[0].centroid
    half = patch_size // 2
    ys, ye = int(y_c - half), int(y_c + half)
    xs, xe = int(x_c - half), int(x_c + half)

    def extract_2d(ch_idx):
        ch = data[ch_idx]
        if ch.ndim == 3:          # (Z, Y, X) — take z_slice
            ch = ch[z_slice]
        patch = ch[ys:ye, xs:xe]
        # Pad if near border
        if patch.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size), dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            return padded
        return patch

    return extract_2d(0), extract_2d(1), extract_2d(2)


# %% helper: load zarr array navigating FOV path
def load_fov_zarr(store_path: Path, fov: str):
    """Open zarr store and navigate to the given FOV sub-array."""
    store = zarr.open(str(store_path), mode="r")
    node = store
    for part in fov.split("/"):
        node = node[part]
    return np.array(node)


# %% main loop over FOVs and tracks
for fov_name, track_ids in viz_tracks.items():
    print(f"\nProcessing FOV: {fov_name}  tracks: {track_ids}")

    # Load zarr arrays for this FOV
    try:
        data_arr = load_fov_zarr(data_zarr_path, fov_name)    # (T, C, [Z,] Y, X)
        track_arr = load_fov_zarr(tracks_zarr_path, fov_name) # (T, Y, X) or (T, Z, Y, X)
    except Exception as exc:
        print(f"  Could not load zarr for {fov_name}: {exc}")
        continue

    n_timepoints = data_arr.shape[0]

    # Embedding subset for this FOV
    fov_emb = emb_df[emb_df["fov_name"] == fov_name].copy()

    for track_id in track_ids:
        safe_fov = fov_name.replace("/", "_")
        frame_dir = output_dir / f"{safe_fov}_track{track_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        for t_idx in range(n_timepoints):
            hpi = start_hpi + t_idx * time_interval_hours

            # ---- cell images ----
            frame_data = data_arr[t_idx]      # (C, [Z,] Y, X)
            seg_frame = track_arr[t_idx]
            if seg_frame.ndim == 3:            # (Z, Y, X)
                seg_frame = seg_frame[z_slice]  # (Y, X)

            phase_patch, gfp_patch, rfp_patch = get_patch(
                frame_data, seg_frame, track_id, patch_size
            )

            # ---- PHATE scatter ----
            with plt.style.context("dark_background"):
                fig = plt.figure(figsize=(10, 5))
                fig.suptitle(
                    f"FOV {fov_name}  |  Track {track_id}  |  t={t_idx}  |  {hpi:.1f} hpi",
                    fontsize=10, color="white",
                )

                # Left: 3 cell image subplots
                ax_phase = fig.add_subplot(3, 2, 1)
                ax_org   = fig.add_subplot(3, 2, 3)
                ax_sens  = fig.add_subplot(3, 2, 5)

                for ax_img, patch, label in zip(
                    [ax_phase, ax_org, ax_sens],
                    [phase_patch, gfp_patch, rfp_patch],
                    CHANNEL_LABELS,
                ):
                    ax_img.imshow(normalise(patch), cmap="gray", interpolation="nearest")
                    ax_img.set_title(label, fontsize=8, color="white")
                    ax_img.axis("off")

                # Right: PHATE scatter
                ax_phate = fig.add_subplot(1, 2, 2)

                # Background: all points at all timepoints (grey)
                ax_phate.scatter(
                    fov_emb["PHATE1"],
                    fov_emb["PHATE2"],
                    c=BACKGROUND_COLOR,
                    s=BACKGROUND_SIZE,
                    alpha=0.3,
                    rasterized=True,
                    zorder=1,
                )

                # Current timepoint: colour by remodeling_status
                current_tp = fov_emb[fov_emb["t"] == t_idx]
                for status, color in REMODELING_PALETTE.items():
                    pts = current_tp[
                        current_tp["remodeling_status"].astype(str).str.lower() == status.lower()
                    ]
                    if not pts.empty:
                        ax_phate.scatter(
                            pts["PHATE1"],
                            pts["PHATE2"],
                            c=color,
                            s=CURRENT_TP_SIZE,
                            alpha=0.8,
                            label=status,
                            zorder=2,
                        )

                # Highlighted cell: larger marker with black edge
                highlighted = fov_emb[
                    (fov_emb["t"] == t_idx) & (fov_emb["track_id"] == track_id)
                ]
                if not highlighted.empty:
                    hl_status = str(highlighted["remodeling_status"].iloc[0]).lower()
                    hl_color = REMODELING_PALETTE.get(hl_status, "yellow")
                    ax_phate.scatter(
                        highlighted["PHATE1"],
                        highlighted["PHATE2"],
                        c=hl_color,
                        s=HIGHLIGHT_SIZE,
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=3,
                        label="current cell",
                    )

                ax_phate.set_xlabel("PHATE1", fontsize=9, color="white")
                ax_phate.set_ylabel("PHATE2", fontsize=9, color="white")
                ax_phate.set_title("PHATE embedding", fontsize=10, color="white")
                ax_phate.legend(fontsize=7, loc="upper right")
                ax_phate.tick_params(colors="white")
                for spine in ax_phate.spines.values():
                    spine.set_edgecolor("white")

                plt.tight_layout()
                frame_path = frame_dir / f"frame_{t_idx:04d}.png"
                fig.savefig(frame_path, format="png", dpi=100, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)

        print(f"  Saved {n_timepoints} frames to {frame_dir}")
