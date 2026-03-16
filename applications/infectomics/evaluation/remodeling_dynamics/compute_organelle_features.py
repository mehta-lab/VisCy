# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import pandas as pd
from iohub import open_ome_zarr

from utils.feature_utils import compute_glcm_features, compute_edge_density
from utils.image_utils import get_patch_2d

# %% paths
input_zarr_path = Path("input_data.zarr")
seg_zarr_path = Path("segmentation_puncta.zarr")
tracks_zarr_path = Path("tracks.zarr")
output_csv = Path("organelle_features.csv")

# Channel names
organelle_channel = "raw GFP EX488 EM525-45"
organelle_seg_channel = "Organelle_mask"

# Patch size in pixels (square; centred on the tracked cell position)
patch_size = 160


# %% helper – append records to CSV incrementally
def append_records_to_csv(records: list, csv_path: Path) -> None:
    """Append *records* (list of dicts) to *csv_path*.
    Creates the file with header on first call; appends without header after."""
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# %% helper – locate a tracks CSV for a given well/position
def find_tracks_csv(tracks_zarr: Path, well_name: str, pos_name: str) -> Path | None:
    """Attempt to locate the tracks CSV for a given position.

    Tries common naming conventions inside the tracks zarr directory tree.

    Parameters
    ----------
    tracks_zarr:
        Root path of the tracks zarr store.
    well_name:
        Well identifier string, e.g. "A/1".
    pos_name:
        Position identifier string, e.g. "0".

    Returns
    -------
    Path or None
        Path to the CSV if found, else None.
    """
    well_flat = well_name.replace("/", "_")
    pos_flat = pos_name.replace("/", "_")
    candidates = [
        tracks_zarr.parent / f"tracks_{well_flat}_{pos_flat}.csv",
        Path(str(tracks_zarr)) / well_name / pos_name / "tracks.csv",
        Path(str(tracks_zarr)) / f"tracks_{well_flat}_{pos_flat}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# %% remove stale output
if output_csv.exists():
    os.remove(output_csv)

# %% main loop
with open_ome_zarr(input_zarr_path, mode="r") as plate, \
     open_ome_zarr(seg_zarr_path, mode="r") as seg_plate:

    for well_name, well in plate.wells():
        for pos_name, pos in well.positions():
            img_data = pos["0"]          # T, C, Z, Y, X
            T, C, Z, Y, X = img_data.shape
            channel_names = pos.channel_names

            if organelle_channel not in channel_names:
                print(f"  [WARN] '{organelle_channel}' not found in {well_name}/{pos_name}. Skipping.")
                continue

            org_ch_idx = channel_names.index(organelle_channel)

            # Load segmentation mask
            seg_pos = seg_plate[f"{well_name}/{pos_name}"]
            seg_data = seg_pos["0"]
            seg_ch_names = seg_pos.channel_names

            if organelle_seg_channel not in seg_ch_names:
                print(f"  [WARN] '{organelle_seg_channel}' not found in seg store "
                      f"{well_name}/{pos_name}. Skipping.")
                continue

            seg_ch_idx = seg_ch_names.index(organelle_seg_channel)

            # Find tracks CSV
            tracks_csv = find_tracks_csv(tracks_zarr_path, well_name, pos_name)
            if tracks_csv is None:
                print(f"  [WARN] No tracks CSV found for {well_name}/{pos_name}. Skipping.")
                continue

            tracks_df = pd.read_csv(tracks_csv)

            batch_records = []

            for _, row in tracks_df.iterrows():
                x_coord = int(row.get("x", row.get("X", -1)))
                y_coord = int(row.get("y", row.get("Y", -1)))
                t_coord = int(row.get("t", row.get("T", 0)))
                track_id = int(row.get("track_id", row.get("id", -1)))

                # Bounds check
                if not (0 <= x_coord < X and 0 <= y_coord < Y):
                    continue
                if not (0 <= t_coord < T):
                    continue

                # Choose the z-slice with maximum organelle signal as the representative plane
                org_stack = np.array(img_data[t_coord, org_ch_idx, :, :, :])  # Z, Y, X
                best_z = int(np.argmax(org_stack.mean(axis=(1, 2))))

                org_frame = org_stack[best_z]                             # Y, X
                seg_frame = np.array(seg_data[t_coord, seg_ch_idx, best_z, :, :])  # Y, X

                # Extract 2-D patches centred on the tracked position
                org_patch = get_patch_2d(org_frame, y_coord, x_coord, patch_size)
                seg_patch = get_patch_2d(seg_frame, y_coord, x_coord, patch_size)

                if org_patch is None or seg_patch is None:
                    continue

                # GLCM texture features
                glcm_feats = compute_glcm_features(org_patch)   # dict
                edge_dens = compute_edge_density(org_patch)

                # Volume (area) and masked intensity of organelle
                seg_binary = seg_patch > 0
                organelle_volume = float(seg_binary.sum())
                organelle_intensity = (
                    float((org_patch * seg_binary).sum() / organelle_volume)
                    if organelle_volume > 0 else 0.0
                )

                record = {
                    "fov_name": f"{well_name}/{pos_name}",
                    "well": well_name,
                    "position": pos_name,
                    "track_id": track_id,
                    "time_point": t_coord,
                    "z_slice": best_z,
                    "x": x_coord,
                    "y": y_coord,
                    "edge_density": edge_dens,
                    "organelle_volume": organelle_volume,
                    "organelle_intensity": organelle_intensity,
                    **glcm_feats,
                }
                batch_records.append(record)

            append_records_to_csv(batch_records, output_csv)
            print(f"  {well_name}/{pos_name}: {len(batch_records)} cells → {output_csv}")

# %% summary
if output_csv.exists():
    final_df = pd.read_csv(output_csv)
    print(f"\nTotal records: {len(final_df)}")
    print(final_df.head())
else:
    print("No records written — check that input paths are correct.")
