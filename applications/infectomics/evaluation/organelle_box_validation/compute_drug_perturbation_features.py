# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import pandas as pd
from iohub import open_ome_zarr

from utils.feature_utils import (
    compute_glcm_features,
    compute_edge_density,
    find_centroid,
    find_mem_label,
    compute_organelle_volume_fraction,
    compute_organelle_count_and_size,
)
from utils.image_utils import get_patch_zyx

# %%  ── experiment configuration ──────────────────────────────────────────────
# Set organelle_type to "TOMM20" or "LAMP1" to select which morphology
# function is called.  All other loop logic is shared between both experiments.
organelle_type = "TOMM20"  # <-- change to "LAMP1" for the LAMP1/Bafilomycin run

# %% paths
input_zarr_path = Path("registered.zarr")         # registered multi-channel images
seg_zarr_path = Path("segmentation.zarr")         # masks: Nucl_mask, Organelle_mask, Membrane_mask
well_map_path = Path("well_map.csv")              # CSV with well → condition mapping
output_csv = Path("feature_list.csv")

# Name of the organelle channel inside input_zarr (as stored in the OME-Zarr metadata)
organelle_channel_name = "TOMM20"  # e.g. "TOMM20" or "LAMP1"

# Patch half-size in ZYX; the full patch will be (2*patch_radius+1)^3 centred on the nucleus
patch_size = 32  # used for get_patch_zyx

# Segmentation channel names (indices in seg zarr)
NUCL_MASK_CHANNEL = "Nucl_mask"
ORGANELLE_MASK_CHANNEL = "Organelle_mask"
MEMBRANE_MASK_CHANNEL = "Membrane_mask"

# For TOMM20: iterate only t=0.  For LAMP1: iterate all timepoints.
if organelle_type == "TOMM20":
    timepoints_to_process = [0]
else:
    timepoints_to_process = None  # will be set per-position based on T dim


# %% helper – write a list of feature dicts to CSV (append mode)
def append_records_to_csv(records: list, csv_path: Path) -> None:
    """Append *records* (list of dicts) to *csv_path*.
    Creates the file with a header on the first call."""
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# %% helper – extract features for a single cell label
def extract_cell_features(
    organelle_data_zyx: np.ndarray,
    nucl_mask_zyx: np.ndarray,
    organelle_mask_zyx: np.ndarray,
    mem_mask_zyx: np.ndarray,
    cell_label: int,
    patch_size: int,
    organelle_type: str,
) -> dict:
    """Return a dict of per-cell organelle features.

    Parameters
    ----------
    organelle_data_zyx:
        Raw organelle channel image (Z, Y, X).
    nucl_mask_zyx:
        Labelled nuclear segmentation mask.
    organelle_mask_zyx:
        Binary (or labelled) organelle segmentation mask.
    mem_mask_zyx:
        Labelled membrane/cell segmentation mask.
    cell_label:
        Integer label of the nucleus to process.
    patch_size:
        Half-size for the ZYX patch extraction.
    organelle_type:
        "TOMM20" → compute_organelle_volume_fraction
        "LAMP1"  → compute_organelle_count_and_size

    Returns
    -------
    dict or None
        Feature dict, or None if the centroid / patch is invalid.
    """
    centroid = find_centroid(nucl_mask_zyx, cell_label)
    if centroid is None:
        return None

    # Extract 3-D patches centred on the nucleus centroid
    org_patch = get_patch_zyx(organelle_data_zyx, centroid, patch_size)
    nucl_patch = get_patch_zyx(nucl_mask_zyx, centroid, patch_size)
    org_mask_patch = get_patch_zyx(organelle_mask_zyx, centroid, patch_size)
    mem_patch = get_patch_zyx(mem_mask_zyx, centroid, patch_size)

    if org_patch is None or org_mask_patch is None:
        return None

    # GLCM texture features (computed on max-Z projection of organelle patch)
    org_patch_2d = org_patch.max(axis=0)
    glcm_feats = compute_glcm_features(org_patch_2d)  # dict: contrast, homogeneity, …
    edge_dens = compute_edge_density(org_patch_2d)

    # Find the matching membrane label for this nucleus
    mem_label = find_mem_label(nucl_patch, mem_patch, cell_label)

    # Organelle-type-specific morphology metrics
    morph_feats = {}
    if organelle_type == "TOMM20":
        vol_frac, org_intensity = compute_organelle_volume_fraction(
            organelle_data_zyx, organelle_mask_zyx, mem_mask_zyx, mem_label
        )
        morph_feats["organelle_volume_fraction"] = vol_frac
        morph_feats["organelle_intensity"] = org_intensity
    else:  # LAMP1
        org_count, mean_org_size = compute_organelle_count_and_size(
            organelle_mask_zyx, mem_mask_zyx, mem_label
        )
        morph_feats["organelle_count"] = org_count
        morph_feats["mean_organelle_size"] = mean_org_size

    return {**glcm_feats, "edge_density": edge_dens, **morph_feats}


# %% main loop
well_map = pd.read_csv(well_map_path)

# Remove stale output so we start fresh
if output_csv.exists():
    os.remove(output_csv)

with open_ome_zarr(input_zarr_path, mode="r") as plate, \
     open_ome_zarr(seg_zarr_path, mode="r") as seg_plate:

    for well_name, well in plate.wells():
        well_id = well_name.replace("/", "")

        # Look up condition from well map (if available)
        condition_rows = well_map[well_map["well_id"] == well_id]
        condition = condition_rows["condition"].values[0] if len(condition_rows) > 0 else "unknown"

        for pos_name, pos in well.positions():
            img_data = pos["0"]           # T, C, Z, Y, X
            T = img_data.shape[0]

            # Resolve which channel index corresponds to the organelle channel
            channel_names = pos.channel_names
            if organelle_channel_name in channel_names:
                org_ch_idx = channel_names.index(organelle_channel_name)
            else:
                print(f"  [WARN] '{organelle_channel_name}' not found in {well_name}/{pos_name}. Skipping.")
                continue

            seg_pos = seg_plate[f"{well_name}/{pos_name}"]
            seg_data = seg_pos["0"]       # T, C, Z, Y, X (segmentation channels)
            seg_channel_names = seg_pos.channel_names

            nucl_ch = seg_channel_names.index(NUCL_MASK_CHANNEL)
            org_mask_ch = seg_channel_names.index(ORGANELLE_MASK_CHANNEL)
            mem_ch = seg_channel_names.index(MEMBRANE_MASK_CHANNEL)

            t_range = timepoints_to_process if timepoints_to_process is not None else range(T)

            batch_records = []

            for t in t_range:
                org_vol = np.array(img_data[t, org_ch_idx, ...])   # Z, Y, X
                nucl_mask = np.array(seg_data[t, nucl_ch, ...])
                org_mask = np.array(seg_data[t, org_mask_ch, ...])
                mem_mask = np.array(seg_data[t, mem_ch, ...])

                # Find max-signal nuclear slice for centroid search
                nucl_z_profile = nucl_mask.sum(axis=(1, 2))
                max_z = int(np.argmax(nucl_z_profile)) if nucl_z_profile.max() > 0 else 0

                cell_labels = np.unique(nucl_mask)
                cell_labels = cell_labels[cell_labels > 0]  # exclude background

                for cell_label in cell_labels:
                    feats = extract_cell_features(
                        org_vol,
                        nucl_mask,
                        org_mask,
                        mem_mask,
                        int(cell_label),
                        patch_size,
                        organelle_type,
                    )
                    if feats is None:
                        continue

                    record = {
                        "well_id": well_id,
                        "fov_id": pos_name,
                        "condition": condition,
                        "label": int(cell_label),
                        "time_point": t,
                        "max_nuclear_z": max_z,
                        **feats,
                    }
                    batch_records.append(record)

            # Append this position's records to the output CSV incrementally
            append_records_to_csv(batch_records, output_csv)
            print(f"  {well_id}/{pos_name}: {len(batch_records)} cells appended to {output_csv}")

# %% summary
final_df = pd.read_csv(output_csv)
print(f"\nTotal records: {len(final_df)}")
print(final_df.head())
