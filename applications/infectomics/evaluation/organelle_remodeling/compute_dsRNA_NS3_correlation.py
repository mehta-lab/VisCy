# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import pandas as pd
from iohub import open_ome_zarr

from utils.correlation_utils import (
    compute_ssim,
    compute_iou,
    compute_pcc,
    compute_patch_status,
)
from utils.image_utils import get_patch_tcyx

# %% paths / replicate configuration
# Each dict describes one replicate experiment.
# channel_start: first channel index of the (NS3, dsRNA, organelle) triplet.
#   Replicate 1 (Oct-15): channels 2,3,4  → channel_start=2
#   Replicate 2 (Oct-22): channels 1,2,3  → channel_start=1
# Add more replicate dicts here to include additional datasets.
replicate_datasets = [
    {
        "replicate": 1,
        "zarr_path": Path("replicate1_registered.zarr"),
        "tracks_path": Path("replicate1_tracks.zarr"),
        "channel_start": 2,
        "organelle": "TOMM20",   # human-readable label
        "condition": "DENV",
    },
    {
        "replicate": 2,
        "zarr_path": Path("replicate2_registered.zarr"),
        "tracks_path": Path("replicate2_tracks.zarr"),
        "channel_start": 1,
        "organelle": "TOMM20",
        "condition": "DENV",
    },
]

patch_size = 80          # pixels; square XY patch centred on the tracked cell
output_csv = Path("dsRNA_NS3_correlation.csv")

# IOU binarisation threshold (fixed, not Otsu)
IOU_THRESHOLD = 300
USE_OTSU = False

# Pixel intensity thresholds used for infection status categorisation
# (derived from the distribution of uninfected cells in the dataset)
dsRNA_THRESHOLD = 870
NS3_THRESHOLD = 670


# %% helper – infection status from dsRNA / NS3 signal intensity
def categorize_infection_status(
    dsRNA_patch: np.ndarray,
    ns3_patch: np.ndarray,
    dsRNA_threshold: float = dsRNA_THRESHOLD,
    ns3_threshold: float = NS3_THRESHOLD,
) -> str:
    """Categorise a cell as uninfected, dsRNA+, NS3+, or dsRNA+/NS3+.

    Parameters
    ----------
    dsRNA_patch:
        2-D (or 3-D) image patch for the dsRNA channel.
    ns3_patch:
        2-D (or 3-D) image patch for the NS3 channel.
    dsRNA_threshold, ns3_threshold:
        Pixel intensity thresholds derived from uninfected cell statistics.

    Returns
    -------
    str
        One of: "uninfected", "dsRNA+", "NS3+", "dsRNA+/NS3+".
    """
    dsRNA_pos = dsRNA_patch.max() > dsRNA_threshold
    ns3_pos = ns3_patch.max() > ns3_threshold

    if dsRNA_pos and ns3_pos:
        return "dsRNA+/NS3+"
    elif dsRNA_pos:
        return "dsRNA+"
    elif ns3_pos:
        return "NS3+"
    else:
        return "uninfected"


# %% helper – process a single replicate
def process_replicate(config: dict) -> pd.DataFrame:
    """Compute per-cell SSIM, IOU, PCC metrics for one replicate.

    Parameters
    ----------
    config:
        Dict from ``replicate_datasets`` containing ``replicate``,
        ``zarr_path``, ``tracks_path``, ``channel_start``,
        ``organelle``, and ``condition``.

    Returns
    -------
    pd.DataFrame
        One row per tracked cell / timepoint.
    """
    rep_id = config["replicate"]
    zarr_path = config["zarr_path"]
    tracks_path = config["tracks_path"]
    ch_start = config["channel_start"]
    organelle = config["organelle"]
    condition = config["condition"]

    # Channel layout within the extracted slice:
    #   slice[:, 0, ...] = NS3
    #   slice[:, 1, ...] = dsRNA
    #   slice[:, 2, ...] = organelle tag
    NS3_CH = 0
    DSRNA_CH = 1
    ORG_CH = 2
    ch_slice = slice(ch_start, ch_start + 3)

    records = []

    with open_ome_zarr(zarr_path, mode="r") as plate, \
         open_ome_zarr(tracks_path, mode="r") as tracks_plate:

        for well_name, well in plate.wells():
            # Parse well components, e.g. "A/1" → well_name="A", well_no="1"
            well_parts = well_name.split("/")
            well_letter = well_parts[0]
            well_no = well_parts[1] if len(well_parts) > 1 else "0"

            for pos_name, pos in well.positions():
                in_data = pos["0"]          # T, C, Z, Y, X
                T, C, Z, Y, X = in_data.shape

                # Build expected tracks CSV filename
                # Convention: tracks_{well_letter}_{well_no}_{pos_id}.csv
                pos_id = pos_name.replace("/", "_")
                tracks_csv_name = f"tracks_{well_letter}_{well_no}_{pos_id}.csv"

                # Tracks may live as an attachment inside the tracks zarr or on disk
                # Try the zarr obs first, then fall back to a CSV sibling
                tracks_csv_path = tracks_path.parent / tracks_csv_name
                if not tracks_csv_path.exists():
                    # Try looking inside the tracks zarr directory
                    tracks_csv_path = Path(str(tracks_path)) / tracks_csv_name

                if not tracks_csv_path.exists():
                    print(f"  [WARN] Tracks CSV not found: {tracks_csv_name}. Skipping.")
                    continue

                tracks_df = pd.read_csv(tracks_csv_path)

                # Load the full (T, 3, Z, Y, X) data stack for these channels
                all_data_stack = np.array(in_data[:, ch_slice, ...])  # T, 3, Z, Y, X

                for _, row in tracks_df.iterrows():
                    x_coord = int(row["x"]) if "x" in row else int(row.get("X", -1))
                    y_coord = int(row["y"]) if "y" in row else int(row.get("Y", -1))
                    t_coord = int(row["t"]) if "t" in row else int(row.get("T", 0))
                    track_id = int(row["track_id"]) if "track_id" in row else int(row.get("id", -1))

                    # Bounds check
                    if not (0 <= x_coord < X and 0 <= y_coord < Y):
                        continue

                    # Extract 2-D patch (using max-Z projection across the z-stack)
                    try:
                        patch_tcyx = get_patch_tcyx(
                            all_data_stack,
                            t_coord,
                            y_coord,
                            x_coord,
                            patch_size,
                        )
                    except Exception:
                        continue

                    if patch_tcyx is None:
                        continue

                    # Max-project along Z if the patch has a Z axis; otherwise use as-is
                    # get_patch_tcyx returns shape (C, Y, X) for a 2-D patch
                    ns3_patch = patch_tcyx[NS3_CH]
                    dsrna_patch = patch_tcyx[DSRNA_CH]
                    org_patch = patch_tcyx[ORG_CH]

                    # Compute correlations: organelle vs dsRNA and organelle vs NS3
                    ssim_org_dsrna = compute_ssim(org_patch, dsrna_patch)
                    ssim_org_ns3 = compute_ssim(org_patch, ns3_patch)

                    iou_org_dsrna = compute_iou(
                        org_patch, dsrna_patch,
                        threshold=IOU_THRESHOLD,
                        use_otsu=USE_OTSU,
                    )
                    iou_org_ns3 = compute_iou(
                        org_patch, ns3_patch,
                        threshold=IOU_THRESHOLD,
                        use_otsu=USE_OTSU,
                    )

                    pcc_org_dsrna = compute_pcc(org_patch, dsrna_patch)
                    pcc_org_ns3 = compute_pcc(org_patch, ns3_patch)

                    patch_status = compute_patch_status(patch_tcyx)

                    infection_status = categorize_infection_status(
                        dsrna_patch, ns3_patch, dsRNA_THRESHOLD, NS3_THRESHOLD
                    )

                    records.append(
                        {
                            "replicate": rep_id,
                            "organelle": organelle,
                            "condition": condition,
                            "well": well_name,
                            "fov": pos_name,
                            "track_id": track_id,
                            "t": t_coord,
                            "x": x_coord,
                            "y": y_coord,
                            "infection_status": infection_status,
                            "patch_status": patch_status,
                            "SSIM_org_dsRNA": ssim_org_dsrna,
                            "SSIM_org_NS3": ssim_org_ns3,
                            "IOU_org_dsRNA": iou_org_dsrna,
                            "IOU_org_NS3": iou_org_ns3,
                            "PCC_org_dsRNA": pcc_org_dsrna,
                            "PCC_org_NS3": pcc_org_ns3,
                        }
                    )

    return pd.DataFrame(records)


# %% main – iterate over replicates and pool results
all_results = []

for config in replicate_datasets:
    print(f"Processing replicate {config['replicate']} ({config['zarr_path']}) ...")
    rep_df = process_replicate(config)
    all_results.append(rep_df)
    print(f"  → {len(rep_df)} cells processed")

pooled_df = pd.concat(all_results, ignore_index=True)
print(f"\nTotal pooled cells: {len(pooled_df)}")

# %% save
pooled_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
print(pooled_df.head())
