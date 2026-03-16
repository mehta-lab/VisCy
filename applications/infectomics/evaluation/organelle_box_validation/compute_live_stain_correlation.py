# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from iohub import open_ome_zarr

from utils.correlation_utils import (
    compute_ncc_3d,
    compute_ssim,
    compute_pcc,
    compute_iou,
    compute_mutual_information,
)
from utils.image_utils import normalize_image

# %% paths
# Each entry is a dict with:
#   zarr_path    : Path to the registered OME-Zarr store
#   platemap_path: Path to the Excel platemap file
# Add new organelle entries here to extend the analysis.
dataset_configs = [
    {
        "zarr_path": Path("LAMP1_registered.zarr"),
        "platemap_path": Path("LAMP1_platemap.xlsx"),
    },
    {
        "zarr_path": Path("TOMM20_registered.zarr"),
        "platemap_path": Path("TOMM20_platemap.xlsx"),
    },
]

output_csv = Path("live_stain_correlation.csv")

# Channel indices inside the zarr array (time=0 assumed)
# pos_data[0, fitc_channel_idx, ...] and pos_data[0, txr_channel_idx, ...]
FITC_CHANNEL_IDX = 2
TXR_CHANNEL_IDX = 3


# %% helper functions
def process_dataset(zarr_path: Path, platemap_path: Path) -> pd.DataFrame:
    """Compute NCC, SSIM, PCC, IOU, MI between FITC and TXR channels
    for every well/FOV in *zarr_path*.

    The platemap Excel is used to look up human-readable ``organelle_tag``
    and ``live_stain`` labels for each well.

    Parameters
    ----------
    zarr_path:
        Path to the OME-Zarr store containing the registered images.
    platemap_path:
        Path to an Excel file with columns: ``Well ID``, ``FITC label``,
        ``TXR label``.

    Returns
    -------
    pd.DataFrame
        One row per FOV with all computed metrics.
    """
    platemap = pd.read_excel(platemap_path)
    # Build a quick-lookup dict keyed by Well ID
    platemap_dict = platemap.set_index("Well ID").to_dict(orient="index")

    records = []

    with open_ome_zarr(zarr_path, mode="r") as plate:
        for well_name, well in plate.wells():
            # Normalise well name to match platemap (e.g. "A/1" → "A1")
            well_id = well_name.replace("/", "")

            organelle_tag = platemap_dict.get(well_id, {}).get("FITC label", "unknown")
            live_stain = platemap_dict.get(well_id, {}).get("TXR label", "unknown")

            for pos_name, pos in well.positions():
                pos_data = pos["0"]  # shape: T, C, Z, Y, X

                fitc = pos_data[0, FITC_CHANNEL_IDX, ...]  # Z, Y, X
                txr = pos_data[0, TXR_CHANNEL_IDX, ...]

                fitc = normalize_image(np.array(fitc))
                txr = normalize_image(np.array(txr))

                ncc = compute_ncc_3d(fitc, txr)
                ssim = compute_ssim(fitc, txr)
                pcc = compute_pcc(fitc, txr)
                iou = compute_iou(fitc, txr)
                mi = compute_mutual_information(fitc, txr)

                records.append(
                    {
                        "zarr_path": str(zarr_path),
                        "well_id": well_id,
                        "fov_id": pos_name,
                        "organelle_tag": organelle_tag,
                        "live_stain": live_stain,
                        "NCC": ncc,
                        "SSIM": ssim,
                        "PCC": pcc,
                        "IOU": iou,
                        "MI": mi,
                    }
                )

    return pd.DataFrame(records)


# %% main – iterate over all dataset configs and pool results
all_results = []

for config in dataset_configs:
    print(f"Processing {config['zarr_path']} ...")
    df = process_dataset(config["zarr_path"], config["platemap_path"])
    all_results.append(df)
    print(f"  → {len(df)} FOVs processed")

results_df = pd.concat(all_results, ignore_index=True)

# %% save
results_df.to_csv(output_csv, index=False)
print(f"Saved {len(results_df)} rows to {output_csv}")
print(results_df.head())
