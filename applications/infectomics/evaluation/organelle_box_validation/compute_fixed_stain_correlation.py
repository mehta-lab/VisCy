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
input_zarr_path = Path("fixed_stain_registered.zarr")
platemap_path = Path("fixed_stain_platemap.xlsx")
output_csv = Path("fixed_stain_correlation.csv")

# Channel indices (fixed/IFA experiment)
# pos_data[0, 1, ...] = FITC (organelle tag)
# pos_data[0, 2, ...] = TXR  (IFA / fixed stain)
FITC_CHANNEL_IDX = 1
TXR_CHANNEL_IDX = 2


# %% compute metrics per well/FOV
def compute_fixed_stain_correlation(
    zarr_path: Path,
    platemap_path: Path,
) -> pd.DataFrame:
    """Compute SSIM, PCC, IOU, MI (and NCC) between the organelle tag
    (FITC channel) and the IFA fixed stain (TXR channel) for every
    well/FOV in the supplied OME-Zarr store.

    Parameters
    ----------
    zarr_path:
        Path to the OME-Zarr store containing registered fixed-cell images.
    platemap_path:
        Excel platemap with columns: ``Well ID``, ``FITC label``.

    Returns
    -------
    pd.DataFrame
        One row per FOV with all computed metrics.
    """
    platemap = pd.read_excel(platemap_path)
    platemap_dict = platemap.set_index("Well ID").to_dict(orient="index")

    records = []

    with open_ome_zarr(zarr_path, mode="r") as plate:
        for well_name, well in plate.wells():
            well_id = well_name.replace("/", "")
            organelle_tag = platemap_dict.get(well_id, {}).get("FITC label", "unknown")

            for pos_name, pos in well.positions():
                pos_data = pos["0"]  # T, C, Z, Y, X

                fitc = np.array(pos_data[0, FITC_CHANNEL_IDX, ...])
                txr = np.array(pos_data[0, TXR_CHANNEL_IDX, ...])

                fitc = normalize_image(fitc)
                txr = normalize_image(txr)

                ncc = compute_ncc_3d(fitc, txr)
                ssim = compute_ssim(fitc, txr)
                pcc = compute_pcc(fitc, txr)
                iou = compute_iou(fitc, txr)
                mi = compute_mutual_information(fitc, txr)

                records.append(
                    {
                        "well_id": well_id,
                        "fov_id": pos_name,
                        "organelle_tag": organelle_tag,
                        "NCC": ncc,
                        "SSIM": ssim,
                        "PCC": pcc,
                        "IOU": iou,
                        "MI": mi,
                    }
                )

    return pd.DataFrame(records)


# %% run
results_df = compute_fixed_stain_correlation(input_zarr_path, platemap_path)

# %% save
results_df.to_csv(output_csv, index=False)
print(f"Saved {len(results_df)} rows to {output_csv}")
print(results_df.head())
