# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from extract_features import (
    extract_features_zyx,
)
from iohub import open_ome_zarr
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from segment_organelles import (
    calculate_nellie_sigmas,
    segment_zyx,
)
from skimage.exposure import rescale_intensity
from tqdm import tqdm


# %% library of feature extraction methods
def get_patch(data, cell_centroid, PATCH_SIZE):

    x_centroid, y_centroid = cell_centroid
    # ensure patch boundaries stay within data dimensions
    x_start = max(0, x_centroid - PATCH_SIZE // 2)
    x_end = min(data.shape[1], x_centroid + PATCH_SIZE // 2)
    y_start = max(0, y_centroid - PATCH_SIZE // 2)
    y_end = min(data.shape[0], y_centroid + PATCH_SIZE // 2)

    # get patch of PATCH_SIZE centered on centroid
    patch = data[int(y_start) : int(y_end), int(x_start) : int(x_end)]
    return patch


def process_position(
    well_id,
    pos_id,
    input_path,
    nuclear_labels_path,
    organelle_channel,
    patch_size,
    frangi_params,
):
    """Process a single position and return the features DataFrame."""
    # Open zarr stores (each worker needs its own file handles)
    input_zarr = open_ome_zarr(input_path, mode="r")

    well_name, well_no = well_id.split("/")

    # Load position data
    position = input_zarr[well_id + "/" + pos_id]
    T, C, Z, Y, X = position.data.shape
    channel_names = position.channel_names
    scale_um = position.scale

    in_data = position.data.numpy()

    # Read the csv stored in each nucl seg zarr folder
    file_name = "tracks_" + well_name + "_" + well_no + "_" + pos_id + ".csv"
    nuclear_labels_csv = os.path.join(
        nuclear_labels_path, well_id + "/" + pos_id + "/" + file_name
    )
    nuclear_labels_df = pd.read_csv(nuclear_labels_csv)
    organelle_data = in_data[:, channel_names.index(organelle_channel)]

    # Initialize an empty list to store values from each row of the csv
    position_features = []
    for idx, row in nuclear_labels_df.iterrows():

        if (
            row["x"] > patch_size // 2
            and row["y"] > patch_size // 2
            and row["x"] < X - patch_size // 2
            and row["y"] < Y - patch_size // 2
        ):
            cell_centroid = row["x"], row["y"]

            timepoint = row["t"]
            organelle_patch = get_patch(
                organelle_data[int(timepoint), 0], cell_centroid, patch_size
            )

            organelle_patch = organelle_patch[np.newaxis]
            organelle_patch = rescale_intensity(
                organelle_patch,
                out_range=(0, 1),
            )
            min_radius_um = 0.15  # 300 nm diameter = ~2 pixels
            max_radius_um = 0.6  # 1 µm diameter = ~6.7 pixels
            sigma_range = calculate_nellie_sigmas(
                min_radius_um,
                max_radius_um,
                scale_um[-1],  # use the highest res axis (XY)
                num_sigma=frangi_params["sigma_steps"],
            )

            labels, vesselness, optimal_sigma = segment_zyx(
                organelle_patch, sigma_range=sigma_range, **frangi_params
            )
            features_df = extract_features_zyx(
                labels_zyx=labels,
                intensity_zyx=organelle_patch,
                frangi_zyx=vesselness,
                spacing=(scale_um[-1], scale_um[-1]),
                extra_properties=[
                    "aspect_ratio",
                    "circularity",
                    "frangi_intensity",
                    "texture",
                    "moments_hu",
                ],
            )

            if not features_df.empty:
                features_df["fov_name"] = well_id + "/" + pos_id
                features_df["track_id"] = row["track_id"]
                features_df["t"] = timepoint
                features_df["x"] = row["x"]
                features_df["y"] = row["y"]

            position_features.append(features_df)

    if position_features:
        # Concatenate the list of DataFrames
        position_df = pd.concat(position_features, ignore_index=True)
        return position_df
    else:
        return pd.DataFrame()

    input_zarr.close()
    # nuclear_labels_zarr.close()


# %%
if __name__ == "__main__":
    input_path = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
    )
    input_zarr = open_ome_zarr(input_path, mode="r")
    in_chans = input_zarr.channel_names
    organelle_channel = "GFP EX488 EM525-45"

    nuclear_labels_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"
    # nuclear_labels_zarr = open_ome_zarr(nuclear_labels_path, mode="r")
    # nuclear_labels_chans = nuclear_labels_zarr.channel_names
    # nuclear_labels = "nuclei_prediction_labels_labels"

    output_root = (
        Path(
            "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output"
        )
        / input_path.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)

    output_csv = output_root / f"{input_path.stem}_mito_features.csv"

    PATCH_SIZE = 160
    frangi_params = {
        "clahe_clip_limit": 0.01,  # Mild contrast enhancement (0.01-0.03 range)
        "sigma_steps": 2,  # Multiple scales to capture size variation
        "auto_optimize_sigma": False,  # Use multi-scale (max across scales)
        "frangi_alpha": 0.5,  # Standard tubular structure sensitivity
        "frangi_beta": 0.5,  # Standard blob rejection
        "threshold_method": "nellie_max",  # CRITICAL: Manual threshold where auto methods fail
        "min_object_size": 10,  # Remove small noise clusters (20-50 pixels)
        "apply_morphology": False,  # Connect fragmented mitochondria
    }

    # Number of parallel jobs - adjust based on your system
    # -1 means use all available cores, or set to a specific number
    n_jobs = -1

    # Collect all (well_id, pos_id) pairs to process
    with open_ome_zarr(input_path, mode="r") as input_zarr:
        positions_to_process = []
        for well_id, well_data in input_zarr.wells():
            for pos_id, pos_data in well_data.positions():
                positions_to_process.append((well_id, pos_id))

    print(
        f"Processing {len(positions_to_process)} positions using {n_jobs} parallel jobs..."
    )

    # Process positions in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_position)(
            well_id=well_id,
            pos_id=pos_id,
            input_path=input_path,
            nuclear_labels_path=nuclear_labels_path,
            organelle_channel=organelle_channel,
            patch_size=PATCH_SIZE,
            frangi_params=frangi_params,
        )
        for well_id, pos_id in positions_to_process
    )

    # Combine all results and save
    all_features = [df for df in results if not df.empty]
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        # Save all at once instead of appending
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Saved {len(final_df)} features to {output_csv}")
