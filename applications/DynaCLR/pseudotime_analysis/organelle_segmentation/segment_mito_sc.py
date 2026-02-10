# %%
import os
from logging import warning
from pathlib import Path

import numpy as np
import pandas as pd
from extract_features import (
    extract_features_zyx,
)
from iohub import open_ome_zarr
from joblib import Parallel, delayed


# %%
def get_patch(data, cell_centroid, PATCH_SIZE):
    """
    Extract a patch of PATCH_SIZE x PATCH_SIZE centered on the cell centroid.
    If the patch would extend beyond image boundaries, slide it to fit while
    keeping the centroid within the patch.

    Returns None if the image is smaller than PATCH_SIZE in any dimension.
    """
    x_centroid, y_centroid = cell_centroid
    height, width = data.shape

    # Check if image is large enough for patch
    if height < PATCH_SIZE or width < PATCH_SIZE:
        return None

    # Calculate ideal patch boundaries centered on centroid
    x_start = x_centroid - PATCH_SIZE // 2
    x_end = x_centroid + PATCH_SIZE // 2
    y_start = y_centroid - PATCH_SIZE // 2
    y_end = y_centroid + PATCH_SIZE // 2

    # Slide patch if it extends beyond left/top boundaries
    if x_start < 0:
        x_start = 0
        x_end = PATCH_SIZE
    if y_start < 0:
        y_start = 0
        y_end = PATCH_SIZE

    # Slide patch if it extends beyond right/bottom boundaries
    if x_end > width:
        x_end = width
        x_start = width - PATCH_SIZE
    if y_end > height:
        y_end = height
        y_start = height - PATCH_SIZE

    # Extract patch (should always be PATCH_SIZE x PATCH_SIZE now)
    patch = data[int(y_start) : int(y_end), int(x_start) : int(x_end)]
    return patch


# TODO add the intesnity zarr and parsing


def process_position(
    well_id,
    pos_id,
    segmentations_zarr,
    nuclear_labels_path,
    patch_size,
):
    """Process a single position and return the features DataFrame."""
    # Open zarr stores (each worker needs its own file handles)
    input_zarr = open_ome_zarr(segmentations_zarr, mode="r")

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

    for chan_name in channel_names:
        if "_labels" in chan_name:
            labels_cidx = channel_names.index(chan_name)
        if "_vesselness" in chan_name:
            vesselness_cidx = channel_names.index(chan_name)

    labels = in_data[:, labels_cidx]
    vesselness = in_data[:, vesselness_cidx]

    # Initialize an empty list to store values from each row of the csv
    position_features = []
    for idx, row in nuclear_labels_df.iterrows():
        cell_centroid = row["x"], row["y"]
        timepoint = row["t"]

        # Extract patches (will slide to fit within boundaries)
        labels_patch = get_patch(labels[int(timepoint), 0], cell_centroid, patch_size)
        vesselness_patch = get_patch(
            vesselness[int(timepoint), 0], cell_centroid, patch_size
        )

        # Skip if patches couldn't be extracted (image too small)
        if labels_patch is None or vesselness_patch is None:
            continue

        label_patch = labels_patch[np.newaxis].astype(np.uint32)
        vesselness_patch = vesselness_patch[np.newaxis]

        features_df = extract_features_zyx(
            labels_zyx=label_patch,
            intensity_zyx=None,
            frangi_zyx=vesselness_patch,
            spacing=(scale_um[-1], scale_um[-1]),
            extra_properties=[
                "aspect_ratio",
                "circularity",
                "eccentricity",
                "solidity",
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

    input_zarr.close()
    if position_features:
        # Concatenate the list of DataFrames
        position_df = pd.concat(position_features, ignore_index=True)
        return position_df
    else:
        warning(f"No valid features extracted for position {well_id}/{pos_id}.")
        return pd.DataFrame()


# %%
if __name__ == "__main__":
    segmentations_zarr = Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/train_test_mito_seg_2.zarr"
    )
    input_zarr = open_ome_zarr(segmentations_zarr, mode="r")
    in_chans = input_zarr.channel_names

    nuclear_labels_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"

    output_root = (
        Path(
            "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output"
        )
        / segmentations_zarr.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)

    output_csv = output_root / f"{segmentations_zarr.stem}_mito_features_nellie.csv"

    PATCH_SIZE = 160

    # Number of parallel jobs - adjust based on your system
    # -1 means use all available cores, or set to a specific number
    n_jobs = -1

    # Collect all (well_id, pos_id) pairs to process
    with open_ome_zarr(segmentations_zarr, mode="r") as input_zarr:
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
            segmentations_zarr=segmentations_zarr,
            nuclear_labels_path=nuclear_labels_path,
            patch_size=PATCH_SIZE,
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

# %%
