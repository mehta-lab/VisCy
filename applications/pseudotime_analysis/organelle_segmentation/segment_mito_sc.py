# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from extract_features import (
    extract_features_zyx,
)
from iohub import open_ome_zarr
from matplotlib import pyplot as plt
from segment_organelles import (
    calculate_nellie_sigmas,
    segment_zyx,
)
from skimage.exposure import rescale_intensity
from tqdm import tqdm

# %%

input_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/train-test/2024_11_21_A549_TOMM20_DENV.zarr"
input_zarr = open_ome_zarr(input_path, mode="r")
in_chans = input_zarr.channel_names
organelle_channel = "GFP EX488 EM525-45"

org_seg_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/1-predictions/quantify_remodeling/segmentation_TOMM20_filaments.zarr"
seg_zarr = open_ome_zarr(org_seg_path, mode="r")
seg_chans = seg_zarr.channel_names
organelle_labels = "Organelle_mask"

nuclear_labels_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/3-track/2024_11_21_A549_TOMM20_DENV_cropped.zarr"
nuclear_labels_zarr = open_ome_zarr(nuclear_labels_path, mode="r")
nuclear_labels_chans = nuclear_labels_zarr.channel_names
nuclear_labels = "nuclei_prediction_labels_labels"

output_root = Path(
    "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output"
)
output_csv = output_root / "2024_11_21_A549_TOMM20_DENV/organelle_seg_features.csv"


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


# %%
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


for well_id, well_data in tqdm(seg_zarr.wells(), desc="Wells", position=0):
    # well_id, well_data = next(seg_zarr.wells())
    well_name, well_no = well_id.split("/")
    # if well_id in wells_to_process:
    for pos_id, pos_data in tqdm(
        well_data.positions(), desc=f"Positions in {well_id}", position=1, leave=False
    ):
        # pos_id, pos_data = next(well_data.positions())
        T, C, Z, Y, X = pos_data.data.shape
        seg_data = pos_data.data.numpy()
        org_seg_mask = seg_data[:, seg_chans.index(organelle_labels)]

        # read the csv stored in each nucl seg zarr folder
        file_name = "tracks_" + well_name + "_" + well_no + "_" + pos_id + ".csv"
        nuclear_labels_csv = os.path.join(
            nuclear_labels_path, well_id + "/" + pos_id + "/" + file_name
        )
        nuclear_labels_df = pd.read_csv(nuclear_labels_csv)

        in_data = input_zarr[well_id + "/" + pos_id].data.numpy()
        scale_um = input_zarr[well_id + "/" + pos_id].scale
        organelle_data = in_data[:, in_chans.index(organelle_channel)]

        # Initialize an empty list to store values from each row of the csv
        position_features = []
        for idx, row in nuclear_labels_df.iterrows():

            if (
                row["x"] > PATCH_SIZE // 2
                and row["y"] > PATCH_SIZE // 2
                and row["x"] < X - PATCH_SIZE // 2
                and row["y"] < Y - PATCH_SIZE // 2
            ):
                cell_centroid = row["x"], row["y"]

                timepoint = row["t"]
                organelle_patch = get_patch(
                    organelle_data[int(timepoint), 0], cell_centroid, PATCH_SIZE
                )

                #
                organelle_patch = organelle_patch[np.newaxis]
                organelle_patch = rescale_intensity(
                    organelle_patch,
                    out_range=(0, 1),
                )
                # FIXME set some defualt for the mitochondria and other organelles
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
                    features_df["fov_name"] = "/" + well_id + "/" + pos_id
                    features_df["track_id"] = row["track_id"]
                    features_df["t"] = timepoint
                    features_df["x"] = row["x"]
                    features_df["y"] = row["y"]

                position_features.append(features_df)

        if position_features:
            # Concatenate the list of DataFrames
            position_df = pd.concat(position_features, ignore_index=True)
            position_df.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
            )


# %%
