# %%
import os

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from tqdm import tqdm

# %% create training and validation dataset
# TODO: Modify path to the input data
input_track_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr"
output_track_path = "/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_ph_2D/classical/data/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV_classical_fake_tracks.zarr"
# TODO: Modify the channel name to the one you are using for the segmentation mask
segmentation_channel_name = "nuclei_prediction_labels_labels"
# TODO: Modify the z-slice to the one you are using for the segmentation mask
Z_SLICE = 0
# %%
"""
Add csvs with fake tracking to tracking data.

The tracking data is a csv with the following columns:
- track_id: from segmentation mask, list of labels
- t: all 0 since there is just one timepoint
- x, y: the coordinates of the centroid of the segmentation mask
- id: must be all unqiue 6 digit numbers starting from 100000
- parent_track_id: all -1
- parent_id: all -1
"""


def create_track_df(seg_mask, time):
    track_id = np.unique(seg_mask)
    track_id = track_id[track_id != 0]
    track_rows = []
    # Get coordinates for each track_id separately
    for tid in track_id:
        y, x = np.where(seg_mask == tid)  # Note: y comes first from np.where
        # Use mean coordinates as centroid
        mean_y = np.mean(y)
        mean_x = np.mean(x)
        track_rows.append(
            {
                "track_id": tid,
                "t": time,
                "y": mean_y,  # Using mean y coordinate
                "x": mean_x,  # Using mean x coordinate
                "id": 100000 + tid,
                "parent_track_id": -1,
                "parent_id": -1,
            }
        )
    track_df = pd.DataFrame(track_rows)
    return track_df


def save_track_df(track_df, well_id, pos_name, out_path):
    folder, subfolder = well_id.split("/")
    out_name = f"{folder}_{subfolder}_{pos_name}_tracks.csv"
    out_path = os.path.join(out_path, folder, subfolder, pos_name, out_name)
    track_df.to_csv(out_path, index=False)


# %%
def main():
    # Load the input segmentation data
    zarr_input = open_ome_zarr(
        input_track_path,
        mode="r",
    )
    chan_names = zarr_input.channel_names
    assert segmentation_channel_name in chan_names, (
        "Channel name not found in the input data"
    )

    # Create the empty store for the tracking data
    position_names = []
    for ds, position in zarr_input.positions():
        position_names.append(tuple(ds.split("/")))

    create_empty_plate(
        store_path=output_track_path,
        position_keys=position_names,
        channel_names=[segmentation_channel_name],
        shape=(1, 1, 1, *position.data.shape[3:]),
        chunks=position.data.chunks,
        scale=position.scale,
    )
    #
    # Populate the tracking data
    with open_ome_zarr(output_track_path, layout="hcs", mode="r+") as track_store:
        # Create progress bar for wells and positions
        for well_id, well_data in tqdm(zarr_input.wells(), desc="Processing wells"):
            for pos_name, pos_data in well_data.positions():
                data = pos_data.data
                T, C, Z, Y, X = data.shape
                track_df_all = pd.DataFrame()
                for time in range(T):
                    seg_mask = data[
                        time, chan_names.index(segmentation_channel_name), Z_SLICE, :, :
                    ]
                    track_pos = track_store[well_id + "/" + pos_name]
                    track_pos["0"][0, 0, 0] = seg_mask
                    track_df = create_track_df(seg_mask, time)
                    track_df_all = pd.concat([track_df_all, track_df])
                save_track_df(track_df_all, well_id, pos_name, output_track_path)
    zarr_input.close()


# %%
if __name__ == "__main__":
    main()
