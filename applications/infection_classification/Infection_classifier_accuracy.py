# %% script to compare the output from the supervised model and human revised annotations to get the accuracy of the model

import numpy as np
from iohub import open_ome_zarr
from scipy.ndimage import label

# %% datapaths

# Path to model output
data_out_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/supervised_test.zarr"

# Path to the human revised annotations
human_corrected_path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred/supervised_test_corrected.zarr"

# %% Load data and compute the number of objects in each class

data_out = open_ome_zarr(data_out_path, layout="hcs", mode="r+")
human_corrected = open_ome_zarr(human_corrected_path, layout="hcs", mode="r+")

out_medians = []
HC_medians = []
for well_id, well_data in data_out.wells():
    well_name, well_no = well_id.split("/")

    for pos_name, pos_data in well_data.positions():

        out_data = pos_data.data.numpy()
        T, C, Z, Y, X = out_data.shape

        HC_data = human_corrected[well_id + "/" + pos_name + "/0"]
        HC_data = HC_data.numpy()

        # Compute the number of objects in the model output
        for t in range(T):
            out_img = out_data[t, 0, 0]

            # Compute the number of objects in the model output
            out_labeled, num_out_objects = label(out_img > 0)

            # Compute the median of pixel values in each object in the model output
            for obj_id in range(1, num_out_objects + 1):
                obj_pixels = out_img[out_labeled == obj_id]
                out_medians.append(np.median(obj_pixels))

            # repeat for human acorrected annotations
            HC_img = HC_data[t, 0, 0]
            HC_labeled, num_HC_objects = label(HC_img > 0)

            for obj_id in range(1, num_HC_objects + 1):
                obj_pixels = HC_img[HC_labeled == obj_id]
                HC_medians.append(np.median(obj_pixels))

# %% Compute the accuracy

num_twos_in_out_medians = out_medians.count(2)
num_twos_in_HC_medians = HC_medians.count(2)
error_inf = (
    (num_twos_in_HC_medians - num_twos_in_out_medians) / num_twos_in_HC_medians
) * 100

num_ones_in_out_medians = out_medians.count(1)
num_ones_in_HC_medians = HC_medians.count(1)
error_uninf = (
    (num_ones_in_HC_medians - num_ones_in_out_medians) / num_ones_in_HC_medians
) * 100

avg_error = (np.abs(error_inf) + np.abs(error_uninf)) / 2

accuracy = 100 - avg_error

# %%
