
# %% script to save 128 by 128 image patches from napari viewer

import napari
import numpy as np
from pathlib import Path
import sys

sys.path.append("/hpc/mydata/soorya.pradeep/scratch/viscy_infection_phenotyping/VisCy")
# from viscy.data.triplet import TripletDataModule
from viscy.representation.evaluation import dataset_of_tracks


# %% input parameters

data_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr"
)

fov_name = '/B/4/8'
track_id = 12
source_channel = ["Phase3D", "RFP"]

# %% load dataset

prediction_dataset = dataset_of_tracks(
    data_path,
    tracks_path,
    [fov_name],
    [track_id],
    source_channel=source_channel,
)
whole = np.stack([p["anchor"] for p in prediction_dataset])
phase = whole[:, 0]
fluor = whole[:, 1]

# use the following if you want to visualize a specific phase slice with max projected fluor
# phase = whole[:, 0, 3]   # 3 is the slice number
# fluor = np.max(whole[:, 1], axis=1)

# load image
v = napari.Viewer()
v.add_image(phase)
v.add_image(fluor)

# %% save patches as png images

# use sliders on napari to get the deisred contrast and make other adjustments
# then use save screenshot if saving the image patch manually
# you can add code to automate the process if desired