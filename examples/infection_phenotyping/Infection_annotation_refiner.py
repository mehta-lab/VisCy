# %% Run this to display napari on the remote server while running the script in local IDE
import os

os.environ["DISPLAY"] = ":1"
# %% use napari to annotate infected cells in segmented data

import napari
from iohub.ngff import open_ome_zarr
import numpy as np
from pathlib import Path

dataset_folder = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets"
)

input_file = dataset_folder / "Exp_2023_09_28_DENV_A2.zarr"
output_file = (
    dataset_folder / "Exp_2023_09_28_DENV_A2_infMarked_test_annotation_pipeline.zarr"
)

zarr_input = open_ome_zarr(
    input_file,
    layout="hcs",
    mode="r+",
)
chan_names = zarr_input.channel_names
# zarr_input.append_channel('Inf_mask',resize_arrays=True)

zarr_output = open_ome_zarr(
    output_file,
    layout="hcs",
    mode="w",
    channel_names=["Sensor", "Nucl_mask", "Inf_mask"],
)

v = napari.Viewer()


# %% Load label image to napari
for well_id, well_data in zarr_input.wells():
    well_name, well_no = well_id.split("/")

    if well_name == "A" and well_no == "2":

        for pos_name, pos_data in well_data.positions():
            # if int(pos_name) > 1:
            v.layers.clear()
            data = pos_data.data

            FITC = data[0, 0, ...]
            v.add_image(FITC, name="FITC", colormap="green", blending="additive")
            Inf_mask = data[0, 1, ...].astype(int)
            v.add_labels(Inf_mask)
            input("Press Enter")

            label_layer = v.layers["Inf_mask"]
            label_array = label_layer.data
            label_array = np.expand_dims(label_array, axis=(0, 1))
            # zarr_input.create_image('Inf_mask',label_array)
            out_data = np.concatenate((data, label_array), axis=1)
            position = zarr_output.create_position(well_name, well_no, pos_name)
            position["0"] = out_data


# %% Template for magicgui based annotation workflow.
from magicgui import magicgui
from napari.types import ImageData


# Create an enumeration of all wells
wells = list(w[0] for w in zarr_input.wells())
well_id, well_data = next(zarr_input.wells())
positions = list(p[0] for p in well_data.positions())
channel_names = zarr_input.channel_names


@magicgui(
    call_button="load data",
    wells={"choices", ["A/1", "A/2", "A/3", "A/4", "A/5"]},
    positions={"choices", ["0", "1", "2", "3", "4"]},
)  # defines the widget.
def load_well(well: str, position: str):  # defines the callback.
    # Load all data from specified well and position
    for well_id, well_data in zarr_input.wells():
        if well_id == well:
            for pos_name, pos_data in well_data.positions():
                if pos_name == position:
                    for i, ch in enumerate(channel_names):
                        data = pos_data.data
                        v.add_image(
                            data[0, i, ...],
                            name=ch,
                            colormap="gray",
                            blending="additive",
                        )
                break
        break


@magicgui(call_button="save annotations")  # defines the widget.
def save_annotations(
    annotation_layer: ImageData, output_path: Path
):  # defines the callback.
    # Save the output to the specified path
    print("save")


# Add both widgets to napari
v.window.add_dock_widget(load_well(wells, "0"))
v.window.add_dock_widget(save_annotations)
# %%
