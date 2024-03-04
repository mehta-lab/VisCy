

#%% use napari to annotate infected cells in segmented data

import napari
from iohub.ngff import open_ome_zarr
import numpy as np

file_in_path = '/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/Exp_2023_09_28_DENV_A2.zarr'
zarr_input = open_ome_zarr(
    file_in_path,
    layout="hcs",
    mode="r+",
)
chan_names = zarr_input.channel_names
# zarr_input.append_channel('Inf_mask',resize_arrays=True)

file_out_path = '/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/Infection_phenotyping_data/Exp_2023_09_28_DENV_A2_infMarked_rev2.zarr'
zarr_output = open_ome_zarr(
    file_out_path,
    layout="hcs",
    mode="w-",
    channel_names=['Sensor','Nucl_mask','Inf_mask'],
)

v = napari.Viewer()


#%% Load label image to napari
for well_id, well_data in zarr_input.wells():
    well_name, well_no = well_id.split("/")

    if well_name == 'A' and well_no == '2':

        for pos_name, pos_data in well_data.positions():
            # if int(pos_name) > 1:
            v.layers.clear()
            data = pos_data.data

            FITC = data[0,0,...]
            v.add_image(FITC, name='FITC', colormap='green', blending='additive')
            Inf_mask = data[0,1,...].astype(int)
            v.add_labels(Inf_mask)
            input("Press Enter")

            label_layer = v.layers['Inf_mask']
            label_array = label_layer.data
            label_array = np.expand_dims(label_array, axis=(0, 1))
            # zarr_input.create_image('Inf_mask',label_array)
            out_data = np.concatenate((data, label_array), axis=1)
            position = zarr_output.create_position(well_name, well_no, pos_name)
            position["0"] = out_data
            

# %%
