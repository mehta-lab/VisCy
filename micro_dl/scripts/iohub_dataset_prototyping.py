#%%
import iohub.ngff as ngff
from torch.utils.data import Dataset
import numpy as np
import zarr
import iohub.ngff_meta as ngff_meta
import os
import sys


sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import micro_dl.input.inference_dataset as inference_dataset
import micro_dl.utils.io_utils as io
from micro_dl.cli.preprocess_script import pre_process
import micro_dl.training.training as train
import micro_dl.cli.torch_inference_script as inference_script
import micro_dl.utils.aux_utils as aux_utils
#%%
########---------------------------------##########
########----------- TOY DATA ------------##########
########---------------------------------##########
position_list = (
    ("A", "1", "0"),
    ("H", 1, "0"),
    ("H", "12", "CannotVisualize"),
    ("Control", "Blank", 0),
)
store_path = f'{os.path.expanduser("~/")}test_hcs.zarr'
with ngff.open_ome_zarr(
    store_path,
    layout="hcs",
    mode="w-",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # Create and write to positions
    # This affects the tile arrangement in visualization
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        
        position[f"0"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )
    # Print dataset summary
    dataset.print_tree()

#%%
########---------------------------------##########
########----- adding untracked array ----##########
########---------------------------------##########
store = zarr.open(store_path)
plate = ngff.open_ome_zarr(
    store_path,
    layout="hcs",
    mode="r",
    channel_names=["DAPI", "GFP", "Brightfield"],
)

for i, (path, pos_object) in enumerate(plate.positions()):
    print(type(pos_object.data))
    break

array_data = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(1, 3, 1, 32, 32), dtype=np.uint16
        )

io.init_untracked_array(
    zarr_dir=store_path,
    position_path=path,
    data_array=array_data,
    name='test',
    overwrite_ok=True
)

# Should display the tree with ff in it
plate.print_tree()

#%%
########---------------------------------##########
########----- adding custom metadata ----##########
########---------------------------------##########

#multiscales should not contain any data for 'test', only '0'
print(plate[path].metadata.dict().keys())

io.write_meta_field(
    zarr_dir=store_path,
    position_path=path,
    metadata={
        'array_name': 'test',
        'channel_ids': [1,2,0]
        },
    field_name='test',
)
metadata = io.read_meta_field(
    zarr_dir=store_path,
    position_path=path,
    field_name='test',
)
print(metadata)
print(plate[path].metadata.dict().keys())
#%%
########---------------------------------##########
########----- reading custom array w/ ---##########
########-------- attached metadata ------##########
########---------------------------------##########
array_slice = io.get_untracked_array_slice(
    zarr_dir=store_path,
    position_path=path,
    meta_field_name='test',
    time_index=0,
    channel_index=0,
    z_index=0,
)
print(array_slice.shape)

#%%
########---------------------------------##########
########----- adding channel to hcs -----##########
########---------------------------------##########
# append a channel
with ngff.open_ome_zarr(store_path, mode="r+") as dataset:
    for name, position in dataset.positions():
        print(f"Appending a channel to position: {name}")
        position.append_channel("Segmentation", resize_arrays=True)
        new_channel_index = len(position.channel_names)
        position["0"][:, 3] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 2, 32, 32), dtype=np.uint16
        )
    dataset.print_tree()

#rewrite that channel
with ngff.open_ome_zarr(store_path, mode="r+") as dataset:
    for name, position in dataset.positions():
        print(f"Replacing a channel at position: {name}")
        new_channel_index = list(position.channel_names).index('Segmentation')
        position["0"][:, new_channel_index] = np.full(shape=(5, 2, 32, 32), fill_value=1)
    dataset.print_tree()

    channel = position["0"][:, new_channel_index]
    result= np.all(np.equal(channel, np.full(shape=(5, 2, 32, 32), fill_value=1)))

print(f"Channel rewritten: {result}")
#%%
########---------------------------------##########
########---------- REAL DATA ------------##########
########---------------------------------##########

zarr_dir = (
    "/hpc/projects/compmicro/projects/automation/"
    "dataFormats/ome_zarr_demo/fish/isim_rachel.zarr"
)

zarr_dir_2 = (
    "/hpc/projects/CompMicro/projects/infected_cell_imaging"
    "/VirtualStaining/VirtualStain_NuclMem_A549_2023_02_07/"
    "Input_Nucl/A549_20X_iohubTest.zarr"
)

plate = ngff.open_ome_zarr(zarr_dir_2,layout='hcs',mode='r')
plate.print_tree()
#%%
########---------------------------------##########
########---------- REAL DATA ------------##########
########---------------------------------##########


config_path = (
    "/hpc/projects/CompMicro/projects/infected_cell_imaging/"
    "VirtualStaining/config_files/torch_gpLoading_2023_02/"
    "torch_config_nucl25D_iohubTest.yml"
)
config = aux_utils.read_config(config_path)
#%%
# Preprocessing
pre_process(config)
#%%
# Training
trainer = train.TorchTrainer(config)

trainer.generate_dataloaders()
trainer.load_model()

# train
trainer.train()
# %%
# Inference
inference_script.main(config, gpu=0, gpu_mem_frac=0.1)