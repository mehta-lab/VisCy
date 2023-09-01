# %%
from iohub import open_ome_zarr
import numpy as np
from tqdm import tqdm

input_data_path = (
    "/home/eduardoh/vs_data/0-raw_data/1-H2B_dataset/input_phase/registered_output.zarr"
)
target_data_path = (
    "/home/eduardoh/vs_data/0-raw_data/1-H2B_dataset/target_fluorescence/deskewed.zarr"
)
appending_dataset = open_ome_zarr(input_data_path, mode="r")
appending_channel_names = appending_dataset.channel_names
with open_ome_zarr(target_data_path, mode="r+") as dataset:
    for name, position in tqdm(dataset.positions()):
        # print(name, position)
        position.append_channel(appending_channel_names[0])
        position["0"][:, 2] = appending_dataset[str(name)][0][:, 0]
        # print(f"Appending a channel to position: {name}")
        # position.append_channel(appending_channel_names, resize_arrays=True)
        # position["0"][:, 2] = appending_channel_names[0]
    dataset.print_tree()

# %%
