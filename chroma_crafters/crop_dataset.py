# %%
from iohub.ngff import open_ome_zarr, Position, TransformationMeta
import numpy as np
import multiprocessing as mp
from natsort import natsorted
import glob
from pathlib import Path
import click
from functools import partial
import itertools
from typing import Tuple

input_data_path = "/home/eduardoh/vs_data/0-raw_data/1-H2B_dataset/target_fluorescence/deskewed.zarr/*/*/*"
output_data_path = "./cropped_dataset_v2.zarr"


def get_output_paths(input_paths: list[Path], output_zarr_path: Path) -> list[Path]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        # Append the same Row/Column/FOV to the output zarr path
        list_output_path.append(Path(output_zarr_path, *path_strings))
    return list_output_path


def create_empty_zarr(
    position_paths: list[Path],
    output_path: Path,
    output_zyx_shape: Tuple[int],
    chunk_zyx_shape: Tuple[int] = None,
    voxel_size: Tuple[int, float] = (1, 1, 1),
) -> None:
    """Create an empty zarr store mirroring another store"""
    DTYPE = np.float32
    MAX_CHUNK_SIZE = 500e6  # in bytes
    bytes_per_pixel = np.dtype(DTYPE).itemsize

    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    click.echo("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + output_zyx_shape
    click.echo(f"Number of positions: {len(position_paths)}")
    click.echo(f"Output shape: {output_shape}")

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    if chunk_zyx_shape is None:
        chunk_zyx_shape = list(output_zyx_shape)
        # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

    chunk_size = 2 * (1,) + chunk_zyx_shape
    click.echo(f"Chunk size: {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for path in position_paths:
        path_strings = Path(path).parts[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        _ = pos.create_zeros(
            name="0",
            shape=output_shape,
            chunks=chunk_size,
            dtype=DTYPE,
            transform=[transform],
        )

    input_dataset.close()


def copy_n_paste(
    position: Position,
    output_path: Path,
    t_idx: int,
    c_idx: int,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Processing c={c_idx}, t={t_idx}")
    data_array = open_ome_zarr(position)

    zyx_data = data_array[0][t_idx, c_idx]

    # Apply transformation
    # TODO:crop here
    # Write to file
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = zyx_data[:, 319:1883, 14:1190]

    data_array.close()
    click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


# %%
# -----------------------------------------------
num_processes = 8

input_data_paths = natsorted(glob.glob(input_data_path))
output_paths = get_output_paths(input_data_paths, output_data_path)
print(input_data_paths, output_paths)
with open_ome_zarr(input_data_paths[0]) as sample_dataset:
    voxel_size = tuple(sample_dataset.scale[-3:])
    T, C, Z, Y, X = sample_dataset[0].shape
    # TODO:crop here
    output_shape_zyx = (Z, 1883 - 319, 1190 - 14)
# %%
create_empty_zarr(
    position_paths=input_data_paths,
    output_path=output_data_path,
    output_zyx_shape=output_shape_zyx,
    chunk_zyx_shape=None,
    voxel_size=voxel_size,
)
# %%
for input_dataset, output_path in zip(input_data_paths, output_paths):
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(copy_n_paste, input_dataset, output_path),
            itertools.product(range(T), range(C)),
        )

# %%
