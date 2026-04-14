"""Zarr store rewriting utilities."""

from pathlib import Path

from iohub.ngff import open_ome_zarr
from tqdm import tqdm


def rewrite_zarr(
    input_path: Path,
    output_path: Path,
    chunks: tuple[int, ...],
    shards_ratio: tuple[int, ...] | None = None,
    version: str = "0.5",
) -> None:
    """Copy an OME-Zarr store with new chunking and sharding.

    Iterates all positions, copies data, channel names, and coordinate
    transforms into a new store with the specified chunk/shard layout.

    Parameters
    ----------
    input_path : Path
        Path to the input OME-Zarr store.
    output_path : Path
        Path for the output OME-Zarr store.
    chunks : tuple[int, ...]
        Chunk dimensions for the output arrays.
    shards_ratio : tuple[int, ...] | None
        Shard-to-chunk ratio. None disables sharding.
    version : str
        Zarr format version (default "0.5").
    """
    with open_ome_zarr(input_path, mode="r", layout="hcs") as old_dataset:
        with open_ome_zarr(
            output_path,
            layout="hcs",
            mode="w",
            channel_names=old_dataset.channel_names,
            version=version,
        ) as new_dataset:
            total_positions = sum(1 for _ in old_dataset.positions())
            for name, old_position in tqdm(old_dataset.positions(), total=total_positions):
                row, col, fov = name.split("/")
                new_position = new_dataset.create_position(row, col, fov)
                old_image = old_position["0"]
                create_kwargs: dict = {
                    "data": old_image.numpy(),
                    "chunks": chunks,
                    "transform": old_position.metadata.multiscales[0].datasets[0].coordinate_transformations,
                }
                if shards_ratio is not None:
                    create_kwargs["shards_ratio"] = shards_ratio
                new_position.create_image("0", **create_kwargs)
