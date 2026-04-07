"""CLI command for preprocessing a cell index parquet (add norm stats, focus slice, remove empties)."""

import click


@click.command()
@click.argument("parquet_path")
@click.option(
    "--output",
    default=None,
    help="Output path. Default: overwrite in place.",
)
@click.option(
    "--focus-channel",
    default=None,
    help="Channel name for focus_slice lookup (e.g. Phase3D). Default: first channel per FOV.",
)
def main(parquet_path, output, focus_channel):
    """Preprocess a cell index parquet: add normalization stats, focus slice, remove empty frames.

    Reads precomputed metadata from zarr zattrs and writes them as parquet
    columns. Requires `viscy preprocess` to have been run on the zarr stores.
    """
    from viscy_data.cell_index import preprocess_cell_index

    preprocess_cell_index(
        parquet_path=parquet_path,
        output_path=output,
        focus_channel=focus_channel,
    )
