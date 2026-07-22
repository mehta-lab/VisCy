"""CLI command for preprocessing a cell index parquet (add norm stats, focus slice, remove empties)."""

import click

from viscy_data.cell_index import preprocess_cell_index


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
@click.option(
    "--csv-dir",
    default=None,
    help=(
        "Read normalization/focus_slice metadata from per-store CSV sidecars under this "
        "directory instead of from zarr zattrs. Use when the zarr stores were preprocessed "
        "with `viscy preprocess --csv_dir ...` / `qc run` configured with `csv_dir`."
    ),
)
def main(parquet_path, output, focus_channel, csv_dir):
    """Preprocess a cell index parquet: add normalization stats, focus slice, remove empty frames.

    Reads precomputed metadata from zarr zattrs (or CSV sidecars, with
    `--csv-dir`) and writes them as parquet columns. Requires `viscy
    preprocess` to have been run on the zarr stores.
    """
    preprocess_cell_index(
        parquet_path=parquet_path,
        output_path=output,
        focus_channel=focus_channel,
        csv_dir=csv_dir,
    )
