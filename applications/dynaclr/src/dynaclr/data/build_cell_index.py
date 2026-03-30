"""CLI command for building a cell index parquet from time-lapse experiments."""

import click


@click.command()
@click.argument("collection_yaml")
@click.argument("output")
@click.option(
    "--include-wells",
    multiple=True,
    default=None,
    help="Wells to include (e.g. A/1). Repeat for multiple.",
)
@click.option(
    "--exclude-fovs",
    multiple=True,
    default=None,
    help="FOVs to exclude (e.g. A/1/0). Repeat for multiple.",
)
@click.option(
    "--num-workers",
    default=-1,
    show_default=True,
    help="Parallel worker processes. -1 = all CPUs, 1 = sequential.",
)
def main(collection_yaml, output, include_wells, exclude_fovs, num_workers):
    """Build cell index parquet from a collection YAML."""
    from viscy_data.cell_index import build_timelapse_cell_index

    build_timelapse_cell_index(
        collection_path=collection_yaml,
        output_path=output,
        include_wells=list(include_wells) or None,
        exclude_fovs=list(exclude_fovs) or None,
        num_workers=num_workers,
    )
