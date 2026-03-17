"""CLI command for building a cell index parquet from time-lapse experiments."""

import click


@click.command()
@click.argument("experiments_yaml")
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
def main(experiments_yaml, output, include_wells, exclude_fovs):
    """Build cell index parquet from time-lapse experiment config."""
    from viscy_data.cell_index import build_timelapse_cell_index

    df = build_timelapse_cell_index(
        experiments_yaml=experiments_yaml,
        output_path=output,
        include_wells=list(include_wells) or None,
        exclude_fovs=list(exclude_fovs) or None,
    )
    click.echo(f"Wrote {len(df)} cell observations to {output}")
