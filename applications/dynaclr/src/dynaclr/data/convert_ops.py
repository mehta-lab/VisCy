"""CLI command for converting OPS merged parquets to DynaCLR cell index format."""

import click


@click.command()
@click.argument("ops_parquet")
@click.argument("output")
@click.option(
    "--store-root",
    default="/hpc/projects/icd.fast.ops",
    show_default=True,
    help="Root directory for OPS zarr stores.",
)
@click.option(
    "--store-suffix",
    default="3-assembly/phenotyping_v3.zarr",
    show_default=True,
    help="Suffix appended after store_key to form store_path.",
)
def main(ops_parquet, output, store_root, store_suffix):
    """Convert an OPS merged parquet to the canonical DynaCLR cell index format."""
    from viscy_data.cell_index import convert_ops_parquet

    convert_ops_parquet(
        ops_parquet_path=ops_parquet,
        output_path=output,
        store_root=store_root,
        store_suffix=store_suffix,
    )
