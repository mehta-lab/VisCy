"""CLI for appending columns from a CSV to the obs of an AnnData zarr store.

Supports any tabular data (human annotations, computed features, predictions,
etc.) by merging on shared key column(s). An optional prefix distinguishes the
source of the new columns (e.g. ``annotated_``, ``predicted_``, ``feature_``).

Usage:
    dynaclr append-obs \
        -e /path/to/embeddings.zarr \
        --csv /path/to/data.csv \
        --prefix annotated_

    dynaclr append-obs \
        -e /path/to/embeddings.zarr \
        --csv /path/to/data.csv \
        --merge-key fov_name --merge-key track_id --merge-key t
"""

from pathlib import Path

import click
from anndata import read_zarr

from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr, merge_csv_into_obs


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-e",
    "--embeddings",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the AnnData zarr store.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to CSV with columns to append.",
)
@click.option(
    "-p",
    "--prefix",
    default="",
    show_default=True,
    help="Prefix for new column names (e.g. 'annotated_', 'predicted_', 'feature_').",
)
@click.option(
    "-c",
    "--columns",
    multiple=True,
    default=None,
    help="Columns to append. If not specified, all new columns from the CSV are used.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output zarr path. Defaults to overwriting the embeddings store.",
)
@click.option(
    "--merge-key",
    multiple=True,
    default=("id",),
    show_default=True,
    help="Column(s) to merge on. Can be specified multiple times for composite keys.",
)
def main(
    embeddings: Path,
    csv_path: Path,
    prefix: str,
    columns: tuple[str, ...],
    output: Path | None,
    merge_key: tuple[str, ...],
):
    """Append columns from a CSV to the obs of an AnnData zarr store."""
    click.echo("=" * 60)
    click.echo("APPEND OBS")
    click.echo("=" * 60)

    write_path = output if output is not None else embeddings
    keys = list(merge_key) if len(merge_key) > 1 else merge_key[0]
    cols = list(columns) if columns else None

    adata = read_zarr(embeddings)
    click.echo(f"\n  Loaded embeddings: {adata.shape}")
    click.echo(f"  CSV: {csv_path}")
    click.echo(f"  Merge key(s): {keys}")
    click.echo(f"  Prefix: '{prefix}'")
    click.echo(f"  Output: {write_path}")

    adata, match_counts = merge_csv_into_obs(
        adata,
        csv_path,
        merge_key=keys,
        columns=cols,
        prefix=prefix,
    )

    for dest, n_matched in match_counts.items():
        click.echo(f"  {dest}: {n_matched}/{len(adata)} matched")

    click.echo(f"\nSaving to: {write_path}")
    append_to_anndata_zarr(write_path, obs=adata.obs)
    click.echo("  Saved.")

    click.echo("\n  Done!")


if __name__ == "__main__":
    main()
