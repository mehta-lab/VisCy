"""CLI for computing MMD² between two groups of cell embeddings.

Usage
-----
    viscy-dynaclr compute-mmd -c config.yaml

The command compares two groups (A and B) defined by obs filters on one or
two AnnData zarrs.  An optional ``group_by`` field splits the comparison into
per-group rows (e.g. per organelle per timepoint).

Example configs
---------------

**Biological signal** (ZIKV vs uninfected, same zarr, per organelle/timepoint):

    zarr_a: /path/to/organelle_embeddings.zarr
    filter_a:
      column: condition
      startswith: ["uninfected"]
    filter_b:
      column: condition
      equals: "ZIKV"
    group_by:
      - organelle
      - timepoint
    use_pca: true
    n_pca: 50
    n_perm: 1000
    max_cells: 2000
    random_seed: 42
    output_csv: mmd_results.csv

**Batch effect** (light-sheet vs confocal, two zarrs):

    zarr_a: /path/to/lightsheet.zarr
    zarr_b: /path/to/confocal.zarr
    filter_a:
      column: fov_name
      startswith: ["C/1/"]
    filter_b:
      column: fov_name
      startswith: ["G3BP1/uninfected"]
    group_by: []
    use_pca: true
    n_pca: 50
    n_perm: 0
    max_cells: 2000
    random_seed: 42
    output_csv: batch_mmd.csv
"""

import logging
from pathlib import Path

import click
from pydantic import ValidationError

from viscy.representation.evaluation.mmd import compute_mmd
from viscy.representation.evaluation.mmd_config import ComputeMMDConfig
from viscy.utils.cli_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file.",
)
def main(config: Path):
    """Compute MMD² between two groups of cell embeddings from AnnData zarrs."""
    click.echo("=" * 60)
    click.echo("MMD COMPUTATION")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        mmd_config = ComputeMMDConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\nConfiguration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nFailed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo(f"\nConfiguration loaded: {config}")
    click.echo(f"  Zarr A:       {mmd_config.zarr_a}")
    click.echo(f"  Zarr B:       {mmd_config.zarr_b or '(same as A)'}")
    click.echo(f"  Filter A:     {mmd_config.filter_a}")
    click.echo(f"  Filter B:     {mmd_config.filter_b}")
    click.echo(f"  Group by:     {mmd_config.group_by or '(none — single overall)'}")
    click.echo(f"  PCA:          {'yes, n=' + str(mmd_config.n_pca) if mmd_config.use_pca else 'no'}")
    click.echo(f"  n_perm:       {mmd_config.n_perm or 'skipped'}")
    click.echo(f"  max_cells:    {mmd_config.max_cells}")
    click.echo(f"  Output:       {mmd_config.output_csv}")

    try:
        results = compute_mmd(
            zarr_a=mmd_config.zarr_a,
            zarr_b=mmd_config.zarr_b,
            filter_a=mmd_config.filter_a.to_dict() if mmd_config.filter_a else None,
            filter_b=mmd_config.filter_b.to_dict() if mmd_config.filter_b else None,
            group_by=mmd_config.group_by or None,
            use_pca=mmd_config.use_pca,
            n_pca=mmd_config.n_pca,
            n_perm=mmd_config.n_perm,
            max_cells=mmd_config.max_cells,
            random_seed=mmd_config.random_seed,
        )

        if results.empty:
            click.echo("\nNo results computed — check filters and group_by columns.")
            raise click.Abort()

        output_path = Path(mmd_config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        click.echo(f"\nResults ({len(results)} rows) written to: {output_path}")
        click.echo("\n" + results.to_string(index=False))

    except click.Abort:
        raise
    except Exception as e:
        click.echo(f"\nMMD computation failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
