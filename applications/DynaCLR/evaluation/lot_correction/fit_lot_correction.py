"""CLI for fitting a LOT batch-correction pipeline on embedding zarrs.

Usage
-----
    viscy-dynaclr fit-lot-correction -c config.yaml

The fitted pipeline (StandardScaler + PCA + LinearTransport) is saved to
the path specified by ``output_pipeline`` in the config file.

Example config (YAML)
---------------------
    source_zarr: /path/to/lightsheet_organelle.zarr
    target_zarr: /path/to/confocal_organelle.zarr
    source_uninf_filter:
      column: fov_name
      startswith:
        - "C/1/"
    target_uninf_filter:
      column: fov_name
      startswith:
        - "G3BP1/uninfected"
    n_pca: 50
    ns_lot: 3000
    random_seed: 42
    output_pipeline: /path/to/lot_pipeline.pkl
"""

import logging
from pathlib import Path

import click
from pydantic import ValidationError

from viscy.representation.evaluation.lot_correction import (
    fit_lot_correction,
    save_lot_pipeline,
)
from viscy.representation.evaluation.lot_correction_config import LotFitConfig
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
    """Fit a LOT batch-correction pipeline on source and target embedding zarrs."""
    click.echo("=" * 60)
    click.echo("LOT BATCH CORRECTION — FIT")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        fit_config = LotFitConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\nConfiguration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nFailed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo(f"\nConfiguration loaded: {config}")
    click.echo(f"  Source zarr:  {fit_config.source_zarr}")
    click.echo(f"  Target zarr:  {fit_config.target_zarr}")
    click.echo(f"  n_pca:        {fit_config.n_pca}")
    click.echo(f"  ns_lot:       {fit_config.ns_lot}")
    click.echo(f"  Random seed:  {fit_config.random_seed}")
    click.echo(f"  Output:       {fit_config.output_pipeline}")

    try:
        pipeline = fit_lot_correction(
            source_zarr=fit_config.source_zarr,
            target_zarr=fit_config.target_zarr,
            source_uninf_filter=fit_config.source_uninf_filter.to_dict(),
            target_uninf_filter=fit_config.target_uninf_filter.to_dict(),
            n_pca=fit_config.n_pca,
            ns_lot=fit_config.ns_lot,
            random_seed=fit_config.random_seed,
        )
        click.echo(
            f"\nPipeline fitted — PCA explained variance: "
            f"{pipeline['pca_variance_explained']:.1f}%"
        )
        save_lot_pipeline(pipeline, fit_config.output_pipeline)
        click.echo(f"Pipeline saved to: {fit_config.output_pipeline}")
    except Exception as e:
        click.echo(f"\nFitting failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
