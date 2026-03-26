"""CLI for applying a fitted LOT pipeline to an embedding zarr.

Usage
-----
    viscy-dynaclr apply-lot-correction -c config.yaml

Transforms all cells through StandardScaler → PCA → LOT and writes a new
zarr whose ``.X`` contains the corrected embeddings (shape n_cells × n_pca).
All ``.obs`` metadata from the input zarr is preserved.

Example config (YAML)
---------------------
    input_zarr: /path/to/lightsheet_organelle.zarr
    pipeline: /path/to/lot_pipeline.pkl
    output_zarr: /path/to/corrected_organelle.zarr
    overwrite: false
"""

import logging
from pathlib import Path

import click
from pydantic import ValidationError

from viscy.representation.evaluation.lot_correction import (
    apply_lot_correction,
    load_lot_pipeline,
)
from viscy.representation.evaluation.lot_correction_config import LotApplyConfig
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
    """Apply a fitted LOT pipeline to correct batch effects in an embedding zarr."""
    click.echo("=" * 60)
    click.echo("LOT BATCH CORRECTION — APPLY")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        apply_config = LotApplyConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\nConfiguration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nFailed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo(f"\nConfiguration loaded: {config}")
    click.echo(f"  Input zarr:   {apply_config.input_zarr}")
    click.echo(f"  Pipeline:     {apply_config.pipeline}")
    click.echo(f"  Output zarr:  {apply_config.output_zarr}")
    click.echo(f"  Overwrite:    {apply_config.overwrite}")

    try:
        pipeline = load_lot_pipeline(apply_config.pipeline)
        click.echo(
            f"\nPipeline loaded — n_pca={pipeline['n_pca']}, "
            f"PCA variance={pipeline.get('pca_variance_explained', float('nan')):.1f}%"
        )
        apply_lot_correction(
            input_zarr=apply_config.input_zarr,
            pipeline=pipeline,
            output_zarr=apply_config.output_zarr,
            overwrite=apply_config.overwrite,
        )
        click.echo(f"\nCorrected zarr written to: {apply_config.output_zarr}")
    except Exception as e:
        click.echo(f"\nApplication failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
