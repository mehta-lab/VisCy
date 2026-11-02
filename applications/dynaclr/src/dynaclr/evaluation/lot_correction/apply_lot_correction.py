"""CLI for applying a fitted LOT pipeline to an embedding zarr.

Usage
-----
    dynaclr apply-lot-correction \
        --pipeline lot_pipeline.pkl \
        --input embeddings.zarr \
        --output corrected.zarr

Transforms all cells through StandardScaler → (optional PCA) → LOT and writes
a new zarr whose ``.X`` contains the corrected embeddings. When the pipeline
was fit with PCA the output has ``n_pca`` columns; otherwise it keeps the
input feature dimension. All ``.obs`` metadata from the input zarr is preserved.
"""

import logging
from pathlib import Path

import click

from dynaclr.evaluation.lot_correction.lot_correction import (
    apply_lot_correction,
    load_lot_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--pipeline",
    "pipeline_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the fitted LOT pipeline (joblib pickle from fit-lot-correction).",
)
@click.option(
    "--input",
    "input_zarr",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the embedding zarr to correct.",
)
@click.option(
    "--output",
    "output_zarr",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to write the corrected embedding zarr.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output zarr if it already exists.",
)
def main(pipeline_path: Path, input_zarr: Path, output_zarr: Path, overwrite: bool):
    """Apply a fitted LOT pipeline to correct batch effects in an embedding zarr."""
    click.echo("=" * 60)
    click.echo("LOT BATCH CORRECTION — APPLY")
    click.echo("=" * 60)
    click.echo(f"  Pipeline:     {pipeline_path}")
    click.echo(f"  Input zarr:   {input_zarr}")
    click.echo(f"  Output zarr:  {output_zarr}")
    click.echo(f"  Overwrite:    {overwrite}")

    try:
        pipeline = load_lot_pipeline(pipeline_path)
        var_exp = pipeline.get("pca_variance_explained")
        var_exp_str = "disabled" if var_exp is None else f"{var_exp:.1f}%"
        channel = pipeline.get("channel") or "(unspecified)"
        click.echo(f"\nPipeline loaded — channel={channel}, n_pca={pipeline['n_pca']}, PCA variance={var_exp_str}")
        apply_lot_correction(
            input_zarr=input_zarr,
            pipeline=pipeline,
            output_zarr=output_zarr,
            overwrite=overwrite,
        )
        click.echo(f"\nCorrected zarr written to: {output_zarr}")
    except Exception as e:
        click.echo(f"\nApplication failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
