"""CLI for fitting a LOT batch-correction pipeline on embedding zarrs.

Usage
-----
    dynaclr fit-lot-correction -c config.yaml

The fitted pipeline (StandardScaler + optional PCA + LinearTransport) is saved
to the path specified by ``output_pipeline`` in the config file.

Multiple datasets can be pooled per side, and PCA can be disabled by setting
``n_pca: null``.

Example config (YAML)
---------------------
    source:
      - zarr: /path/to/lightsheet_organelle_rep1.zarr
        filter:
          column: fov_name
          startswith:
            - "C/1/"
      - zarr: /path/to/lightsheet_organelle_rep2.zarr
        filter:
          column: fov_name
          startswith:
            - "C/1/"
    target:
      - zarr: /path/to/confocal_organelle.zarr
        filter:
          column: fov_name
          startswith:
            - "G3BP1/uninfected"
    channel: Phase3D
    n_pca: 50
    ns_lot: 3000
    random_seed: 42
    output_pipeline: /path/to/lot_pipeline.pkl
"""

import logging
from pathlib import Path

import anndata as ad
import click
import numpy as np
from pydantic import ValidationError

from dynaclr.evaluation.lot_correction.config import DatasetSpec, LotFitConfig
from dynaclr.evaluation.lot_correction.lot_correction import (
    fit_lot_correction,
    save_lot_pipeline,
)
from viscy_utils.cli_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_logger = logging.getLogger(__name__)


def _apply_filter(obs, filter_spec: dict) -> np.ndarray:
    """Return a boolean mask for rows of *obs* matching *filter_spec*.

    Parameters
    ----------
    obs : pd.DataFrame
        AnnData ``.obs`` table.
    filter_spec : dict
        Must contain ``"column"`` plus one of:

        * ``"startswith"`` – str or list[str]: keep rows where the column
          value starts with any of the given prefixes.
        * ``"equals"`` – str: keep rows where the column value equals the
          given string.

    Returns
    -------
    np.ndarray of bool
        Boolean mask with the same length as *obs*.
    """
    col = filter_spec["column"]
    values = obs[col].astype(str)

    if "startswith" in filter_spec:
        prefixes = filter_spec["startswith"]
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        mask = np.zeros(len(obs), dtype=bool)
        for p in prefixes:
            mask |= values.str.startswith(p).values
        return mask

    if "equals" in filter_spec:
        return (values == str(filter_spec["equals"])).values

    raise ValueError(f"filter_spec must contain either 'startswith' or 'equals'. Got: {list(filter_spec.keys())}")


def _load_datasets(specs: list[DatasetSpec], side: str) -> list[ad.AnnData]:
    """Load and filter each dataset in *specs* to its reference population.

    Parameters
    ----------
    specs : list[DatasetSpec]
        Dataset specifications (zarr path + optional filter).
    side : str
        Human-readable label for logging (e.g. ``"source"``).

    Returns
    -------
    list[AnnData]
        Filtered AnnData objects, one per spec.
    """
    adatas = []
    for spec in specs:
        _logger.info("Loading %s zarr: %s", side, spec.zarr)
        adata = ad.read_zarr(spec.zarr)
        adata.obs_names_make_unique()
        if spec.filter is not None:
            mask = _apply_filter(adata.obs, spec.filter.to_dict())
            _logger.info("  Filtered %s cells: %d / %d", side, int(mask.sum()), adata.n_obs)
            adata = adata[mask].copy()
        else:
            _logger.info("  No filter — using all %d %s cells", adata.n_obs, side)
        adatas.append(adata)
    return adatas


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file.",
)
def main(config: Path):
    """Fit a LOT batch-correction pipeline on pooled source and target zarrs."""
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
    click.echo(f"  Channel:         {fit_config.channel if fit_config.channel is not None else '(unspecified)'}")
    click.echo(f"  Source datasets: {len(fit_config.source)}")
    click.echo(f"  Target datasets: {len(fit_config.target)}")
    click.echo(f"  n_pca:           {fit_config.n_pca if fit_config.n_pca is not None else 'disabled'}")
    click.echo(f"  ns_lot:          {fit_config.ns_lot}")
    click.echo(f"  Random seed:     {fit_config.random_seed}")
    click.echo(f"  Output:          {fit_config.output_pipeline}")

    try:
        source_adatas = _load_datasets(fit_config.source, "source")
        target_adatas = _load_datasets(fit_config.target, "target")
        pipeline = fit_lot_correction(
            source_adatas=source_adatas,
            target_adatas=target_adatas,
            channel=fit_config.channel,
            n_pca=fit_config.n_pca,
            ns_lot=fit_config.ns_lot,
            random_seed=fit_config.random_seed,
        )
        if pipeline["pca_variance_explained"] is not None:
            click.echo(f"\nPipeline fitted — PCA explained variance: {pipeline['pca_variance_explained']:.1f}%")
        else:
            click.echo("\nPipeline fitted — PCA disabled (LOT in scaled embedding space)")
        save_lot_pipeline(pipeline, fit_config.output_pipeline)
        click.echo(f"Pipeline saved to: {fit_config.output_pipeline}")
    except Exception as e:
        click.echo(f"\nFitting failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
