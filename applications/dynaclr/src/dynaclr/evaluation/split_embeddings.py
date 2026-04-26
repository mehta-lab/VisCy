"""Split a combined embeddings zarr into one zarr per experiment.

Reads the combined embeddings.zarr produced by the predict step, groups rows
by obs["experiment"], and writes one AnnData zarr per experiment under
output_dir/{experiment}.zarr. The combined zarr is removed after splitting.

Usage
-----
dynaclr split-embeddings -c split.yaml

Or with inline arguments:

dynaclr split-embeddings --input /path/to/embeddings.zarr --output-dir /path/to/embeddings/
"""

from __future__ import annotations

from pathlib import Path

import click


def split_embeddings(input_path: Path, output_dir: Path) -> list[Path]:
    """Split combined embeddings zarr into one zarr per experiment.

    Parameters
    ----------
    input_path : Path
        Path to the combined embeddings zarr (AnnData format).
        Must have obs["experiment"] column.
    output_dir : Path
        Directory to write per-experiment zarrs.
        Each experiment is written to output_dir/{experiment}.zarr.

    Returns
    -------
    list[Path]
        Paths to the written per-experiment zarrs.
    """
    import anndata as ad

    if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    import pandas as pd

    pd.options.future.infer_string = False

    click.echo(f"Loading embeddings from {input_path}")
    adata = ad.read_zarr(input_path)
    click.echo(f"  {adata.n_obs} cells, {adata.n_vars} features")

    if "experiment" not in adata.obs.columns:
        raise ValueError(
            "embeddings zarr obs is missing 'experiment' column. "
            "Re-run the predict step with the updated pipeline to include metadata."
        )

    experiments = adata.obs["experiment"].unique().tolist()
    click.echo(f"  {len(experiments)} experiments: {experiments}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for exp in experiments:
        mask = adata.obs["experiment"] == exp
        adata_exp = adata[mask].copy()
        out_path = output_dir / f"{exp}.zarr"
        click.echo(f"  Writing {exp}: {adata_exp.n_obs} cells → {out_path}")
        adata_exp.write_zarr(out_path)
        written.append(out_path)

    click.echo(f"\nRemoving combined zarr: {input_path}")
    import shutil

    shutil.rmtree(input_path)

    click.echo(f"\nWrote {len(written)} per-experiment zarrs to {output_dir}")
    return written


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to combined embeddings zarr",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write per-experiment zarrs",
)
def main(input_path: Path, output_dir: Path) -> None:
    """Split a combined embeddings zarr into one zarr per experiment."""
    split_embeddings(input_path, output_dir)


if __name__ == "__main__":
    main()
