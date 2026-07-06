"""Split a combined embeddings zarr into one zarr per group.

Reads the combined embeddings.zarr produced by the predict step, groups rows
by obs[group_by] (``experiment`` by default), and writes one AnnData zarr per
group under output_dir/{group}.zarr. The combined zarr is removed after
splitting.

Usage
-----
dynaclr split-embeddings -c split.yaml

Or with inline arguments:

dynaclr split-embeddings --input /path/to/embeddings.zarr --output-dir /path/to/embeddings/

Split by a different column (e.g. one zarr per marker):

dynaclr split-embeddings --input /path/to/embeddings.zarr --output-dir /path/to/embeddings/ --group-by marker
"""

from __future__ import annotations

from pathlib import Path

import click


def split_embeddings(
    input_path: Path,
    output_dir: Path,
    group_by: str = "experiment",
    prefix_by: str | None = None,
) -> list[Path]:
    """Split combined embeddings zarr into one zarr per group.

    Parameters
    ----------
    input_path : Path
        Path to the combined embeddings zarr (AnnData format).
        Must have the ``group_by`` (and ``prefix_by``, if set) column in obs.
    output_dir : Path
        Directory to write per-group zarrs.
        Each group value is written to ``output_dir/{group}.zarr``, or
        ``output_dir/{prefix}_{group}.zarr`` when ``prefix_by`` is set.
    group_by : str, optional
        obs column to group rows by. By default ``"experiment"``.
    prefix_by : str or None, optional
        obs column whose value prefixes each output filename as
        ``{prefix}_{group}.zarr`` (e.g. ``prefix_by="experiment"`` with
        ``group_by="marker"`` yields ``{dataset}_{marker}.zarr``). The prefix
        column must be constant within each group. By default ``None`` (no
        prefix).

    Returns
    -------
    list[Path]
        Paths to the written per-group zarrs.
    """
    import anndata as ad

    if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    import pandas as pd

    pd.options.future.infer_string = False

    click.echo(f"Loading embeddings from {input_path}")
    adata = ad.read_zarr(input_path)
    click.echo(f"  {adata.n_obs} cells, {adata.n_vars} features")

    for col in filter(None, [group_by, prefix_by]):
        if col not in adata.obs.columns:
            raise ValueError(
                f"embeddings zarr obs is missing '{col}' column. "
                f"Available columns: {sorted(adata.obs.columns)}. "
                "Re-run the predict step with the updated pipeline to include metadata."
            )

    groups = adata.obs[group_by].unique().tolist()
    click.echo(f"  {len(groups)} {group_by} groups: {groups}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for group in groups:
        mask = adata.obs[group_by] == group
        adata_group = adata[mask].copy()
        if prefix_by is not None:
            prefixes = adata_group.obs[prefix_by].unique().tolist()
            if len(prefixes) != 1:
                raise ValueError(
                    f"'{prefix_by}' is not constant within {group_by}={group!r}: "
                    f"found {prefixes}. Cannot build a '{{prefix}}_{{group}}' filename."
                )
            name = f"{prefixes[0]}_{group}"
        else:
            name = f"{group}"
        out_path = output_dir / f"{name}.zarr"
        click.echo(f"  Writing {name}: {adata_group.n_obs} cells → {out_path}")
        adata_group.write_zarr(out_path)
        written.append(out_path)

    click.echo(f"\nRemoving combined zarr: {input_path}")
    import shutil

    shutil.rmtree(input_path)

    click.echo(f"\nWrote {len(written)} per-{group_by} zarrs to {output_dir}")
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
    help="Directory to write per-group zarrs",
)
@click.option(
    "--group-by",
    default="experiment",
    show_default=True,
    help="obs column to group rows by (e.g. 'marker' for one zarr per marker)",
)
@click.option(
    "--prefix-by",
    default=None,
    help="obs column to prefix filenames as {prefix}_{group}.zarr "
    "(e.g. 'experiment' with --group-by marker gives {dataset}_{marker}.zarr)",
)
def main(input_path: Path, output_dir: Path, group_by: str, prefix_by: str | None) -> None:
    """Split a combined embeddings zarr into one zarr per group."""
    split_embeddings(input_path, output_dir, group_by=group_by, prefix_by=prefix_by)


if __name__ == "__main__":
    main()
