"""Split a combined embeddings zarr into one zarr per group.

Reads the combined embeddings.zarr produced by the predict step, groups rows
by obs[group_by] (``experiment`` by default), and writes one AnnData zarr per
group under output_dir/{group}.zarr.

Usage
-----
dynaclr split-embeddings -c split.yaml

Or with inline arguments:

dynaclr split-embeddings --input /path/to/embeddings.zarr --output-dir /path/to/embeddings/

Split by a different column (e.g. one zarr per marker):

dynaclr split-embeddings --input /path/to/embeddings.zarr --output-dir /path/to/embeddings/ --group-by marker

Route into the dataset-centric tree (parquet spine → same layout as predict-triplet):

dynaclr split-embeddings --input /path/to/embeddings.zarr --route-by-dataset \\
    --model-family M --run R --ckpt-name C [--keep-combined]
"""

from __future__ import annotations

from pathlib import Path

import click

from dynaclr.evaluation.paths import DATASETS_ROOT, embedding_store


def split_embeddings(
    input_path: Path,
    output_dir: Path | None = None,
    group_by: str = "experiment",
    prefix_by: str | None = None,
    *,
    route_by_dataset: bool = False,
    model_family: str | None = None,
    run: str | None = None,
    ckpt_name: str | None = None,
    datasets_root: str | Path = DATASETS_ROOT,
    keep_combined: bool = False,
) -> list[Path]:
    """Split combined embeddings zarr into one zarr per group.

    Two output layouts:

    - **Flat** (default): one zarr per ``group_by`` value under ``output_dir``,
      named ``{group}.zarr`` or ``{prefix}_{group}.zarr`` when ``prefix_by`` is
      set.
    - **Dataset-centric** (``route_by_dataset=True``): group by
      ``experiment`` x ``marker`` and route each into the canonical tree
      ``{dataset}/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr``
      via :func:`dynaclr.evaluation.paths.embedding_store`, so the parquet spine
      lands in the same layout as ``predict-triplet``.

    Parameters
    ----------
    input_path : Path
        Path to the combined embeddings zarr (AnnData format).
    output_dir : Path or None
        Directory for flat output. Required unless ``route_by_dataset`` is set.
    group_by : str, optional
        obs column to group rows by (flat mode). By default ``"experiment"``.
    prefix_by : str or None, optional
        obs column whose value prefixes each flat filename as
        ``{prefix}_{group}.zarr``. Must be constant within each group.
    route_by_dataset : bool, optional
        If ``True``, ignore ``output_dir``/``group_by``/``prefix_by`` and write
        the dataset-centric tree keyed by obs ``experiment`` and ``marker``.
    model_family, run, ckpt_name : str or None
        Provenance identity for the dataset-centric tree. Required when
        ``route_by_dataset`` is set.
    datasets_root : str or Path, optional
        Base under which datasets live (dataset-centric mode). Defaults to
        :data:`dynaclr.evaluation.paths.DATASETS_ROOT`.
    keep_combined : bool, optional
        If ``False`` (default), remove the combined ``input_path`` after
        splitting. Set ``True`` to keep it so downstream eval can re-run
        without re-predicting.

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

    if route_by_dataset:
        written = _write_dataset_centric(
            adata,
            model_family=model_family,
            run=run,
            ckpt_name=ckpt_name,
            datasets_root=datasets_root,
        )
    else:
        if output_dir is None:
            raise ValueError("--output-dir is required unless --route-by-dataset is set.")
        written = _write_flat(adata, output_dir, group_by=group_by, prefix_by=prefix_by)

    if keep_combined:
        click.echo(f"\nKeeping combined zarr: {input_path}")
    else:
        click.echo(f"\nRemoving combined zarr: {input_path}")
        import shutil

        shutil.rmtree(input_path)

    click.echo(f"\nWrote {len(written)} zarrs")
    return written


def _require_obs_columns(adata, columns: list[str]) -> None:
    """Raise if any required obs column is missing."""
    for col in columns:
        if col not in adata.obs.columns:
            raise ValueError(
                f"embeddings zarr obs is missing '{col}' column. "
                f"Available columns: {sorted(adata.obs.columns)}. "
                "Re-run the predict step with the updated pipeline to include metadata."
            )


def _write_flat(adata, output_dir: Path, *, group_by: str, prefix_by: str | None) -> list[Path]:
    """Write one zarr per ``group_by`` value under ``output_dir`` (flat layout)."""
    _require_obs_columns(adata, [c for c in (group_by, prefix_by) if c])

    groups = adata.obs[group_by].unique().tolist()
    click.echo(f"  {len(groups)} {group_by} groups: {groups}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for group in groups:
        adata_group = adata[adata.obs[group_by] == group].copy()
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
    return written


def _write_dataset_centric(
    adata,
    *,
    model_family: str | None,
    run: str | None,
    ckpt_name: str | None,
    datasets_root: str | Path,
) -> list[Path]:
    """Route each (experiment, marker) group into the dataset-centric tree."""
    if not (model_family and run and ckpt_name):
        raise ValueError("--route-by-dataset requires --model-family, --run, and --ckpt-name.")
    _require_obs_columns(adata, ["experiment", "marker"])

    written: list[Path] = []
    pairs = adata.obs[["experiment", "marker"]].drop_duplicates().itertuples(index=False)
    for dataset, marker in pairs:
        mask = (adata.obs["experiment"] == dataset) & (adata.obs["marker"] == marker)
        adata_group = adata[mask].copy()
        out_path = embedding_store(str(dataset), model_family, run, ckpt_name, str(marker), datasets_root=datasets_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"  Writing {dataset}/{marker}: {adata_group.n_obs} cells → {out_path}")
        adata_group.write_zarr(out_path)
        written.append(out_path)
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
    default=None,
    help="Directory to write per-group zarrs (flat layout; required unless --route-by-dataset).",
)
@click.option(
    "--group-by",
    default="experiment",
    show_default=True,
    help="obs column to group rows by (flat layout; e.g. 'marker' for one zarr per marker)",
)
@click.option(
    "--prefix-by",
    default=None,
    help="obs column to prefix filenames as {prefix}_{group}.zarr "
    "(e.g. 'experiment' with --group-by marker gives {dataset}_{marker}.zarr)",
)
@click.option(
    "--route-by-dataset",
    is_flag=True,
    default=False,
    help="Write the dataset-centric tree "
    "<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr "
    "keyed by obs experiment x marker (ignores --output-dir/--group-by/--prefix-by).",
)
@click.option("--model-family", default=None, help="Model family (required with --route-by-dataset).")
@click.option("--run", default=None, help="Training run/version (required with --route-by-dataset).")
@click.option("--ckpt-name", default=None, help="Checkpoint label (required with --route-by-dataset).")
@click.option(
    "--datasets-root",
    type=click.Path(path_type=Path),
    default=DATASETS_ROOT,
    show_default=True,
    help="Base under which datasets live (with --route-by-dataset).",
)
@click.option(
    "--keep-combined",
    is_flag=True,
    default=False,
    help="Keep the combined input zarr instead of deleting it after splitting.",
)
def main(
    input_path: Path,
    output_dir: Path | None,
    group_by: str,
    prefix_by: str | None,
    route_by_dataset: bool,
    model_family: str | None,
    run: str | None,
    ckpt_name: str | None,
    datasets_root: Path,
    keep_combined: bool,
) -> None:
    """Split a combined embeddings zarr into one zarr per group."""
    split_embeddings(
        input_path,
        output_dir,
        group_by=group_by,
        prefix_by=prefix_by,
        route_by_dataset=route_by_dataset,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        keep_combined=keep_combined,
    )


if __name__ == "__main__":
    main()
