"""CLI for appending annotation columns to per-experiment AnnData zarr stores.

Reads per-experiment annotation CSVs and writes task columns (e.g. infection_state,
organelle_state) directly into each zarr's obs. This persists ground truth labels
alongside the embeddings so downstream plots can color by annotation.

Called as a step in the Nextflow evaluation pipeline after split-embeddings.
Annotation sources are shared with the linear_classifiers step config.

Usage
-----
dynaclr append-annotations -c append_annotations.yaml
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import click

from dynaclr.evaluation.evaluate_config import AnnotationSource, TaskSpec
from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.annotation import load_annotation_anndata
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr


def append_annotations(
    embeddings_path: Path,
    annotations: list[AnnotationSource],
    tasks: list[TaskSpec],
) -> None:
    """Append annotation columns to per-experiment zarr obs.

    For each experiment in ``annotations``, loads the matching per-experiment
    zarr, joins task columns from the annotation CSV, and persists the
    updated obs back to zarr.

    When ``tasks`` is empty, auto-discovers task columns from the
    annotation CSV (every column except the join keys ``fov_name``, ``t``,
    ``track_id``, ``id``). This supports Wave-2 datasets that publish
    annotations independently of any LC training task list.

    Parameters
    ----------
    embeddings_path : Path
        Directory containing per-experiment zarrs named ``{experiment}.zarr``.
    annotations : list[AnnotationSource]
        Per-experiment annotation CSV sources. Each entry maps an experiment
        name to a CSV path with task columns.
    tasks : list[TaskSpec]
        Tasks to join (e.g. infection_state, organelle_state). Empty list →
        auto-discover from the CSV.
    """
    import pandas as pd

    explicit_tasks = [t.task for t in tasks]
    join_keys = {"fov_name", "t", "track_id", "id"}

    if explicit_tasks:
        click.echo(f"Appending annotations for {len(annotations)} experiments, tasks: {explicit_tasks}")
    else:
        click.echo(
            f"Appending annotations for {len(annotations)} experiments, "
            "tasks auto-discovered per-CSV (all non-join-key columns)"
        )

    for ann_src in annotations:
        experiment = ann_src.experiment
        zarr_path = embeddings_path / f"{experiment}.zarr"

        if not zarr_path.exists():
            click.echo(f"  [{experiment}] zarr not found, skipping: {zarr_path}", err=True)
            continue

        ann_path = Path(ann_src.path)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {ann_src.path}")

        # Resolve task list: explicit if provided, else discover from this CSV.
        if explicit_tasks:
            task_names = explicit_tasks
        else:
            csv_cols = pd.read_csv(ann_path, nrows=0).columns.tolist()
            task_names = [c for c in csv_cols if c not in join_keys]
            click.echo(f"  [{experiment}] discovered tasks from CSV: {task_names}")

        click.echo(f"\n  [{experiment}]")
        adata = ad.read_zarr(zarr_path)
        click.echo(f"    Loaded {adata.n_obs} cells")

        n_joined = 0
        for task_name in task_names:
            try:
                adata = load_annotation_anndata(adata, str(ann_path), task_name)
                n_valid = int(adata.obs[task_name].notna().sum())
                click.echo(f"    {task_name}: {n_valid}/{adata.n_obs} labeled")
                n_joined += 1
            except KeyError:
                click.echo(f"    {task_name}: not in {ann_path.name}, skipping")

        if n_joined == 0:
            click.echo(f"    No tasks found in {ann_path.name}, skipping zarr write")
            continue

        append_to_anndata_zarr(zarr_path, obs=adata.obs)
        click.echo(f"    Saved obs to {zarr_path}")

    click.echo("\nDone.")


class _AppendAnnotationsConfig:
    def __init__(self, raw: dict):
        self.embeddings_path = Path(raw["embeddings_path"])
        self.annotations = [AnnotationSource(**a) for a in raw["annotations"]]
        self.tasks = [TaskSpec(**t) for t in raw["tasks"]]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Append annotation columns to per-experiment AnnData zarr stores."""
    click.echo("=" * 60)
    click.echo("APPEND ANNOTATIONS")
    click.echo("=" * 60)
    raw = load_config(config)
    cfg = _AppendAnnotationsConfig(raw)
    append_annotations(cfg.embeddings_path, cfg.annotations, cfg.tasks)


if __name__ == "__main__":
    main()
