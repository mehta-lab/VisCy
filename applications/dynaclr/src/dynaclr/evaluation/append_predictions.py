"""CLI for applying saved linear classifiers to per-experiment AnnData zarr stores.

Reads the pipelines manifest written by ``dynaclr run-linear-classifiers``,
applies each saved classifier to ALL cells with the matching marker in each
per-experiment zarr, and writes predictions back to obs/obsm/uns.

This enables plots colored by predicted labels (e.g. predicted_infection_state)
for every cell, including unannotated ones.

Called as a step in the Nextflow evaluation pipeline after linear classifiers
have been trained (LINEAR_CLASSIFIERS step).

Usage
-----
dynaclr append-predictions -c append_predictions.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import click
import joblib
import numpy as np

from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr


def append_predictions(
    embeddings_path: Path,
    pipelines_dir: Path,
) -> None:
    """Apply saved classifiers to all cells and write predictions to zarrs.

    ``pipelines_dir`` may be a ``latest`` symlink into the central LC registry
    (e.g. ``/hpc/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/latest``).
    The symlink is resolved **once** at startup so the whole run is consistent
    even if a new version is published mid-run.

    For each per-experiment zarr, loads all saved classifier pipelines and
    applies each one to cells with the matching marker. Results are merged
    per task (one ``predicted_{task}`` column per task regardless of how
    many marker-specific classifiers contributed), then persisted to zarr.

    Parameters
    ----------
    embeddings_path : Path
        Directory containing per-experiment zarrs named ``{experiment}.zarr``.
    pipelines_dir : Path
        Directory containing ``manifest.json`` and ``{task}_{marker}.joblib``
        pipeline files. If this is the ``latest`` symlink, it is resolved
        to a ``vN/`` target before loading.
    """
    resolved = pipelines_dir.resolve()
    version_tag = resolved.name
    # Registry layout: {registry_root}/{model_name}/vN/. Two levels up from
    # vN is the registry root; one level up is the per-model dir (== model
    # name). This is the feature_space identifier.
    feature_space = resolved.parent.name if resolved.parent != resolved else "<unversioned>"
    click.echo(f"LC pipelines: {pipelines_dir} -> {resolved}")
    click.echo(f"  feature_space={feature_space}  version={version_tag}")

    manifest_path = resolved / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pipeline manifest not found: {manifest_path}. Run dynaclr run-linear-classifiers first."
        )

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # New-format manifest: dict with {trained_at, pipelines: [...]}.
    if not isinstance(manifest_data, dict) or "pipelines" not in manifest_data:
        raise ValueError(
            f"Manifest at {manifest_path} is not in the expected format "
            "(dict with 'pipelines' key). Re-train with the current "
            "run-linear-classifiers to produce a compatible bundle."
        )
    manifest_entries = manifest_data["pipelines"]
    trained_at = manifest_data.get("trained_at", "<unknown>")
    click.echo(f"  trained_at={trained_at}")

    if not manifest_entries:
        click.echo("No pipelines in manifest, nothing to do.")
        return

    click.echo(f"  {len(manifest_entries)} pipeline(s):")
    for entry in manifest_entries:
        click.echo(f"    {entry['task']} / marker={entry['marker_filter']}")

    manifest_markers = {e["marker_filter"] for e in manifest_entries}

    zarr_paths = sorted(embeddings_path.glob("*.zarr"))
    if not zarr_paths:
        raise FileNotFoundError(f"No .zarr files found in {embeddings_path}")

    click.echo(f"\nProcessing {len(zarr_paths)} per-experiment zarr(s)...")

    for zarr_path in zarr_paths:
        click.echo(f"\n  {zarr_path.stem}")
        adata = ad.read_zarr(zarr_path)
        zarr_markers = set(adata.obs["marker"].unique().tolist())
        click.echo(f"    {adata.n_obs} cells, markers: {sorted(zarr_markers)}")

        # Coverage report: which zarr markers are predictable from this bundle?
        covered = sorted(zarr_markers & manifest_markers)
        missing = sorted(zarr_markers - manifest_markers)
        click.echo(
            f"    LC coverage: {len(covered)}/{len(zarr_markers)} markers predictable"
            + (f"; missing: {missing}" if missing else "")
        )

        # Group manifest entries by task
        tasks_seen: set[str] = {entry["task"] for entry in manifest_entries}

        new_obsm: dict[str, np.ndarray] = {}

        for task in sorted(tasks_seen):
            task_entries = [e for e in manifest_entries if e["task"] == task]

            first_pipeline = joblib.load(resolved / task_entries[0]["path"])
            n_classes = len(first_pipeline.classifier.classes_)
            classes = first_pipeline.classifier.classes_.tolist()

            all_pred = np.full(adata.n_obs, np.nan, dtype=object)
            all_proba = np.full((adata.n_obs, n_classes), np.nan)

            for entry in task_entries:
                marker_filter = entry["marker_filter"]
                pipeline_path = resolved / entry["path"]

                if not pipeline_path.exists():
                    click.echo(f"    Pipeline not found: {pipeline_path}, skipping", err=True)
                    continue

                marker_mask = (adata.obs["marker"] == marker_filter).to_numpy()
                n_matching = int(marker_mask.sum())
                if n_matching == 0:
                    click.echo(f"    {task}/{marker_filter}: no matching cells, skipping")
                    continue

                pipeline = joblib.load(pipeline_path)
                adata_subset = adata[marker_mask]

                X_subset = adata_subset.X if isinstance(adata_subset.X, np.ndarray) else adata_subset.X.toarray()
                preds = pipeline.predict(X_subset)
                probas = pipeline.predict_proba(X_subset)

                all_pred[marker_mask] = preds
                all_proba[marker_mask] = probas
                click.echo(f"    {task}/{marker_filter}: predicted {n_matching} cells")

            adata.obs[f"predicted_{task}"] = all_pred
            adata.uns[f"predicted_{task}_classes"] = classes
            adata.uns[f"predicted_{task}_lc_version"] = version_tag
            adata.uns[f"predicted_{task}_lc_feature_space"] = feature_space
            adata.uns[f"predicted_{task}_lc_path"] = str(resolved)
            new_obsm[f"predicted_{task}_proba"] = all_proba

        if not new_obsm:
            click.echo("    No predictions written (no matching markers)")
            continue

        append_to_anndata_zarr(zarr_path, obs=adata.obs, obsm=new_obsm, uns=adata.uns)
        click.echo(f"    Saved predictions to {zarr_path}")

    click.echo("\nDone.")


class _AppendPredictionsConfig:
    def __init__(self, raw: dict):
        self.embeddings_path = Path(raw["embeddings_path"])
        self.pipelines_dir = Path(raw["pipelines_dir"])


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Apply saved linear classifiers to per-experiment zarrs and write predictions."""
    click.echo("=" * 60)
    click.echo("APPEND PREDICTIONS")
    click.echo("=" * 60)
    raw = load_config(config)
    cfg = _AppendPredictionsConfig(raw)
    append_predictions(cfg.embeddings_path, cfg.pipelines_dir)


if __name__ == "__main__":
    main()
