"""CLI for applying trained linear classifiers to new embeddings.

Usage:
    dynaclr apply-linear-classifier -c path/to/config.yaml
"""

from pathlib import Path

import click
from anndata import read_zarr
from pydantic import ValidationError

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.linear_classifier import (
    load_pipeline_from_wandb,
    predict_with_classifier,
)
from viscy_utils.evaluation.linear_classifier_config import (
    LinearClassifierInferenceConfig,
)


def format_predictions_markdown(adata, task: str) -> str:
    """Format prediction summary as markdown.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with predictions.
    task : str
        Task name.

    Returns
    -------
    str
        Markdown-formatted summary.
    """
    lines = ["## Prediction Summary", ""]

    pred_col = f"predicted_{task}"
    if pred_col in adata.obs.columns:
        lines.append("### Class Distribution")
        lines.append("")
        counts = adata.obs[pred_col].value_counts().sort_index()
        class_counts = {str(k): int(v) for k, v in counts.items()}
        lines.append(format_markdown_table(class_counts, headers=["Class", "Count"]).strip())
        lines.append("")

        lines.append(f"**Total predictions:** {len(adata)}")
        lines.append("")

    proba_key = f"predicted_{task}_proba"
    if proba_key in adata.obsm.keys():
        lines.append(f"**Probability matrix shape:** {adata.obsm[proba_key].shape}")
        lines.append("")

    classes_key = f"predicted_{task}_classes"
    if classes_key in adata.uns.keys():
        lines.append(f"**Classes:** {', '.join(adata.uns[classes_key])}")
        lines.append("")

    artifact_key = f"classifier_{task}_artifact"
    if artifact_key in adata.uns.keys():
        provenance = {
            "Artifact": adata.uns[artifact_key],
        }
        id_key = f"classifier_{task}_id"
        if id_key in adata.uns.keys():
            provenance["Artifact ID"] = adata.uns[id_key]
        version_key = f"classifier_{task}_version"
        if version_key in adata.uns.keys():
            provenance["Artifact Version"] = adata.uns[version_key]
        lines.append(format_markdown_table(provenance, title="Classifier Provenance", headers=["Key", "Value"]).strip())
        lines.append("")

    return "\n".join(lines)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """Apply trained linear classifiers to embeddings."""
    click.echo("=" * 60)
    click.echo("LINEAR CLASSIFIER INFERENCE")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        inference_config = LinearClassifierInferenceConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\n Configuration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n Failed to load configuration: {e}", err=True)
        raise click.Abort()

    write_path = (
        Path(inference_config.output_path)
        if inference_config.output_path is not None
        else Path(inference_config.embeddings_path)
    )

    click.echo(f"\n Configuration loaded: {config}")
    click.echo(f"  W&B project: {inference_config.wandb_project}")
    click.echo(f"  Models: {len(inference_config.models)}")
    for spec in inference_config.models:
        click.echo(f"    - {spec.model_name} ({spec.version})")
    click.echo(f"  Embeddings: {inference_config.embeddings_path}")
    click.echo(f"  Output: {write_path}")

    try:
        click.echo(f"\nLoading embeddings from: {inference_config.embeddings_path}")
        adata = read_zarr(inference_config.embeddings_path)
        click.echo(f" Loaded embeddings: {adata.shape}")

        for i, spec in enumerate(inference_config.models, 1):
            click.echo(f"\n--- Model {i}/{len(inference_config.models)}: {spec.model_name} ---")

            pipeline, loaded_config, artifact_metadata = load_pipeline_from_wandb(
                wandb_project=inference_config.wandb_project,
                model_name=spec.model_name,
                version=spec.version,
                wandb_entity=inference_config.wandb_entity,
            )

            task = loaded_config["task"]
            marker = loaded_config.get("marker")
            task_key = f"{task}_{marker}" if marker else task

            if spec.include_wells:
                click.echo(f"  Well filter: {spec.include_wells}")

            adata = predict_with_classifier(
                adata,
                pipeline,
                task_key,
                artifact_metadata=artifact_metadata,
                include_wells=spec.include_wells,
            )

            click.echo(format_predictions_markdown(adata, task_key))

        write_path.parent.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nSaving predictions to: {write_path}")
        adata.write_zarr(write_path)
        click.echo(" Saved predictions")

        click.echo("\n Inference complete!")

    except Exception as e:
        click.echo(f"\n Inference failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
