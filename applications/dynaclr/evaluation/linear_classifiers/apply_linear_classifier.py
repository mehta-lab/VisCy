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
    """Apply a trained linear classifier to new embeddings."""
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

    click.echo(f"\n Configuration loaded: {config}")
    click.echo(f"  Model: {inference_config.model_name}")
    click.echo(f"  Version: {inference_config.version}")
    click.echo(f"  Embeddings: {inference_config.embeddings_path}")
    click.echo(f"  Output: {inference_config.output_path}")

    try:
        pipeline, loaded_config = load_pipeline_from_wandb(
            wandb_project=inference_config.wandb_project,
            model_name=inference_config.model_name,
            version=inference_config.version,
            wandb_entity=inference_config.wandb_entity,
        )

        task = loaded_config["task"]

        click.echo(f"\nLoading embeddings from: {inference_config.embeddings_path}")
        adata = read_zarr(inference_config.embeddings_path)
        click.echo(f" Loaded embeddings: {adata.shape}")

        adata = predict_with_classifier(adata, pipeline, task)

        output_path = Path(inference_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nSaving predictions to: {output_path}")
        adata.write_zarr(output_path)
        click.echo(" Saved predictions")

        click.echo("\n" + format_predictions_markdown(adata, task))

        click.echo("\n Inference complete!")

    except Exception as e:
        click.echo(f"\n Inference failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
