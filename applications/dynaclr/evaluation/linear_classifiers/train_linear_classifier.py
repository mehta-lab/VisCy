"""CLI for training linear classifiers on cell embeddings.

Usage:
    dynaclr train-linear-classifier -c path/to/config.yaml
"""

from pathlib import Path

import click
from pydantic import ValidationError

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.linear_classifier import (
    load_and_combine_datasets,
    save_pipeline_to_wandb,
    train_linear_classifier,
)
from viscy_utils.evaluation.linear_classifier_config import (
    LinearClassifierTrainConfig,
)


def format_metrics_markdown(metrics: dict) -> str:
    """Format metrics as markdown table.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = ["## Classification Metrics", ""]

    train_metrics = {k.replace("train_", ""): v for k, v in metrics.items() if k.startswith("train_")}
    val_metrics = {k.replace("val_", ""): v for k, v in metrics.items() if k.startswith("val_")}

    if train_metrics:
        lines.append("### Training Set")
        lines.append("")
        lines.append(format_markdown_table(train_metrics).strip())
        lines.append("")

    if val_metrics:
        lines.append("### Validation Set")
        lines.append("")
        lines.append(format_markdown_table(val_metrics).strip())
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
    """Train a linear classifier on cell embeddings."""
    click.echo("=" * 60)
    click.echo("LINEAR CLASSIFIER TRAINING")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        train_config = LinearClassifierTrainConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\n Configuration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n Failed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo(f"\n Configuration loaded: {config}")
    click.echo(f"  Task: {train_config.task}")
    click.echo(f"  Input channel: {train_config.input_channel}")
    if train_config.marker:
        click.echo(f"  Marker: {train_config.marker}")
    click.echo(f"  Embedding model: {train_config.embedding_model_name} ({train_config.embedding_model_version})")
    click.echo(f"  W&B project: {train_config.wandb_project}")
    click.echo(f"  Datasets: {len(train_config.train_datasets)}")

    try:
        click.echo("\n" + "=" * 60)
        click.echo("LOADING TRAINING DATA")
        click.echo("=" * 60)

        combined_adata = load_and_combine_datasets(
            train_config.train_datasets,
            train_config.task,
        )

        classifier_params = {
            "max_iter": train_config.max_iter,
            "class_weight": train_config.class_weight,
            "solver": train_config.solver,
            "random_state": train_config.random_seed,
        }

        pipeline, metrics = train_linear_classifier(
            adata=combined_adata,
            task=train_config.task,
            use_scaling=train_config.use_scaling,
            use_pca=train_config.use_pca,
            n_pca_components=train_config.n_pca_components,
            classifier_params=classifier_params,
            split_train_data=train_config.split_train_data,
            random_seed=train_config.random_seed,
        )

        click.echo("\n" + format_metrics_markdown(metrics))

        full_config = train_config.model_dump()

        artifact_name = save_pipeline_to_wandb(
            pipeline=pipeline,
            metrics=metrics,
            config=full_config,
            wandb_project=train_config.wandb_project,
            wandb_entity=train_config.wandb_entity,
            tags=train_config.wandb_tags,
        )

        click.echo(f"\n Training complete! Artifact: {artifact_name}")

    except Exception as e:
        click.echo(f"\n Training failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
