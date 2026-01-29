"""
CLI tool for comparing previously saved evaluation results.

This tool loads CSV results from multiple evaluation runs and creates
comparison tables and summaries.

Usage
-----
python -m applications.DynaCLR.evaluation.compare_models --config compare_config.yaml
"""

from pathlib import Path

import click
import pandas as pd

from .utils import format_comparison_summary, format_results_table, load_config


@click.command("compare-models")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """
    Compare previously saved evaluation results.

    This tool loads CSV result files from multiple models and creates
    a comparison table. Useful for comparing results from different
    evaluation runs or combining results from different metrics.
    """
    # Load config
    click.echo("Loading configuration...")
    config_dict = load_config(config)

    if "result_files" not in config_dict:
        click.echo("✗ Config must contain 'result_files' section", err=True)
        return

    result_files = config_dict["result_files"]
    comparison_config = config_dict.get("comparison", {})

    metrics = comparison_config.get(
        "metrics",
        [
            "smoothness_score",
            "dynamic_range",
            "adjacent_frame_mean",
            "adjacent_frame_peak",
            "random_frame_mean",
            "random_frame_peak",
        ],
    )
    output_format = comparison_config.get("output_format", "markdown")

    # Load all result files
    all_results = {}

    for file_config in result_files:
        file_path = Path(file_config["path"])
        label = file_config["label"]

        if not file_path.exists():
            click.echo(f"⚠ Warning: Result file not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Convert first row to dictionary
                all_results[label] = df.iloc[0].to_dict()
            else:
                click.echo(f"⚠ Warning: Empty result file: {file_path}")
        except Exception as e:
            click.echo(f"⚠ Warning: Error reading {file_path}: {e}")
            continue

    if not all_results:
        click.echo("✗ No valid result files were loaded", err=True)
        return

    # Output comparison table
    click.echo("\n" + "=" * 80)
    click.echo("MODEL COMPARISON")
    click.echo("=" * 80 + "\n")

    # Create markdown table
    table = format_results_table(all_results, metrics)
    click.echo(table)

    # Add metric interpretation for smoothness metrics
    if "smoothness_score" in metrics or "dynamic_range" in metrics:
        click.echo("\n**Metrics Interpretation**")
        if "smoothness_score" in metrics:
            click.echo(
                "- Smoothness Score: Lower is better (adjacent frames are closer)"
            )
        if "dynamic_range" in metrics:
            click.echo(
                "- Dynamic Range: Higher is better (more separation between adjacent and random)"
            )

    # Add best model summaries for key metrics
    click.echo("\n**Best Models**")
    for metric in metrics:
        if metric == "smoothness_score":
            click.echo(
                format_comparison_summary(all_results, metric, lower_is_better=True)
            )
        elif metric == "dynamic_range":
            click.echo(
                format_comparison_summary(all_results, metric, lower_is_better=False)
            )

    click.echo("\n" + "=" * 80)
    click.echo(f"✓ Compared {len(all_results)} models")
    click.echo("=" * 80)

    # Save combined results if output path specified
    output_path = comparison_config.get("output_path")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df = pd.DataFrame(all_results).T
        combined_df.index.name = "model"

        if output_format == "csv":
            combined_df.to_csv(output_path)
            click.echo(f"✓ Results saved to: {output_path}")
        elif output_format == "json":
            combined_df.to_json(output_path, orient="index", indent=2)
            click.echo(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
