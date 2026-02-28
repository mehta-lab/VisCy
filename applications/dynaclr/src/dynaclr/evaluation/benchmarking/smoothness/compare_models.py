"""
CLI tool for comparing previously saved evaluation results.

Loads CSV results from multiple evaluation runs and creates
comparison tables and summaries.

Usage
-----
dynaclr compare-models -c compare_config.yaml
"""

from pathlib import Path

import click
import pandas as pd

from viscy_utils.cli_utils import format_markdown_table, load_config

from .config import CompareModelsConfig
from .utils import format_comparison_summary


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """Compare previously saved evaluation results."""
    click.echo("Loading configuration...")
    raw_config = load_config(config)
    config = CompareModelsConfig(
        result_files=raw_config.get("result_files", []),
        **raw_config.get("comparison", {}),
    )

    all_results = {}

    for file_entry in config.result_files:
        file_path = Path(file_entry.path)
        label = file_entry.label

        if not file_path.exists():
            click.echo(f"Warning: Result file not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                all_results[label] = df.iloc[0].to_dict()
            else:
                click.echo(f"Warning: Empty result file: {file_path}")
        except Exception as e:
            click.echo(f"Warning: Error reading {file_path}: {e}")
            continue

    if not all_results:
        click.echo("No valid result files were loaded", err=True)
        return

    # Build comparison table
    table_data = [
        {"model": label, **{col: metrics.get(col) for col in config.metrics}} for label, metrics in all_results.items()
    ]

    click.echo("\n" + "=" * 80)
    click.echo("MODEL COMPARISON")
    click.echo("=" * 80 + "\n")
    click.echo(format_markdown_table(table_data, headers=["model"] + config.metrics))

    if "smoothness_score" in config.metrics or "dynamic_range" in config.metrics:
        click.echo("**Metrics Interpretation**")
        if "smoothness_score" in config.metrics:
            click.echo("- Smoothness Score: Lower is better (adjacent frames are closer)")
        if "dynamic_range" in config.metrics:
            click.echo("- Dynamic Range: Higher is better (more separation between adjacent and random)")

    click.echo("\n**Best Models**")
    for metric in config.metrics:
        if metric == "smoothness_score":
            click.echo(format_comparison_summary(all_results, metric, lower_is_better=True))
        elif metric == "dynamic_range":
            click.echo(format_comparison_summary(all_results, metric, lower_is_better=False))

    click.echo("\n" + "=" * 80)
    click.echo(f"Compared {len(all_results)} models")
    click.echo("=" * 80)

    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df = pd.DataFrame(all_results).T
        combined_df.index.name = "model"

        if config.output_format == "csv":
            combined_df.to_csv(output_path)
            click.echo(f"Results saved to: {output_path}")
        elif config.output_format == "json":
            combined_df.to_json(output_path, orient="index", indent=2)
            click.echo(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
