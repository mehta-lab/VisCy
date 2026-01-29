"""
CLI tool for evaluating temporal smoothness of representation learning models.

This tool computes temporal smoothness metrics for embeddings from multiple models
and outputs a markdown-formatted comparison table. It uses the memory-optimized
smoothness computation from viscy.representation.evaluation.smoothness.

Usage
-----
python -m applications.DynaCLR.evaluation.evaluate_smoothness --config smoothness_config.yaml
"""

import gc
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from viscy.representation.evaluation.smoothness import compute_embeddings_smoothness

from .utils import (
    format_comparison_summary,
    format_results_table,
    load_config,
    load_embedding,
    save_results,
    validate_embedding,
    validate_smoothness_config,
)

matplotlib.use("Agg")


@click.command("evaluate-smoothness")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """
    Evaluate temporal smoothness of representation learning models.

    This tool loads embeddings from multiple models, computes temporal smoothness
    metrics, and outputs a comparison table in markdown format.

    Metrics computed:
    - Smoothness Score: Ratio of adjacent frame distance to random frame distance (lower is better)
    - Dynamic Range: Separation between random and adjacent peaks (higher is better)
    - Adjacent/Random frame statistics: mean, std, median, peak

    The tool uses a memory-optimized computation that avoids creating the full
    pairwise distance matrix, reducing memory usage from 100+ GB to ~1GB for large datasets.
    """
    # Load and validate config
    click.echo("Loading configuration...")
    config_dict = load_config(config)
    validate_smoothness_config(config_dict)

    # Extract config parameters
    models = config_dict["models"]
    eval_config = config_dict["evaluation"]

    distance_metric = eval_config.get("distance_metric", "cosine")
    time_offsets = eval_config.get("time_offsets", [1])
    output_dir = Path(eval_config["output_dir"])
    eval_config.get("output_format", "markdown")
    save_plots = eval_config.get("save_plots", True)
    save_distributions = eval_config.get("save_distributions", False)
    use_optimized = eval_config.get("use_optimized", True)
    verbose = eval_config.get("verbose", False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each model
    all_results = {}
    all_distributions = {}

    for i, model_config in enumerate(models, 1):
        model_path = Path(model_config["path"])
        model_label = model_config["label"]

        click.echo(f"\nProcessing {i}/{len(models)}: {model_label}...")

        try:
            # Load embedding
            if verbose:
                click.echo(f"  Loading embeddings from {model_path}...")
            features_ad = load_embedding(model_path)
            validate_embedding(features_ad)

            if verbose:
                click.echo(
                    f"  Loaded {features_ad.shape[0]:,} samples with {features_ad.shape[1]} features"
                )

            # Compute smoothness
            stats, distributions, _ = compute_embeddings_smoothness(
                features_ad,
                distance_metric=distance_metric,
                time_offsets=time_offsets,
                use_optimized=use_optimized,
                verbose=verbose,
            )

            all_results[model_label] = stats
            all_distributions[model_label] = distributions

            # Save individual results
            save_results(
                stats,
                output_dir / f"{model_label}_smoothness_stats.csv",
                format="csv",
            )

            if save_distributions:
                # Save distributions as numpy arrays
                np.save(
                    output_dir / f"{model_label}_adjacent_distribution.npy",
                    distributions["adjacent_frame_distribution"],
                )
                np.save(
                    output_dir / f"{model_label}_random_distribution.npy",
                    distributions["random_frame_distribution"],
                )

            # Create plot
            if save_plots:
                if verbose:
                    click.echo("  Creating plots...")
                _create_smoothness_plot(
                    distributions,
                    stats,
                    model_label,
                    distance_metric,
                    output_dir,
                )

            click.echo(f"  ✓ {model_label} processed successfully")

            # Cleanup
            del features_ad, stats, distributions
            gc.collect()

        except Exception as e:
            click.echo(f"  ✗ Error processing {model_label}: {e}", err=True)
            continue

    # Output comparison table
    if not all_results:
        click.echo("\n✗ No models were successfully processed.", err=True)
        return

    click.echo("\n" + "=" * 80)
    click.echo("TEMPORAL SMOOTHNESS EVALUATION")
    click.echo("=" * 80 + "\n")

    # Create markdown table
    columns = [
        "smoothness_score",
        "dynamic_range",
        "adjacent_frame_mean",
        "adjacent_frame_peak",
        "random_frame_mean",
        "random_frame_peak",
    ]

    table = format_results_table(all_results, columns)
    click.echo(table)

    # Add metric interpretation
    click.echo("\n**Metrics Interpretation**")
    click.echo("- Smoothness Score: Lower is better (adjacent frames are closer)")
    click.echo(
        "- Dynamic Range: Higher is better (more separation between adjacent and random)"
    )

    # Add best model summaries
    click.echo("\n**Best Models**")
    click.echo(
        format_comparison_summary(all_results, "smoothness_score", lower_is_better=True)
    )
    click.echo(
        format_comparison_summary(all_results, "dynamic_range", lower_is_better=False)
    )

    click.echo("\n" + "=" * 80)
    click.echo(f"✓ All {len(all_results)} models processed successfully")
    click.echo(f"✓ Results saved to: {output_dir}")
    click.echo("=" * 80)

    # Save combined results
    combined_df = pd.DataFrame(all_results).T
    combined_df.index.name = "model"
    combined_df.to_csv(output_dir / "combined_smoothness_stats.csv")


def _create_smoothness_plot(
    distributions: dict,
    stats: dict,
    label: str,
    distance_metric: str,
    output_dir: Path,
) -> None:
    """
    Create and save smoothness distribution plots.

    Parameters
    ----------
    distributions : dict
        Dictionary containing adjacent and random frame distributions
    stats : dict
        Dictionary containing computed statistics
    label : str
        Model label for plot title
    distance_metric : str
        Distance metric used
    output_dir : Path
        Output directory for saving plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use colorblind-friendly colors (blue and orange)
    sns.histplot(
        distributions["adjacent_frame_distribution"],
        bins=30,
        kde=True,
        color="#1f77b4",  # Blue
        alpha=0.5,
        stat="density",
        label="Adjacent Frame",
        ax=ax,
    )
    sns.histplot(
        distributions["random_frame_distribution"],
        bins=30,
        kde=True,
        color="#ff7f0e",  # Orange
        alpha=0.5,
        stat="density",
        label="Random Sample",
        ax=ax,
    )

    # Add peak markers
    ax.axvline(
        x=stats["adjacent_frame_peak"],
        color="#1f77b4",
        linestyle="--",
        alpha=0.8,
        label="Adjacent Peak",
    )
    ax.axvline(
        x=stats["random_frame_peak"],
        color="#ff7f0e",
        linestyle="--",
        alpha=0.8,
        label="Random Peak",
    )

    ax.set_xlabel(f"{distance_metric.capitalize()} Distance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(
        f"{label}\nSmoothness: {stats['smoothness_score']:.3f}, Dynamic Range: {stats['dynamic_range']:.3f}"
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"{label}_smoothness.pdf", dpi=300)
    plt.savefig(output_dir / f"{label}_smoothness.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
