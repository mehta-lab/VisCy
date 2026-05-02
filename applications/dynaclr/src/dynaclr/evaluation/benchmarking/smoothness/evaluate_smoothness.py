"""
CLI tool for evaluating temporal smoothness of representation learning models.

Computes temporal smoothness metrics for embeddings from multiple models
and outputs a markdown-formatted comparison table.

Usage
-----
dynaclr evaluate-smoothness -c smoothness_config.yaml
"""

import gc
from pathlib import Path

import anndata as ad
import click
import numpy as np
import pandas as pd

from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.smoothness import compute_embeddings_smoothness

from .config import SmoothnessEvalConfig
from .utils import format_comparison_summary, save_results, validate_embedding


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path):
    """Evaluate temporal smoothness of representation learning models."""
    click.echo("Loading configuration...")
    raw_config = load_config(config)
    config = SmoothnessEvalConfig(
        **raw_config.pop("evaluation", {}),
        models=raw_config.get("models", []),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_distributions = {}

    for i, model_entry in enumerate(config.models, 1):
        model_path = Path(model_entry.path)
        model_label = model_entry.label
        experiment_name = model_path.stem

        click.echo(f"\nProcessing {i}/{len(config.models)}: {model_label}...")

        try:
            features_ad = ad.read_zarr(model_path)
            validate_embedding(features_ad)

            if config.verbose:
                click.echo(f"  Loaded {features_ad.shape[0]:,} samples with {features_ad.shape[1]} features")

            group_col = config.group_by
            if group_col and group_col in features_ad.obs.columns:
                groups = features_ad.obs[group_col].unique().tolist()
                click.echo(f"  Computing smoothness per {group_col}: {groups}")

                per_group_rows = []
                group_stats_list = []
                group_distributions = {}

                for group_val in groups:
                    mask = features_ad.obs[group_col] == group_val
                    group_ad = features_ad[mask].copy()

                    if config.verbose:
                        click.echo(f"    {group_col}={group_val}: {group_ad.shape[0]:,} cells")

                    g_stats, g_dists, _ = compute_embeddings_smoothness(
                        group_ad,
                        distance_metric=config.distance_metric,
                        verbose=config.verbose,
                    )
                    per_group_rows.append({group_col: group_val, **g_stats})
                    group_stats_list.append(g_stats)
                    group_distributions[group_val] = g_dists

                    if config.save_plots:
                        _create_smoothness_plot(
                            g_dists,
                            g_stats,
                            f"{model_label}_{experiment_name}_{group_val}",
                            config.distance_metric,
                            output_dir,
                        )

                per_group_df = pd.DataFrame(per_group_rows)
                per_group_df.insert(0, "experiment", experiment_name)
                per_group_df.to_csv(
                    output_dir / f"{model_label}_{experiment_name}_per_{group_col}_smoothness.csv", index=False
                )
                click.echo(f"  Per-{group_col} stats saved.")

                # Aggregate: mean ± std across groups
                metric_cols = [c for c in per_group_df.columns if c != group_col]
                agg_means = per_group_df[metric_cols].mean()
                agg_stds = per_group_df[metric_cols].std()
                stats = agg_means.to_dict()
                stats_std = {f"{k}_std": v for k, v in agg_stds.to_dict().items()}
                stats.update(stats_std)

                # Concatenate distributions across groups for the combined plot
                distributions = {
                    "adjacent_frame_distribution": np.concatenate(
                        [d["adjacent_frame_distribution"] for d in group_distributions.values()]
                    ),
                    "random_frame_distribution": np.concatenate(
                        [d["random_frame_distribution"] for d in group_distributions.values()]
                    ),
                }
            else:
                stats, distributions, _ = compute_embeddings_smoothness(
                    features_ad,
                    distance_metric=config.distance_metric,
                    verbose=config.verbose,
                )

            all_results[model_label] = stats
            all_distributions[model_label] = distributions

            save_results(
                stats,
                output_dir / f"{model_label}_{experiment_name}_smoothness_stats.csv",
                format="csv",
            )

            if config.save_distributions:
                np.save(
                    output_dir / f"{model_label}_{experiment_name}_adjacent_distribution.npy",
                    distributions["adjacent_frame_distribution"],
                )
                np.save(
                    output_dir / f"{model_label}_{experiment_name}_random_distribution.npy",
                    distributions["random_frame_distribution"],
                )

            if config.save_plots:
                if config.verbose:
                    click.echo("  Creating plots...")
                _create_smoothness_plot(
                    distributions,
                    stats,
                    f"{model_label}_{experiment_name}",
                    config.distance_metric,
                    output_dir,
                )

            click.echo(f"  {model_label} processed successfully")

            del features_ad, stats, distributions
            gc.collect()

        except Exception as e:
            click.echo(f"  Error processing {model_label}: {e}", err=True)
            continue

    if not all_results:
        click.echo("\nNo models were successfully processed.", err=True)
        return

    # Build comparison table
    columns = [
        "smoothness_score",
        "dynamic_range",
        "adjacent_frame_mean",
        "adjacent_frame_peak",
        "random_frame_mean",
        "random_frame_peak",
    ]

    table_data = [
        {"model": label, **{col: metrics.get(col) for col in columns}} for label, metrics in all_results.items()
    ]

    click.echo("\n" + "=" * 80)
    click.echo("TEMPORAL SMOOTHNESS EVALUATION")
    click.echo("=" * 80 + "\n")
    click.echo(format_markdown_table(table_data, headers=["model"] + columns))

    click.echo("**Metrics Interpretation**")
    click.echo("- Smoothness Score: Lower is better (adjacent frames are closer)")
    click.echo("- Dynamic Range: Higher is better (more separation between adjacent and random)")

    click.echo("\n**Best Models**")
    click.echo(format_comparison_summary(all_results, "smoothness_score", lower_is_better=True))
    click.echo(format_comparison_summary(all_results, "dynamic_range", lower_is_better=False))

    click.echo("\n" + "=" * 80)
    click.echo(f"All {len(all_results)} models processed successfully")
    click.echo(f"Results saved to: {output_dir}")
    click.echo("=" * 80)

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
    """Create and save smoothness distribution plots."""
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        distributions["adjacent_frame_distribution"],
        bins=30,
        kde=True,
        color="#1f77b4",
        alpha=0.5,
        stat="density",
        label="Adjacent Frame",
        ax=ax,
    )
    sns.histplot(
        distributions["random_frame_distribution"],
        bins=30,
        kde=True,
        color="#ff7f0e",
        alpha=0.5,
        stat="density",
        label="Random Sample",
        ax=ax,
    )

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
    ax.set_title(f"{label}\nSmoothness: {stats['smoothness_score']:.3f}, Dynamic Range: {stats['dynamic_range']:.3f}")

    plt.tight_layout()
    plt.savefig(output_dir / f"{label}_smoothness.pdf", dpi=300)
    plt.savefig(output_dir / f"{label}_smoothness.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
