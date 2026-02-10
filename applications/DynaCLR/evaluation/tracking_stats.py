"""CLI tool for computing tracking statistics from zarr or CSV files."""

import logging
from pathlib import Path

import click

from viscy.representation.pseudotime import CytoDtw
from viscy.utils.cli_utils import format_markdown_table

from .utils import load_tracking_data

logger = logging.getLogger("viscy")


def _format_fov_stats(fov_stats) -> list[dict]:
    """Convert FOV stats DataFrame to list of dicts for markdown formatting."""
    rows = []
    for _, row in fov_stats.iterrows():
        std_val = row["std_total_timepoints"]
        std_str = f"{std_val:.1f}" if std_val == std_val else "N/A"
        rows.append(
            {
                "fov_name": row["fov_name"],
                "n_lineages": row["n_lineages"],
                "mean_length": f"{row['mean_total_timepoints']:.1f}",
                "std": std_str,
            }
        )
    return rows


def _format_well_stats(well_stats) -> list[dict]:
    """Convert well stats DataFrame to list of dicts for markdown formatting."""
    rows = []
    for _, row in well_stats.iterrows():
        std_val = row["std_lineage_length"]
        std_str = f"{std_val:.1f}" if std_val == std_val else "N/A"
        rows.append(
            {
                "well_id": row["well_id"],
                "n_fovs": row["n_fovs"],
                "n_lineages": row["n_lineages"],
                "mean_length": f"{row['mean_lineage_length']:.1f}",
                "std": std_str,
            }
        )
    return rows


@click.command(
    "tracking-stats", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--min-timepoints",
    type=int,
    default=0,
    help="Minimum timepoints for a lineage to be included",
)
@click.option(
    "--levels",
    type=click.Choice(["fov", "well", "global", "all"]),
    multiple=True,
    default=["all"],
    help="Statistics levels to compute (can specify multiple)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path for markdown results",
)
def main(input_path: Path, min_timepoints: int, levels: tuple, output: Path | None):
    """Compute tracking statistics from zarr or CSV files.

    INPUT_PATH: Path to tracking data (.zarr or .csv)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Expand 'all' to all levels
    if "all" in levels:
        output_levels = ["fov", "well", "global"]
    else:
        output_levels = list(levels)

    logger.info(f"Loading tracking data from: {input_path}")
    adata = load_tracking_data(input_path)
    logger.info(f"Loaded {adata.shape[0]} samples")

    cytodtw = CytoDtw(adata)
    lineages = cytodtw.get_lineages(min_timepoints)
    logger.info(
        f"Identified {len(lineages)} lineages (min_timepoints={min_timepoints})"
    )

    output_lines = ["# Tracking Statistics\n"]

    # FOV-level statistics
    if "fov" in output_levels:
        fov_stats = cytodtw.get_track_statistics(lineages, per_fov=True)
        fov_data = _format_fov_stats(fov_stats)
        output_lines.append(
            format_markdown_table(
                fov_data,
                title="FOV-Level Statistics",
                headers=["fov_name", "n_lineages", "mean_length", "std"],
            )
        )

    # Well-level statistics
    if "well" in output_levels:
        well_stats = cytodtw.get_well_statistics(lineages)
        well_data = _format_well_stats(well_stats)
        output_lines.append(
            format_markdown_table(
                well_data,
                title="Well-Level Statistics",
                headers=["well_id", "n_fovs", "n_lineages", "mean_length", "std"],
            )
        )

    # Global statistics
    if "global" in output_levels:
        well_stats = cytodtw.get_well_statistics(lineages)
        fov_stats = cytodtw.get_track_statistics(lineages, per_fov=True)

        total_wells = well_stats["well_id"].nunique() if not well_stats.empty else 0
        total_fovs = fov_stats["fov_name"].nunique() if not fov_stats.empty else 0
        total_lineages = len(lineages)

        # Compute global mean/std from per-lineage data
        lineage_stats = cytodtw.get_track_statistics(lineages, per_fov=False)
        if not lineage_stats.empty:
            global_mean = lineage_stats["total_timepoints"].mean()
            global_std = lineage_stats["total_timepoints"].std()
        else:
            global_mean = 0
            global_std = 0

        global_data = {
            "total_wells": total_wells,
            "total_fovs": total_fovs,
            "total_lineages": total_lineages,
            "mean_lineage_length": f"{global_mean:.1f} ± {global_std:.1f}",
        }
        output_lines.append(format_markdown_table(global_data, title="Global Summary"))

    # Print output
    output_text = "\n".join(output_lines)
    logger.info("\n" + output_text)

    # Save output if specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(output_text)
        logger.info(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
