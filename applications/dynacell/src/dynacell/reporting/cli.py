"""Reporting CLI entry point for the DynaCell benchmark.

Hydra-based entry point that generates benchmark comparison tables
and figures from evaluation CSV outputs.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dynacell.reporting.figures import metric_comparison_barplot
from dynacell.reporting.tables import comparison_table, to_latex

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path="_configs",
    config_name="base",
)
def generate_report(cfg: DictConfig) -> None:
    """Generate benchmark tables and figures from evaluation results.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``results_dirs``, ``output_dir``, ``metrics``,
        and ``figure_format``.  See ``configs/report/base.yaml``.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = dict(cfg.results_dirs)
    if not model_results:
        logger.warning("No results_dirs provided. Nothing to report.")
        return

    path_results = {k: Path(v) for k, v in model_results.items()}
    all_metrics = list(cfg.metrics.pixel) + list(cfg.metrics.mask)

    logger.info("Generating comparison table for %d models...", len(model_results))
    table = comparison_table(path_results, metrics=all_metrics)
    logger.info("Comparison table:\n%s", table.to_string())

    latex = to_latex(table, bold_best=True)
    latex_path = output_dir / "comparison_table.tex"
    latex_path.write_text(latex)
    logger.info("LaTeX table written to %s", latex_path)

    figure_path = output_dir / f"comparison_barplot.{cfg.figure_format}"
    fig = metric_comparison_barplot(path_results, metrics=all_metrics, save_path=figure_path)
    # Import plt here (not at module level) so the Agg backend set by
    # dynacell.reporting.figures is active before pyplot is first loaded.
    import matplotlib.pyplot as _plt

    _plt.close(fig)
    logger.info("Comparison figure written to %s", figure_path)


if __name__ == "__main__":
    generate_report()
