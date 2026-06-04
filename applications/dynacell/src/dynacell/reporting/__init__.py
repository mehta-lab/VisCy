"""Benchmark reporting: tables and figures from evaluation outputs."""

from dynacell.reporting.figures import metric_comparison_barplot
from dynacell.reporting.tables import (
    FEATURE_METRICS,
    HIGHER_IS_BETTER,
    MASK_METRICS,
    PIXEL_METRICS,
    aggregate_metrics,
    comparison_table,
    load_and_aggregate,
    load_eval_results,
    to_latex,
)

__all__ = [
    "FEATURE_METRICS",
    "HIGHER_IS_BETTER",
    "MASK_METRICS",
    "PIXEL_METRICS",
    "aggregate_metrics",
    "comparison_table",
    "load_and_aggregate",
    "load_eval_results",
    "metric_comparison_barplot",
    "to_latex",
]
