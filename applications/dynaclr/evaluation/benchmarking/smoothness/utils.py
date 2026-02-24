"""Smoothness-specific evaluation utilities."""

import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


def validate_embedding(features_ad: ad.AnnData) -> None:
    """
    Check required metadata columns in embedding.

    Parameters
    ----------
    features_ad : ad.AnnData
        AnnData object to validate

    Raises
    ------
    ValueError
        If required metadata columns are missing or embedding is empty
    """
    required_columns = ["fov_name", "track_id", "t"]
    missing_columns = [col for col in required_columns if col not in features_ad.obs.columns]

    if missing_columns:
        raise ValueError(
            f"Embedding missing required metadata columns: {missing_columns}. "
            f"Available columns: {list(features_ad.obs.columns)}"
        )

    if features_ad.shape[0] == 0:
        raise ValueError("Embedding has no samples")


def save_results(results: dict[str, Any], output_path: Path, format: str = "csv") -> None:
    """
    Save results dictionary to CSV or JSON.

    Parameters
    ----------
    results : dict
        Results dictionary to save
    output_path : Path
        Output file path
    format : str, optional
        Output format ('csv' or 'json'), by default "csv"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")


def format_comparison_summary(results: dict[str, dict], metric: str, lower_is_better: bool = True) -> str:
    """
    Highlight best model for a given metric.

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary mapping model labels to their metric dictionaries
    metric : str
        Metric name to compare
    lower_is_better : bool, optional
        Whether lower values are better, by default True

    Returns
    -------
    str
        Formatted summary string
    """
    if not results:
        return "No results to compare."

    metric_values = {label: metrics.get(metric) for label, metrics in results.items() if metric in metrics}

    if not metric_values:
        return f"Metric '{metric}' not found in results."

    if lower_is_better:
        best_label = min(metric_values, key=metric_values.get)
        comparison = "lowest"
    else:
        best_label = max(metric_values, key=metric_values.get)
        comparison = "highest"

    best_value = metric_values[best_label]
    return f"**Best {metric}**: {best_label} ({comparison}: {best_value:.4f})"
