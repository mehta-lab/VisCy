"""Shared utilities for evaluation CLIs."""

import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """
    Load and validate YAML config file.

    Parameters
    ----------
    config_path : Path
        Path to YAML config file

    Returns
    -------
    dict
        Loaded configuration dictionary

    Raises
    ------
    FileNotFoundError
        If config file does not exist
    yaml.YAMLError
        If config file is malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    return config


def validate_smoothness_config(config: dict) -> None:
    """
    Validate smoothness evaluation config structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Raises
    ------
    ValueError
        If config is missing required fields or has invalid structure
    """
    if "models" not in config:
        raise ValueError("Config must contain 'models' section")

    if not isinstance(config["models"], list) or len(config["models"]) == 0:
        raise ValueError("'models' must be a non-empty list")

    for i, model in enumerate(config["models"]):
        if "path" not in model:
            raise ValueError(f"Model {i} missing required 'path' field")
        if "label" not in model:
            raise ValueError(f"Model {i} missing required 'label' field")

    if "evaluation" not in config:
        raise ValueError("Config must contain 'evaluation' section")

    eval_config = config["evaluation"]
    if "output_dir" not in eval_config:
        raise ValueError("Evaluation config missing required 'output_dir' field")


def load_embedding(path: Path) -> ad.AnnData:
    """
    Load embeddings with error handling.

    Parameters
    ----------
    path : Path
        Path to zarr file containing embeddings

    Returns
    -------
    ad.AnnData
        Loaded AnnData object

    Raises
    ------
    FileNotFoundError
        If embedding file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    try:
        features_ad = ad.read_zarr(path)
    except Exception as e:
        raise RuntimeError(f"Error loading embedding from {path}: {e}")

    return features_ad


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
        If required metadata columns are missing
    """
    required_columns = ["fov_name", "track_id", "t"]
    missing_columns = [
        col for col in required_columns if col not in features_ad.obs.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Embedding missing required metadata columns: {missing_columns}. "
            f"Available columns: {list(features_ad.obs.columns)}"
        )

    # Check for empty tracks
    n_samples = features_ad.shape[0]
    if n_samples == 0:
        raise ValueError("Embedding has no samples")


def save_results(
    results: dict[str, Any], output_path: Path, format: str = "csv"
) -> None:
    """
    Save results to CSV or JSON.

    Parameters
    ----------
    results : dict
        Results dictionary to save
    output_path : Path
        Output file path
    format : str, optional
        Output format ('csv' or 'json'), by default "csv"

    Raises
    ------
    ValueError
        If format is not supported
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


def format_results_table(results: dict[str, dict], columns: list[str]) -> str:
    """
    Create markdown table from results.

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary mapping model labels to their metric dictionaries
    columns : list[str]
        List of metric column names to include

    Returns
    -------
    str
        Markdown-formatted table
    """
    if not results:
        return "No results to display."

    # Create header
    header = "| Model | " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"

    # Create rows
    rows = []
    for label, metrics in results.items():
        row_values = [
            f"{metrics.get(col, 'N/A'):.4f}"
            if isinstance(metrics.get(col), (int, float))
            else str(metrics.get(col, "N/A"))
            for col in columns
        ]
        row = f"| {label} | " + " | ".join(row_values) + " |"
        rows.append(row)

    return "\n".join([header, separator] + rows)


def format_comparison_summary(
    results: dict[str, dict], metric: str, lower_is_better: bool = True
) -> str:
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

    metric_values = {
        label: metrics.get(metric)
        for label, metrics in results.items()
        if metric in metrics
    }

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
