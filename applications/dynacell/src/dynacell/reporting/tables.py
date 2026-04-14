"""Benchmark comparison tables from evaluation CSV outputs.

Reads the per-FOV, per-timepoint CSVs written by
``dynacell_paper.evaluation.pipeline`` and aggregates them into benchmark-ready
tables for the paper.
"""

from pathlib import Path

import pandas as pd

PIXEL_METRICS = ["PCC", "SSIM", "NRMSE", "PSNR", "Spectral_PCC", "MicroMS3IM"]
MASK_METRICS = ["Dice", "IoU", "Precision", "Recall"]
FEATURE_METRICS = [
    "CP_Median_Cosine_Similarity",
    "DINOv3_Median_Cosine_Similarity",
    "DynaCLR_Median_Cosine_Similarity",
    "CP_FID",
    "DINOv3_FID",
    "DynaCLR_FID",
]

HIGHER_IS_BETTER = {
    "PCC",
    "SSIM",
    "PSNR",
    "Spectral_PCC",
    "MicroMS3IM",
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "Accuracy",
    "CP_Median_Cosine_Similarity",
    "DINOv3_Median_Cosine_Similarity",
    "DynaCLR_Median_Cosine_Similarity",
}


def load_eval_results(
    results_dir: Path,
    pixel_csv: str = "pixel_metrics.csv",
    mask_csv: str = "mask_metrics.csv",
    feature_csv: str = "feature_metrics.csv",
) -> dict[str, pd.DataFrame]:
    """Load evaluation CSV files from a results directory.

    Parameters
    ----------
    results_dir
        Directory containing the CSV files.
    pixel_csv, mask_csv, feature_csv
        Filenames (overridable for legacy layouts).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"pixel"``, ``"mask"``, and ``"feature"`` (if present).
    """
    results_dir = Path(results_dir)
    result = {}
    for key, filename in [
        ("pixel", pixel_csv),
        ("mask", mask_csv),
        ("feature", feature_csv),
    ]:
        path = results_dir / filename
        if path.exists():
            result[key] = pd.read_csv(path)
    return result


def aggregate_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-FOV/timepoint metrics to mean and std.

    Parameters
    ----------
    df
        Raw per-FOV, per-timepoint DataFrame.
    metrics
        Subset of metric columns. Default: all numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``mean`` and ``std`` for each metric.
    """
    if metrics is None:
        metrics = [c for c in df.columns if c not in ("FOV", "Timepoint")]
    agg = df[metrics].agg(["mean", "std"])
    return agg


def load_and_aggregate(
    results_dir: Path,
    metrics: list[str],
    pixel_csv: str = "pixel_metrics.csv",
    mask_csv: str = "mask_metrics.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Load eval CSVs, combine, and aggregate to mean/std.

    Parameters
    ----------
    results_dir
        Directory containing evaluation CSV files.
    metrics
        Desired metric columns.
    pixel_csv, mask_csv
        CSV filenames to load.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Aggregated DataFrame (rows: mean/std, cols: metrics) and the
        list of available metric names.
    """
    data = load_eval_results(Path(results_dir), pixel_csv=pixel_csv, mask_csv=mask_csv)
    if not data:
        return pd.DataFrame(), []
    dfs = list(data.values())
    key_cols = ["FOV", "Timepoint"]
    if len(dfs) > 1:
        for label, df in zip(data.keys(), dfs):
            missing = [k for k in key_cols if k not in df.columns]
            if missing:
                raise ValueError(
                    f"{results_dir}/{label}: missing key columns {missing}. "
                    f"Cannot merge CSVs without FOV and Timepoint."
                )
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.merge(df, on=key_cols, how="outer", validate="one_to_one")
    else:
        combined = dfs[0]
    available = [m for m in metrics if m in combined.columns]
    return aggregate_metrics(combined, metrics=available), available


def comparison_table(
    model_results: dict[str, Path],
    metrics: list[str] | None = None,
    pixel_csv: str = "pixel_metrics.csv",
    mask_csv: str = "mask_metrics.csv",
) -> pd.DataFrame:
    """Build a model-comparison table (models as rows, metrics as columns).

    Parameters
    ----------
    model_results
        Mapping of model display name to results directory path.
    metrics
        Metric columns to include. Default: PIXEL_METRICS + MASK_METRICS.
    pixel_csv, mask_csv
        CSV filenames to load.

    Returns
    -------
    pd.DataFrame
        Index is model name, columns are metric names, values are
        ``"mean +/- std"`` formatted strings.
    """
    if metrics is None:
        metrics = PIXEL_METRICS + MASK_METRICS

    rows = {}
    for model_name, results_dir in model_results.items():
        agg, available = load_and_aggregate(results_dir, metrics, pixel_csv=pixel_csv, mask_csv=mask_csv)
        row = {}
        for m in available:
            mean = agg.loc["mean", m]
            std = agg.loc["std", m]
            row[m] = f"{mean:.4f} +/- {std:.4f}"
        rows[model_name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def to_latex(
    df: pd.DataFrame,
    bold_best: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    r"""Render a comparison table as a LaTeX tabular fragment.

    Parameters
    ----------
    df
        DataFrame from :func:`comparison_table`.
    bold_best
        Whether to bold the best value in each column.
    caption, label
        Optional LaTeX caption and label.

    Returns
    -------
    str
        LaTeX string suitable for ``\input{tables/...}``.
    """
    if bold_best and len(df) > 1:
        formatted = df.copy()
        for col in formatted.columns:
            vals: list[float | None] = []
            for cell in formatted[col]:
                try:
                    mean_str = cell.split(" +/- ")[0]
                    vals.append(float(mean_str))
                except (ValueError, AttributeError):
                    vals.append(None)

            if all(v is None for v in vals):
                continue

            higher = col in HIGHER_IS_BETTER
            if higher:
                numeric = [v if v is not None else float("-inf") for v in vals]
            else:
                numeric = [-v if v is not None else float("-inf") for v in vals]
            best_idx = max(range(len(numeric)), key=lambda i: numeric[i])
            original = formatted.iloc[best_idx][col]
            formatted.iloc[best_idx, formatted.columns.get_loc(col)] = f"\\textbf{{{original}}}"
        df = formatted

    latex = df.to_latex(escape=False)

    if caption or label:
        lines = ["\\begin{table}[ht]", "\\centering"]
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append(latex)
        lines.append("\\end{table}")
        return "\n".join(lines)

    return latex
