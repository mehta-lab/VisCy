"""Rotating test-set cross-validation for training dataset impact analysis.

Leave-one-dataset-out as test (rotating): for each dataset D as test, train
on the remaining pool, then do leave-one-out on the training pool. Impact
is aggregated across ALL test folds for unbiased generalization scores.

Usage::

    python scripts/cross_validation.py -c configs/cross_validate_example.yaml
    python scripts/cross_validation.py -c config.yaml --report
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from dynaclr.evaluation.linear_classifiers.src.utils import (
    find_channel_zarrs,
    get_available_tasks,
    resolve_task_channels,
)
from viscy_utils.cli_utils import format_markdown_table, load_config
from viscy_utils.evaluation.annotation import load_annotation_anndata
from viscy_utils.evaluation.linear_classifier import (
    load_and_combine_datasets,
    predict_with_classifier,
    train_linear_classifier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cv_pairs(datasets: list[dict], channel: str, task: str) -> list[tuple[dict, dict]]:
    """Build (dataset_meta, training_dict) pairs for a channel and task.

    Parameters
    ----------
    datasets : list[dict]
        Dataset dicts from config with 'name', 'embeddings_dir', 'annotations'.
    channel : str
        Channel to look for in embeddings_dir.
    task : str
        Task column to require in the annotations CSV.

    Returns
    -------
    list[tuple[dict, dict]]
        Each tuple is (original dataset dict, {"embeddings": ..., "annotations": ...}).
    """
    result = []
    for ds in datasets:
        embeddings_dir = Path(ds["embeddings_dir"])
        annotations_path = Path(ds["annotations"])
        channel_zarrs = find_channel_zarrs(embeddings_dir, [channel])
        if channel not in channel_zarrs:
            continue
        available_tasks = get_available_tasks(annotations_path)
        if task not in available_tasks:
            continue
        result.append(
            (
                ds,
                {
                    "embeddings": str(channel_zarrs[channel]),
                    "annotations": str(annotations_path),
                },
            )
        )
    return result


def _resolve_task_channels_from_datasets(config: dict) -> dict[str, list[str]]:
    """Resolve task -> channels from intersection across all datasets."""
    annotation_csvs = []
    for model_spec in config["models"].values():
        for ds in model_spec["datasets"]:
            annotation_csvs.append(Path(ds["annotations"]))
    return resolve_task_channels(config.get("task_channels"), annotation_csvs)


def _check_class_safety(
    datasets_for_combo: list[dict],
    task: str,
    min_class_samples: int,
) -> bool:
    """Check if the dataset subset has enough samples per class."""
    all_labels: list[str] = []
    for ds in datasets_for_combo:
        ann = pd.read_csv(ds["annotations"])
        if task in ann.columns:
            valid = ann[task].dropna()
            valid = valid[valid != "unknown"]
            all_labels.extend(valid.tolist())

    if not all_labels:
        return False
    class_counts = pd.Series(all_labels).value_counts()
    return bool((class_counts >= min_class_samples).all())


def _get_class_counts(datasets_for_combo: list[dict], task: str) -> dict[str, int]:
    """Count per-class samples across datasets."""
    all_labels: list[str] = []
    for ds in datasets_for_combo:
        ann = pd.read_csv(ds["annotations"])
        if task in ann.columns:
            valid = ann[task].dropna()
            valid = valid[valid != "unknown"]
            all_labels.extend(valid.tolist())
    return dict(pd.Series(all_labels).value_counts())


def _detect_n_features(datasets: list[dict], channel: str) -> int | None:
    """Detect embedding dimensionality from the first available zarr."""
    for ds in datasets:
        embeddings_dir = Path(ds["embeddings_dir"])
        channel_zarrs = find_channel_zarrs(embeddings_dir, [channel])
        if channel in channel_zarrs:
            adata = ad.read_zarr(channel_zarrs[channel])
            return adata.shape[1]
    return None


# ---------------------------------------------------------------------------
# Core rotating CV unit
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    config: dict,
    model_label: str,
    task: str,
    channel: str,
    train_datasets: list[dict],
    test_dataset: dict,
    test_dataset_name: str,
    seed: int,
    excluded_dataset: str | None = None,
) -> dict[str, Any]:
    """Train on train_datasets and evaluate on test_dataset.

    Parameters
    ----------
    config : dict
        Full CV config dict.
    model_label : str
        Model label (e.g. "2D").
    task : str
        Classification task.
    channel : str
        Input channel.
    train_datasets : list[dict]
        Training dataset dicts with 'embeddings' and 'annotations' keys.
    test_dataset : dict
        Test dataset dict with 'embeddings' and 'annotations' keys.
    test_dataset_name : str
        Name of the test dataset.
    seed : int
        Random seed for this run.
    excluded_dataset : str or None
        Name of the excluded dataset (None for baseline).

    Returns
    -------
    dict
        Flat result dict with metrics and metadata.
    """
    row: dict[str, Any] = {
        "model": model_label,
        "task": task,
        "channel": channel,
        "excluded_dataset": excluded_dataset or "baseline",
        "test_dataset": test_dataset_name,
        "seed": seed,
        "n_train_datasets": len(train_datasets),
    }

    class_counts = _get_class_counts(train_datasets, task)
    for cls, cnt in class_counts.items():
        row[f"train_class_{cls}"] = cnt

    if class_counts:
        minority_class = min(class_counts, key=class_counts.get)
        row["minority_class"] = minority_class
        row["minority_class_count"] = class_counts[minority_class]
    else:
        row["minority_class"] = None
        row["minority_class_count"] = 0

    use_scaling = config.get("use_scaling", True)
    n_pca = config.get("n_pca_components")
    use_pca = n_pca is not None
    split_train_data = config.get("split_train_data", 0.8)

    try:
        combined_adata = load_and_combine_datasets(train_datasets, task)

        classifier_params = {
            "max_iter": config.get("max_iter", 1000),
            "class_weight": config.get("class_weight", "balanced"),
            "solver": config.get("solver", "liblinear"),
            "random_state": seed,
        }

        pipeline, metrics = train_linear_classifier(
            adata=combined_adata,
            task=task,
            use_scaling=use_scaling,
            use_pca=use_pca,
            n_pca_components=n_pca,
            classifier_params=classifier_params,
            split_train_data=split_train_data,
            random_seed=seed,
        )

        row.update(metrics)

        test_adata = ad.read_zarr(test_dataset["embeddings"])
        test_adata = predict_with_classifier(test_adata, pipeline, task)

        annotated = load_annotation_anndata(test_adata, str(test_dataset["annotations"]), task)

        mask = annotated.obs[task].notna() & (annotated.obs[task] != "unknown")
        eval_subset = annotated[mask]

        if len(eval_subset) == 0:
            row["auroc"] = np.nan
            row["error"] = "no annotated test cells"
            return row

        pred_col = f"predicted_{task}"
        y_true = eval_subset.obs[task].values
        y_pred = eval_subset.obs[pred_col].values

        # AUROC
        proba_key = f"predicted_{task}_proba"
        classes_key = f"predicted_{task}_classes"
        if proba_key in annotated.obsm and classes_key in annotated.uns:
            y_proba = annotated[mask].obsm[proba_key]
            classes = annotated.uns[classes_key]
            n_classes = len(classes)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if n_classes == 2:
                        auroc = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                except ValueError:
                    auroc = np.nan
            row["auroc"] = auroc

            _compute_temporal_metrics(row, eval_subset, task, y_proba, classes)
        else:
            row["auroc"] = np.nan

        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        row["test_accuracy"] = report["accuracy"]
        row["test_weighted_f1"] = report["weighted avg"]["f1-score"]
        row["test_weighted_precision"] = report["weighted avg"]["precision"]
        row["test_weighted_recall"] = report["weighted avg"]["recall"]
        row["test_n_samples"] = len(eval_subset)

        for class_name in sorted(set(y_true) | set(y_pred)):
            if class_name in report:
                row[f"test_{class_name}_f1"] = report[class_name]["f1-score"]
                row[f"test_{class_name}_precision"] = report[class_name]["precision"]
                row[f"test_{class_name}_recall"] = report[class_name]["recall"]

        if row.get("minority_class") and row["minority_class"] in report:
            mc = row["minority_class"]
            row["minority_f1"] = report[mc]["f1-score"]
            row["minority_recall"] = report[mc]["recall"]
            row["minority_precision"] = report[mc]["precision"]

    except Exception as e:
        row["auroc"] = np.nan
        row["error"] = str(e)
        logger.warning(f"CV fold failed: {excluded_dataset}, seed={seed}: {e}")

    return row


def _compute_temporal_metrics(
    row: dict,
    eval_subset: ad.AnnData,
    task: str,
    y_proba: np.ndarray,
    classes: list,
    n_bins: int = 10,
) -> None:
    """Compute AUROC and F1 macro per normalized-time bin."""
    if "t" not in eval_subset.obs.columns:
        row["temporal_metrics"] = None
        return

    t_values = eval_subset.obs["t"].values.astype(float)
    if len(np.unique(t_values)) < 2:
        row["temporal_metrics"] = None
        return

    t_norm = (t_values - t_values.min()) / (t_values.max() - t_values.min())
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(t_norm, bin_edges[1:-1]), 0, n_bins - 1)

    y_true = eval_subset.obs[task].values
    pred_col = f"predicted_{task}"
    y_pred = eval_subset.obs[pred_col].values
    n_classes = len(classes)

    auroc_list: list[float | None] = []
    f1_list: list[float | None] = []
    n_samples_list: list[int] = []

    for b in range(n_bins):
        mask_b = bins == b
        n_b = int(mask_b.sum())
        n_samples_list.append(n_b)

        if n_b == 0:
            auroc_list.append(None)
            f1_list.append(None)
            continue

        y_true_b = y_true[mask_b]
        y_pred_b = y_pred[mask_b]
        proba_b = y_proba[mask_b]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1_val = float(f1_score(y_true_b, y_pred_b, average="macro"))
        f1_list.append(f1_val)

        n_unique = len(np.unique(y_true_b))
        if n_unique < 2:
            auroc_list.append(None)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if n_classes == 2:
                    auroc_val = float(roc_auc_score(y_true_b, proba_b[:, 1]))
                else:
                    auroc_val = float(roc_auc_score(y_true_b, proba_b, multi_class="ovr", average="macro"))
            except ValueError:
                auroc_val = None
        auroc_list.append(auroc_val)

    row["temporal_metrics"] = json.dumps(
        {
            "bin_edges": bin_edges.tolist(),
            "auroc": auroc_list,
            "f1_macro": f1_list,
            "n_samples": n_samples_list,
        }
    )


# ---------------------------------------------------------------------------
# Main rotating CV loop
# ---------------------------------------------------------------------------


def cross_validate(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rotating test-set cross-validation.

    Parameters
    ----------
    config : dict
        CV configuration parsed from YAML. Expected keys:
        - models: dict of model specs with 'datasets' lists
        - output_dir: str path
        - ranking_metric: str (default "auroc")
        - n_bootstrap: int (default 5)
        - min_class_samples: int or None
        - use_scaling, n_pca_components, max_iter, class_weight,
          solver, split_train_data, random_seed

    Returns
    -------
    pd.DataFrame
        Raw results (one row per fold x seed).
    pd.DataFrame
        Aggregated summary with impact labels.
    """
    ranking_metric = config.get("ranking_metric", "auroc")
    n_bootstrap = config.get("n_bootstrap", 5)
    min_class_samples = config.get("min_class_samples")

    tc = _resolve_task_channels_from_datasets(config)
    if not tc:
        raise ValueError("No valid tasks found across datasets.")

    n_pca = config.get("n_pca_components")
    if min_class_samples is None:
        min_class_samples = n_pca if n_pca else 16
        print(f"  Auto-detected min_class_samples={min_class_samples}")

    base_seed = config.get("random_seed", 42)
    seeds = [base_seed + i for i in range(n_bootstrap)]

    all_rows: list[dict[str, Any]] = []

    for model_label, model_spec in config["models"].items():
        print(f"\n## Rotating CV: {model_label} ({model_spec.get('name', model_label)})")

        datasets = model_spec["datasets"]
        for task, channels in tc.items():
            for channel in channels:
                print(f"\n### {task} / {channel}")

                all_pairs = _build_cv_pairs(datasets, channel, task)
                if len(all_pairs) < 3:
                    print(f"  Only {len(all_pairs)} dataset(s), need >= 3. Skipping.")
                    continue

                for test_idx, (test_ds, test_dict) in enumerate(all_pairs):
                    test_name = test_ds["name"]
                    train_pool = [(ds, d) for j, (ds, d) in enumerate(all_pairs) if j != test_idx]
                    train_dicts = [d for _, d in train_pool]

                    print(f"\n  Test fold: {test_name}")

                    # BASELINE: train on full training pool
                    print(f"    Baseline: {len(train_dicts)} datasets, {n_bootstrap} seeds")
                    for seed in seeds:
                        row = _train_and_evaluate(
                            config,
                            model_label,
                            task,
                            channel,
                            train_dicts,
                            test_dict,
                            test_name,
                            seed,
                            excluded_dataset=None,
                        )
                        all_rows.append(row)

                    # Leave-one-out from training pool
                    for loo_idx, (loo_ds, _) in enumerate(train_pool):
                        loo_name = loo_ds["name"]
                        remaining = [d for j, (_, d) in enumerate(train_pool) if j != loo_idx]

                        safe = _check_class_safety(remaining, task, min_class_samples)
                        if not safe:
                            print(f"    Excluding {loo_name}: UNSAFE (class threshold)")
                            for seed in seeds:
                                unsafe_row = {
                                    "model": model_label,
                                    "task": task,
                                    "channel": channel,
                                    "excluded_dataset": loo_name,
                                    "test_dataset": test_name,
                                    "seed": seed,
                                    "n_train_datasets": len(remaining),
                                    "impact": "unsafe",
                                    "auroc": np.nan,
                                }
                                all_rows.append(unsafe_row)
                            continue

                        print(f"    Excluding {loo_name}: {len(remaining)} remaining, {n_bootstrap} seeds")
                        for seed in seeds:
                            row = _train_and_evaluate(
                                config,
                                model_label,
                                task,
                                channel,
                                remaining,
                                test_dict,
                                test_name,
                                seed,
                                excluded_dataset=loo_name,
                            )
                            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(all_rows)
    summary_df = _compute_summary(results_df, ranking_metric)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)

    recommendations = _get_recommended_subsets(summary_df)
    if not recommendations.empty:
        recommendations.to_csv(output_dir / "cv_recommended_subsets.csv", index=False)

    _print_markdown_summary(summary_df, ranking_metric)

    return results_df, summary_df


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _compute_summary(
    results_df: pd.DataFrame,
    ranking_metric: str = "auroc",
) -> pd.DataFrame:
    """Aggregate raw rotating CV results using paired within-fold deltas.

    For each (model, task, channel, excluded_dataset), computes deltas
    relative to the baseline within each test fold, then averages across
    shared test folds to control for test-fold difficulty.
    """
    if results_df.empty:
        return pd.DataFrame()

    group_cols = ["model", "task", "channel"]
    summary_rows = []

    for group_key, group_df in results_df.groupby(group_cols):
        model, task, channel = group_key

        baseline = group_df[group_df["excluded_dataset"] == "baseline"]

        bl_fold_means: dict[str, float] = {}
        for td, td_df in baseline.groupby("test_dataset"):
            vals = td_df[ranking_metric].dropna()
            if not vals.empty:
                bl_fold_means[td] = vals.mean()

        baseline_mean = np.mean(list(bl_fold_means.values())) if bl_fold_means else np.nan

        n_test_folds = group_df["test_dataset"].nunique()

        for exc_ds, exc_df in group_df.groupby("excluded_dataset"):
            exc_overall_mean = exc_df[ranking_metric].mean()
            exc_overall_std = exc_df[ranking_metric].std()

            if exc_ds == "baseline":
                summary_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "channel": channel,
                        "excluded_dataset": exc_ds,
                        f"mean_{ranking_metric}": baseline_mean,
                        f"std_{ranking_metric}": exc_overall_std,
                        "baseline_mean": baseline_mean,
                        "delta": 0.0,
                        "impact": "baseline",
                        "n_test_folds": len(bl_fold_means),
                    }
                )
                continue

            if exc_df.get("impact", pd.Series()).eq("unsafe").any():
                summary_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "channel": channel,
                        "excluded_dataset": exc_ds,
                        f"mean_{ranking_metric}": exc_overall_mean,
                        f"std_{ranking_metric}": exc_overall_std,
                        "baseline_mean": baseline_mean,
                        "delta": np.nan,
                        "impact": "unsafe",
                        "n_test_folds": n_test_folds,
                    }
                )
                continue

            paired_deltas = []
            exc_fold_means: dict[str, float] = {}
            for td, td_df in exc_df.groupby("test_dataset"):
                vals = td_df[ranking_metric].dropna()
                if not vals.empty:
                    exc_fold_means[td] = vals.mean()

            shared_folds = set(bl_fold_means) & set(exc_fold_means)
            for td in shared_folds:
                paired_deltas.append(exc_fold_means[td] - bl_fold_means[td])

            n_shared = len(shared_folds)

            if not paired_deltas:
                delta = np.nan
                delta_std = np.nan
            else:
                delta = np.mean(paired_deltas)
                delta_std = np.std(paired_deltas, ddof=1) if n_shared > 1 else 0.0

            if np.isnan(delta) or n_shared < 2:
                impact = "uncertain"
            else:
                sem = delta_std / np.sqrt(n_shared) if n_shared > 0 else 0.0
                if sem == 0:
                    impact = "uncertain"
                elif delta > 0 and delta > sem:
                    impact = "hurts"
                elif delta < 0 and abs(delta) > sem:
                    impact = "helps"
                else:
                    impact = "uncertain"

            shared_exc_mean = np.mean([exc_fold_means[td] for td in shared_folds]) if shared_folds else exc_overall_mean
            shared_bl_mean = np.mean([bl_fold_means[td] for td in shared_folds]) if shared_folds else baseline_mean

            summary_rows.append(
                {
                    "model": model,
                    "task": task,
                    "channel": channel,
                    "excluded_dataset": exc_ds,
                    f"mean_{ranking_metric}": shared_exc_mean,
                    f"std_{ranking_metric}": exc_overall_std,
                    "baseline_mean": shared_bl_mean,
                    "delta": delta,
                    "delta_std": delta_std,
                    "impact": impact,
                    "n_test_folds": n_shared,
                }
            )

    return pd.DataFrame(summary_rows)


def _print_markdown_summary(summary_df: pd.DataFrame, ranking_metric: str) -> None:
    """Print a markdown-formatted summary table."""
    if summary_df.empty:
        print("\nNo cross-validation results to summarize.")
        return

    print("\n## Cross-Validation Impact Summary\n")

    headers = [
        "Excluded Dataset",
        f"Mean {ranking_metric.upper()}",
        "Paired Delta",
        "Delta Std",
        "Impact",
        "Folds",
    ]

    for (model, task, channel), group in summary_df.groupby(["model", "task", "channel"]):
        rows = []
        for _, row in group.sort_values("delta", ascending=False, na_position="last").iterrows():
            mean_val = row.get(f"mean_{ranking_metric}", np.nan)
            delta = row.get("delta", np.nan)
            delta_std = row.get("delta_std", np.nan)

            rows.append(
                {
                    headers[0]: row["excluded_dataset"],
                    headers[1]: (f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"),
                    headers[2]: (f"{delta:+.4f}" if not np.isnan(delta) else "N/A"),
                    headers[3]: (
                        f"{delta_std:.4f}" if not (isinstance(delta_std, float) and np.isnan(delta_std)) else "-"
                    ),
                    headers[4]: row.get("impact", "?"),
                    headers[5]: row.get("n_test_folds", "?"),
                }
            )

        print(f"\n### {model} / {task} / {channel}\n")
        print(format_markdown_table(rows, headers=headers))


# ---------------------------------------------------------------------------
# Recommended subsets
# ---------------------------------------------------------------------------


def _get_recommended_subsets(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Derive recommended training subsets per (model, task, channel)."""
    non_baseline = summary_df[summary_df["excluded_dataset"] != "baseline"]
    baseline = summary_df[summary_df["excluded_dataset"] == "baseline"]

    rows = []
    for (model, task, channel), group in non_baseline.groupby(["model", "task", "channel"]):
        bl = baseline[(baseline["model"] == model) & (baseline["task"] == task) & (baseline["channel"] == channel)]
        bl_auroc = bl["baseline_mean"].values[0] if len(bl) > 0 else np.nan

        included = []
        excluded = []
        for _, row in group.iterrows():
            ds = row["excluded_dataset"]
            impact = row["impact"]
            if impact == "hurts":
                excluded.append((ds, impact, row.get("delta", np.nan)))
            elif impact == "unsafe":
                excluded.append((ds, impact, np.nan))
            else:
                included.append((ds, impact, row.get("delta", np.nan)))

        rows.append(
            {
                "model": model,
                "task": task,
                "channel": channel,
                "baseline_auroc": bl_auroc,
                "n_included": len(included),
                "n_excluded": len(excluded),
                "included_datasets": ", ".join(d for d, _, _ in included),
                "excluded_datasets": ", ".join(
                    (f"{d} ({imp}, {delta:+.4f})" if not np.isnan(delta) else f"{d} ({imp})")
                    for d, imp, delta in excluded
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotating test-set leave-one-dataset-out cross-validation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate PDF report",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["output_dir"])
    print(f"Output: {output_dir}")
    for label, spec in config["models"].items():
        n_ds = len(spec["datasets"])
        print(f"  {label}: {n_ds} datasets (all rotate as test)")

    results_df, summary_df = cross_validate(config)

    if args.report and not results_df.empty:
        from dynaclr.evaluation.linear_classifiers.src.report import generate_cv_report

        config_summary = {
            "use_scaling": config.get("use_scaling", True),
            "n_pca_components": config.get("n_pca_components"),
            "solver": config.get("solver", "liblinear"),
            "class_weight": config.get("class_weight", "balanced"),
            "max_iter": config.get("max_iter", 1000),
            "split_train_data": config.get("split_train_data", 0.8),
        }
        generate_cv_report(
            output_dir,
            results_df,
            summary_df,
            config_summary,
            ranking_metric=config.get("ranking_metric", "auroc"),
        )
