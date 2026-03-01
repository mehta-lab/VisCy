"""Evaluation pipeline comparing embedding models on a held-out test dataset.

Trains linear classifiers on cross-dataset embeddings, applies them to a
held-out test set, evaluates predictions, and optionally generates a PDF
comparison report.

Usage::

    python scripts/evaluate_dataset.py -c configs/evaluate_dataset_example.yaml
    python scripts/evaluate_dataset.py -c config.yaml --report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import pandas as pd
from sklearn.metrics import classification_report

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
    save_pipeline_to_wandb,
    train_linear_classifier,
)

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def run_evaluation(config: dict) -> None:
    """Run the full evaluation pipeline: train, infer, evaluate, report.

    Parameters
    ----------
    config : dict
        Evaluation config parsed from YAML. Expected keys:
        - dataset_name: str
        - test_annotations_csv: str path
        - output_dir: str path
        - models: dict of model specs
        - task_channels: dict or None (auto-detect from test CSV)
        - use_scaling, n_pca_components, max_iter, class_weight, solver,
          split_train_data, random_seed
        - wandb_logging: bool (default True)
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = Path(config["test_annotations_csv"])
    tc = resolve_task_channels(config.get("task_channels"), [test_csv])
    if not tc:
        raise ValueError("No valid tasks found in test annotations CSV.")

    model_labels = list(config["models"].keys())

    print("## Evaluation Pipeline")
    print(f"  Test dataset: {config['dataset_name']}")
    print(f"  Task-channels: {tc}")
    print(f"  Models: {model_labels}")

    use_scaling = config.get("use_scaling", True)
    n_pca = config.get("n_pca_components")
    use_pca = n_pca is not None
    split_train_data = config.get("split_train_data", 0.8)
    random_seed = config.get("random_seed", 42)
    wandb_logging = config.get("wandb_logging", True)

    classifier_params = {
        "max_iter": config.get("max_iter", 1000),
        "class_weight": config.get("class_weight", "balanced"),
        "solver": config.get("solver", "liblinear"),
        "random_state": random_seed,
    }

    train_results: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    eval_results: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for model_label, model_spec in config["models"].items():
        print(f"\n### Model: {model_label} ({model_spec.get('name', model_label)})")
        model_train: dict[tuple[str, str], dict[str, Any]] = {}
        model_eval: dict[tuple[str, str], dict[str, Any]] = {}
        model_output_dir = output_dir / model_label
        model_output_dir.mkdir(parents=True, exist_ok=True)

        test_embeddings_dir = Path(model_spec["test_embeddings_dir"])

        for task, channels in tc.items():
            test_channel_zarrs = find_channel_zarrs(test_embeddings_dir, channels)

            for channel in channels:
                combo_key = (task, channel)
                print(f"\n  {task} / {channel}:")

                # --- Train ---
                try:
                    datasets_for_combo = _build_train_datasets(model_spec["train_datasets"], task, channel)
                    if not datasets_for_combo:
                        print("    No training datasets available, skipping.")
                        continue

                    print(f"    Training on {len(datasets_for_combo)} dataset(s)")
                    combined_adata = load_and_combine_datasets(datasets_for_combo, task)

                    pipeline, metrics = train_linear_classifier(
                        adata=combined_adata,
                        task=task,
                        use_scaling=use_scaling,
                        use_pca=use_pca,
                        n_pca_components=n_pca,
                        classifier_params=classifier_params,
                        split_train_data=split_train_data,
                        random_seed=random_seed,
                    )

                    pipeline_path = model_output_dir / f"{task}_{channel}_pipeline.joblib"
                    joblib.dump(pipeline, pipeline_path)
                    print(f"    Pipeline saved: {pipeline_path.name}")

                    artifact_name = f"{model_spec.get('name', model_label)}_{task}_{channel}_local"
                    if wandb_logging and "wandb_project" in model_spec:
                        wandb_config = {
                            "task": task,
                            "input_channel": channel,
                            "marker": None,
                            "embedding_model": f"{model_spec['name']}-{model_spec['version']}",
                            "test_dataset": config["dataset_name"],
                            "use_scaling": use_scaling,
                            "use_pca": use_pca,
                            "n_pca_components": n_pca,
                            "max_iter": classifier_params["max_iter"],
                            "class_weight": classifier_params["class_weight"],
                            "solver": classifier_params["solver"],
                            "split_train_data": split_train_data,
                            "random_seed": random_seed,
                        }
                        wandb_tags = [
                            config["dataset_name"],
                            model_spec["name"],
                            model_spec["version"],
                            channel,
                            task,
                            "cross-dataset",
                        ]
                        artifact_name = save_pipeline_to_wandb(
                            pipeline=pipeline,
                            metrics=metrics,
                            config=wandb_config,
                            wandb_project=model_spec["wandb_project"],
                            tags=wandb_tags,
                        )

                    model_train[combo_key] = {
                        "pipeline": pipeline,
                        "metrics": metrics,
                        "artifact_name": artifact_name,
                    }

                    val_acc = metrics.get("val_accuracy")
                    val_f1 = metrics.get("val_weighted_f1")
                    if val_acc is not None:
                        print(f"    Val accuracy: {val_acc:.3f}  Val F1: {val_f1:.3f}")

                except Exception as e:
                    print(f"    TRAIN FAILED: {e}")
                    continue

                # --- Infer + Evaluate ---
                if channel not in test_channel_zarrs:
                    print(f"    No test zarr for {channel}, skipping inference.")
                    continue

                try:
                    print("    Loading test embeddings...")
                    test_adata = ad.read_zarr(test_channel_zarrs[channel])

                    artifact_metadata = {
                        "artifact_name": artifact_name,
                        "artifact_id": artifact_name,
                        "artifact_version": "local",
                    }
                    test_adata = predict_with_classifier(
                        test_adata,
                        pipeline,
                        task,
                        artifact_metadata=artifact_metadata,
                    )

                    pred_path = model_output_dir / f"{task}_{channel}_predictions.zarr"
                    test_adata.write_zarr(pred_path)
                    print(f"    Saved predictions: {pred_path.name}")

                    # Evaluate against ground truth
                    annotated = load_annotation_anndata(test_adata, str(test_csv), task)
                    mask = annotated.obs[task].notna() & (annotated.obs[task] != "unknown")
                    eval_subset = annotated[mask]

                    if len(eval_subset) == 0:
                        print("    No annotated test cells after filtering.")
                        continue

                    pred_col = f"predicted_{task}"
                    y_true = eval_subset.obs[task].values
                    y_pred = eval_subset.obs[pred_col].values

                    report = classification_report(y_true, y_pred, digits=3, output_dict=True)

                    test_metrics = {
                        "test_accuracy": report["accuracy"],
                        "test_weighted_precision": report["weighted avg"]["precision"],
                        "test_weighted_recall": report["weighted avg"]["recall"],
                        "test_weighted_f1": report["weighted avg"]["f1-score"],
                        "test_n_samples": len(eval_subset),
                    }

                    for class_name in sorted(set(y_true) | set(y_pred)):
                        if class_name in report:
                            test_metrics[f"test_{class_name}_precision"] = report[class_name]["precision"]
                            test_metrics[f"test_{class_name}_recall"] = report[class_name]["recall"]
                            test_metrics[f"test_{class_name}_f1"] = report[class_name]["f1-score"]

                    annotated_path = model_output_dir / f"{task}_{channel}_annotated.zarr"
                    annotated.write_zarr(annotated_path)

                    model_eval[combo_key] = {
                        "metrics": test_metrics,
                        "annotated_adata": annotated,
                    }

                    acc = test_metrics["test_accuracy"]
                    f1 = test_metrics["test_weighted_f1"]
                    n = test_metrics["test_n_samples"]
                    print(f"    Test: acc={acc:.3f}  F1={f1:.3f}  (n={n})")

                except Exception as e:
                    print(f"    EVAL FAILED: {e}")
                    continue

        train_results[model_label] = model_train
        eval_results[model_label] = model_eval

        # Save per-model metrics CSV
        _save_metrics_csv(
            model_train,
            model_eval,
            model_output_dir / "metrics_summary.csv",
        )

    # Save combined comparison CSVs
    _save_comparison_csv(train_results, output_dir / "train_metrics_comparison.csv")
    _save_eval_comparison_csv(eval_results, output_dir / "test_metrics_comparison.csv")

    # Print markdown summary
    _print_summary(train_results, eval_results, tc)

    return train_results, eval_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_train_datasets(train_datasets: list[dict], task: str, channel: str) -> list[dict]:
    """Filter and build training dataset dicts for a (task, channel) combo.

    Parameters
    ----------
    train_datasets : list[dict]
        Raw dataset entries from config, each with 'embeddings_dir' and 'annotations'.
    task : str
        Classification task to check for.
    channel : str
        Channel to look for in embeddings_dir.

    Returns
    -------
    list[dict]
        Filtered list with 'embeddings' and 'annotations' keys.
    """
    result = []
    for ds in train_datasets:
        embeddings_dir = Path(ds["embeddings_dir"])
        annotations_path = Path(ds["annotations"])

        channel_zarrs = find_channel_zarrs(embeddings_dir, [channel])
        if channel not in channel_zarrs:
            print(f"    Skipping {embeddings_dir.parent.name} - no {channel} zarr")
            continue

        available_tasks = get_available_tasks(annotations_path)
        if task not in available_tasks:
            print(f"    Skipping {embeddings_dir.parent.name} - no {task} column")
            continue

        result.append(
            {
                "embeddings": str(channel_zarrs[channel]),
                "annotations": str(annotations_path),
            }
        )
    return result


def _save_metrics_csv(
    train_results: dict[tuple[str, str], dict[str, Any]],
    eval_results: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
) -> None:
    """Save combined train + eval metrics for one model."""
    rows = []
    all_keys = set(train_results.keys()) | set(eval_results.keys())
    for combo_key in sorted(all_keys):
        task, channel = combo_key
        row = {"task": task, "channel": channel}
        if combo_key in train_results:
            row.update(train_results[combo_key]["metrics"])
        if combo_key in eval_results:
            row.update(eval_results[combo_key]["metrics"])
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)


def _save_comparison_csv(
    all_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save combined train metrics comparison across models."""
    rows = []
    for model_label, model_results in all_results.items():
        for (task, channel), result in model_results.items():
            row = {"model": model_label, "task": task, "channel": channel}
            row.update(result["metrics"])
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)


def _save_eval_comparison_csv(
    all_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save combined test metrics comparison across models."""
    rows = []
    for model_label, model_results in all_results.items():
        for (task, channel), result in model_results.items():
            row = {"model": model_label, "task": task, "channel": channel}
            row.update(result["metrics"])
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)


def _print_summary(
    train_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    eval_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    task_channels: dict[str, list[str]],
) -> None:
    """Print markdown summary table of all results."""
    headers = ["Task", "Channel"]
    model_labels = list(train_results.keys())
    for label in model_labels:
        headers += [
            f"{label} Val Acc",
            f"{label} Val F1",
            f"{label} Test Acc",
            f"{label} Test F1",
        ]

    rows = []
    for task, channels in task_channels.items():
        for channel in channels:
            row_dict = {"Task": task, "Channel": channel}
            for label in model_labels:
                tr = train_results.get(label, {}).get((task, channel))
                ev = eval_results.get(label, {}).get((task, channel))
                if tr:
                    row_dict[f"{label} Val Acc"] = f"{tr['metrics'].get('val_accuracy', float('nan')):.3f}"
                    row_dict[f"{label} Val F1"] = f"{tr['metrics'].get('val_weighted_f1', float('nan')):.3f}"
                else:
                    row_dict[f"{label} Val Acc"] = "-"
                    row_dict[f"{label} Val F1"] = "-"
                if ev:
                    row_dict[f"{label} Test Acc"] = f"{ev['metrics'].get('test_accuracy', float('nan')):.3f}"
                    row_dict[f"{label} Test F1"] = f"{ev['metrics'].get('test_weighted_f1', float('nan')):.3f}"
                else:
                    row_dict[f"{label} Test Acc"] = "-"
                    row_dict[f"{label} Test F1"] = "-"
            rows.append(row_dict)

    print(format_markdown_table(rows, title="Evaluation Summary", headers=headers))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embedding models on a held-out test dataset")
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
        help="Generate PDF comparison report",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Dataset: {config['dataset_name']}")
    print(f"Output: {config['output_dir']}")
    for label, spec in config["models"].items():
        n_train = len(spec["train_datasets"])
        print(f"  {label}: {n_train} training dataset(s)")

    train_results, eval_results = run_evaluation(config)

    if args.report:
        from dynaclr.evaluation.linear_classifiers.src.report import generate_comparison_report

        test_csv = Path(config["test_annotations_csv"])
        tc = resolve_task_channels(config.get("task_channels"), [test_csv])
        tasks = list(tc.keys())
        channels = sorted({ch for chs in tc.values() for ch in chs})

        generate_comparison_report(
            output_dir=Path(config["output_dir"]),
            dataset_name=config["dataset_name"],
            model_labels=list(config["models"].keys()),
            tasks=tasks,
            channels=channels,
            train_results=train_results,
            eval_results=eval_results,
        )
