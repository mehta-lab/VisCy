"""Evaluation pipeline comparing 2D vs 3D linear classifiers on cell embeddings.

Trains linear classifiers on cross-dataset embeddings, runs inference on a
held-out test dataset, evaluates predictions, and generates a PDF comparison
report. Each block can be run independently or chained via ``run_evaluation()``.
"""

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from viscy.representation.evaluation.linear_classifier import (
    load_and_combine_datasets,
    save_pipeline_to_wandb,
    train_linear_classifier,
)
from viscy.representation.evaluation.linear_classifier_config import (
    VALID_CHANNELS,
    VALID_TASKS,
)

CHANNELS = list(VALID_CHANNELS.__args__)
TASKS = list(VALID_TASKS.__args__)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class TrainDataset(BaseModel):
    """A single training dataset with embeddings and annotations.

    Parameters
    ----------
    embeddings_dir : Path
        Version directory containing per-channel ``.zarr`` files.
    annotations : Path
        Path to the annotation CSV for this dataset.
    """

    embeddings_dir: Path
    annotations: Path

    @model_validator(mode="after")
    def validate_paths(self):
        if not self.embeddings_dir.exists():
            raise ValueError(f"Embeddings directory not found: {self.embeddings_dir}")
        if not self.annotations.exists():
            raise ValueError(f"Annotations file not found: {self.annotations}")
        return self


class ModelSpec(BaseModel):
    """Specification for one embedding model to evaluate.

    Parameters
    ----------
    name : str
        Model name (e.g. ``"DynaCLR-3D-BagOfChannels-timeaware"``).
    train_datasets : list[TrainDataset]
        Training datasets (excluding held-out test).
    test_embeddings_dir : Path
        Version directory with ``.zarr`` files for the held-out test dataset.
    version : str
        Model version string (e.g. ``"v1"``).
    wandb_project : str
        Weights & Biases project name.
    """

    name: str = Field(..., min_length=1)
    train_datasets: list[TrainDataset] = Field(..., min_length=1)
    test_embeddings_dir: Path
    version: str = Field(..., min_length=1)
    wandb_project: str = Field(..., min_length=1)

    @field_validator("test_embeddings_dir")
    @classmethod
    def validate_test_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Test embeddings directory not found: {v}")
        return v


class DatasetEvalConfig(BaseModel):
    """Configuration for a single-dataset evaluation run.

    Parameters
    ----------
    dataset_name : str
        Held-out test dataset name.
    test_annotations_csv : Path
        Path to annotation CSV for the test dataset.
    models : dict[str, ModelSpec]
        Models to compare, keyed by label (e.g. ``"2D"``, ``"3D"``).
    output_dir : Path
        Root output directory for results.
    channels : list[str]
        Channels to evaluate.
    tasks : list[str] or None
        Tasks to evaluate. ``None`` to auto-detect from test annotations.
    split_train_data : float
        Train/val split ratio within the training pool.
    use_scaling : bool
        Whether to apply StandardScaler.
    max_iter : int
        Max iterations for LogisticRegression.
    class_weight : str or None
        Class weighting strategy.
    solver : str
        LogisticRegression solver.
    random_seed : int
        Random seed for reproducibility.
    """

    dataset_name: str = Field(..., min_length=1)
    test_annotations_csv: Path
    models: dict[str, ModelSpec] = Field(..., min_length=1)
    output_dir: Path
    channels: list[str] = Field(
        default_factory=lambda: ["phase", "sensor", "organelle"]
    )
    tasks: Optional[list[str]] = None
    split_train_data: float = Field(default=0.8, gt=0.0, le=1.0)
    use_scaling: bool = True
    max_iter: int = Field(default=1000, gt=0)
    class_weight: Optional[str] = "balanced"
    solver: str = "liblinear"
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_config(self):
        if not self.test_annotations_csv.exists():
            raise ValueError(
                f"Test annotations CSV not found: {self.test_annotations_csv}"
            )
        if self.tasks is not None:
            for task in self.tasks:
                if task not in TASKS:
                    raise ValueError(f"Invalid task '{task}'. Must be one of {TASKS}")
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_channel_zarrs(embeddings_dir: Path, channels: list[str]) -> dict[str, Path]:
    """Find per-channel zarr files in a predictions directory."""
    from glob import glob

    from natsort import natsorted

    channel_zarrs = {}
    for channel in channels:
        matches = natsorted(glob(str(embeddings_dir / f"*{channel}*.zarr")))
        if matches:
            channel_zarrs[channel] = Path(matches[0])
    return channel_zarrs


def _get_available_tasks(csv_path: Path) -> list[str]:
    """Read CSV header and return which valid task columns are present."""
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [t for t in TASKS if t in columns]


def _resolve_tasks(config: DatasetEvalConfig) -> list[str]:
    """Resolve tasks to evaluate, auto-detecting from test CSV if needed."""
    if config.tasks is not None:
        return config.tasks
    return _get_available_tasks(config.test_annotations_csv)


# ---------------------------------------------------------------------------
# Block 1: Train classifiers
# ---------------------------------------------------------------------------


def train_classifiers(
    config: DatasetEvalConfig,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Train linear classifiers for all model x task x channel combinations.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.

    Returns
    -------
    dict[str, dict[tuple[str, str], dict]]
        Nested dict: ``model_label -> (task, channel) -> result_dict``.
        Each ``result_dict`` has keys: ``"pipeline"``, ``"metrics"``,
        ``"artifact_name"``.
    """
    tasks = _resolve_tasks(config)
    if not tasks:
        raise ValueError("No valid tasks found in test annotations CSV.")

    print("## Training classifiers")
    print(f"  Tasks: {tasks}")
    print(f"  Channels: {config.channels}")
    print(f"  Models: {list(config.models.keys())}")

    all_results: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for model_label, model_spec in config.models.items():
        print(f"\n### Model: {model_label} ({model_spec.name})")
        model_results: dict[tuple[str, str], dict[str, Any]] = {}
        model_output_dir = config.output_dir / model_label
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            for channel in config.channels:
                combo_key = (task, channel)
                print(f"\n  {task} / {channel}:")

                try:
                    # Build training dataset list for this (task, channel)
                    datasets_for_combo = []
                    for train_ds in model_spec.train_datasets:
                        channel_zarrs = _find_channel_zarrs(
                            train_ds.embeddings_dir, [channel]
                        )
                        if channel not in channel_zarrs:
                            print(
                                f"    Skipping {train_ds.embeddings_dir.parent.name}"
                                f" - no {channel} zarr"
                            )
                            continue

                        available_tasks = _get_available_tasks(train_ds.annotations)
                        if task not in available_tasks:
                            print(
                                f"    Skipping {train_ds.embeddings_dir.parent.name}"
                                f" - no {task} column"
                            )
                            continue

                        datasets_for_combo.append(
                            {
                                "embeddings": str(channel_zarrs[channel]),
                                "annotations": str(train_ds.annotations),
                            }
                        )

                    if not datasets_for_combo:
                        print("    No training datasets available, skipping.")
                        continue

                    print(f"    Training on {len(datasets_for_combo)} dataset(s)")

                    combined_adata = load_and_combine_datasets(datasets_for_combo, task)

                    classifier_params = {
                        "max_iter": config.max_iter,
                        "class_weight": config.class_weight,
                        "solver": config.solver,
                        "random_state": config.random_seed,
                    }

                    pipeline, metrics = train_linear_classifier(
                        adata=combined_adata,
                        task=task,
                        use_scaling=config.use_scaling,
                        use_pca=False,
                        classifier_params=classifier_params,
                        split_train_data=config.split_train_data,
                        random_seed=config.random_seed,
                    )

                    # Save pipeline to disk
                    pipeline_path = (
                        model_output_dir / f"{task}_{channel}_pipeline.joblib"
                    )
                    joblib.dump(pipeline, pipeline_path)
                    print(f"    Pipeline saved: {pipeline_path.name}")

                    # Build wandb config with provenance
                    embedding_model_label = f"{model_spec.name}-{model_spec.version}"
                    train_dataset_names = [
                        str(ds.embeddings_dir.parent.name)
                        for ds in model_spec.train_datasets
                    ]
                    wandb_config = {
                        "task": task,
                        "input_channel": channel,
                        "marker": None,
                        "embedding_model": embedding_model_label,
                        "embedding_model_version": model_spec.version,
                        "test_dataset": config.dataset_name,
                        "train_dataset_names": train_dataset_names,
                        "use_scaling": config.use_scaling,
                        "use_pca": False,
                        "max_iter": config.max_iter,
                        "class_weight": config.class_weight,
                        "solver": config.solver,
                        "split_train_data": config.split_train_data,
                        "random_seed": config.random_seed,
                    }

                    wandb_tags = [
                        config.dataset_name,
                        model_spec.name,
                        model_spec.version,
                        channel,
                        task,
                        "cross-dataset",
                    ]

                    artifact_name = save_pipeline_to_wandb(
                        pipeline=pipeline,
                        metrics=metrics,
                        config=wandb_config,
                        wandb_project=model_spec.wandb_project,
                        tags=wandb_tags,
                    )

                    model_results[combo_key] = {
                        "pipeline": pipeline,
                        "metrics": metrics,
                        "artifact_name": artifact_name,
                    }

                    # Print val metrics summary
                    val_acc = metrics.get("val_accuracy")
                    val_f1 = metrics.get("val_weighted_f1")
                    if val_acc is not None:
                        print(f"    Val accuracy: {val_acc:.3f}  Val F1: {val_f1:.3f}")

                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue

        all_results[model_label] = model_results

        # Save per-model metrics summary CSV
        _save_metrics_csv(model_results, model_output_dir / "metrics_summary.csv")

    # Save combined metrics comparison CSV
    _save_comparison_csv(all_results, config.output_dir / "metrics_comparison.csv")

    # Print markdown summary
    _print_training_summary(all_results, tasks, config.channels)

    return all_results


def _save_metrics_csv(
    model_results: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
) -> None:
    """Save per-model metrics to CSV."""
    rows = []
    for (task, channel), result in model_results.items():
        row = {"task": task, "channel": channel}
        row.update(result["metrics"])
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def _save_comparison_csv(
    all_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save combined metrics comparison across models."""
    rows = []
    for model_label, model_results in all_results.items():
        for (task, channel), result in model_results.items():
            row = {"model": model_label, "task": task, "channel": channel}
            row.update(result["metrics"])
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def _print_training_summary(
    all_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    tasks: list[str],
    channels: list[str],
) -> None:
    """Print markdown summary table of training results."""
    print("\n## Training Summary\n")
    header = "| Task | Channel |"
    sep = "|------|---------|"
    for model_label in all_results:
        header += f" {model_label} Val Acc | {model_label} Val F1 |"
        sep += "------------|-----------|"
    print(header)
    print(sep)

    for task in tasks:
        for channel in channels:
            row = f"| {task} | {channel} |"
            for model_label, model_results in all_results.items():
                result = model_results.get((task, channel))
                if result:
                    acc = result["metrics"].get("val_accuracy", float("nan"))
                    f1 = result["metrics"].get("val_weighted_f1", float("nan"))
                    row += f" {acc:.3f} | {f1:.3f} |"
                else:
                    row += " - | - |"
            print(row)
