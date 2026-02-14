"""Evaluation pipeline comparing 2D vs 3D linear classifiers on cell embeddings.

Trains linear classifiers on cross-dataset embeddings, runs inference on a
held-out test dataset, evaluates predictions, and generates a PDF comparison
report. Each block can be run independently or chained via ``run_evaluation()``.
"""

from pathlib import Path
from typing import Any, Optional

import anndata as ad
import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.metrics import classification_report

from viscy.representation.evaluation.linear_classifier import (
    LinearClassifierPipeline,
    load_and_combine_datasets,
    predict_with_classifier,
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


# ---------------------------------------------------------------------------
# Block 2: Inference
# ---------------------------------------------------------------------------


def infer_classifiers(
    config: DatasetEvalConfig,
    trained: dict[str, dict[tuple[str, str], dict[str, Any]]] | None = None,
) -> dict[str, dict[tuple[str, str], ad.AnnData]]:
    """Apply trained classifiers to held-out test embeddings.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    trained : dict or None
        Output from ``train_classifiers()``. If ``None``, loads pipelines
        from disk.

    Returns
    -------
    dict[str, dict[tuple[str, str], ad.AnnData]]
        ``model_label -> (task, channel) -> adata`` with predictions.
    """
    tasks = _resolve_tasks(config)
    print("\n## Running inference on test dataset")

    all_predictions: dict[str, dict[tuple[str, str], ad.AnnData]] = {}

    for model_label, model_spec in config.models.items():
        print(f"\n### Model: {model_label} ({model_spec.name})")
        model_predictions: dict[tuple[str, str], ad.AnnData] = {}
        model_output_dir = config.output_dir / model_label
        model_output_dir.mkdir(parents=True, exist_ok=True)

        test_channel_zarrs = _find_channel_zarrs(
            model_spec.test_embeddings_dir, config.channels
        )

        for task in tasks:
            for channel in config.channels:
                combo_key = (task, channel)

                if channel not in test_channel_zarrs:
                    print(f"  {task} / {channel}: no test zarr, skipping.")
                    continue

                # Get pipeline: from in-memory results or disk
                pipeline = None
                artifact_metadata = None

                if trained and model_label in trained:
                    result = trained[model_label].get(combo_key)
                    if result:
                        pipeline = result["pipeline"]
                        artifact_name = result.get("artifact_name")
                        if artifact_name:
                            artifact_metadata = {
                                "artifact_name": artifact_name,
                                "artifact_id": artifact_name,
                                "artifact_version": "local",
                            }

                if pipeline is None:
                    pipeline_path = (
                        model_output_dir / f"{task}_{channel}_pipeline.joblib"
                    )
                    if not pipeline_path.exists():
                        print(f"  {task} / {channel}: no trained pipeline, skipping.")
                        continue
                    pipeline = joblib.load(pipeline_path)
                    if not isinstance(pipeline, LinearClassifierPipeline):
                        print(f"  {task} / {channel}: invalid pipeline file, skipping.")
                        continue

                try:
                    print(f"  {task} / {channel}: loading test embeddings...")
                    adata = ad.read_zarr(test_channel_zarrs[channel])

                    adata = predict_with_classifier(
                        adata,
                        pipeline,
                        task,
                        artifact_metadata=artifact_metadata,
                    )

                    # Save predictions zarr
                    pred_path = model_output_dir / f"{task}_{channel}_predictions.zarr"
                    adata.write_zarr(pred_path)
                    print(f"  {task} / {channel}: saved {pred_path.name}")

                    model_predictions[combo_key] = adata

                except Exception as e:
                    print(f"  {task} / {channel}: FAILED - {e}")
                    continue

        all_predictions[model_label] = model_predictions

    return all_predictions


# ---------------------------------------------------------------------------
# Block 3: Evaluate predictions
# ---------------------------------------------------------------------------


def evaluate_predictions(
    config: DatasetEvalConfig,
    predictions: dict[str, dict[tuple[str, str], ad.AnnData]] | None = None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Evaluate predictions against held-out test annotations.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    predictions : dict or None
        Output from ``infer_classifiers()``. If ``None``, loads prediction
        zarrs from disk.

    Returns
    -------
    dict[str, dict[tuple[str, str], dict]]
        ``model_label -> (task, channel) -> eval_dict`` with keys:
        ``"metrics"``, ``"annotated_adata"``.
    """
    from viscy.representation.evaluation import load_annotation_anndata

    tasks = _resolve_tasks(config)
    print("\n## Evaluating predictions on test set")

    all_eval: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for model_label in config.models:
        print(f"\n### Model: {model_label}")
        model_eval: dict[tuple[str, str], dict[str, Any]] = {}
        model_output_dir = config.output_dir / model_label

        for task in tasks:
            for channel in config.channels:
                combo_key = (task, channel)

                # Get prediction adata: from in-memory or disk
                adata = None
                if predictions and model_label in predictions:
                    adata = predictions[model_label].get(combo_key)

                if adata is None:
                    pred_path = model_output_dir / f"{task}_{channel}_predictions.zarr"
                    if not pred_path.exists():
                        print(f"  {task} / {channel}: no predictions found, skipping.")
                        continue
                    adata = ad.read_zarr(pred_path)

                try:
                    annotated = load_annotation_anndata(
                        adata, str(config.test_annotations_csv), task
                    )

                    # Filter to cells with non-NaN ground truth
                    mask = annotated.obs[task].notna() & (
                        annotated.obs[task] != "unknown"
                    )
                    eval_subset = annotated[mask]

                    if len(eval_subset) == 0:
                        print(
                            f"  {task} / {channel}: "
                            "no annotated cells after filtering, skipping."
                        )
                        continue

                    pred_col = f"predicted_{task}"
                    y_true = eval_subset.obs[task].values
                    y_pred = eval_subset.obs[pred_col].values

                    report = classification_report(
                        y_true, y_pred, digits=3, output_dict=True
                    )

                    test_metrics = {
                        "test_accuracy": report["accuracy"],
                        "test_weighted_precision": report["weighted avg"]["precision"],
                        "test_weighted_recall": report["weighted avg"]["recall"],
                        "test_weighted_f1": report["weighted avg"]["f1-score"],
                        "test_n_samples": len(eval_subset),
                    }

                    for class_name in sorted(set(y_true) | set(y_pred)):
                        if class_name in report:
                            test_metrics[f"test_{class_name}_precision"] = report[
                                class_name
                            ]["precision"]
                            test_metrics[f"test_{class_name}_recall"] = report[
                                class_name
                            ]["recall"]
                            test_metrics[f"test_{class_name}_f1"] = report[class_name][
                                "f1-score"
                            ]

                    # Save annotated adata
                    annotated_path = (
                        model_output_dir / f"{task}_{channel}_annotated.zarr"
                    )
                    annotated.write_zarr(annotated_path)

                    model_eval[combo_key] = {
                        "metrics": test_metrics,
                        "annotated_adata": annotated,
                    }

                    acc = test_metrics["test_accuracy"]
                    f1 = test_metrics["test_weighted_f1"]
                    n = test_metrics["test_n_samples"]
                    print(f"  {task} / {channel}: acc={acc:.3f}  F1={f1:.3f}  (n={n})")

                except Exception as e:
                    print(f"  {task} / {channel}: FAILED - {e}")
                    continue

        all_eval[model_label] = model_eval

        # Save per-model test metrics CSV
        rows = []
        for (task, channel), result in model_eval.items():
            row = {"task": task, "channel": channel}
            row.update(result["metrics"])
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(model_output_dir / "test_metrics_summary.csv", index=False)

    # Save combined test metrics comparison
    rows = []
    for model_label, model_eval in all_eval.items():
        for (task, channel), result in model_eval.items():
            row = {"model": model_label, "task": task, "channel": channel}
            row.update(result["metrics"])
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(config.output_dir / "test_metrics_comparison.csv", index=False)

    return all_eval


# ---------------------------------------------------------------------------
# Block 4: Report generation
# ---------------------------------------------------------------------------


def generate_report(
    config: DatasetEvalConfig,
    train_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
    eval_results: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> Path:
    """Generate a PDF comparison report.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    train_results : dict
        Output from ``train_classifiers()``.
    eval_results : dict
        Output from ``evaluate_predictions()``.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    from applications.DynaCLR.evaluation.linear_classifiers.report import (
        generate_comparison_report,
    )

    return generate_comparison_report(config, train_results, eval_results)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_evaluation(
    config: DatasetEvalConfig,
    skip_train: bool = False,
    skip_infer: bool = False,
) -> Path:
    """Run the evaluation pipeline.

    Parameters
    ----------
    config : DatasetEvalConfig
        Evaluation configuration.
    skip_train : bool
        Skip training. Loads pipelines from disk for inference.
    skip_infer : bool
        Skip inference. Loads prediction zarrs from disk for evaluation.
        Implies ``skip_train=True``.

    Returns
    -------
    Path
        Path to the generated PDF report.
    """
    trained = None
    if not skip_train and not skip_infer:
        trained = train_classifiers(config)

    predictions = None
    if not skip_infer:
        predictions = infer_classifiers(config, trained=trained)

    eval_results = evaluate_predictions(config, predictions=predictions)
    report_path = generate_report(config, trained or {}, eval_results)
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

EMBEDDINGS_BASE = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
ANNOTATIONS_BASE = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")
OUTPUT_BASE = Path(
    "/hpc/projects/organelle_phenotyping/models/bag_of_channels/"
    "h2b_caax_tomm_sec61_g3bp1_sensor_phase/evaluation/predictions"
)

TEST_DATASET = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"

# Training datasets per model (excluding test dataset)
TRAIN_DATASETS_2D = [
    "2024_11_07_A549_SEC61_DENV",
    "2025_01_24_A549_G3BP1_DENV",
    "2025_01_28_A549_G3BP1_ZIKV_DENV",
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV",
    "2025_08_26_A549_SEC61_TOMM20_ZIKV",
]

TRAIN_DATASETS_3D = [
    "2024_11_07_A549_SEC61_DENV",
    "2025_01_28_A549_G3BP1_ZIKV_DENV",
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV",
    "2025_08_26_A549_SEC61_TOMM20_ZIKV",
]


def _find_predictions_dir(dataset_name: str, model_name: str, version: str) -> Path:
    """Locate the predictions version directory for a dataset."""
    from glob import glob

    from natsort import natsorted

    dataset_dir = EMBEDDINGS_BASE / dataset_name
    pattern = str(dataset_dir / "*phenotyping*" / "*prediction*" / model_name / version)
    matches = natsorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No predictions found for {dataset_name}/{model_name}/{version}"
        )
    return Path(matches[0])


def _build_train_datasets(
    dataset_names: list[str], model_name: str, version: str
) -> list[TrainDataset]:
    """Build TrainDataset list from dataset names."""
    from applications.DynaCLR.evaluation.linear_classifiers.utils import (
        find_annotation_csv,
    )

    datasets = []
    for name in dataset_names:
        try:
            emb_dir = _find_predictions_dir(name, model_name, version)
            csv_path = find_annotation_csv(ANNOTATIONS_BASE, name)
            if csv_path is None:
                print(f"  Skipping {name}: no annotation CSV found")
                continue
            datasets.append(TrainDataset(embeddings_dir=emb_dir, annotations=csv_path))
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            continue
    return datasets


def build_default_config() -> DatasetEvalConfig:
    """Build the default evaluation config for the 2D vs 3D comparison."""
    from applications.DynaCLR.evaluation.linear_classifiers.utils import (
        find_annotation_csv,
    )

    test_csv = find_annotation_csv(ANNOTATIONS_BASE, TEST_DATASET)
    if test_csv is None:
        raise FileNotFoundError(f"No annotation CSV for test dataset: {TEST_DATASET}")

    model_2d = ModelSpec(
        name="DynaCLR-2D-BagOfChannels-timeaware",
        train_datasets=_build_train_datasets(
            TRAIN_DATASETS_2D, "DynaCLR-2D-BagOfChannels-timeaware", "v3"
        ),
        test_embeddings_dir=_find_predictions_dir(
            TEST_DATASET, "DynaCLR-2D-BagOfChannels-timeaware", "v3"
        ),
        version="v3",
        wandb_project="DynaCLR-2D-linearclassifiers",
    )

    model_3d = ModelSpec(
        name="DynaCLR-3D-BagOfChannels-timeaware",
        train_datasets=_build_train_datasets(
            TRAIN_DATASETS_3D, "DynaCLR-3D-BagOfChannels-timeaware", "v1"
        ),
        test_embeddings_dir=_find_predictions_dir(
            TEST_DATASET, "DynaCLR-3D-BagOfChannels-timeaware", "v1"
        ),
        version="v1",
        wandb_project="DynaCLR-3D-linearclassifiers",
    )

    return DatasetEvalConfig(
        dataset_name=TEST_DATASET,
        test_annotations_csv=test_csv,
        models={"2D": model_2d, "3D": model_3d},
        output_dir=OUTPUT_BASE / TEST_DATASET,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run eval pipeline")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, load pipelines from disk",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference, load predictions from disk (implies --skip-train)",
    )
    args = parser.parse_args()

    config = build_default_config()
    print(f"Output: {config.output_dir}")
    for label, spec in config.models.items():
        print(f"  {label}: {len(spec.train_datasets)} training datasets")

    report = run_evaluation(
        config, skip_train=args.skip_train, skip_infer=args.skip_infer
    )
    print(f"\nDone! Report: {report}")
