"""Shared discovery functions for finding predictions, annotations, and gaps."""

# %%
from glob import glob
from pathlib import Path

import pandas as pd
from natsort import natsorted

from viscy_utils.evaluation.linear_classifier_config import (
    VALID_CHANNELS,
    VALID_TASKS,
)

CHANNELS = list(VALID_CHANNELS.__args__)
TASKS = list(VALID_TASKS.__args__)


def discover_predictions(
    embeddings_dir: Path,
    model_name: str,
    version: str,
) -> dict[str, Path]:
    """Find datasets that have a predictions folder for the given model/version.

    Searches for paths matching:
        {embeddings_dir}/{dataset}/*phenotyping*/*prediction*/{model_glob}/{version}/

    Parameters
    ----------
    embeddings_dir : Path
        Base directory containing dataset folders.
    model_name : str
        Model directory name (supports glob patterns).
    version : str
        Version subdirectory (e.g. "v3").

    Returns
    -------
    dict[str, Path]
        Mapping of dataset_name -> resolved predictions version directory.
    """
    pattern = str(embeddings_dir / "*" / "*phenotyping*" / "*prediction*" / model_name / version)
    matches = natsorted(glob(pattern))

    results = {}
    for match in matches:
        match_path = Path(match)
        dataset_name = match_path.relative_to(embeddings_dir).parts[0]
        results[dataset_name] = match_path

    return results


def find_channel_zarrs(
    predictions_dir: Path,
    channels: list[str] | None = None,
) -> dict[str, Path]:
    """Find embedding zarr files for each channel in a predictions directory.

    Parameters
    ----------
    predictions_dir : Path
        Path to the version directory containing zarr files.
    channels : list[str] or None
        Channel names to search for. Defaults to CHANNELS.

    Returns
    -------
    dict[str, Path]
        Mapping of channel_name -> zarr path (only channels with a match).
    """
    if channels is None:
        channels = CHANNELS
    channel_zarrs = {}
    for channel in channels:
        matches = natsorted(glob(str(predictions_dir / f"*{channel}*.zarr")))
        if matches:
            channel_zarrs[channel] = Path(matches[0])
    return channel_zarrs


def find_annotation_csv(annotations_dir: Path, dataset_name: str) -> Path | None:
    """Find the annotation CSV for a dataset.

    Parameters
    ----------
    annotations_dir : Path
        Base annotations directory.
    dataset_name : str
        Dataset folder name.

    Returns
    -------
    Path or None
        Path to CSV if found, None otherwise.
    """
    dataset_dir = annotations_dir / dataset_name
    if not dataset_dir.is_dir():
        return None
    csvs = natsorted(glob(str(dataset_dir / "*.csv")))
    return Path(csvs[0]) if csvs else None


def get_available_tasks(csv_path: Path) -> list[str]:
    """Read CSV header and return which valid task columns are present.

    Parameters
    ----------
    csv_path : Path
        Path to annotation CSV.

    Returns
    -------
    list[str]
        Task names found in the CSV columns.
    """
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [t for t in TASKS if t in columns]


def build_registry(
    embeddings_dir: Path,
    annotations_dir: Path,
    model_name: str,
    version: str,
) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """Build a registry of datasets with predictions and annotations.

    Parameters
    ----------
    embeddings_dir : Path
        Base directory containing dataset folders with embeddings.
    annotations_dir : Path
        Base directory containing dataset annotation folders.
    model_name : str
        Model directory name (supports glob patterns).
    version : str
        Version subdirectory (e.g. "v3").

    Returns
    -------
    registry : list[dict]
        Datasets with both predictions and annotations.
    skipped : list[dict]
        Datasets with predictions but missing annotations or tasks.
    annotations_only : list[str]
        Annotation datasets with no matching predictions.
    predictions_only : list[str]
        Prediction datasets with no matching annotations.
    """
    predictions = discover_predictions(embeddings_dir, model_name, version)

    registry: list[dict] = []
    skipped: list[dict] = []

    for dataset_name, pred_dir in predictions.items():
        channel_zarrs = find_channel_zarrs(pred_dir)
        csv_path = find_annotation_csv(annotations_dir, dataset_name)

        if not csv_path:
            skipped.append({"dataset": dataset_name, "reason": "No annotation CSV"})
            continue
        if not channel_zarrs:
            skipped.append({"dataset": dataset_name, "reason": "No channel zarrs"})
            continue

        available_tasks = get_available_tasks(csv_path)
        if not available_tasks:
            skipped.append({"dataset": dataset_name, "reason": "No valid task columns in CSV"})
            continue

        registry.append(
            {
                "dataset": dataset_name,
                "predictions_dir": pred_dir,
                "channel_zarrs": channel_zarrs,
                "annotations_csv": csv_path,
                "available_tasks": available_tasks,
            }
        )

    annotation_datasets = set(d.name for d in annotations_dir.iterdir() if d.is_dir())
    prediction_datasets = set(predictions.keys())

    annotations_only = natsorted(annotation_datasets - prediction_datasets)
    predictions_only = natsorted(prediction_datasets - annotation_datasets)

    return registry, skipped, annotations_only, predictions_only


def print_registry_summary(
    registry: list[dict],
    skipped: list[dict],
    annotations_only: list[str],
    predictions_only: list[str],
):
    """Print a markdown summary of the dataset registry and gaps."""
    print("## Dataset Registry\n")
    print("| Dataset | Annotations | Channels | Tasks |")
    print("|---------|-------------|----------|-------|")
    for entry in registry:
        channels_str = ", ".join(sorted(entry["channel_zarrs"].keys()))
        tasks_str = ", ".join(entry["available_tasks"])
        print(f"| {entry['dataset']} | {entry['annotations_csv'].name} | {channels_str} | {tasks_str} |")

    if annotations_only or predictions_only or skipped:
        print("\n## Gaps\n")
        print("| Dataset | Status |")
        print("|---------|--------|")
        for d in annotations_only:
            print(f"| {d} | Annotations only (missing predictions) |")
        for d in predictions_only:
            print(f"| {d} | Predictions only (missing annotations) |")
        for s in skipped:
            print(f"| {s['dataset']} | {s['reason']} |")


# %%
