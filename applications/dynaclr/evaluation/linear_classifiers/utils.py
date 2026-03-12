"""Shared utilities for the linear_classifiers workflow.

Constants, path resolution, config generation, dataset discovery,
and focus/z-range helpers used by both ``generate_batch_predictions.py``
and ``generate_train_config.py``.
"""

# %%
import re
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

# ---------------------------------------------------------------------------
# Model templates
# ---------------------------------------------------------------------------

MODEL_3D_BAG_TIMEAWARE = {
    "name": "DynaCLR-3D-BagOfChannels-timeaware",
    "in_stack_depth": 30,
    "stem_kernel_size": [5, 4, 4],
    "stem_stride": [5, 4, 4],
    "patch_size": 192,
    "data_path_type": "2-assemble",
    "z_range": "auto",
    # Fraction of z slices below the focus plane (0.33 = 1/3 below, 2/3 above).
    "focus_below_fraction": 1 / 3,
    "logger_base": "/hpc/projects/organelle_phenotyping/models/tb_logs",
}

MODEL_2D_BAG_TIMEAWARE = {
    "name": "DynaCLR-2D-BagOfChannels-timeaware",
    "in_stack_depth": 1,
    "stem_kernel_size": [1, 4, 4],
    "stem_stride": [1, 4, 4],
    "patch_size": 160,
    "data_path_type": "train-test",
    "z_range": [0, 1],
    "logger_base": "/hpc/projects/organelle_phenotyping/models/embedding_logs",
}

# ---------------------------------------------------------------------------
# Channel defaults
# ---------------------------------------------------------------------------

CHANNEL_DEFAULTS: dict[str, dict] = {
    "organelle": {
        "keyword": "GFP",
        "yaml_alias": "fluor",
        "normalization_class": "viscy.transforms.ScaleIntensityRangePercentilesd",
        "normalization_args": {
            "lower": 50,
            "upper": 99,
            "b_min": 0.0,
            "b_max": 1.0,
        },
        "batch_size": {"2d": 32, "3d": 64},
        "num_workers": {"2d": 8, "3d": 16},
    },
    "phase": {
        "keyword": "Phase",
        "yaml_alias": "Ph",
        "normalization_class": "viscy.transforms.NormalizeSampled",
        "normalization_args": {
            "level": "fov_statistics",
            "subtrahend": "mean",
            "divisor": "std",
        },
        "batch_size": {"2d": 64, "3d": 64},
        "num_workers": {"2d": 16, "3d": 16},
    },
    "sensor": {
        "keyword": "mCherry",
        "yaml_alias": "fluor",
        "normalization_class": "viscy.transforms.ScaleIntensityRangePercentilesd",
        "normalization_args": {
            "lower": 50,
            "upper": 99,
            "b_min": 0.0,
            "b_max": 1.0,
        },
        "batch_size": {"2d": 32, "3d": 64},
        "num_workers": {"2d": 8, "3d": 16},
    },
}

# ---------------------------------------------------------------------------
# Focus parameters (microscope-specific defaults)
# ---------------------------------------------------------------------------

FOCUS_PARAMS = {
    "NA_det": 1.35,
    "lambda_ill": 0.450,
    "pixel_size": 0.1494,
    "device": "cuda",
}


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def extract_epoch(ckpt_path: str) -> str:
    """Extract epoch number from a checkpoint filename.

    ``epoch=32-step=33066.ckpt`` -> ``"32"``
    """
    m = re.search(r"epoch=(\d+)", Path(ckpt_path).stem)
    if m:
        return m.group(1)
    return Path(ckpt_path).stem


# ---------------------------------------------------------------------------
# Channel utilities
# ---------------------------------------------------------------------------


def resolve_channel_name(
    channel_names: list[str],
    channel_type: str,
    channel_overrides: dict[str, str] | None = None,
) -> str | None:
    """Find the full channel name by keyword substring match.

    When multiple channels match the keyword, the ``raw`` variant is
    preferred (e.g. ``"raw GFP EX488 EM525-45"`` over ``"GFP EX488 EM525-45"``).

    Parameters
    ----------
    channel_names : list[str]
        Channel names from the zarr dataset.
    channel_type : str
        One of "organelle", "phase", "sensor".
    channel_overrides : dict[str, str] or None
        Optional mapping of channel_type -> keyword override.

    Returns
    -------
    str or None
        Matched channel name, or None if not found.
    """
    keyword = channel_overrides.get(channel_type) if channel_overrides else None
    if keyword is None:
        keyword = CHANNEL_DEFAULTS[channel_type]["keyword"]
    matches = [name for name in channel_names if keyword in name]
    if not matches:
        return None
    # Prefer the "raw" variant when both raw and processed exist
    raw = [m for m in matches if m.lower().startswith("raw")]
    return raw[0] if raw else matches[0]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_dataset_paths(
    dataset_name: str,
    base_dir: Path,
    model_config: dict,
) -> dict:
    """Resolve data_path and tracks_path for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset folder name.
    base_dir : Path
        Base directory containing all datasets.
    model_config : dict
        Model template (used to determine data_path_type).

    Returns
    -------
    dict
        Keys: data_path, tracks_path (both as Path objects).

    Raises
    ------
    FileNotFoundError
        If required paths cannot be found.
    """
    dataset_dir = base_dir / dataset_name

    # Data path
    if model_config["data_path_type"] == "train-test":
        matches = natsorted(glob(str(dataset_dir / "*phenotyping*" / "*train-test*" / f"{dataset_name}*.zarr")))
        if not matches:
            raise FileNotFoundError(f"No train-test zarr found for {dataset_name}")
        data_path = Path(matches[0])
    else:
        matches = natsorted(glob(str(dataset_dir / "2-assemble" / f"{dataset_name}*.zarr")))
        if not matches:
            raise FileNotFoundError(f"No 2-assemble zarr found for {dataset_name}")
        data_path = Path(matches[0])

    # Tracks path
    tracks_matches = natsorted(
        glob(str(dataset_dir / "1-preprocess" / "label-free" / "3-track" / f"{dataset_name}*cropped.zarr"))
    )
    if not tracks_matches:
        raise FileNotFoundError(f"No tracking zarr found for {dataset_name}")
    tracks_path = Path(tracks_matches[0])

    return {"data_path": data_path, "tracks_path": tracks_path}


def find_phenotyping_predictions_dir(
    dataset_dir: Path,
    model_name: str,
    version: str,
) -> Path:
    """Locate or create the predictions output directory for a dataset."""
    pheno_matches = natsorted(glob(str(dataset_dir / "*phenotyping*")))
    if not pheno_matches:
        pheno_dir = dataset_dir / "4-phenotyping"
    else:
        pheno_dir = Path(pheno_matches[0])

    pred_matches = natsorted(glob(str(pheno_dir / "*prediction*")))
    pred_parent = Path(pred_matches[0]) if pred_matches else pheno_dir / "predictions"

    return pred_parent / model_name / version


# ---------------------------------------------------------------------------
# Focus / z-range
# ---------------------------------------------------------------------------


def get_z_range(
    data_path: str | Path,
    model_config: dict,
    focus_params: dict | None = None,
    phase_channel: str | None = None,
) -> list[int]:
    """Determine z_range for prediction.

    For models with ``z_range="auto"``, reads focus_slice metadata from the
    zarr. If metadata is missing, computes it on the fly.

    Parameters
    ----------
    data_path : str or Path
        Path to the OME-Zarr dataset.
    model_config : dict
        Model template dictionary.
    focus_params : dict or None
        Parameters for on-the-fly focus computation.
    phase_channel : str or None
        Name of the phase channel in the zarr. Used to look up focus_slice
        metadata. If None, auto-detected by keyword match.

    Returns
    -------
    list[int]
        [z_start, z_end] range for prediction.
    """
    from iohub import open_ome_zarr

    if model_config["z_range"] != "auto":
        return list(model_config["z_range"])

    plate = open_ome_zarr(str(data_path), mode="r")

    # Resolve phase channel name if not provided
    if phase_channel is None:
        phase_channel = resolve_channel_name(list(plate.channel_names), "phase")
    if phase_channel is None:
        plate.close()
        raise ValueError(f"Cannot determine z_range: no phase channel found in {data_path}")

    focus_data = plate.zattrs.get("focus_slice", {})
    phase_stats = focus_data.get(phase_channel, {}).get("dataset_statistics", {})
    z_focus_mean = phase_stats.get("z_focus_mean")

    # Get total z depth from first position
    for _, pos in plate.positions():
        z_total = pos["0"].shape[2]
        break
    plate.close()

    if z_focus_mean is None:
        print(f"  Focus metadata missing for {Path(data_path).name}, computing...")
        z_focus_mean = _compute_focus(str(data_path), focus_params or FOCUS_PARAMS, phase_channel)

    depth = model_config["in_stack_depth"]
    below_frac = model_config.get("focus_below_fraction", 0.5)
    slices_below = int(round(depth * below_frac))
    z_center = int(round(z_focus_mean))
    z_start = max(0, z_center - slices_below)
    z_end = min(z_total, z_start + depth)
    # Re-adjust start if we hit the ceiling
    z_start = max(0, z_end - depth)

    return [z_start, z_end]


def _compute_focus(zarr_path: str, focus_params: dict, phase_channel: str) -> float:
    """Compute focus_slice metadata and write it to the zarr.

    Returns the dataset-level z_focus_mean.
    """
    from iohub import open_ome_zarr

    from qc.focus import FocusSliceMetric
    from qc.qc_metrics import generate_qc_metadata

    metric = FocusSliceMetric(
        NA_det=focus_params["NA_det"],
        lambda_ill=focus_params["lambda_ill"],
        pixel_size=focus_params["pixel_size"],
        channel_names=[phase_channel],
        device=focus_params.get("device", "cpu"),
    )
    generate_qc_metadata(zarr_path, [metric])

    plate = open_ome_zarr(zarr_path, mode="r")
    z_focus_mean = plate.zattrs["focus_slice"][phase_channel]["dataset_statistics"]["z_focus_mean"]
    plate.close()
    return z_focus_mean


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def model_dim_key(model_config: dict) -> str:
    """Return '2d' or '3d' based on model template."""
    return "2d" if model_config["in_stack_depth"] == 1 else "3d"


def generate_yaml(
    dataset_name: str,
    data_path: Path,
    tracks_path: Path,
    model_config: dict,
    channel_type: str,
    channel_name: str,
    z_range: list[int],
    ckpt_path: str,
    output_dir: Path,
    version: str,
) -> str:
    """Generate a prediction YAML config string.

    Uses YAML anchors to match the existing config style.
    """
    dim = model_dim_key(model_config)
    ch_cfg = CHANNEL_DEFAULTS[channel_type]
    patch = model_config["patch_size"]
    depth = model_config["in_stack_depth"]
    epoch = extract_epoch(ckpt_path)
    yaml_alias = ch_cfg["yaml_alias"]

    output_zarr = output_dir / f"timeaware_{channel_type}_{patch}patch_{epoch}ckpt.zarr"

    # Build normalization block
    norm_class = ch_cfg["normalization_class"]
    norm_args = dict(ch_cfg["normalization_args"])

    # Format normalization init_args as YAML lines
    norm_lines = [f"          keys: [*{yaml_alias}]"]
    for k, v in norm_args.items():
        norm_lines.append(f"          {k}: {v}")
    norm_block = "\n".join(norm_lines)

    logger_base = model_config["logger_base"]
    model_name = model_config["name"]
    logger_save_dir = f"{logger_base}/{dataset_name}"
    logger_name = f"{model_name}/{version}/{channel_type}"

    yaml_str = f"""\
seed_everything: 42
trainer:
  accelerator: gpu
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 32-true
  callbacks:
    - class_path: viscy.representation.embedding_writer.EmbeddingWriter
      init_args:
        output_path: "{output_zarr}"
  logger:
    save_dir: "{logger_save_dir}"
    name: "{logger_name}"
  inference_mode: true
model:
  class_path: viscy.representation.engine.ContrastiveModule
  init_args:
    encoder:
      class_path: viscy.representation.contrastive.ContrastiveEncoder
      init_args:
        backbone: convnext_tiny
        in_channels: 1
        in_stack_depth: {depth}
        stem_kernel_size: {model_config["stem_kernel_size"]}
        stem_stride: {model_config["stem_stride"]}
        embedding_dim: 768
        projection_dim: 32
        drop_path_rate: 0.0
    example_input_array_shape: [1, 1, {depth}, {patch}, {patch}]
data:
  class_path: viscy.data.triplet.TripletDataModule
  init_args:
    data_path: {data_path}
    tracks_path: {tracks_path}
    source_channel:
      - &{yaml_alias} {channel_name}
    z_range: {z_range}
    batch_size: {ch_cfg["batch_size"][dim]}
    num_workers: {ch_cfg["num_workers"][dim]}
    initial_yx_patch_size: [{patch}, {patch}]
    final_yx_patch_size: [{patch}, {patch}]
    normalizations:
      - class_path: {norm_class}
        init_args:
{norm_block}
return_predictions: false
ckpt_path: {ckpt_path}
"""
    return yaml_str


def generate_slurm_script(
    channel_type: str,
    output_dir: Path,
    suffix: str = "",
) -> str:
    """Generate a SLURM submission shell script."""
    config_file = output_dir / f"predict_{channel_type}{suffix}.yml"
    slurm_out = output_dir / "slurm_out" / "pred_%j.out"

    return f"""\
#!/bin/bash

#SBATCH --job-name=dynaclr_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-02:00:00
#SBATCH --output={slurm_out}

module load anaconda/latest
conda activate viscy

cat {config_file}
srun viscy predict -c {config_file}
"""


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


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
