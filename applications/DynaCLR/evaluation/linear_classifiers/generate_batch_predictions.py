# %%
"""Batch DynaCLR prediction config & SLURM script generator.

Generates prediction YAML configs and SLURM submission scripts for
multiple datasets, channels, and checkpoints. Automatically resolves
z_range from focus_slice metadata (computing it on the fly if missing)
and detects source channel names from the zarr.

Usage: run cells interactively or execute as a script.
"""

import re
from glob import glob
from pathlib import Path

from iohub import open_ome_zarr
from natsort import natsorted

# %%
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

# %%
# ---------------------------------------------------------------------------
# Channel defaults
# ---------------------------------------------------------------------------
# keyword: substring matched against the zarr channel_names
# yaml_alias: short name used as a YAML &alias/*reference for the channel

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

# %%
# ---------------------------------------------------------------------------
# Focus parameters (for computing focus_slice on the fly when metadata is
# missing). These are microscope-specific.
# ---------------------------------------------------------------------------

FOCUS_PARAMS = {
    "NA_det": 1.35,
    "lambda_ill": 0.450,
    "pixel_size": 0.1494,
    "device": "cuda",
}

# %%
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_epoch(ckpt_path: str) -> str:
    """Extract epoch number from a checkpoint filename.

    ``epoch=32-step=33066.ckpt`` -> ``"32"``
    """
    m = re.search(r"epoch=(\d+)", Path(ckpt_path).stem)
    if m:
        return m.group(1)
    return Path(ckpt_path).stem


def resolve_channel_name(
    channel_names: list[str],
    channel_type: str,
    channel_overrides: dict[str, str] | None = None,
) -> str | None:
    """Find the full channel name by keyword substring match.

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
    for name in channel_names:
        if keyword in name:
            return name
    return None


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
    if model_config["z_range"] != "auto":
        return list(model_config["z_range"])

    plate = open_ome_zarr(str(data_path), mode="r")

    # Resolve phase channel name if not provided
    if phase_channel is None:
        phase_channel = resolve_channel_name(list(plate.channel_names), "phase")
    if phase_channel is None:
        plate.close()
        raise ValueError(
            f"Cannot determine z_range: no phase channel found in {data_path}"
        )

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
        z_focus_mean = _compute_focus(
            str(data_path), focus_params or FOCUS_PARAMS, phase_channel
        )

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
    from viscy.preprocessing.focus import FocusSliceMetric
    from viscy.preprocessing.qc_metrics import generate_qc_metadata

    metric = FocusSliceMetric(
        NA_det=focus_params["NA_det"],
        lambda_ill=focus_params["lambda_ill"],
        pixel_size=focus_params["pixel_size"],
        channel_names=[phase_channel],
        device=focus_params.get("device", "cpu"),
    )
    generate_qc_metadata(zarr_path, [metric])

    plate = open_ome_zarr(zarr_path, mode="r")
    z_focus_mean = plate.zattrs["focus_slice"][phase_channel]["dataset_statistics"][
        "z_focus_mean"
    ]
    plate.close()
    return z_focus_mean


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
        matches = natsorted(
            glob(
                str(
                    dataset_dir
                    / "*phenotyping*"
                    / "*train-test*"
                    / f"{dataset_name}*.zarr"
                )
            )
        )
        if not matches:
            raise FileNotFoundError(f"No train-test zarr found for {dataset_name}")
        data_path = Path(matches[0])
    else:
        matches = natsorted(
            glob(str(dataset_dir / "2-assemble" / f"{dataset_name}*.zarr"))
        )
        if not matches:
            raise FileNotFoundError(f"No 2-assemble zarr found for {dataset_name}")
        data_path = Path(matches[0])

    # Tracks path
    tracks_matches = natsorted(
        glob(
            str(
                dataset_dir
                / "1-preprocess"
                / "label-free"
                / "3-track"
                / f"{dataset_name}*cropped.zarr"
            )
        )
    )
    if not tracks_matches:
        raise FileNotFoundError(f"No tracking zarr found for {dataset_name}")
    tracks_path = Path(tracks_matches[0])

    return {"data_path": data_path, "tracks_path": tracks_path}


def _model_dim_key(model_config: dict) -> str:
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
    dim = _model_dim_key(model_config)
    ch_cfg = CHANNEL_DEFAULTS[channel_type]
    patch = model_config["patch_size"]
    depth = model_config["in_stack_depth"]
    epoch = _extract_epoch(ckpt_path)
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
    suffix: str = "",
) -> str:
    """Generate a SLURM submission shell script."""
    config_file = f"predict_{channel_type}{suffix}.yml"

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
#SBATCH --output=./slurm_out/pred_%j.out

module load anaconda/latest
conda activate viscy

cat {config_file}
srun viscy predict -c {config_file}
"""


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


# %%
# ===========================================================================
# USER CONFIGURATION — edit this cell
# ===========================================================================

BASE_DIR = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")

# Choose model template
MODEL = MODEL_3D_BAG_TIMEAWARE
# MODEL = MODEL_2D_BAG_TIMEAWARE

VERSION = "v1"

CHANNELS = ["phase", "organelle", "sensor"]

CHECKPOINTS = [
    "/hpc/projects/organelle_phenotyping/models/bag_of_channels/h2b_caax_tomm_sec61_g3bp1_sensor_phase/tb_logs/dynaclr3d_bag_channels_v1/version_2/checkpoints/epoch=40-step=44746.ckpt",
]

# Datasets to process. Set to [] to auto-discover from annotations_only.
DATASETS = ["2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"]

# Per-dataset channel keyword overrides.
# E.g., {"2025_04_10_...": {"organelle": "Cy5"}}
CHANNEL_OVERRIDES: dict[str, dict[str, str]] = {}

# Annotations directory (used for auto-discovery when DATASETS is empty).
ANNOTATIONS_DIR = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")

# Set to True for a dry run (preview only, no files written).
DRY_RUN = False

# Set to True to overwrite existing config files. False to skip them.
OVERWRITE_FILES = True

# %%
# ===========================================================================
# Discovery & validation
# ===========================================================================

# Auto-discover datasets from annotations when DATASETS is empty
if not DATASETS:
    from dataset_discovery import build_registry, print_registry_summary

    registry, skipped, annotations_only, predictions_only = build_registry(
        BASE_DIR, ANNOTATIONS_DIR, MODEL["name"], VERSION
    )
    print_registry_summary(registry, skipped, annotations_only, predictions_only)
    DATASETS = annotations_only
    print(f"\nAuto-discovered {len(DATASETS)} datasets missing predictions.\n")

print("## Batch Prediction Config Generator\n")
print(f"- **Model**: `{MODEL['name']}`")
print(f"- **Version**: `{VERSION}`")
print(f"- **Channels**: {CHANNELS}")
print(f"- **Checkpoints**: {len(CHECKPOINTS)}")
print(f"- **Datasets**: {len(DATASETS)}")
print(f"- **Dry run**: {DRY_RUN}\n")

validated: list[dict] = []
errors: list[dict] = []

for ds in DATASETS:
    try:
        paths = resolve_dataset_paths(ds, BASE_DIR, MODEL)
        print(f"Resolving {ds}...")

        # Read channel names from data zarr
        plate = open_ome_zarr(str(paths["data_path"]), mode="r")
        zarr_channels = list(plate.channel_names)
        plate.close()

        # Resolve channel names
        ds_overrides = CHANNEL_OVERRIDES.get(ds)
        available = {}
        for ch_type in CHANNELS:
            ch_name = resolve_channel_name(zarr_channels, ch_type, ds_overrides)
            if ch_name:
                available[ch_type] = ch_name
            else:
                print(f"  WARNING: channel '{ch_type}' not found in {ds}")

        # Resolve z_range (may compute focus on the fly)
        phase_ch = available.get("phase")
        z_range = get_z_range(
            paths["data_path"], MODEL, FOCUS_PARAMS, phase_channel=phase_ch
        )
        print(f"  z_range: {z_range}")

        validated.append(
            {
                "dataset": ds,
                "paths": paths,
                "z_range": z_range,
                "channels": available,
            }
        )

    except Exception as e:
        errors.append({"dataset": ds, "error": str(e)})
        print(f"  ERROR: {e}")

# %%
# ===========================================================================
# Summary before generation
# ===========================================================================

print("\n### Validated Datasets\n")
print("| Dataset | z_range | Channels | data_path |")
print("|---------|---------|----------|-----------|")
for v in validated:
    ch_str = ", ".join(sorted(v["channels"].keys()))
    print(
        f"| {v['dataset']} | {v['z_range']} | {ch_str} | `{v['paths']['data_path'].name}` |"
    )

if errors:
    print("\n### Errors\n")
    print("| Dataset | Error |")
    print("|---------|-------|")
    for e in errors:
        print(f"| {e['dataset']} | {e['error']} |")

print(
    f"\n**Will generate**: {len(validated)} datasets "
    f"x {len(CHECKPOINTS)} checkpoints "
    f"= {len(validated) * len(CHECKPOINTS)} config sets"
)

# %%
# ===========================================================================
# Generate configs and scripts
# ===========================================================================

generated: list[dict] = []

for entry in validated:
    ds = entry["dataset"]
    paths = entry["paths"]
    z_range = entry["z_range"]
    channels = entry["channels"]

    output_dir = find_phenotyping_predictions_dir(BASE_DIR / ds, MODEL["name"], VERSION)

    # TODO: support multiple checkpoints (namespace files by epoch or subdirs)
    for ckpt in CHECKPOINTS:
        epoch = _extract_epoch(ckpt)
        suffix = ""
        files_written = []

        for ch_type, ch_name in channels.items():
            yml_content = generate_yaml(
                ds,
                paths["data_path"],
                paths["tracks_path"],
                MODEL,
                ch_type,
                ch_name,
                z_range,
                ckpt,
                output_dir,
                VERSION,
            )
            sh_content = generate_slurm_script(ch_type, suffix=suffix)

            yml_path = output_dir / f"predict_{ch_type}{suffix}.yml"
            sh_path = output_dir / f"predict_{ch_type}{suffix}.sh"

            if not OVERWRITE_FILES and yml_path.exists():
                print(f"  Skipping {yml_path.name} (exists)")
                continue

            if not DRY_RUN:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "slurm_out").mkdir(exist_ok=True)
                yml_path.write_text(yml_content)
                sh_path.write_text(sh_content)
                sh_path.chmod(0o755)

            files_written.append(
                {
                    "channel": ch_type,
                    "yml": yml_path,
                    "sh": sh_path,
                    "yml_content": yml_content,
                    "sh_content": sh_content,
                }
            )

        # predict_all.sh
        if files_written:
            run_all_lines = ["#!/bin/bash", ""]
            for f in files_written:
                run_all_lines.append(f"sbatch {f['sh'].name}")
            run_all_content = "\n".join(run_all_lines) + "\n"

            run_all_name = f"predict_all{suffix}.sh"
            run_all_path = output_dir / run_all_name
            if not DRY_RUN:
                run_all_path.write_text(run_all_content)
                run_all_path.chmod(0o755)

            generated.append(
                {
                    "dataset": ds,
                    "checkpoint": ckpt,
                    "epoch": epoch,
                    "output_dir": output_dir,
                    "files": files_written,
                }
            )

# %%
# ===========================================================================
# Generation summary
# ===========================================================================

action = "Generated" if not DRY_RUN else "Would generate (DRY RUN)"
print(f"\n## {action}\n")
print("| Dataset | Epoch | Channels | Output Dir |")
print("|---------|-------|----------|------------|")
for g in generated:
    ch_str = ", ".join(f["channel"] for f in g["files"])
    print(f"| {g['dataset']} | {g['epoch']} | {ch_str} | `{g['output_dir']}` |")

print("\n### Files\n")
for g in generated:
    print(f"**{g['dataset']}** (epoch {g['epoch']}):")
    for f in g["files"]:
        print(f"  - `{f['yml']}`")
        print(f"  - `{f['sh']}`")
    print(f"  - `{g['output_dir'] / 'predict_all.sh'}`")

if DRY_RUN and generated:
    print("\n### Preview (first config)\n")
    print("```yaml")
    print(generated[0]["files"][0]["yml_content"])
    print("```")
    print("\nSet `DRY_RUN = False` to write files.")

# %%
