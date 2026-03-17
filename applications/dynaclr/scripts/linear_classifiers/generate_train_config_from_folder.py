# %%
"""Generate linear classifier training configs from a model predictions folder.

Works with any embedding model (DynaCLR, DINOv3, OpenPhenom, etc.) by
pointing directly at prediction folders rather than hardcoding model
templates.

Usage: run cells interactively or execute as a script.
"""

from pathlib import Path

import yaml

from dynaclr.evaluation.linear_classifiers.utils import (
    TASKS,
    find_annotation_csv,
    find_channel_zarrs,
    get_available_tasks,
)

# %%
# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Prediction folders to include in training.
# Each entry maps to a single dataset's version directory containing *.zarr
# embeddings. All datasets listed here will be combined for training.
PREDICTION_FOLDERS = [
    Path(
        "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
        "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/"
        "DINOv3/convnext-tiny-lvd1689m"
    ),
    # Add more dataset folders to combine for training:
    # Path(".../another_dataset/4-phenotyping/predictions/DINOv3/convnext-tiny-lvd1689m"),
]

# Embedding model identity — used for the W&B project name:
#   linearclassifiers-{embedding_model_name}-{embedding_model_version}
# Set to None to auto-derive from the folder structure (parent.name / folder.name).
EMBEDDING_MODEL_NAME = None  # e.g. "DINOv3"
EMBEDDING_MODEL_VERSION = None  # e.g. "convnext-tiny-lvd1689m"

# Annotations directory
ANNOTATIONS_DIR = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")

# Channels to train on (only matching zarrs will be used)
CHANNELS = ["Phase3D"]

# Tasks to train (None = all tasks found in annotations)
TASKS_TO_TRAIN: list[str] | None = None

# Output directory for generated configs
OUTPUT_DIR = None  # None = write configs next to PREDICTION_FOLDERS[0]

# Classifier hyperparameters
USE_SCALING = True
USE_PCA = False
N_PCA_COMPONENTS = None
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"
SOLVER = "liblinear"
SPLIT_TRAIN_DATA = 0.8
RANDOM_SEED = 42

# W&B
WANDB_ENTITY = "computational_imaging"
WANDB_TAGS: list[str] = []

# Set to True for a dry run (preview only, no files written)
DRY_RUN = False

# %%
# ===========================================================================
# Resolve model identity and discover data
# ===========================================================================

first_folder = PREDICTION_FOLDERS[0]
embedding_model_name = EMBEDDING_MODEL_NAME or first_folder.parent.name
embedding_model_version = EMBEDDING_MODEL_VERSION or first_folder.name
output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else first_folder

print("## Generate Classifier Training Configs\n")
print(f"- **Embedding model**: `{embedding_model_name}`")
print(f"- **Version**: `{embedding_model_version}`")
print(f"- **Channels**: {CHANNELS}")
print(f"- **W&B project**: `linearclassifiers-{embedding_model_name}-{embedding_model_version}`")
print(f"- **Prediction folders**: {len(PREDICTION_FOLDERS)}")

# %%
# ===========================================================================
# Build dataset entries: find zarrs + annotations per folder
# ===========================================================================

# Infer dataset name from folder path:
#   .../DATASET_NAME/4-phenotyping/predictions/MODEL/VERSION
#   parts[-5] is the dataset name
datasets: list[dict] = []
errors: list[dict] = []

for folder in PREDICTION_FOLDERS:
    try:
        parts = folder.parts
        # Walk up to find the dataset name (first dir above *phenotyping*)
        dataset_name = None
        for i, part in enumerate(parts):
            if "phenotyping" in part:
                dataset_name = parts[i - 1]
                break
        if dataset_name is None:
            raise ValueError(f"Cannot infer dataset name from {folder}")

        channel_zarrs = find_channel_zarrs(folder, CHANNELS)
        if not channel_zarrs:
            raise ValueError(f"No zarrs matching channels {CHANNELS} in {folder}")

        annotations_csv = find_annotation_csv(ANNOTATIONS_DIR, dataset_name)
        if not annotations_csv:
            raise ValueError(f"No annotations CSV found for {dataset_name}")

        available_tasks = get_available_tasks(annotations_csv)
        tasks_to_use = TASKS_TO_TRAIN or [t for t in TASKS if t in available_tasks]
        tasks_to_use = [t for t in tasks_to_use if t in available_tasks]

        datasets.append(
            {
                "dataset_name": dataset_name,
                "folder": folder,
                "channel_zarrs": channel_zarrs,
                "annotations_csv": annotations_csv,
                "tasks": tasks_to_use,
            }
        )
    except Exception as e:
        errors.append({"folder": str(folder), "error": str(e)})

# %%
# ===========================================================================
# Summary
# ===========================================================================

print("\n### Discovered Datasets\n")
print("| Dataset | Channels | Tasks | Annotations |")
print("|---------|----------|-------|-------------|")
for ds in datasets:
    ch_str = ", ".join(sorted(ds["channel_zarrs"].keys()))
    task_str = ", ".join(ds["tasks"])
    print(f"| {ds['dataset_name']} | {ch_str} | {task_str} | `{ds['annotations_csv'].name}` |")

if errors:
    print("\n### Errors\n")
    print("| Folder | Error |")
    print("|--------|-------|")
    for e in errors:
        print(f"| `{e['folder']}` | {e['error']} |")

if not datasets:
    raise RuntimeError("No valid datasets found.")

# Collect all tasks across datasets
all_tasks = sorted(set(t for ds in datasets for t in ds["tasks"]))
all_channels = sorted(set(ch for ds in datasets for ch in ds["channel_zarrs"]))

print(f"\n- **Tasks to train**: {all_tasks}")
print(f"- **Channels available**: {all_channels}")

# %%
# ===========================================================================
# Generate training configs: one per (task, channel)
# ===========================================================================

generated: list[dict] = []

for task in all_tasks:
    for channel in all_channels:
        train_datasets = []
        for ds in datasets:
            if task in ds["tasks"] and channel in ds["channel_zarrs"]:
                train_datasets.append(
                    {
                        "embeddings": str(ds["channel_zarrs"][channel]),
                        "annotations": str(ds["annotations_csv"]),
                    }
                )

        if not train_datasets:
            continue

        config = {
            "task": task,
            "input_channel": channel,
            "embedding_model_name": embedding_model_name,
            "embedding_model_version": embedding_model_version,
            "train_datasets": train_datasets,
            "use_scaling": USE_SCALING,
            "use_pca": USE_PCA,
            "n_pca_components": N_PCA_COMPONENTS,
            "max_iter": MAX_ITER,
            "class_weight": CLASS_WEIGHT,
            "solver": SOLVER,
            "split_train_data": SPLIT_TRAIN_DATA,
            "random_seed": RANDOM_SEED,
            "wandb_entity": WANDB_ENTITY,
            "wandb_tags": WANDB_TAGS,
        }

        filename = f"train_{task}_{channel}.yaml"
        generated.append(
            {
                "task": task,
                "channel": channel,
                "n_datasets": len(train_datasets),
                "filename": filename,
                "config": config,
            }
        )

# %%
# ===========================================================================
# Generate SLURM script
# ===========================================================================

WORKSPACE_DIR = "/hpc/mydata/eduardo.hirata/repos/viscy"

slurm_lines = [
    "#!/bin/bash",
    "",
    "#SBATCH --job-name=train_lc",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --partition=cpu",
    "#SBATCH --cpus-per-task=16",
    "#SBATCH --mem-per-cpu=8G",
    "#SBATCH --time=0-01:00:00",
    f"#SBATCH --output={output_dir}/slurm_out/slurm_%j.out",
    "",
    "export PYTHONNOUSERSITE=1",
    "",
    f"WORKSPACE_DIR={WORKSPACE_DIR}",
    "",
    "scontrol show job $SLURM_JOB_ID",
    "",
]

for entry in generated:
    yml_path = output_dir / entry["filename"]
    slurm_lines.append(f'echo "=== {entry["task"]} / {entry["channel"]} ==="')
    slurm_lines.append('uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \\')
    slurm_lines.append(f"    dynaclr train-linear-classifier -c {yml_path}")
    slurm_lines.append("")

slurm_content = "\n".join(slurm_lines)
slurm_path = output_dir / "train_classifiers_all.sh"

# %%
# ===========================================================================
# Print generation summary
# ===========================================================================

action = "Generated" if not DRY_RUN else "Would generate (DRY RUN)"
print(f"\n### {action}\n")
print("| Task | Channel | Datasets | Config |")
print("|------|---------|----------|--------|")
for entry in generated:
    print(f"| {entry['task']} | {entry['channel']} | {entry['n_datasets']} | `{entry['filename']}` |")
print(f"\n- **SLURM script**: `{slurm_path.name}`")
print(f"- **Output dir**: `{output_dir}`")

# %%
# ===========================================================================
# Write files
# ===========================================================================

if not DRY_RUN:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm_out").mkdir(exist_ok=True)

    for entry in generated:
        out_path = output_dir / entry["filename"]
        with open(out_path, "w") as f:
            yaml.dump(entry["config"], f, default_flow_style=False, sort_keys=False)
        print(f"  Wrote {out_path}")

    slurm_path.write_text(slurm_content)
    slurm_path.chmod(0o755)
    print(f"  Wrote {slurm_path}")

# %%
