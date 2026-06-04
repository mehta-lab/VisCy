# %%
"""Generate linear classifier inference configs and SLURM scripts.

Given a model predictions folder (e.g.
.../DynaCLR-2D-BagOfChannels-timeaware/v3/), discovers embedding zarrs
for each channel and generates a YAML config + SLURM script to apply
all matching classifiers.

Usage: run cells interactively or execute as a script.
"""

from pathlib import Path

import yaml

from dynaclr.evaluation.linear_classifiers.utils import (
    CHANNELS,
    TASKS,
    find_channel_zarrs,
)

# %%
# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

# Path to the model version folder containing *.zarr embedding files
MODEL_FOLDER = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_01_24_A549_G3BP1_DENV/4-phenotyping/predictions/"
    "DynaCLR-2D-BagOfChannels-timeaware/v3"
)

# Embedding model identity (derived from folder structure if not set)
EMBEDDING_MODEL_NAME = None  # e.g. "DynaCLR-2D-BagOfChannels-timeaware", None = auto
EMBEDDING_MODEL_VERSION = None  # e.g. "v3", None = auto

# W&B entity
WANDB_ENTITY = "computational_imaging"

# Tasks to generate classifiers for (None = all known tasks)
TASKS_TO_APPLY: list[str] | None = None

# Channels to process (None = auto-discover from zarrs)
CHANNELS_TO_PROCESS: list[str] | None = None

# Classifier version to use
CLASSIFIER_VERSION = "latest"

# Set to True for a dry run (preview only, no files written)
DRY_RUN = False

# Set to True to overwrite existing config files
OVERWRITE = True

# Set to True to submit SLURM jobs after generating
SUBMIT_JOBS = False

WORKSPACE_DIR = "/hpc/mydata/eduardo.hirata/repos/viscy"

# %%
# ===========================================================================
# Resolve model identity from folder structure
# ===========================================================================

embedding_model_name = EMBEDDING_MODEL_NAME or MODEL_FOLDER.parent.name
embedding_model_version = EMBEDDING_MODEL_VERSION or MODEL_FOLDER.name

tasks = TASKS_TO_APPLY or list(TASKS)
channels = CHANNELS_TO_PROCESS or list(CHANNELS)

print("## Generate Classifier Inference Configs\n")
print(f"- **Model folder**: `{MODEL_FOLDER}`")
print(f"- **Embedding model**: `{embedding_model_name}`")
print(f"- **Version**: `{embedding_model_version}`")
print(f"- **Tasks**: {tasks}")
print(f"- **W&B entity**: `{WANDB_ENTITY}`")

# %%
# ===========================================================================
# Discover channel zarrs
# ===========================================================================

channel_zarrs = find_channel_zarrs(MODEL_FOLDER, channels)

if not channel_zarrs:
    raise RuntimeError(f"No channel zarrs found in {MODEL_FOLDER}")

print("\n### Discovered Channels\n")
print("| Channel | Zarr Path |")
print("|---------|-----------|")
for ch, zpath in sorted(channel_zarrs.items()):
    print(f"| {ch} | `{zpath.name}` |")

# %%
# ===========================================================================
# Generate configs per channel
# ===========================================================================

generated: list[dict] = []

for channel, zarr_path in sorted(channel_zarrs.items()):
    models = []
    for task in tasks:
        model_name = f"linear-classifier-{task}-{channel}"
        models.append({"model_name": model_name, "version": CLASSIFIER_VERSION})

    config = {
        "embedding_model_name": embedding_model_name,
        "embedding_model_version": embedding_model_version,
        "wandb_entity": WANDB_ENTITY,
        "channel": channel,
        "embeddings_path": str(zarr_path),
        "overwrite": False,
        "models": models,
    }

    yml_path = MODEL_FOLDER / f"linear_classifier_inference_{channel}.yml"
    generated.append(
        {
            "channel": channel,
            "yml_path": yml_path,
            "config": config,
            "n_models": len(models),
        }
    )

# %%
# ===========================================================================
# Generate SLURM script
# ===========================================================================

slurm_lines = [
    "#!/bin/bash",
    "",
    "#SBATCH --job-name=dynaclr_apply_lc",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --partition=cpu",
    "#SBATCH --cpus-per-task=16",
    "#SBATCH --mem-per-cpu=8G",
    "#SBATCH --time=0-01:00:00",
    f"#SBATCH --output={MODEL_FOLDER}/slurm_out/slurm_%j.out",
    "",
    "export PYTHONNOUSERSITE=1",
    "",
    f"WORKSPACE_DIR={WORKSPACE_DIR}",
    "",
    "scontrol show job $SLURM_JOB_ID",
    "",
]

for entry in generated:
    yml = entry["yml_path"]
    slurm_lines.append(f'echo "=== {entry["channel"]} ==="')
    slurm_lines.append('uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \\')
    slurm_lines.append(f"    dynaclr apply-linear-classifier -c {yml}")
    slurm_lines.append("")

slurm_content = "\n".join(slurm_lines)
slurm_path = MODEL_FOLDER / "apply_classifiers_all.sh"

# %%
# ===========================================================================
# Summary
# ===========================================================================

action = "Generated" if not DRY_RUN else "Would generate (DRY RUN)"
print(f"\n### {action}\n")
print("| Channel | Models | Config |")
print("|---------|--------|--------|")
for entry in generated:
    print(f"| {entry['channel']} | {entry['n_models']} | `{entry['yml_path'].name}` |")
print(f"\n- **SLURM script**: `{slurm_path.name}`")

# %%
# ===========================================================================
# Write files
# ===========================================================================

if not DRY_RUN:
    (MODEL_FOLDER / "slurm_out").mkdir(exist_ok=True)
    for entry in generated:
        yml_path = entry["yml_path"]
        if not OVERWRITE and yml_path.exists():
            print(f"  Skipping {yml_path.name} (exists)")
            continue
        with open(yml_path, "w") as f:
            yaml.dump(entry["config"], f, default_flow_style=False, sort_keys=False)
        print(f"  Wrote {yml_path.name}")

    slurm_path.write_text(slurm_content)
    slurm_path.chmod(0o755)
    print(f"  Wrote {slurm_path.name}")

# %%
# ===========================================================================
# Submit SLURM job
# ===========================================================================

if SUBMIT_JOBS and not DRY_RUN:
    import subprocess

    print("\n## Submitting SLURM job\n")
    result = subprocess.run(
        ["sbatch", str(slurm_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")
    else:
        print(result.stdout.strip())
elif SUBMIT_JOBS and DRY_RUN:
    print("\n**SUBMIT_JOBS is True but DRY_RUN is also True -- skipping.**")

# %%
