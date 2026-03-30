#!/bin/bash

#SBATCH --job-name=cytoland_fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-20:00:00

# ── Run identity ─────────────────────────────────────────────────────────────
# Model naming convention: {Architecture}-{Loss}-{Dataset}
# Examples:
#   UNeXt2-MSE-HEK
#   FNet3D-Spotlight-A549
#   FNet3D-MSE-HEK-A549
RUN_NAME="FNet3D-Spotlight-TODO"

# TODO: set the project output root
PROJECT_DIR="/hpc/projects/TODO"
RUN_DIR="${PROJECT_DIR}/models/${RUN_NAME}"

# ── Paths ─────────────────────────────────────────────────────────────────────
# TODO: point to your uv workspace
WORKSPACE_DIR=/path/to/viscy

# TODO: point to a model config (e.g., vscyto3d/finetune.yml, fnet3d/fit.yml)
CONFIG="$(dirname "$0")/vscyto3d/finetune.yml"

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1   # prevent ~/.local from shadowing conda/uv env
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup completed."
}
trap cleanup EXIT

# Print job info and config for reproducibility
scontrol show job $SLURM_JOB_ID
cat "$CONFIG"

# ── Launch ────────────────────────────────────────────────────────────────────
uv run --project "$WORKSPACE_DIR" viscy fit \
  --config "$CONFIG" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
  --trainer.callbacks[1].init_args.dirpath="${RUN_DIR}/checkpoints"
