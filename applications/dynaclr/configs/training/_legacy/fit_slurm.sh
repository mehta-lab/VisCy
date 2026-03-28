#!/bin/bash

#SBATCH --job-name=dynaclr_fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-20:00:00

# ── Run identity ─────────────────────────────────────────────────────────────
# Model naming convention: {ModelType}-{Dimensionality}-{Features}
# Examples:
#   DynaCLR-2D-Phase
#   DynaCLR-2D-BagOfChannels
#   DynaCLR-3D-Phase-Reporter
# WandB handles versioning automatically — no version suffix needed here.
MODEL_TYPE="DynaCLR"
DIMS="2D"                  # 2D or 3D
FEATURES="Phase"           # Phase, BagOfChannels, Phase-Reporter, etc.
RUN_NAME="${MODEL_TYPE}-${DIMS}-${FEATURES}"

# TODO: set the project output root
PROJECT_DIR="/hpc/projects/TODO"
RUN_DIR="${PROJECT_DIR}/models/${RUN_NAME}"  # checkpoints + logs live here

# ── Paths ─────────────────────────────────────────────────────────────────────
# TODO: point to your uv workspace
WORKSPACE_DIR=/path/to/viscy

# TODO: point to the config file (fit.yml or multi_experiment_fit.yml)
CONFIG="$(dirname "$0")/fit.yml"

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
srun uv run --project "$WORKSPACE_DIR" viscy fit \
  --config "$CONFIG" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
  --trainer.callbacks[1].init_args.dirpath="${RUN_DIR}/checkpoints" \
