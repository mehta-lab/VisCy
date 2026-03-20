#!/bin/bash

#SBATCH --job-name=dinov3_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-20:00:00

# ── Run identity ─────────────────────────────────────────────────────────────
# Model naming convention: {ModelType}-{Dimensionality}-{Features}
# DINOv3-finetune: frozen backbone + trainable MLP adapter
RUN_NAME="DINOv3-temporal-MLP-2D-BagOfChannels"
RUN_DIR="/hpc/projects/intracellular_dashboard/organelle_dynamics/models/${RUN_NAME}"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy

CONFIG="$(dirname "$0")/dinov3_finetune_temporal_MLP_fit.yml"

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1   # prevent ~/.local from shadowing conda/uv env
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
uv run --project "$WORKSPACE_DIR" --package viscy-utils viscy fit \
  --config "$CONFIG"
