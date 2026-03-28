#!/bin/bash

#SBATCH --job-name=dynaclr_ops373
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=14G
#SBATCH --time=3-00:00:00

# ── Run identity ─────────────────────────────────────────────────────────────
MODEL_TYPE="DynaCLR"
DIMS="2D"
FEATURES="OPS-373genes-BagOfChannels-GeneClassifier"
RUN_NAME="${MODEL_TYPE}-${DIMS}-${FEATURES}"

PROJECT_DIR="/hpc/mydata/eduardo.hirata/logs/dynaclr"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="${WORKSPACE_DIR}/applications/dynaclr/configs/training/ops_373genes_fit.yml"

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup completed."
}
trap cleanup EXIT

scontrol show job $SLURM_JOB_ID
cat "$CONFIG"

# ── Launch ────────────────────────────────────────────────────────────────────
srun uv run --project "$WORKSPACE_DIR" viscy fit \
  --config "$CONFIG" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
