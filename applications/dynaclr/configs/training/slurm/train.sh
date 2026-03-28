#!/bin/bash
# DynaCLR composable training launcher
# =====================================
# GPU type and count are passed via sbatch flags (not hardcoded here).
#
# Required env vars:
#   PROJECT    WandB project name (also model output directory name)
#   RUN_NAME   WandB run name
#   CONFIGS    Space-separated list of config files (relative to WORKSPACE_DIR)
#
# Optional env vars:
#   WORKSPACE_DIR  Repo root (default: /hpc/mydata/eduardo.hirata/repos/viscy)
#   MODEL_ROOT     Model output root (default: /hpc/projects/organelle_phenotyping/models)
#   EXTRA_ARGS     Extra CLI args passed to viscy fit
#
# Example:
#   PROJECT=DynaCLR-3D-BagOfChannels-v2 \
#   RUN_NAME=phase1-ntxent-4gpu \
#   CONFIGS="applications/dynaclr/configs/training/_base/common.yml \
#            applications/dynaclr/configs/training/arch/3d_z16.yml \
#            applications/dynaclr/configs/training/_base/aug_boc_3d.yml \
#            applications/dynaclr/configs/training/experiments/DynaCLR-3D-BagOfChannels-v2.yml" \
#   sbatch --gres=gpu:h200:4 --ntasks-per-node=4 --cpus-per-task=15 --mem-per-cpu=8G \
#     applications/dynaclr/configs/training/slurm/train.sh

#SBATCH --job-name=dynaclr
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=0-22:00:00

PROJECT="${PROJECT:?Set PROJECT (WandB project name)}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME (WandB run name)}"
CONFIGS="${CONFIGS:?Set CONFIGS (space-separated config paths)}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
MODEL_ROOT="${MODEL_ROOT:-/hpc/projects/organelle_phenotyping/models}"
RUN_DIR="${MODEL_ROOT}/${PROJECT}/${RUN_NAME}"

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup completed."
}
trap cleanup EXIT

# Create per-run directory and save config copies for reproducibility
mkdir -p "${RUN_DIR}/checkpoints"
for cfg in $CONFIGS; do
  cp "${WORKSPACE_DIR}/${cfg}" "${RUN_DIR}/"
done

scontrol show job $SLURM_JOB_ID

# Build --config flags from space-separated CONFIGS
CONFIG_FLAGS=""
for cfg in $CONFIGS; do
  CONFIG_FLAGS="${CONFIG_FLAGS} --config ${WORKSPACE_DIR}/${cfg}"
done

srun uv run --project "$WORKSPACE_DIR" viscy fit \
  ${CONFIG_FLAGS} \
  --trainer.logger.init_args.project="${PROJECT}" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
  --trainer.callbacks[1].init_args.dirpath="${RUN_DIR}/checkpoints" \
  ${EXTRA_ARGS}
