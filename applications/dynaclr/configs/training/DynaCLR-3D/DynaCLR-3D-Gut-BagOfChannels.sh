#!/bin/bash
# DynaCLR-3D-Gut-BagOfChannels (Phase 1) — Zuben gut cells, bag-of-channels SimCLR.
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-Gut-BagOfChannels.sh
# Resume:
#   CKPT_PATH=.../last.ckpt WANDB_RUN_ID=<id> sbatch .../DynaCLR-3D-Gut-BagOfChannels.sh

#SBATCH --job-name=dynaclr_gut_boc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# Set WORKSPACE_DIR to YOUR clone of the repo before submitting, e.g.
#   WORKSPACE_DIR=/hpc/mydata/<you>/repos/VisCy sbatch <this>.sh
# MODEL_ROOT is where checkpoints/configs are written (defaults to your clone's
# models/ dir; override to a shared project path if desired).
WORKSPACE_DIR="${WORKSPACE_DIR:?Set WORKSPACE_DIR to your repo clone path}"

export PROJECT="DynaCLR-3D-Gut-BagOfChannels"
export RUN_NAME="gut-3d-z24-cellz-64-ntxent-t0p2-self"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-Gut-BagOfChannels.yml"
export MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}/models}"
# Point at the dynaclr-pinned venv in your clone (avoids the shared .venv sync race).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE_DIR}/.venv-dynaclr}"

# The shared trainer recipe logs to the `computational_imaging` W&B entity. Set
# WANDB_ENTITY to your own entity to log there instead (EXTRA_ARGS overrides the
# recipe). Leave unset to keep the default.
if [ -n "${WANDB_ENTITY:-}" ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:-} --trainer.logger.init_args.entity=${WANDB_ENTITY}"
fi

# Absolute path (SLURM spools this script, so $(dirname "$0") would break).
source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
