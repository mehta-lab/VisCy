#!/bin/bash
# DynaCLR-3D-Gut-MultiChannel (Phase 2) — Zuben gut cells, 4-channel input.
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-Gut-MultiChannel.sh
# Resume:
#   CKPT_PATH=.../last.ckpt WANDB_RUN_ID=<id> sbatch .../DynaCLR-3D-Gut-MultiChannel.sh

#SBATCH --job-name=dynaclr_gut_4ch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# Set WORKSPACE_DIR to YOUR clone of the repo before submitting, e.g.
#   WORKSPACE_DIR=/hpc/mydata/<you>/repos/VisCy sbatch <this>.sh
WORKSPACE_DIR="${WORKSPACE_DIR:?Set WORKSPACE_DIR to your repo clone path}"

export PROJECT="DynaCLR-3D-Gut-MultiChannel"
export RUN_NAME="gut-3d-4ch-z24-cellz-64-ntxent-t0p2-self"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-Gut-MultiChannel.yml"
export MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}/models}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE_DIR}/.venv-dynaclr}"

# W&B writes sample images to a temp dir under $TMPDIR before upload. On some
# SLURM nodes the default /tmp is per-job and gets swept mid-run, causing
# `FileNotFoundError: .../wandb-media/*.png` at validation image logging. Pin
# TMPDIR + WANDB_DIR to persistent paths we create so they can't disappear.
export TMPDIR="${TMPDIR:-${MODEL_ROOT}/tmp}"
export WANDB_DIR="${WANDB_DIR:-${MODEL_ROOT}/wandb}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

# The shared trainer recipe logs to the `computational_imaging` W&B entity. Set
# WANDB_ENTITY to your own entity to log there instead. Leave unset for default.
if [ -n "${WANDB_ENTITY:-}" ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:-} --trainer.logger.init_args.entity=${WANDB_ENTITY}"
fi

# Absolute path (SLURM spools this script, so $(dirname "$0") would break).
source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
