#!/bin/bash
# DynaCLR-3D-BagOfChannels-v2
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-3D-BagOfChannels-v2.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR:
#   sbatch /hpc/projects/.../3d-z16-.../DynaCLR-3D-BagOfChannels-v2.sh

#SBATCH --job-name=dynaclr_3d_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="DynaCLR-3D-BagOfChannels-v2"
export RUN_NAME="3d-z16-ntxent-t0p2-lr2e5-bs512-192to160-zext45"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-3D-BagOfChannels-v2.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH="/hpc/projects/organelle_phenotyping/models/DynaCLR-3D-BagOfChannels-v2/3d-z16-ntxent-t0p2-lr2e5-bs512-192to160-zext45/checkpoints/last.ckpt"
# export WANDB_RUN_ID="20260329-063341"

source "$(dirname "$0")/slurm/train.sh"
