#!/bin/bash
# DynaCLR-3D-BagOfChannels-v2
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR:
#   sbatch /hpc/projects/.../3d-z32-.../DynaCLR-3D-BagOfChannels-v2.sh

#SBATCH --job-name=dynaclr_3d_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="DynaCLR-3D-BagOfChannels-v2"
export RUN_NAME="3d-z32-256to228to160-ntxent-t0p2-mixed-markers"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-3D/DynaCLR-3D-BagOfChannels-v2.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# Commented out for fresh A/B comparison run against single-marker variant.
# export CKPT_PATH="/hpc/projects/organelle_phenotyping/models/DynaCLR-3D-BagOfChannels-v2/3d-z32-256to228to160-ntxent-t0p2/DynaCLR-3D-BagOfChannels-v2/20260402-185442/checkpoints/last.ckpt"
# export WANDB_RUN_ID="20260402-185442"

source "$(dirname "$0")/../slurm/train.sh"
