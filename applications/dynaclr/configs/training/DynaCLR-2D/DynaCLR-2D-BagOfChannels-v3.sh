#!/bin/bash
# DynaCLR-2D-BagOfChannels-v3
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-BagOfChannels-v3.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR.

#SBATCH --job-name=dynaclr_2d_v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="DynaCLR-2D-BagOfChannels-v3"
export RUN_NAME="phase1-ntxent-temp0p2"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-BagOfChannels-v3.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

source "$(dirname "$0")/../slurm/train.sh"
