#!/bin/bash
# CELL-DINO-temporal-MLP-2D-BagOfChannels
#
# New run:
#   sbatch applications/dynaclr/configs/training/CELL-DINO/CELL-DINO-temporal-MLP-2D-BagOfChannels.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR:
#   sbatch /hpc/projects/.../CELL-DINO-temporal-MLP-2D-BagOfChannels.sh

#SBATCH --job-name=cell_dino_mlp_2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="CELL-DINO-temporal-MLP-2D-BagOfChannels-v1"
export RUN_NAME="cell-dino-mlp-2d-mip-ntxent-t0p5-lr1e4-bs512"
export CONFIGS="applications/dynaclr/configs/training/CELL-DINO/CELL-DINO-temporal-MLP-2D-BagOfChannels.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
