#!/bin/bash
# DINOv3-temporal-MLP-2D-BagOfChannels
#
# New run:
#   sbatch applications/dynaclr/configs/training/DINOv3/DINOv3-temporal-MLP-2D-BagOfChannels.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR:
#   sbatch /hpc/projects/.../DINOv3-temporal-MLP-2D-BagOfChannels.sh

#SBATCH --job-name=dinov3_mlp_2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="DINOv3-temporal-MLP-2D-BagOfChannels-v1"
export RUN_NAME="dinov3-mlp-2d-mip-ntxent-t0p5-lr1e4-bs512"
export CONFIGS="applications/dynaclr/configs/training/DINOv3/DINOv3-temporal-MLP-2D-BagOfChannels.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
export CKPT_PATH="/hpc/projects/organelle_phenotyping/models/DINOv3-temporal-MLP-2D-BagOfChannels-v1/dinov3-mlp-2d-mip-ntxent-t0p5-lr1e4-bs512/DINOv3-temporal-MLP-2D-BagOfChannels-v1/20260403-223550/checkpoints/last.ckpt"
export WANDB_RUN_ID="20260403-223550"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
