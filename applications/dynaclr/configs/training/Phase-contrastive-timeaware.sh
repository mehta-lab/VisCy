#!/bin/bash
# Phase contrastive timeaware — DINOv3 frozen backbone + temporal MLP
#
# New run:
#   sbatch applications/dynaclr/configs/training/Phase-contrastive-timeaware.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR.

#SBATCH --job-name=phase_dinov3_mlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="Phase-contrastive-timeaware"
export RUN_NAME="dinov3-mlp-temp0p5"
export CONFIGS="applications/dynaclr/configs/training/Phase-contrastive-timeaware.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
