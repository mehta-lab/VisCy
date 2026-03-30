#!/bin/bash
# OPS 373-gene DynaCLR with gene classifier head
#
# New run:
#   sbatch applications/dynaclr/configs/training/OPS-373genes.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR.

#SBATCH --job-name=dynaclr_ops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="dynaclr"
export RUN_NAME="OPS-373genes-GeneClassifier"
export CONFIGS="applications/dynaclr/configs/training/OPS-373genes.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

source "$(dirname "$0")/slurm/train.sh"
