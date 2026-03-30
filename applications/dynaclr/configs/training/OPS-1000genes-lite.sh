#!/bin/bash
# OPS 1000-gene DynaCLR with cosine gene classifier head (lite dataset)
#
# New run:
#   sbatch applications/dynaclr/configs/training/OPS-1000genes-lite.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR.

#SBATCH --job-name=dynaclr_ops_1k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="OPS"
export RUN_NAME="OPS-1000genes-lite-CosineClassifier"
export EXTRA_ARGS="--trainer.logger.init_args.project=OPS-1000genes-lite-CosineClassifier"
export CONFIGS="applications/dynaclr/configs/training/OPS-1000genes-lite.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
