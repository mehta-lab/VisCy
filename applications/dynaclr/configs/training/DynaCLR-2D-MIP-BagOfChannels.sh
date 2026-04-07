#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels
# Multi-cell-type 2D contrastive learning with channel-wise z-reduction.
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D-MIP-BagOfChannels.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch.

#SBATCH --job-name=dynaclr_2d_mip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D-MIP-BagOfChannels.yml"

# ── Resume (uncomment to continue from checkpoint) ────────────────────
export CKPT_PATH="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11/DynaCLR-2D-MIP-BagOfChannels/20260403-150013/checkpoints/last.ckpt"
export WANDB_RUN_ID="20260403-150013"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
