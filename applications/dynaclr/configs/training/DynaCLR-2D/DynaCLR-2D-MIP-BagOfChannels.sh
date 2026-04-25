#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels
# Multi-cell-type 2D contrastive learning with channel-wise z-reduction.
#
# New run:
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.sh
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
#SBATCH --time=3-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
# Warm-started from prior mixed-markers run (s1f8kgtp/last.ckpt, Apr 22)
# at epoch 0. Picks up the FlexibleBatchSampler reshuffle fix
# (commit f4f40c38) and the profiling-pass defaults (nw=4, ts.Batch
# overlap, file_io_concurrency=128, z_extraction_window=16, cuDNN
# benchmark, TF32 matmul). Optimizer state and epoch counter reset so
# AdamW moments don't carry over biased gradients from the broken sampler.
export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-mixed-markers-fix-shuffler"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml"

# ── Warm-start at epoch 0 (state_dict only — not Lightning's full resume) ──
export EXTRA_ARGS="--model.init_args.ckpt_path=/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-mixed-markers/DynaCLR-2D-MIP-BagOfChannels/s1f8kgtp/checkpoints/last.ckpt"

# ── Resume (Lightning full state, NOT what we want here) ──────────────
# export CKPT_PATH="/path/to/last.ckpt"
# export WANDB_RUN_ID="<timestamp>"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
