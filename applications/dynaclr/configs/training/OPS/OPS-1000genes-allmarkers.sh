#!/bin/bash
# OPS 1000-gene × ALL-markers DynaCLR — single-marker SupCon batches with
# sqrt-weighted marker sampling, warm-started from OPS-1000genes-lite epoch 192.
#
# New run:
#   sbatch applications/dynaclr/configs/training/OPS/OPS-1000genes-allmarkers.sh
#
# Resume: edit CKPT_PATH and WANDB_RUN_ID below, then sbatch from RUN_DIR.

#SBATCH --job-name=dynaclr_ops_allmk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --constraint="h100|h200"
# gpu-h-5 has pathological NFS read performance on /hpc/projects/ —
# FOV-split Arrow-take takes ~26 min vs ~15s on gpu-h-2/gpu-f-4.
# Exclude it until the underlying storage issue is fixed or we move the
# dataset to faster storage.
#SBATCH --exclude=gpu-h-5
#SBATCH --cpus-per-task=15
# 16 GB/CPU × 60 CPUs = 960 GB/node. Needed because the 81M-row OPS
# cell_index × 4 DDP ranks × dataloader worker fork-copies blows past the
# original 480 GB budget. Pandas reference-counting defeats CoW so workers
# end up duplicating the full cached DataFrame.
#SBATCH --mem-per-cpu=14G
#SBATCH --time=3-00:00:00

# ── Run identity ──────────────────────────────────────────────────────
# Warm-started from prior OPS run (t89f7q4n/last.ckpt, Apr 20) at epoch 0.
# Picks up the FlexibleBatchSampler reshuffle fix (commit f4f40c38) plus
# the profiling-pass defaults (file_io_concurrency=128, ts.Batch overlap,
# cuDNN benchmark, TF32). num_workers stays at 1 for OPS due to per-rank
# memory pressure on the 81M-row cell_index. Optimizer state and epoch
# counter reset.
export PROJECT="OPS"
export RUN_NAME="OPS-1000genes-allmarkers-fix-shuffler"
WARMSTART_CKPT="/hpc/projects/organelle_phenotyping/models/OPS/OPS-1000genes-allmarkers/OPS-1000genes-allmarkers/t89f7q4n/checkpoints/last.ckpt"
export EXTRA_ARGS="--trainer.logger.init_args.project=OPS-1000genes-allmarkers --model.init_args.ckpt_path=${WARMSTART_CKPT}"
export CONFIGS="applications/dynaclr/configs/training/OPS/OPS-1000genes-allmarkers.yml"

# ── Resume (Lightning full state, NOT what we want here) ──────────────
# export CKPT_PATH=""
# export WANDB_RUN_ID=""

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
