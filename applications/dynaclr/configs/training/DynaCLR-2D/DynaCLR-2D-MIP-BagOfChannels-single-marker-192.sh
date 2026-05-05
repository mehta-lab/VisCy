#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels SINGLE-MARKER 192px variant.
# Same recipe as single-marker.sh but with 384->256->192 crops instead
# of 256->192->160. Larger final input preserves more subcellular detail
# at ~2x the I/O cost per batch.
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker-192.sh

#SBATCH --job-name=dynaclr_2d_sm192
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
# 17 GB/CPU × 15 CPUs = 255 GB/rank, 510 GB/node on 2 GPUs. Bumped from 14G
# after rank 3 host-RAM OOM on 384² patches (job 31449149). Combined with
# prefetch_factor=1 in datamodule. Dropped from 4 GPUs to 2 to ease queueing;
# batch_size kept at 256/rank — host RAM was the OOM driver (cgroup), not
# VRAM, and that scales with workers × prefetch, not batch_size. If this
# still OOMs, suspect a real leak (loky semaphores, tensorstore decoder
# scratch) — investigate rather than papering over with more RAM.
#SBATCH --mem-per-cpu=17G
#SBATCH --time=3-00:00:00

export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-ntxent-t0p2-lr2e5-bs256-384to192-zext16-single-marker-fix-shuffler"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker-192.yml"

# Warm-start disabled: prior attempt 31442612 hit a 30-min NCCL all-reduce
# timeout in optimizer.step. Suspected interaction between the warm-start
# (160-input encoder weights loaded into a 192-input model) and the
# augmentation pipeline causing rank divergence. Train from random init
# to remove that confound; if the fresh-init run trains cleanly we can
# revisit warm-start in v2.
# export EXTRA_ARGS="--model.init_args.ckpt_path=/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/DynaCLR-2D-MIP-BagOfChannels/0rhpwh77/checkpoints/last.ckpt"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
