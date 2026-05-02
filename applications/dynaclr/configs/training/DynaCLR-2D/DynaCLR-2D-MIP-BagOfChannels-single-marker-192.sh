#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels SINGLE-MARKER 192px variant.
# Same recipe as single-marker.sh but with 384->256->192 crops instead
# of 256->192->160. Larger final input preserves more subcellular detail
# at ~2x the I/O cost per batch.
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker-192.sh

#SBATCH --job-name=dynaclr_2d_sm192
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
# 14 GB/CPU × 15 CPUs = 210 GB/rank, 840 GB/node. Bumped from 8G after
# rank 2 OOM'd on the 384² patches × prefetch buffers (job 31447592).
# Matches the OPS sbatch which has the same multi-GB-batch profile.
#SBATCH --mem-per-cpu=14G
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
