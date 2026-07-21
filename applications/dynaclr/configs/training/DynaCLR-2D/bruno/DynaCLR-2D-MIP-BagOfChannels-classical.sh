#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels CLASSICAL (SimCLR-style) variant.
# Anchor and positive are the same crop; augmentation creates two views.
# Same v3 parquet and single-marker batching as the temporal variant.
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-classical.sh

#SBATCH --job-name=dynaclr_2d_cls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-00:00:00

export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-classical-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-classical.yml"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
