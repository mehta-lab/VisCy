#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels SINGLE-MARKER variant.
# Every batch contains only one marker (OPS-style).
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker.sh

#SBATCH --job-name=dynaclr_2d_sm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-00:00:00

export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker.yml"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
