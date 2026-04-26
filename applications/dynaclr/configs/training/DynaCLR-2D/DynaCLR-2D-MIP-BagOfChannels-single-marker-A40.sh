#!/bin/bash
# DynaCLR-2D-MIP-BagOfChannels single-marker — A40 interactive single-GPU variant.
# For smoke-testing and small-scale iteration on the interactive partition
# without queueing on the gpu partition.
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker-A40.sh

#SBATCH --job-name=dynaclr_2d_sm_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=14G
#SBATCH --time=4-00:00:00

export PROJECT="DynaCLR-2D-MIP-BagOfChannels"
export RUN_NAME="2d-mip-ntxent-t0p2-lr2e5-bs128-A40-single-marker"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker.yml applications/dynaclr/configs/training/DynaCLR-2D/DynaCLR-2D-MIP-BagOfChannels-single-marker-A40.yml"

source "$(dirname "$0")/../slurm/train.sh"
