#!/bin/bash
# DynaCLR-3D-BagOfChannels-v2 SINGLE-MARKER variant (fresh, no resume).
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-3D-BagOfChannels-v2-single-marker.sh

#SBATCH --job-name=dynaclr_3d_sm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4-00:00:00

export PROJECT="DynaCLR-3D-BagOfChannels-v2"
export RUN_NAME="3d-z32-256to228to160-ntxent-t0p2-single-marker"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-3D-BagOfChannels-v2.yml applications/dynaclr/configs/training/DynaCLR-3D-BagOfChannels-v2-single-marker.yml"

source /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/training/slurm/train.sh
