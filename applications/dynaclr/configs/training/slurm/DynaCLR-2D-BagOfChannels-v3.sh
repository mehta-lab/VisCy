#!/bin/bash
# DynaCLR-2D-BagOfChannels-v3 training
#
# Usage:
#   sbatch applications/dynaclr/configs/training/slurm/DynaCLR-2D-BagOfChannels-v3.sh

#SBATCH --job-name=dynaclr_2d_v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

TRAINING_DIR="applications/dynaclr/configs/training"

export PROJECT="DynaCLR-2D-BagOfChannels-v3"
export RUN_NAME="${RUN_NAME:-phase1-ntxent-temp0p2}"
export CONFIGS="\
${TRAINING_DIR}/_base.yml \
${TRAINING_DIR}/arch/2d_z1.yml \
${TRAINING_DIR}/data/boc_2d_temporal_stratify-perturbation-marker.yml"
export EXTRA_ARGS="${EXTRA_ARGS} \
  --trainer.devices 2 \
  --data.init_args.cell_index_path /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/DynaCLR-2D-BagOfChannels-v3.parquet \
  --data.init_args.batch_size 256 \
  --model.init_args.pca_color_keys '[perturbation,hours_post_perturbation]'"

source "$(dirname "$0")/train.sh"
