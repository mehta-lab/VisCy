#!/bin/bash
# DynaCLR-3D-BagOfChannels-v2 training
#
# Usage:
#   sbatch applications/dynaclr/configs/training/slurm/DynaCLR-3D-BagOfChannels-v2.sh
#
# Override run name:
#   RUN_NAME=phase2-hcl-beta0p5 sbatch .../DynaCLR-3D-BagOfChannels-v2.sh
#
# Sweep example:
#   for TEMP in 0.1 0.2 0.5; do
#     RUN_NAME="sweep-temp${TEMP}" \
#     EXTRA_ARGS="--model.init_args.loss_function.init_args.temperature ${TEMP}" \
#     sbatch .../DynaCLR-3D-BagOfChannels-v2.sh
#   done

#SBATCH --job-name=dynaclr_3d_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100|h200"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

TRAINING_DIR="applications/dynaclr/configs/training"

export PROJECT="DynaCLR-3D-BagOfChannels-v2"
export RUN_NAME="${RUN_NAME:-phase1-ntxent-temp0p2}"
export CONFIGS="\
${TRAINING_DIR}/_base.yml \
${TRAINING_DIR}/arch/3d_z16.yml \
${TRAINING_DIR}/data/boc_3d_temporal_stratify-perturbation.yml"
export EXTRA_ARGS="${EXTRA_ARGS} \
  --trainer.devices 4 \
  --data.init_args.collection_path applications/dynaclr/configs/collections/DynaCLR-3D-BagOfChannels-v2.yml \
  --data.init_args.cell_index_path /hpc/projects/organelle_phenotyping/models/collections/DynaCLR-3D-BagOfChannels-v2.parquet \
  --data.init_args.batch_size 512 \
  --data.init_args.reference_pixel_size_xy_um 0.1494 \
  --data.init_args.reference_pixel_size_z_um 0.174 \
  --data.init_args.focus_channel Phase3D \
  --model.init_args.pca_color_keys '[perturbation,hours_post_perturbation,experiment,marker]'"

source "$(dirname "$0")/train.sh"
