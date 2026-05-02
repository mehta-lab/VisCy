#!/bin/bash
# Predict embeddings for microglia and ALFI datasets
# Uses DynaCLR-2D-MIP-BagOfChannels checkpoint.
#
# Usage:
#   sbatch applications/dynaclr/configs/evaluation/predict_microglia_alfi.sh

#SBATCH --job-name=dynaclr_predict_microglia_alfi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:00:00

export PYTHONNOUSERSITE=1
WORKSPACE_DIR="/hpc/mydata/eduardo.hirata/repos/viscy"

# echo "=== Microglia predict ==="
# srun uv run --project /hpc/mydata/eduardo.hirata/repos/viscy viscy predict --config /hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/evaluations/microglia/configs/predict.yml

echo "=== ALFI predict ==="
srun uv run --project /hpc/mydata/eduardo.hirata/repos/viscy viscy predict --config /hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/evaluations/alfi/configs/predict.yml
