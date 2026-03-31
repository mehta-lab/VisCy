#!/bin/bash

#SBATCH --job-name=FNet3D_SEC61B
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint=h200
#SBATCH --output=slurm_log/%j.out
#SBATCH --error=slurm_log/%j.err

mkdir -p -m 775 slurm_log

ml uv

export PYTHONUNBUFFERED=1

nvidia-smi
uv run python -m cytoland fit --config applications/cytoland/examples/configs/dynacell/fit_fnet3d_sec61b.yml
