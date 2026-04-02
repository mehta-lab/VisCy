#!/bin/bash

#SBATCH --job-name=UNeXt2_SEC61B
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint=h200
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/checkpoints

ml uv

export PYTHONUNBUFFERED=1

nvidia-smi
uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_unext2.yml
