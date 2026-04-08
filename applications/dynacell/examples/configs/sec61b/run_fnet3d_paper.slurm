#!/bin/bash

#SBATCH --job-name=FNet3DPaper_SEC61B
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/fnet3d_paper/checkpoints

ml uv

export PYTHONUNBUFFERED=1

nvidia-smi
uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_fnet3d_paper.yml
