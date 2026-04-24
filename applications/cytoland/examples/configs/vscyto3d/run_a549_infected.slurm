#!/bin/bash
# Full finetune: VSCyto3D warm-started on A549 infection data from three
# microscopes (D1/D2/D3). Run from repo root: sbatch applications/cytoland/examples/configs/vscyto3d/run_a549_infected.slurm

#SBATCH --job-name=VSCyto3D_A549_infected
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=1024G
#SBATCH --constraint=h200
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/cytoland/a549_infected/vscyto3d/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/cytoland/a549_infected/vscyto3d/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/cytoland/a549_infected/vscyto3d/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/cytoland/a549_infected/vscyto3d/checkpoints

ml uv

export PYTHONUNBUFFERED=1

nvidia-smi
srun uv run python -m cytoland fit --config applications/cytoland/examples/configs/vscyto3d/finetune_a549_infected.yml
