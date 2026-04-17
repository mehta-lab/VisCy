#!/bin/bash

#SBATCH --job-name=UNetViT3D_SEC61B
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint="h200"
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy/ipsc/sec61b/unetvit3d/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy/ipsc/sec61b/unetvit3d/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy/ipsc/sec61b/unetvit3d/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy/ipsc/sec61b/unetvit3d/checkpoints

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID /dev/shm/$SLURM_JOB_ID
  echo "Cleanup Completed."
}
trap cleanup EXIT

ml uv

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

scontrol show job $SLURM_JOB_ID
nvidia-smi
srun uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_unetvit3d.yml
