#!/bin/bash

#SBATCH --job-name=CELLDiff_SEC61B
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --constraint="a100_80|h100|h200"
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/celldiff/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/celldiff/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/celldiff/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/celldiff/checkpoints

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
srun uv run python -m dynacell fit --config applications/dynacell/examples/configs/sec61b/fit_celldiff.yml
