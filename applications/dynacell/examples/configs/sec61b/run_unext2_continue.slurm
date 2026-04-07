#!/bin/bash

#SBATCH --job-name=UNeXt2_SEC61B_cont
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=12G
#SBATCH --constraint="a100_80|h100|h200"
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/checkpoints

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
srun uv run python -m dynacell fit \
  --config applications/dynacell/examples/configs/sec61b/fit_unext2_continue.yml \
  --ckpt_path /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/sec61b/unext2/checkpoints/last-v1.ckpt
