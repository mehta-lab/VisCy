#!/bin/bash
# 4-GPU DDP on OPS tiny (346k rows) WITHOUT fast_dev_run.
# Isolates DDP+wandb+val_check_interval from dataset-size effects.

#SBATCH --job-name=ops_tiny_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
# Drop GPU-type constraint to clear the queue faster. nodes=1 guarantees
# the two ranks share a single GPU model, which is what matters for DDP.
#SBATCH --exclude=gpu-h-5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=/hpc/mydata/eduardo.hirata/repos/viscy/tmp/ops_tiny_ddp_%j.out

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=WARN

cd "$(dirname "$0")/../../../../.."

srun uv run --project . viscy fit \
  --config applications/dynaclr/configs/training/OPS/OPS-1000genes-allmarkers.yml \
  --config applications/dynaclr/configs/training/OPS-1000genes-allmarkers-tiny-ddp.yml
