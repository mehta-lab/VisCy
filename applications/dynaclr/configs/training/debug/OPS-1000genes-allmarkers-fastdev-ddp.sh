#!/bin/bash
# Minimal OPS fast_dev_run on 4-GPU DDP to localize the post-LOCAL_RANK hang.
# Strips callbacks, logger, wandb.

#SBATCH --job-name=ops_fastdev_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --constraint="h100|h200"
#SBATCH --exclude=gpu-h-5
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=14G
#SBATCH --time=0-01:00:00
#SBATCH --output=/hpc/mydata/eduardo.hirata/repos/viscy/tmp/ops_fastdev_ddp_%j.out

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=WARN

cd /hpc/mydata/eduardo.hirata/repos/viscy

srun uv run --project . viscy fit \
  --config applications/dynaclr/configs/training/OPS/OPS-1000genes-allmarkers.yml \
  --config applications/dynaclr/configs/training/OPS-1000genes-allmarkers-fastdev-ddp.yml
