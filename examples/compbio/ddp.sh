#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-12:00:00


# debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


module load anaconda/2022.05
conda activate viscy

srun python ddp.py