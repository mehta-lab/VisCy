#!/bin/bash
# Fast-dev-run smoke test of BoC training on 4-GPU DDP.
# Goal: validate sampler generator + FOV split + NCCL init + first
# batch end-to-end with the 20k-row boc_tiny parquet.

#SBATCH --job-name=boc_fastdev_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=/hpc/mydata/eduardo.hirata/repos/viscy/tmp/boc_fastdev_ddp_%j.out

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=WARN

cd /hpc/mydata/eduardo.hirata/repos/viscy

srun uv run --project . viscy fit \
  --config applications/dynaclr/configs/training/DynaCLR-2D-MIP-BagOfChannels.yml \
  --config applications/dynaclr/configs/training/DynaCLR-2D-MIP-BagOfChannels-fastdev-ddp.yml
