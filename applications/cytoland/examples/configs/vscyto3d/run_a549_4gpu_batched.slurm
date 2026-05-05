#!/bin/bash
# 4-GPU production training for VSCyto3D A549 infected-cell finetune.
# Uses cropped zarrs (~323 GB total) so mmap_preload stages to /tmp
# without OOM. Mirrors dynacell joint training pattern.
#
#   sbatch applications/cytoland/examples/configs/vscyto3d/run_a549_4gpu_batched.slurm
#
# Architecture: BatchedConcatDataModule + zarr-python + mmap_preload + fork.
# - mmap_preload stages cropped FOVs to /tmp (~323 GB; node /tmp is 28 TB)
# - During training, DataLoader workers (fork) read from MemoryMappedTensor
#   instead of zarr — no fork-after-asyncio issue.

#SBATCH --job-name=VSCyto3D_A549_4gpu_batched
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=1024G
#SBATCH --constraint='h200|h100'
#SBATCH --output=/home/eduardo.hirata/repos/viscy/slurm_logs/cytoland_4gpu_batched/run_%j.out
#SBATCH --error=/home/eduardo.hirata/repos/viscy/slurm_logs/cytoland_4gpu_batched/run_%j.err

set -euo pipefail

mkdir -p /home/eduardo.hirata/repos/viscy/slurm_logs/cytoland_4gpu_batched
mkdir -p /hpc/mydata/eduardo.hirata/cytoland/a549_infected_4gpu_batched/checkpoints

ml uv

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# Limit threading libs so DataLoader workers don't oversubscribe the
# 8-core/rank allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/hpc/mydata/eduardo.hirata/repos/viscy
cd "${REPO}"

nvidia-smi
df -h /tmp

srun uv run python -m cytoland fit \
  --config applications/cytoland/examples/configs/vscyto3d/finetune_a549_infected_4gpu_batched.yml
