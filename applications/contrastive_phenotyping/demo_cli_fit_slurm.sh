#!/bin/bash

#SBATCH --job-name=contrastive_origin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-20:00:00

# debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup Completed."
}

trap cleanup EXIT

module load anaconda/2022.05
conda activate viscy

scontrol show job $SLURM_JOB_ID

config=./demo_cli_fit.yml
cat $config
srun python -m viscy.cli.contrastive_triplet fit -c $config
