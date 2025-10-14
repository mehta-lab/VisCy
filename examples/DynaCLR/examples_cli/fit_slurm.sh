#!/bin/bash

#SBATCH --job-name=contrastive_origin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-20:00:00

# NOTE: debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup Completed."
}
trap cleanup EXIT


# TODO: Activate the conda environment - specfic to your installation!
# TODO: You'll need to replace this path with path to your own conda environment
module load anaconda/latest
conda activate dynaclr

# TODO: point to the path to the config file
config=./fit.yml

# Printing this to the stdout lets us connect the job id to config.
scontrol show job $SLURM_JOB_ID
cat $config

# Run the training CLI
viscy fit -c $config

# Tips:
# 1. Run this script with `sbatch fit_slurm.sh`
# 2. Check the status of the job with `squeue -u $USER`
# 3. Use turm to monitor the job with `turm -u first.last`. Use module load turm to load the turm module.
