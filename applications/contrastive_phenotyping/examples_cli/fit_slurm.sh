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


# Cleanup function to remove the temporary files
function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup Completed."
}

trap cleanup EXIT
# trap the EXIT signal sent to the process and invoke the cleanup.

# Activate the conda environment - specfic to your installation!
module load anaconda/2022.05
# You'll need to replace this path with path to your own conda environment.
conda activate /hpc/mydata/$USER/envs/viscy

config=./demo_cli_fit.yml

# Printing this to the stdout lets us connect the job id to config.
scontrol show job $SLURM_JOB_ID
cat $config

# Run the training CLI
srun python -m viscy.cli.contrastive_triplet fit -c $config

# Tips:
# 1. run this script with `sbatch demo_cli_fit_slurm.sh`
# 2. check the status of the job with `squeue -u $USER`
# 3. use turm to monitor the job with `turm -u first.last`. Use module load turm to load the turm module.
