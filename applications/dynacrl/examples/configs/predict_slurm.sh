#!/bin/bash

#SBATCH --job-name=contrastive_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-01:00:00

# TODO: point to the path to your uv workspace
WORKSPACE_DIR=/path/to/viscy

scontrol show job $SLURM_JOB_ID

# use absolute path in production
config=./predict.yml
cat $config

# Run the prediction CLI (viscy is provided by viscy-utils)
uv run --project "$WORKSPACE_DIR" --package dynacrl viscy predict -c $config
