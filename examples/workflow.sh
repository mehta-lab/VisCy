#!/bin/bash

# slurm resources
#SBATCH --job-name=viscy_example
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBTACH --mem-per-cpu=15G
#SBTACH --time=0-01:00

# slurm module
# module load anaconda/2022.05

conda activate viscy

# preprocess
python -m viscy.cli.preprocess_script --config preprocess.yml

# train/validate
viscy fit --config fit.yml

# test
viscy test --config test.yml

# export
viscy export --config export.yml
