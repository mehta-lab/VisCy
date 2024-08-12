#!/bin/bash

#SBATCH --job-name=contrastive_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-01:00:00

module load anaconda/2022.05
conda activate viscy

scontrol show job $SLURM_JOB_ID

config=/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/contrastive_tune_augmentations/predict/predict.yml
cat $config
srun python -m viscy.cli.contrastive_triplet predict -c $config
