#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=gpu-c-1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=dataloader_test
#SBATCH --output=slurm-%j.out

# Make sure that /hpc/mydata/$USER/slurm_logs/ exists!!

# Activate viscy and run the dataloader_test.py script
module load anaconda/2022.05
conda activate /hpc/mydata/$USER/envs/viscy/
python /hpc/mydata/$USER/VisCy/viscy/applications/contrastive_phenotyping/training_script.py