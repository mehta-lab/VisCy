#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpc/mydata/$USER/slurm_logs/dataloader_test_%j.txt
#SBATCH --nodelist=gpu-b-[3-4,6]


# Activate viscy and run the dataloader_test.py script
conda activate /hpc/mydata/$USER/envs/viscy/
python /hpc/mydata/$USER/viscy/applications/contrastive_phenotyping/dataloader_test.py