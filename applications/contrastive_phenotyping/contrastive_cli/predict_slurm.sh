#!/bin/bash

#SBATCH --job-name=contrastive_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-01:00:00

module load anaconda/2022.05
# Update to use the actual prefix
conda activate $MYDATA/envs/viscy

scontrol show job $SLURM_JOB_ID

# use absolute path in production
config=./predict.yml
cat $config
srun python -m viscy.cli.contrastive_triplet predict -c $config
