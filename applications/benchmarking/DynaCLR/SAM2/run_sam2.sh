#!/bin/bash

#SBATCH --job-name=dynaclr_imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=7G
#SBATCH --time=0-02:00:00
#SBATCH --output=./slurm_logs/%j_dynaclr_sam2.out


module load anaconda/latest
conda activate viscy

CONFIG_PATH=/home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/SAM2/sam2_sensor_only.yml
python /home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/SAM2/sam2_embeddings.py -c $CONFIG_PATH
