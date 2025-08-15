#!/bin/bash
#SBATCH --job-name=optuna_vae_search
#SBATCH --output=/slurm_out/optuna_vae_%j.out
#SBATCH --error=/slurm_out/optuna_vae_%j.err
#SBATCH --time=0-20:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load modules/environment
module load anaconda/25.3.1
conda activate viscy

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to repo directory
ROOT_DIR="/home/eduardo.hirata/repos/viscy"
cd $ROOT_DIR

# Create output directory for this job
OUTPUT_DIR="/hpc/projects/organelle_phenotyping/models/SEC61B/vae/optuna_results/job_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Run Optuna optimization
echo "Starting Optuna VAE hyperparameter search..."
python viscy/scripts/optimization/optuna_vae_search.py \
    --output_dir $OUTPUT_DIR \
    --n_trials 50 \
    --timeout 86400

# Copy results to output directory
cp optuna_vae_study.db $OUTPUT_DIR/
cp best_vae_config.yml $OUTPUT_DIR/ 2>/dev/null || echo "No best config generated yet"

echo "Job completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"