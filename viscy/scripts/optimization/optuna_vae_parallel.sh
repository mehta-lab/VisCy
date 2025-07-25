#!/bin/bash
#SBATCH --job-name=optuna_vae_parallel
#SBATCH --output=optuna_vae_parallel_%A_%a.out
#SBATCH --error=optuna_vae_parallel_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6G
#SBATCH --array=1-4  # Run 4 parallel workers


# Print job info
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Change to repo directory
module load anaconda/25.3.1
conda activate viscy

# Load environment
OPTUNA_SCRIPT='/home/eduardo.hirata/repos/viscy/viscy/scripts/optimization/optuna_vae_search.py'

# Shared storage for Optuna study (all workers use same database)
SHARED_DB="/hpc/projects/organelle_phenotyping/models/SEC61B/vae/optuna_results/optuna_parallel_study.db"
OUTPUT_DIR="/hpc/projects/organelle_phenotyping/models/SEC61B/vae//optuna_results/parallel_job_${SLURM_ARRAY_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Each worker runs a portion of trials
TRIALS_PER_WORKER=15  # 4 workers × 15 trials = 60 total trials

echo "Worker $SLURM_ARRAY_TASK_ID starting $TRIALS_PER_WORKER trials..."

# Run Optuna optimization (all workers share the same database)
python  \
    --storage_url "sqlite:///$pp" \
    --n_trials $TRIALS_PER_WORKER \
    --timeout 43200 \
    --study_name "vae_parallel_optimization"

# Only the first worker saves the final results
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Worker 1 saving final results..."
    sleep 60  # Wait for other workers to finish
    cp $SHARED_DB $OUTPUT_DIR/
    cp best_vae_config.yml $OUTPUT_DIR/ 2>/dev/null || echo "No best config generated yet"
fi

echo "Worker $SLURM_ARRAY_TASK_ID completed at: $(date)"