#!/bin/bash

#SBATCH --job-name=dynaclr_apply_lc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="$(dirname "$0")/example_linear_classifier_inference.yaml"
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Config: $CONFIG"
cat "$CONFIG"
echo ""

uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
    dynaclr apply-linear-classifier -c "$CONFIG"
