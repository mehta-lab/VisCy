#!/bin/bash

#SBATCH --job-name=dynaclr_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG=${1:?Usage: sbatch cross_validate_slurm.sh <config.yaml> [--task <task_name>]}
EXTRA_ARGS="${@:2}"  # optional: --task infection_state --report
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Config: $CONFIG"
echo "Extra:  $EXTRA_ARGS"
echo ""

uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
    dynaclr cross-validate -c "$CONFIG" $EXTRA_ARGS
