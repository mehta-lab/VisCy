#!/bin/bash

#SBATCH --job-name=dynaclr_compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="$(dirname "$0")/example_compare.yaml"
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Config: $CONFIG"
cat "$CONFIG"
echo ""

uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
    dynaclr compare-models -c "$CONFIG"
