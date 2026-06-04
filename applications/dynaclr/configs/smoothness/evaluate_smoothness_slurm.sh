#!/bin/bash

#SBATCH --job-name=dynaclr_smoothness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="$(dirname "$0")/example_smoothness.yaml"
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Config: $CONFIG"
cat "$CONFIG"
echo ""

uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
    dynaclr evaluate-smoothness -c "$CONFIG"
