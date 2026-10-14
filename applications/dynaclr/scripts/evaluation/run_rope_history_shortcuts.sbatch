#!/bin/bash
# Fixed-checkpoint CELL-DINO context-5 history intervention, five held-out folds.

#SBATCH --job-name=pal_celldino_rope_c5_history_s17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/hpc/projects/intracellular_dashboard/organelle_dynamics/logs/%x-%A_%a.out
#SBATCH --error=/hpc/projects/intracellular_dashboard/organelle_dynamics/logs/%x-%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

HELD_OUT_DATASETS=(
  07_01_SEC61B_TOMM20_G3BP1
  03_24_SEC61B
  04_08_G3BP1
  04_10_TOMM20
  07_14_MAP4
)
HELD_OUT="${HELD_OUT_DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "held_out=$HELD_OUT host=$(hostname)"
srun "$WORKSPACE_DIR/.venv-dynaclr/bin/python" \
  applications/dynaclr/scripts/evaluation/evaluate_rope_history_shortcuts.py \
  run-fold \
  --held-out-dataset "$HELD_OUT" \
  --seed 17 \
  --random-seed 17 \
  --device cuda
