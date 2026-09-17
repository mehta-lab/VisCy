#!/bin/bash
# Direct evaluation worker for `dynaclr run-matrix-eval` (no Nextflow).
# Resources are supplied by the launcher through sbatch command-line flags.
#
# Positional args:
#   $1 plan YAML, $2 model family, $3 run, $4 checkpoint, $5 evaluation,
#   $6 embedding key (X = raw .X), $7 overwrite (0|1)
#SBATCH --job-name=dynaclr_matrix_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

PLAN="$1"
MODEL_FAMILY="$2"
RUN="$3"
CKPT_NAME="$4"
EVALUATION="$5"
EMBEDDING_KEY="$6"
OVERWRITE="${7:-0}"

OVERWRITE_FLAG=(--no-overwrite)
if [ "$OVERWRITE" = "1" ]; then
  OVERWRITE_FLAG=(--overwrite)
fi

srun uv run --project "$WORKSPACE_DIR" dynaclr evaluate-matrix-unit \
  --config "$PLAN" \
  --model-family "$MODEL_FAMILY" \
  --run "$RUN" \
  --ckpt-name "$CKPT_NAME" \
  --evaluation "$EVALUATION" \
  --embedding-key "$EMBEDDING_KEY" \
  "${OVERWRITE_FLAG[@]}"
