#!/bin/bash
# Eval link of the model matrix (dynaclr run-matrix chains this after predict).
# Runs `dynaclr eval`, which launches the Nextflow eval_from_embeddings entry
# over the frozen embeddings for this (model_family, run, ckpt_name). Nextflow
# submits its own downstream SLURM steps under the executor, so this job is
# light (a launcher) — it stays alive to drive the Nextflow run.
#
# Positional args (passed by `dynaclr run-matrix`):
#   $1 eval_config     evaluation config YAML (what to compute)
#   $2 model_family    embedding-tree key
#   $3 run             embedding-tree key
#   $4 ckpt_name       embedding-tree key
#   $5 datasets_root   base under which datasets live
#
#SBATCH --job-name=dynaclr_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

module load nextflow/24.10.5

EVAL_CONFIG="$1"
MODEL_FAMILY="$2"
RUN="$3"
CKPT_NAME="$4"
DATASETS_ROOT="$5"

uv run --project "$WORKSPACE_DIR" dynaclr eval \
  --eval-config "$EVAL_CONFIG" \
  --model-family "$MODEL_FAMILY" \
  --run "$RUN" \
  --ckpt-name "$CKPT_NAME" \
  --datasets-root "$DATASETS_ROOT"
