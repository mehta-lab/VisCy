#!/bin/bash
# Predict link of the model matrix (dynaclr run-matrix chains this after train).
# Runs `dynaclr predict-triplet` for one model over its collection via
# `dynaclr predict-batch`.
#
# Positional args (passed by `dynaclr run-matrix`):
#   $1 collection      collection YAML (predict-triplet input)
#   $2 model_family    PROJECT / embedding-tree key
#   $3 run             RUN_NAME / embedding-tree key
#   $4 ckpt_name       checkpoint label (last | epochN-stepM)
#   $5 datasets_root   base under which datasets live
#   $6 checkpoint      (optional) explicit checkpoint path; empty = derive from run dir
#
#SBATCH --job-name=dynaclr_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

COLLECTION="$1"
MODEL_FAMILY="$2"
RUN="$3"
CKPT_NAME="$4"
DATASETS_ROOT="$5"
CHECKPOINT="${6:-}"
MARKERS="${7:-}"

CKPT_FLAG=()
if [ -n "$CHECKPOINT" ]; then
  CKPT_FLAG=(--checkpoint "$CHECKPOINT")
fi

MARKER_FLAG=()
if [ -n "$MARKERS" ]; then
  MARKER_FLAG=(--markers)
  IFS=',' read -ra _M <<< "$MARKERS"
  MARKER_FLAG+=("${_M[@]}")
fi

srun uv run --project "$WORKSPACE_DIR" dynaclr predict-batch \
  -c "$COLLECTION" \
  --model-family "$MODEL_FAMILY" \
  --run "$RUN" \
  --ckpt-name "$CKPT_NAME" \
  --datasets-root "$DATASETS_ROOT" \
  "${CKPT_FLAG[@]}" \
  "${MARKER_FLAG[@]}" \
  --num-workers 0
