#!/bin/bash
# Normalization link of the model matrix (chained after prediction).
# Fits one shared, time-matched control median/MAD -> pooled marker PCA80
# representation and selectively writes X_normalized_pca80 + provenance to
# every expected embedding store for this model checkpoint.
#
# Positional args (passed by `dynaclr run-matrix`):
#   $1 collection      full collection YAML (never the skip-existing subset)
#   $2 model_family    embedding-tree key
#   $3 run             embedding-tree key
#   $4 ckpt_name       embedding-tree key
#   $5 datasets_root   base under which prediction outputs live
#   $6 markers         (optional) comma-separated marker subset
#   $7 recipe          pooled representation recipe
#
#SBATCH --job-name=dynaclr_normalize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

COLLECTION="$1"
MODEL_FAMILY="$2"
RUN="$3"
CKPT_NAME="$4"
DATASETS_ROOT="$5"
MARKERS="${6:-}"
RECIPE="${7:-applications/dynaclr/configs/evaluation/recipes/witness_gmm_pooled_joint_pca80.yaml}"

MARKER_FLAG=()
if [ -n "$MARKERS" ]; then
  IFS=',' read -ra _M <<< "$MARKERS"
  for _m in "${_M[@]}"; do
    MARKER_FLAG+=(--markers "$_m")
  done
fi

srun uv run --project "$WORKSPACE_DIR" dynaclr normalize-embeddings \
  --collection "$COLLECTION" \
  --model-family "$MODEL_FAMILY" \
  --run "$RUN" \
  --ckpt-name "$CKPT_NAME" \
  --datasets-root "$DATASETS_ROOT" \
  --recipe "$RECIPE" \
  "${MARKER_FLAG[@]}" \
  --overwrite
