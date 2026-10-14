#!/bin/bash
# DynaFormer: self-supervised temporal head on frozen embeddings + baseline ladder.
#   $1 recipe  $2 output_root  $3 tag  $4.. input_paths (one Phase3D.zarr per dataset)
#SBATCH --job-name=dynaformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/hpc/projects/intracellular_dashboard/organelle_dynamics/matrix_evaluations/dynaformer_infection_onset/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1
WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"
RECIPE="$1"; OUTPUT_ROOT="$2"; TAG="$3"; shift 3
INPUT_FLAGS=(); for path in "$@"; do INPUT_FLAGS+=(--input-path "$path"); done
srun uv run --project "$WORKSPACE_DIR" dynaclr dynaformer -c "$RECIPE" --output-root "$OUTPUT_ROOT" --tag "$TAG" "${INPUT_FLAGS[@]}"
