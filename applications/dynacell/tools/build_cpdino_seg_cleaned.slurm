#!/bin/bash
#SBATCH --job-name=cpdino_seg_cleaned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=./slurm_logs/cpdino_seg-%A_%a.out

# Build cpdino whole-cell _seg_cleaned masks from the 9 A549 mantis test _vs
# stores (SEC61B/TOMM20/dual_nucl_memb x mock/ZIKV/DENV). One array task per
# store. Requires the _vs.zarr predictions to exist (5-predict stage).
#
# Usage (submit from this dir, or via --chdir):
#   sbatch --chdir=<this dir> build_cpdino_seg_cleaned.slurm

set -euo pipefail

mkdir -p slurm_logs

VISCY_ROOT=/hpc/mydata/alex.kalinin/VisCy
CPDINO_VENV=/hpc/mydata/alex.kalinin/VisCy-venvs/cpdino-eval
HF_HUB_CACHE=/hpc/projects/comp.micro/virtual_staining/models/dynacell/evaluation/hf_cache
TEST_DIR=/hpc/projects/virtual_staining/training/dynacell/a549/mantis/test

VS_STORES=(
    "$TEST_DIR/SEC61B_DENV_vs.zarr"
    "$TEST_DIR/SEC61B_ZIKV_vs.zarr"
    "$TEST_DIR/SEC61B_mock_vs.zarr"
    "$TEST_DIR/TOMM20_DENV_vs.zarr"
    "$TEST_DIR/TOMM20_ZIKV_vs.zarr"
    "$TEST_DIR/TOMM20_mock_vs.zarr"
    "$TEST_DIR/dual_nucl_memb_DENV_vs.zarr"
    "$TEST_DIR/dual_nucl_memb_ZIKV_vs.zarr"
    "$TEST_DIR/dual_nucl_memb_mock_vs.zarr"
)

VS_STORE="${VS_STORES[$SLURM_ARRAY_TASK_ID]}"
echo "Array task $SLURM_ARRAY_TASK_ID: $VS_STORE"

export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="$CPDINO_VENV"
export HF_HUB_CACHE="$HF_HUB_CACHE"

ml uv 2>/dev/null || true

# The builder refuses an existing output rather than truncating it. To rebuild
# the nine stores in place, add --force to the line below -- deliberately not on
# by default, since these are the canonical io.cell_segmentation_path stores.
uv run --no-sync --directory "$VISCY_ROOT" python \
    "$VISCY_ROOT/applications/dynacell/tools/build_cpdino_seg_cleaned.py" \
    --vs-store "$VS_STORE"
