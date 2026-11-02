#!/bin/bash
# Witness-GMM phase classifier — end-to-end Stage A -> Stage B on the ZIKV plate.
#
# Reproduces Soorya's sec61b_timelapse_zikv_classifier.py flow through the
# frameworked two-stage pipeline, on dataset
# 2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV (mantis v2):
#
#   Stage A  dynaclr witness-gmm-labels   (SEC61B organelle -> remodeling labels)
#   Stage B  dynaclr run-linear-classifiers (train Phase3D classifier on those labels)
#
# Both stages are CPU-only (MMD/GMM + sklearn LR). Run directly on a login/compute
# shell, or submit with `sbatch` (SLURM header below).
#
# Usage:
#   bash run_witness_gmm_phase_classifier.sh          # run both stages
#   bash run_witness_gmm_phase_classifier.sh --stage-a  # Stage A only
#   bash run_witness_gmm_phase_classifier.sh --stage-b  # Stage B only (needs Stage A output)
#   sbatch run_witness_gmm_phase_classifier.sh        # submit as a batch job
#
#SBATCH --job-name=witness_gmm_phase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
CONFIG_DIR="$WORKSPACE_DIR/applications/dynaclr/configs/evaluation/witness_gmm_phase_classifier"
STAGE_A_CONFIG="$CONFIG_DIR/01_stage_a_sec61b_labels.yml"
STAGE_B_PHASE_CONFIG="$CONFIG_DIR/02_stage_b_train_phase.yml"
STAGE_B_SEC61B_CONFIG="$CONFIG_DIR/02_stage_b_train_sec61b.yml"

cd "$WORKSPACE_DIR"

RUN_A=1
RUN_B=1
case "${1:-}" in
    --stage-a) RUN_B=0 ;;
    --stage-b) RUN_A=0 ;;
    "") ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac

if [[ $RUN_A -eq 1 ]]; then
    echo "## Stage A — witness-GMM labels (SEC61B organelle)"
    uv run --project "$WORKSPACE_DIR" dynaclr witness-gmm-labels -c "$STAGE_A_CONFIG"
fi

if [[ $RUN_B -eq 1 ]]; then
    echo "## Stage B — train Phase3D classifier from SEC61B labels (teacher/student)"
    uv run --project "$WORKSPACE_DIR" dynaclr run-linear-classifiers -c "$STAGE_B_PHASE_CONFIG"
    echo "## Stage B — train SEC61B classifier from SEC61B labels (same-modality)"
    uv run --project "$WORKSPACE_DIR" dynaclr run-linear-classifiers -c "$STAGE_B_SEC61B_CONFIG"
fi

echo "## Done."
