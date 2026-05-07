#!/bin/bash
# Joint-trained (iPSC + A549 mantis) FNet3D nucleus eval — fills the two
# missing slots in the 6 train-set × 2 organelle eval matrix:
#   array task 0 → iPSC test set        (1 eval)
#   array task 1 → A549 test set × 3    (mock + DENV + ZIKV)
#
# Submit as:
#   sbatch applications/dynacell/tools/run_eval_jointtrained_nucleus.slurm
#
# Outputs land under:
#   ipsc/joint_evaluations/eval_fnet3d_joint_nucleus
#   a549/joint_evaluations/eval_fnet3d_joint_nucleus_<infection>

#SBATCH --job-name=eval_jointtrained_nucleus
#SBATCH --array=0-1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=/hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/slurm/%x_%A_%a.out
#SBATCH --error=/hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/slurm/%x_%A_%a.err

set -euo pipefail

REPO=/hpc/mydata/alex.kalinin/VisCy
EVAL_DIR="${REPO}/applications/dynacell/configs/evaluations"

SCRIPTS=(
    "${EVAL_DIR}/fnet3d/run_eval_fnet3d_jointtrained_ipsc.sh"
    "${EVAL_DIR}/fnet3d/run_eval_fnet3d_jointtrained_a549.sh"
)

mkdir -p /hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/slurm
mkdir -p /hpc/projects/virtual_staining/training/dynacell/ipsc/joint_evaluations

cd "${REPO}"

bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
