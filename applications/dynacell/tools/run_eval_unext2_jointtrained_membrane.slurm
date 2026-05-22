#!/bin/bash
# UNeXt2 joint-trained membrane eval — fills the missing UNeXt2 column of
# the joint training pool, mirroring the FNet3D joint nucleus runner:
#   array task 0 → iPSC test set        (1 eval)
#   array task 1 → A549 test set × 3    (mock + DENV + ZIKV)
#
# Submit as:
#   sbatch applications/dynacell/tools/run_eval_unext2_jointtrained_membrane.slurm
#
# Outputs:
#   ipsc/joint_evaluations/eval_unext2_joint_membrane
#   a549/joint_evaluations/eval_unext2_joint_membrane_<infection>

#SBATCH --job-name=eval_unext2_joint_memb
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

# Cap BLAS/OMP threads to the SLURM allocation before any C extension loads.
# applications/dynacell/CLAUDE.md "Eval runtime / parallelism" section.
export DYNACELL_THREADS_PER_WORKER="${SLURM_CPUS_PER_TASK}"
# To enable FOV-level parallelism in a future multi-FOV run, pass these
# overrides to the wrapped eval invocation (see the .sh script):
#   runtime.executor=process runtime.fov_workers=4 runtime.threads_per_worker=4

REPO=/hpc/mydata/alex.kalinin/VisCy
EVAL_DIR="${REPO}/applications/dynacell/configs/evaluations"

SCRIPTS=(
    "${EVAL_DIR}/unext2/run_eval_unext2_jointtrained_ipsc.sh"
    "${EVAL_DIR}/unext2/run_eval_unext2_jointtrained_a549.sh"
)

mkdir -p /hpc/projects/virtual_staining/training/dynacell/a549/joint_evaluations/slurm
mkdir -p /hpc/projects/virtual_staining/training/dynacell/ipsc/joint_evaluations

cd "${REPO}"

bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
