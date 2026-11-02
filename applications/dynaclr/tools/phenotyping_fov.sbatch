#!/bin/bash
# Initialize the attempt once with --init, then submit one FOV per array task.
# Arguments: recipe YAML, FOV list, workspace, Python, optional frozen runtime.
#SBATCH --job-name=cpdino_cells
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
PHENO_CONFIG="${1:?recipe YAML required}"
PHENO_FOV_LIST="${2:?FOV list required}"
PHENO_WORKSPACE="${3:?workspace required}"
PHENO_PYTHON="${4:?phenotyping Python required}"
PHENO_RUNTIME="${5:-}"
PHENO_TASK="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
PHENO_FOV="$(sed -n "$((PHENO_TASK + 1))p" "$PHENO_FOV_LIST")"
test -n "$PHENO_FOV"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
if test -n "$PHENO_RUNTIME"; then
    export PYTHONPATH="$PHENO_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
fi
cd "$PHENO_WORKSPACE"
srun "$PHENO_PYTHON" -u -m dynaclr.phenotyping.pipeline --config "$PHENO_CONFIG" --resume --fov "$PHENO_FOV" --no-summary
