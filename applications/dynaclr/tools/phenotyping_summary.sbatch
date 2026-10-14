#!/bin/bash
# Submit with --dependency=afterok:<segmentation-array-job>.
# Arguments: attempt directory, workspace, Python, marker (default TOMM20).
#SBATCH --job-name=cpdino_summary
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
PHENO_ATTEMPT="${1:?attempt directory required}"
PHENO_WORKSPACE="${2:?workspace required}"
PHENO_PYTHON="${3:?phenotyping Python required}"
PHENO_MARKER="${4:-TOMM20}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export PYTHONPATH="$PHENO_ATTEMPT/runtime${PYTHONPATH:+:$PYTHONPATH}"
cd "$PHENO_WORKSPACE"
srun "$PHENO_PYTHON" -u -m dynaclr.phenotyping.pipeline --config "$PHENO_ATTEMPT/pilot.yml" --summarize
srun "$PHENO_PYTHON" -u "$PHENO_ATTEMPT/provenance/review_cell_signal.py" --attempt "$PHENO_ATTEMPT" --marker "$PHENO_MARKER"
