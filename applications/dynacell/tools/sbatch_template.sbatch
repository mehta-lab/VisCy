#!/bin/bash

@@sbatch_directives

mkdir -p -m 775 @@run_root/slurm
mkdir -p -m 775 @@run_root/checkpoints

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID /dev/shm/$SLURM_JOB_ID
  echo "Cleanup Completed."
}
trap cleanup EXIT

ml uv

@@env_block

scontrol show job $SLURM_JOB_ID
nvidia-smi
srun uv run python -m dynacell @@mode --config @@resolved_config@@overrides
