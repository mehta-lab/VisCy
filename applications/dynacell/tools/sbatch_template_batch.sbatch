#!/bin/bash

@@sbatch_directives

# Shared-group writes: make every dir/file created by this job (mkdirs below,
# wandb run folders, Lightning ModelCheckpoint, HCSPredictionWriter) g+w.
umask 0002

mkdir -p -m 775 @@run_root/slurm
mkdir -p -m 775 @@run_root/checkpoints

function cleanup() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    rm -rf "/tmp/${SLURM_JOB_ID}" "/dev/shm/${SLURM_JOB_ID}"
  fi
  echo "Cleanup Completed."
}
trap cleanup EXIT

ml uv

cd @@repo_root

@@env_block

scontrol show job $SLURM_JOB_ID
nvidia-smi

# Predict-only batch job: each invocation is independent. No NCCL preflight
# needed (single-GPU predict). The MASTER_ADDR/PORT dance from the train
# template is unnecessary for predict — Lightning's predict runs on a
# single rank by default.

@@predict_invocations
