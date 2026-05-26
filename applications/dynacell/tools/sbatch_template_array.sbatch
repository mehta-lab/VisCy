#!/bin/bash

@@sbatch_directives

# Shared-group writes: make every dir/file created by this job (mkdirs below,
# wandb run folders, Lightning ModelCheckpoint, HCSPredictionWriter) g+w.
umask 0002

mkdir -p -m 775 @@run_root/slurm
mkdir -p -m 775 @@run_root/checkpoints

function cleanup() {
  # Guard the job-id expansion: outside Slurm SLURM_JOB_ID is unset, and
  # `rm -rf /tmp/ /dev/shm/` would wipe shared temp locations.
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    rm -rf "/tmp/${SLURM_JOB_ID}" "/dev/shm/${SLURM_JOB_ID}"
  fi
  echo "Cleanup Completed."
}
trap cleanup EXIT

ml uv

# uv resolves project context from CWD; pin to repo_root so the compute job
# uses the worktree's pyproject + .venv regardless of where sbatch was invoked.
cd @@repo_root

@@env_block

scontrol show job $SLURM_JOB_ID
nvidia-smi

# Each array task picks its resolved config from CONFIGS[SLURM_ARRAY_TASK_ID].
# All entries share the SBATCH directives baked into the array (same hardware
# profile per array submission); --allow-mixed-directives would have routed
# differently-shaped leaves into separate array submissions.
CONFIGS=(
@@configs_list
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set; this template requires sbatch --array=..."
  exit 1
fi

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "$CFG" ]]; then
  echo "ERROR: no config for array task $SLURM_ARRAY_TASK_ID (CONFIGS has ${#CONFIGS[@]} entries)"
  exit 1
fi
echo "[array] task $SLURM_ARRAY_TASK_ID / $((${#CONFIGS[@]} - 1)) → $CFG"

# Predict-only: single-rank, no NCCL preflight (matches sbatch_template_batch.sbatch).
srun uv run python -m dynacell predict --config "$CFG"
