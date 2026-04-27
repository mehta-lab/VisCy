#!/bin/bash

@@sbatch_directives

# Shared-group writes: make every dir/file created by this job (mkdirs below,
# wandb run folders, Lightning ModelCheckpoint, HCSPredictionWriter) g+w.
# Compute-node login umask is not reliably 0002, so set it explicitly.
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

# Pre-flight NCCL smoke test: if the intra-node peer link is broken, catch
# it in ~60 s instead of burning a 30-minute PyTorch watchdog timeout on the
# DDP initial-parameter broadcast. Fails fast; operator resubmits with
# --override launcher.sbatch.exclude=<bad-node>.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))

echo "[preflight] NCCL smoke test..."
srun --kill-on-bad-exit=1 uv run python @@repo_root/applications/dynacell/tools/nccl_smoke_test.py
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "[preflight] smoke test FAILED on $SLURM_JOB_NODELIST (rc=$SMOKE_RC); exiting before main srun" >&2
    exit $SMOKE_RC
fi

# Lightning's SLURMEnvironment picks its own port from SLURM_JOB_ID; don't
# leak the preflight values into the training srun.
unset MASTER_ADDR MASTER_PORT

srun uv run python -m dynacell @@mode --config @@resolved_config
