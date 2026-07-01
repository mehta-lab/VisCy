#!/bin/bash
# DynaCLR training launcher
# ==========================
# Handles run directory setup, config copying, and srun dispatch.
# Each training run is defined by a single YAML + a thin SLURM shell script.
#
# Required env vars:
#   PROJECT    WandB project name (also model output directory name)
#   RUN_NAME   WandB run name
#   CONFIGS    Space-separated list of config files (relative to WORKSPACE_DIR)
#
# Optional env vars:
#   WORKSPACE_DIR  Repo root (default: /hpc/mydata/eduardo.hirata/repos/viscy)
#   MODEL_ROOT     Model output root (default: /hpc/projects/organelle_phenotyping/models)
#   EXTRA_ARGS     Extra CLI args passed to dynaclr fit
#   CKPT_PATH      Path to checkpoint to resume from (appends --ckpt_path)
#   WANDB_RUN_ID   W&B run ID to resume (continues metrics on same run)
#   CALLER_SCRIPT  Set automatically by train.sh — the sbatch script that sourced us

PROJECT="${PROJECT:?Set PROJECT (WandB project name)}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME (WandB run name)}"
CONFIGS="${CONFIGS:?Set CONFIGS (space-separated config paths)}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
MODEL_ROOT="${MODEL_ROOT:-/hpc/projects/organelle_phenotyping/models}"
RUN_DIR="${MODEL_ROOT}/${PROJECT}/${RUN_NAME}"

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# bf16-mixed already lets matmuls use TF32; "high" instructs Lightning to
# enable TF32 for any remaining float32 matmuls (silences the runtime warning).
export TORCH_FLOAT32_MATMUL_PRECISION=high

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup completed."
}
trap cleanup EXIT

mkdir -p "${RUN_DIR}/checkpoints"

# Rotate existing config.yaml before Lightning overwrites it
if [ -f "${RUN_DIR}/config.yaml" ]; then
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  cp "${RUN_DIR}/config.yaml" "${RUN_DIR}/config.${TIMESTAMP}.yaml"
fi

for cfg in $CONFIGS; do
  cp "${WORKSPACE_DIR}/${cfg}" "${RUN_DIR}/"
done
# Copy the calling SLURM script for reproducibility (sbatch from RUN_DIR to resume)
if [ -n "${BASH_SOURCE[1]:-}" ] && [ -f "${BASH_SOURCE[1]}" ]; then
  cp "${BASH_SOURCE[1]}" "${RUN_DIR}/"
fi

scontrol show job $SLURM_JOB_ID

CONFIG_FLAGS=""
for cfg in $CONFIGS; do
  CONFIG_FLAGS="${CONFIG_FLAGS} --config ${WORKSPACE_DIR}/${cfg}"
done

# Auto-resume after SLURM preemption/requeue: if no explicit CKPT_PATH was
# given but a checkpoint exists in the (stable) run directory, resume from it.
# A fresh run has no last.ckpt, so the first launch trains from scratch.
if [ -z "${CKPT_PATH:-}" ] && [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
  CKPT_PATH="${RUN_DIR}/checkpoints/last.ckpt"
  echo "Resuming from ${CKPT_PATH}"
fi

CKPT_FLAG=""
if [ -n "${CKPT_PATH:-}" ]; then
  CKPT_FLAG="--ckpt_path ${CKPT_PATH}"
fi

# Persist the W&B run id so a requeued job continues the same run (continuous
# metrics across preemptions). The id is generated once on first launch and
# reused on every resume. Generating it shell-side (rather than reading it back
# from wandb) avoids touching logger.experiment in a callback, which can
# deadlock DDP.
WANDB_ID_FILE="${RUN_DIR}/.wandb_run_id"
if [ -z "${WANDB_RUN_ID:-}" ]; then
  if [ -f "${WANDB_ID_FILE}" ]; then
    WANDB_RUN_ID="$(cat "${WANDB_ID_FILE}")"
  else
    WANDB_RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
    echo "${WANDB_RUN_ID}" > "${WANDB_ID_FILE}"
  fi
fi

# resume=allow (not must) so the first launch can create the run; subsequent
# requeues find the existing id and continue it.
WANDB_ID_FLAG="--trainer.logger.init_args.id=${WANDB_RUN_ID} --trainer.logger.init_args.resume=allow"

srun uv run --project "$WORKSPACE_DIR" dynaclr fit \
  ${CONFIG_FLAGS} \
  --slurm_auto_requeue \
  --trainer.default_root_dir="${RUN_DIR}" \
  --trainer.logger.init_args.project="${PROJECT}" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
  ${CKPT_FLAG} \
  ${WANDB_ID_FLAG} \
  ${EXTRA_ARGS}
