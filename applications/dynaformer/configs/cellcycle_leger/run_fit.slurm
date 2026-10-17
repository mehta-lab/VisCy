#!/bin/bash
# Train one cell-cycle model on the Leger et al. RPE1 tracks. Submit with submit.sh, which also queues the evaluation
# (run_eval.slurm).
#
# Usage:  sbatch [--time=HH:MM:SS] run_fit.slurm <experiment leaf, e.g. regression_c-radio.yml>
#
# Short allocations are fine: before the time limit SLURM sends SIGUSR1, Lightning saves an hpc_ckpt_*.ckpt in the run
# directory and requeues the job, and the restarted fit resumes from it.
# Invariant: --ntasks-per-node must equal trainer.devices (1, recipes/trainer.yml) and --gpus.
#SBATCH --job-name=ccl_fit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle_leger/%x-%j.out

set -euo pipefail
# Node-local /tmp: multiprocessing cleans up its socket folders badly on the NFS default TMPDIR.
export TMPDIR=/tmp
LEAF="$1"
ROOT=/hpc/projects/comp.micro/virtual_staining/models/dynaformer
CONFIG="$ROOT/VisCy/applications/dynaformer/configs/cellcycle_leger/experiments/$LEAF"
RUNDIR="$ROOT/runs/cellcycle_leger/${LEAF%.yml}"
# The VAST copy of the workspace environment: importing torch from the comp.micro .venv takes minutes.
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"

mkdir -p "$RUNDIR"
# Checkpoints, hpc checkpoints and wandb files follow the working directory (recipes use relative paths).
cd "$RUNDIR"
echo "host=$(hostname) leaf=$LEAF job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} rundir=$RUNDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Resume: Lightning restores an hpc_ckpt_*.ckpt by itself (saved on SIGUSR1 before the time limit). A preempted job
# gets SIGTERM instead and leaves none, so it resumes from the last validation's checkpoint.
RESUME=()
if ! compgen -G "hpc_ckpt_*.ckpt" > /dev/null && [ -f checkpoints/last.ckpt ]; then
  RESUME=(--ckpt_path checkpoints/last.ckpt)
fi
srun "$PYTHON" -u -m dynaformer fit -c "$CONFIG" "${RESUME[@]}"
