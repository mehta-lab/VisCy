#!/bin/bash
# Train dynaformer on the Leger et al. cell-cycle dataset, brightfield input.
#
# Usage:  sbatch run_fit.slurm [leaf.yml]
#         defaults to fit_unext2_bf.yml; pass fit_unext2_bf_single_frame.yml for the ablation.
#
# Invariant: --ntasks-per-node must equal trainer.devices (1, from recipes/topology/single_gpu.yml)
# and --gpus. Lightning's SLURMEnvironment rejects --ntasks and kills the job at trainer init.
#SBATCH --job-name=dynaformer_leger
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/%x-%j.out

set -euo pipefail

LEAF="${1:-fit_unext2_bf.yml}"
ROOT=/hpc/projects/comp.micro/virtual_staining/models/dynaformer
REPO="$ROOT/VisCy"
CONFIG="$REPO/applications/dynaformer/configs/examples/leger/$LEAF"
RUNDIR="$ROOT/runs/${LEAF%.yml}"

mkdir -p "$ROOT/logs" "$RUNDIR"
# Checkpoints, wandb files and logs all follow the working directory. Running from the run
# directory keeps them out of the config tree without overriding the logger on the command line --
# overriding `trainer.logger` from the CLI wedged the process during startup (no output, 4 s of CPU
# in 14 min, GPU never allocated), so the logger is left exactly as the recipe declares it.
# `base:` paths resolve relative to the config file, so an absolute -c works from anywhere.
cd "$RUNDIR"

echo "host=$(hostname)  leaf=$LEAF  job=${SLURM_JOB_ID:-none}  rundir=$RUNDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$REPO/.venv/bin/python" -u -m dynaformer fit -c "$CONFIG"
