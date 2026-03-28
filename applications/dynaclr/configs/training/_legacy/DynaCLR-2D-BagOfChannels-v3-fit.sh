#!/bin/bash

#SBATCH --job-name=dynaclr_boc_v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=14G
#SBATCH --time=0-22:00:00

WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="${WORKSPACE_DIR}/applications/dynaclr/configs/training/DynaCLR-2D-BagOfChannels-v3-fit.yml"
RUN_NAME="DynaCLR-2D-BagOfChannels-v3-test"
RUN_DIR="/hpc/projects/organelle_phenotyping/models/${RUN_NAME}"

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

function cleanup() {
  rm -rf /tmp/$SLURM_JOB_ID/*.zarr
  echo "Cleanup completed."
}
trap cleanup EXIT

scontrol show job $SLURM_JOB_ID
cat "$CONFIG"

srun uv run --project "$WORKSPACE_DIR" viscy fit \
  --config "$CONFIG" \
  --trainer.logger.init_args.name="${RUN_NAME}" \
  --trainer.logger.init_args.save_dir="${RUN_DIR}" \
