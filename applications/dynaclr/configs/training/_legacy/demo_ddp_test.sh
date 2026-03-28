#!/bin/bash
# Quick DDP smoke test using the demo bag-of-channels v3 collection.
# Small dataset (~4 experiments), fast setup, validates 4-GPU DDP end-to-end.
# Usage: sbatch demo_ddp_test.sh

#SBATCH --job-name=dynaclr_ddp_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=14G
#SBATCH --time=0-00:30:00
#SBATCH --constraint=a100|a40|a6000

WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
CONFIG="${WORKSPACE_DIR}/applications/dynaclr/configs/training/demo_bag_of_channels_v3_fit.yml"

export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

scontrol show job $SLURM_JOB_ID

CELL_INDEX="${WORKSPACE_DIR}/applications/dynaclr/configs/cell_index/demo_bag_of_channels_v3.parquet"

srun uv run --project "$WORKSPACE_DIR" viscy fit \
  --config "$CONFIG" \
  --data.init_args.cell_index_path="$CELL_INDEX" \
  --trainer.devices=2 \
  --trainer.strategy=ddp \
  --trainer.fast_dev_run=true \
  --trainer.enable_model_summary=false \
  --trainer.use_distributed_sampler=false \
