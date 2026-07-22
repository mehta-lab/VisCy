#!/bin/bash
# DynaCLR-2D-MIP-pretrain CLASSICAL (SimCLR-style) variant — Reef/Kelp smoke test.
# Adapted from
# applications/dynaclr/configs/training/DynaCLR-2D/bruno/DynaCLR-2D-MIP-BagOfChannels-classical.sh
# per docs/clusters/reef.md: named GPU partition instead of --constraint,
# explicit --qos (Reef requires one). Uses the requeing_slurm branch's
# checkpoint+requeue support (train.sh always passes --slurm_auto_requeue;
# it only attaches SLURMEnvironment(auto_requeue=True) when actually running
# under SLURM). --requeue/--signal are harmless at --qos dev (non-preemptible)
# and keep this script ready to resubmit at --qos mid/low unchanged.
#
#   sbatch applications/dynaclr/configs/training/DynaCLR-2D/reef/DynaCLR-2D-MIP-pretrain-classical.sh

#SBATCH --job-name=dynaclr_2d_pretrain_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --partition=h100-reserved
#SBATCH --qos=dev
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

export WORKSPACE_DIR="/mnt/main0/home/eduardo.hirata/repos/VisCy"
export MODEL_ROOT="/bio/projects/compimaging/models"
export PROJECT="DynaCLR-2D-MIP-pretrain"
export RUN_NAME="2d-mip-classical-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-reef-smoke"
export CONFIGS="applications/dynaclr/configs/training/DynaCLR-2D/reef/DynaCLR-2D-MIP-pretrain.yml applications/dynaclr/configs/training/DynaCLR-2D/reef/DynaCLR-2D-MIP-pretrain-classical.yml"

# Smoke test: cap epochs/batches so we quickly get a checkpoint to validate
# against (single dirpath, no duplication) and to test SLURM preemption
# (auto-requeue + resume). Drop this override once the pipeline is validated.
export EXTRA_ARGS="--trainer.max_epochs=30 --trainer.limit_train_batches=5 --trainer.limit_val_batches=3"

source "${WORKSPACE_DIR}/applications/dynaclr/configs/training/slurm/train.sh"
