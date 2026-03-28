#!/bin/bash
# OPS 373-gene DynaCLR with gene classifier head
#
# Usage:
#   sbatch applications/dynaclr/configs/training/slurm/ops-373genes.sh

#SBATCH --job-name=dynaclr_ops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-22:00:00

TRAINING_DIR="applications/dynaclr/configs/training"

export PROJECT="dynaclr"
export RUN_NAME="${RUN_NAME:-OPS-373genes-GeneClassifier}"
export CONFIGS="\
${TRAINING_DIR}/_base.yml \
${TRAINING_DIR}/arch/2d_z1.yml \
${TRAINING_DIR}/data/boc_2d_gene-reporter_stratify-marker.yml"
export EXTRA_ARGS="${EXTRA_ARGS} \
  --trainer.devices 4 \
  --trainer.max_epochs 300 \
  --trainer.limit_train_batches 400 \
  --trainer.limit_val_batches 100 \
  --trainer.log_every_n_steps 5 \
  --trainer.callbacks='[{class_path: lightning.pytorch.callbacks.LearningRateMonitor, init_args: {logging_interval: step}},{class_path: lightning.pytorch.callbacks.ModelCheckpoint, init_args: {monitor: loss/val, every_n_epochs: 1, save_top_k: 5, save_last: true}}]' \
  --model.init_args.encoder.init_args.projection_dim 256 \
  --model.init_args.encoder.init_args.drop_path_rate 0.0 \
  --model.init_args.loss_function.init_args.temperature 0.5 \
  --model.init_args.lr 0.0001 \
  --model.init_args.ckpt_path /hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr/ops_bagofchannels_gene_n_reporter_grouped_reporter_256proj_373genes_convnext_tiny_temp0p5_512bs_lr1e-4_pretrained_self/version_0/checkpoints/last.ckpt \
  --model.init_args.log_batches_per_epoch 8 \
  --model.init_args.log_samples_per_batch 1 \
  --model.init_args.example_input_array_shape='[1,1,1,128,128]' \
  --data.init_args.cell_index_path /hpc/mydata/eduardo.hirata/repos/viscy/applications/dynaclr/configs/cell_index/ops_373genes.parquet \
  --data.init_args.yx_patch_size='[224,224]' \
  --data.init_args.final_yx_patch_size='[128,128]' \
  --data.init_args.batch_size 512 \
  --data.init_args.label_columns='{gene_label: perturbation}'"

# NOTE: auxiliary_heads is too complex for CLI flags.
# For the gene classifier head, use an override YAML instead:
#   --config ${TRAINING_DIR}/overrides/ops_gene_head.yml

source "$(dirname "$0")/train.sh"
