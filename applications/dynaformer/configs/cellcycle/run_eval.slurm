#!/bin/bash
# Score each best checkpoint of a trained cell-cycle model on the validation and test splits.
#
# Usage:  sbatch run_eval.slurm <experiment leaf>   (submit.sh queues it after the fit)
# Writes runs/cellcycle/<leaf>/eval/<checkpoint>/{predictions,metrics}_{val,test}.{csv,json}.
#SBATCH --job-name=cc_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle/%x-%j.out

set -euo pipefail
export TMPDIR=/tmp
LEAF="$1"
ROOT=/hpc/projects/comp.micro/virtual_staining/models/dynaformer
CONFIG="$ROOT/VisCy/applications/dynaformer/configs/cellcycle/experiments/$LEAF"
RUNDIR="$ROOT/runs/cellcycle/${LEAF%.yml}"
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"

cd "$RUNDIR"
echo "host=$(hostname) leaf=$LEAF job=${SLURM_JOB_ID:-none}"
for ckpt in checkpoints/best_*.ckpt; do
  which=$(basename "$ckpt" .ckpt)
  for split in val test; do
    srun "$PYTHON" -u -m dynaformer test -c "$CONFIG" --ckpt_path "$ckpt" --trainer.logger=false \
      --data.init_args.test_split "$split" --model.init_args.output_dir "$RUNDIR/eval/$which"
  done
done
