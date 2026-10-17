#!/bin/bash
# Score each best checkpoint of a Leger cell-cycle model on the validation movie and both test splits, then export the
# test predictions to the Leger baseline harness (Andrea_mlbi/code/results/predictions).
#
# Usage:  sbatch run_eval.slurm <experiment leaf>   (submit.sh queues it after the fit)
# Writes runs/cellcycle_leger/<leaf>/eval/<checkpoint>/{predictions,metrics}_{val,test,test_drugged}.{csv,json}.
# The model is small and reads cached embeddings, so it runs on CPU (float32) rather than wait for a GPU.
#SBATCH --job-name=ccl_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle_leger/%x-%j.out

set -euo pipefail
export TMPDIR=/tmp
LEAF="$1"
ROOT=/hpc/projects/comp.micro/virtual_staining/models/dynaformer
CONFIG="$ROOT/VisCy/applications/dynaformer/configs/cellcycle_leger/experiments/$LEAF"
RUNDIR="$ROOT/runs/cellcycle_leger/${LEAF%.yml}"
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"

# Run from eval/: the CLI saves its config.yaml in the working directory, and the fit's one stays as it was.
mkdir -p "$RUNDIR/eval"
cd "$RUNDIR/eval"
echo "host=$(hostname) leaf=$LEAF job=${SLURM_JOB_ID:-none}"
for ckpt in "$RUNDIR"/checkpoints/best_*.ckpt; do
  which=$(basename "$ckpt" .ckpt)
  for split in val test test_drugged; do
    srun "$PYTHON" -u -m dynaformer test -c "$CONFIG" --ckpt_path "$ckpt" --trainer.logger=false \
      --trainer.accelerator=cpu --trainer.precision=32-true \
      --data.init_args.test_split "$split" --model.init_args.output_dir "$RUNDIR/eval/$which"
  done
done
"$PYTHON" -u -m dynaformer.leger_export --run-dir "$RUNDIR" --leaf "$LEAF"
