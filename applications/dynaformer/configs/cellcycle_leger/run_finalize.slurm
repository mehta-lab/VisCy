#!/bin/bash
# Check every chunk and every row of an encoder's Leger embedding cache, then write <encoder>.json, without which no
# fit reads the cache. embed.sh queues this after the encoder's chunk jobs.
#
# Usage:  sbatch --dependency=afterok:<chunk jobs> run_finalize.slurm <encoder>
#SBATCH --job-name=ccl_finalize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle_leger/%x-%j.out

set -euo pipefail
export TMPDIR=/tmp
ENCODER="$1"
DATA=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"

echo "host=$(hostname) encoder=$ENCODER job=${SLURM_JOB_ID:-none}"
"$PYTHON" -u -m dynaformer.leger_embed finalize --out "$DATA/embeddings" --encoder "$ENCODER"
