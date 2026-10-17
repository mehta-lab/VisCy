#!/bin/bash
# Embed one planned chunk of the Leger et al. frames with a frozen foundation model (dynaformer.leger_embed chunk).
# embed.sh plans the chunks, submits this once per chunk and queues run_finalize.slurm after them.
#
# Usage:  sbatch run_embed.slurm <encoder> <chunk>
#SBATCH --job-name=ccl_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle_leger/%x-%j.out

set -euo pipefail
# Node-local /tmp: multiprocessing cleans up its socket folders badly on the NFS default TMPDIR.
export TMPDIR=/tmp
ENCODER="$1"
CHUNK="$2"
DATA=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data
# The VAST copy of the workspace environment: importing torch from the comp.micro .venv takes minutes.
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"
# Every model is in the local HuggingFace cache; offline mode keeps a slow hub from stalling the job.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) encoder=$ENCODER chunk=$CHUNK job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u -m dynaformer.leger_embed chunk --data-dir "$DATA" --out "$DATA/embeddings" --encoder "$ENCODER" \
  --chunk "$CHUNK" --num-workers 8
