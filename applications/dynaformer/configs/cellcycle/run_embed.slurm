#!/bin/bash
# Cache a frozen foundation model's embedding of every track point (8 views each) for the cell-cycle models.
#
# Usage:  sbatch run_embed.slurm <morphem | cell-dino | dinov3 | c-radio | vit-untrained>
#SBATCH --job-name=cc_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/logs/cellcycle/%x-%j.out

set -euo pipefail
# Node-local /tmp: multiprocessing cleans up its socket folders badly on the NFS default TMPDIR.
export TMPDIR=/tmp
ENCODER="$1"
DATA=/hpc/projects/virtual_staining/datasets/huang-lab/Bill/5-cellcycle
# The VAST copy of the workspace environment: importing torch from the comp.micro .venv takes minutes.
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"
# Every model is in the local HuggingFace cache; offline mode keeps a slow hub from stalling the job.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) encoder=$ENCODER job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u -m dynaformer.embed --data-dir "$DATA" --encoder "$ENCODER" --out "$DATA/embeddings" --num-workers 8
