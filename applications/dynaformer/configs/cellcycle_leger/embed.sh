#!/bin/bash
# Plan one encoder's embedding cache of the Leger et al. frames, then queue a GPU job per chunk and the finalize step.
# Run once after `python -m dynaformer.leger_embed index` has written index.csv and labels.npy.
#
# Usage:  ./embed.sh <encoder> <frames per chunk> [sbatch options for the chunks, e.g. --constraint="h200|h100|a100"]
# Prints the finalize job last: queue the encoder's fits after it (./submit.sh --dependency=afterok:<id> -- ...).
set -euo pipefail
cd "$(dirname "$0")"
ENCODER="$1"
FRAMES="$2"
shift 2
DATA=/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data
PYTHON="${DYNAFORMER_PYTHON:-/hpc/mydata/dihan.zheng/venvs/viscy-dynaformer/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Brightfield z-scores span about -6 to 4 (0.1st to 99.95th percentile of the stored frames); the phase window of the
# NIH3T3 caches, (-4, 8), would leave a third of the RGB models' input range unused. MorphEm and CELL-DINO normalize
# their input themselves and ignore it.
n=$("$PYTHON" -m dynaformer.leger_embed plan --out "$DATA/embeddings" --encoder "$ENCODER" \
  --frames-per-chunk "$FRAMES" --rgb-window -6 4 | tail -n 1)
ids=()
for ((k = 0; k < n; k++)); do
  ids+=("$(sbatch --parsable --job-name="ccl_embed_${ENCODER}_$k" "$@" run_embed.slurm "$ENCODER" "$k")")
done
chunks=$(IFS=:; echo "${ids[*]}")
final=$(sbatch --parsable --job-name="ccl_finalize_$ENCODER" --dependency="afterok:$chunks" \
  run_finalize.slurm "$ENCODER")
echo "$ENCODER: chunks ${ids[*]} finalize $final"
