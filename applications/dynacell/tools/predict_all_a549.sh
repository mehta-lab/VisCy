#!/bin/bash
# Submit one sbatch job that runs every per-plate A549 predict leaf for a
# given (organelle, model) pair, in series. Path-1 batching: amortizes
# queue submission + GPU allocation; total compute is the same as N
# per-plate jobs.
#
# Usage:
#   predict_all_a549.sh <organelle> <model> [extra args to submit_benchmark_batch.py...]
#
# Examples:
#   predict_all_a549.sh er  fnet3d_paper             # submit
#   predict_all_a549.sh er  fnet3d_paper --dry-run   # render only
#   predict_all_a549.sh mito celldiff --time 10:00:00
#
# Discovers all `predict__a549_mantis_*.yml` leaves under
#   configs/benchmarks/virtual_staining/<organelle>/<model>/ipsc_confocal/
# and passes them to submit_benchmark_batch.py.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <organelle> <model> [submit_benchmark_batch.py args...]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
shift 2

VISCY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LEAF_DIR=$VISCY_ROOT/applications/dynacell/configs/benchmarks/virtual_staining/$ORGANELLE/$MODEL/ipsc_confocal

if [ ! -d "$LEAF_DIR" ]; then
  echo "error: leaf directory does not exist: $LEAF_DIR" >&2
  exit 1
fi

mapfile -t LEAVES < <(ls "$LEAF_DIR"/predict__a549_mantis_*.yml 2>/dev/null | sort)
if [ ${#LEAVES[@]} -eq 0 ]; then
  echo "error: no per-plate predict__a549_mantis_*.yml leaves found in $LEAF_DIR" >&2
  exit 1
fi

JOB_NAME="${MODEL}_PRED_${ORGANELLE}_ON_A549_ALL"

echo "[predict_all_a549] organelle=$ORGANELLE model=$MODEL leaves=${#LEAVES[@]}"
for leaf in "${LEAVES[@]}"; do
  echo "  - $(basename "$leaf")"
done
echo "[predict_all_a549] composite job_name=$JOB_NAME"

cd "$VISCY_ROOT"
exec uv run python applications/dynacell/tools/submit_benchmark_batch.py \
  "${LEAVES[@]}" \
  --job-name "$JOB_NAME" \
  "$@"
