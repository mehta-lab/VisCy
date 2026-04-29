#!/bin/bash
# Run a sequence of A549 predict leaves locally on the current host's GPU.
#
# Stages each leaf through submit_benchmark_job.py --dry-run (so the
# resolved YAML lands under launcher.run_root/resolved/ with a
# timestamped filename for provenance), then invokes
# `uv run dynacell predict -c <resolved>` directly — no sbatch.
#
# Usage:
#   predict_local_a549.sh <organelle> <model> [--overwrite] [--parallel N]
#
# Examples:
#   predict_local_a549.sh er  fnet3d_paper                  # serial
#   predict_local_a549.sh er  fnet3d_paper --parallel 2     # 2 plates concurrent on one GPU
#   predict_local_a549.sh er  fnet3d_paper --overwrite      # force-replace existing channels
#   predict_local_a549.sh mito celldiff --overwrite --parallel 2
#
# Notes:
# - Discovers all `predict__a549_mantis_*.yml` leaves under
#     configs/benchmarks/virtual_staining/<organelle>/<model>/ipsc_confocal/
# - --overwrite passes through to submit_benchmark_job.py, which sets
#   HCSPredictionWriter.init_args.overwrite=True in the resolved YAML.
# - --parallel N runs N predicts concurrently on the same GPU, waiting
#   between batches of N. Tune to fit VRAM (~2 fnet predicts fit on an A40).
# - Logs land at $run_root/slurm/local_<TS>_<organelle>_<model>_<plate>.log
# - Fail-fast: a failing plate aborts the script. Re-run remaining plates
#   manually if needed.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <organelle> <model> [--overwrite] [--parallel N]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
shift 2

OVERWRITE=""
PARALLEL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --overwrite)   OVERWRITE="--overwrite"; shift ;;
    --parallel)    PARALLEL="$2"; shift 2 ;;
    --parallel=*)  PARALLEL="${1#--parallel=}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --parallel must be a positive integer (got '$PARALLEL')" >&2
  exit 2
fi

VISCY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LEAF_DIR=$VISCY_ROOT/applications/dynacell/configs/benchmarks/virtual_staining/$ORGANELLE/$MODEL/ipsc_confocal
if [ ! -d "$LEAF_DIR" ]; then
  echo "error: leaf directory does not exist: $LEAF_DIR" >&2
  exit 1
fi

mapfile -t LEAVES < <(ls "$LEAF_DIR"/predict__a549_mantis_*.yml 2>/dev/null | sort)
if [ ${#LEAVES[@]} -eq 0 ]; then
  echo "error: no per-plate predict__a549_mantis_*.yml leaves in $LEAF_DIR" >&2
  exit 1
fi

cd "$VISCY_ROOT"

# Extract launcher.{run_root,job_name} from each leaf in one python call
# so we can locate the resolved YAML staged below by job_name suffix.
META=$(uv run python - "${LEAVES[@]}" <<'PY'
import sys, yaml
for path in sys.argv[1:]:
    with open(path) as f:
        d = yaml.safe_load(f)
    print(f"{path}\t{d['launcher']['job_name']}\t{d['launcher']['run_root']}")
PY
)

RUN_ROOT=$(echo "$META" | head -1 | cut -f3)
mkdir -p "$RUN_ROOT/slurm"

# Stage every resolved YAML up-front (so resolved/ is populated before any
# predict starts). One pass means we tolerate timestamp collisions across
# leaves cleanly: each call writes a distinct {mode}_{job_name}_{TS}.yml.
echo "[stage] composing ${#LEAVES[@]} leaves ($ORGANELLE/$MODEL)${OVERWRITE:+ + overwrite}"
for leaf in "${LEAVES[@]}"; do
  uv run python applications/dynacell/tools/submit_benchmark_job.py \
    "$leaf" $OVERWRITE --dry-run >/dev/null
done

# Pick the runner: unbuffer keeps Lightning's TQDM progress bar visible
# in the log (Python sees a pseudo-tty); fall back to plain uv if missing.
if command -v unbuffer >/dev/null 2>&1; then
  RUNNER=(unbuffer uv run dynacell predict)
else
  RUNNER=(uv run dynacell predict)
  echo "[warn] 'unbuffer' not found — TQDM progress bar will be hidden in logs"
fi

# Trap to kill any backgrounded predicts if we exit early (Ctrl+C, error).
trap 'kill 0 2>/dev/null || true' EXIT

# Single timestamp for the whole batch so all logs share a clock label.
TS=$(date +%Y%m%d-%H%M%S)

echo "[run] parallel=$PARALLEL leaves=${#LEAVES[@]} run_root=$RUN_ROOT"
i=0
while IFS=$'\t' read -r leaf job_name _run_root; do
  resolved=$(ls -t "$RUN_ROOT/resolved/predict_${job_name}_"*.yml 2>/dev/null | head -1)
  if [ -z "$resolved" ]; then
    echo "error: no resolved yaml found for job_name=$job_name" >&2
    exit 1
  fi
  plate=$(basename "$leaf" .yml | sed 's/^predict__a549_mantis_//')
  log="$RUN_ROOT/slurm/local_${TS}_${ORGANELLE}_${MODEL}_${plate}.log"

  echo "  [start] plate=$plate log=$(basename "$log")"
  "${RUNNER[@]}" -c "$resolved" >"$log" 2>&1 &

  # Use pure-assignment increment, NOT ((i++)). With set -e, the
  # arithmetic expression i++ returns its pre-increment value (0 on
  # the first iteration), which bash treats as exit status 1 → the
  # EXIT trap fires kill 0 → script + just-started child both die.
  i=$((i + 1))
  if [ $((i % PARALLEL)) -eq 0 ]; then
    wait
  fi
done <<< "$META"

wait
trap - EXIT
echo "[done] all leaves complete; logs: $RUN_ROOT/slurm/local_${TS}_${ORGANELLE}_${MODEL}_*.log"
