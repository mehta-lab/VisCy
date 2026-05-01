#!/bin/bash
# Run the ipsc_confocal predict leaf locally on the current host's GPU.
#
# Stages the single leaf
#   <organelle>/<model>/ipsc_confocal/predict__ipsc_confocal.yml
# through submit_benchmark_job.py --dry-run (resolved YAML lands under
# launcher.run_root/resolved/) and invokes
# `uv run dynacell predict -c <resolved>` directly — no sbatch.
#
# Usage:
#   predict_local_ipsc.sh <organelle> <model> [--overwrite]
#
# Examples:
#   predict_local_ipsc.sh mito fcmae_vscyto3d_scratch
#   predict_local_ipsc.sh nucleus fcmae_vscyto3d_pretrained --overwrite
#
# Notes:
# - Unlike predict_local_a549.sh there's only one leaf here, so no
#   --parallel option.
# - --overwrite passes through to submit_benchmark_job.py, which sets
#   HCSPredictionWriter.init_args.overwrite=True in the resolved YAML.
# - Log lands at $run_root/slurm/local_<TS>_<organelle>_<model>_ipsc_confocal.log

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <organelle> <model> [--overwrite]" >&2
  exit 2
fi

ORGANELLE=$1
MODEL=$2
shift 2

OVERWRITE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --overwrite) OVERWRITE="--overwrite"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

VISCY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LEAF=$VISCY_ROOT/applications/dynacell/configs/benchmarks/virtual_staining/$ORGANELLE/$MODEL/ipsc_confocal/predict__ipsc_confocal.yml
if [ ! -f "$LEAF" ]; then
  echo "error: leaf does not exist: $LEAF" >&2
  exit 1
fi

cd "$VISCY_ROOT"

# Read launcher.{run_root,job_name} + model.init_args.ckpt_path so we can
# locate the resolved YAML staged below by job_name suffix AND fail fast
# (before the slow compose / Lightning init) if ckpt_path is a placeholder
# or missing on disk.
META=$(uv run python - "$LEAF" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    d = yaml.safe_load(f)
ckpt = d.get("model", {}).get("init_args", {}).get("ckpt_path", "") or ""
print(f"{d['launcher']['job_name']}\t{d['launcher']['run_root']}\t{ckpt}")
PY
)
JOB_NAME=$(echo "$META" | cut -f1)
RUN_ROOT=$(echo "$META" | cut -f2)
CKPT=$(echo "$META" | cut -f3)

if [ -z "$CKPT" ]; then
  echo "error: leaf has no model.init_args.ckpt_path: $LEAF" >&2
  exit 1
fi
if echo "$CKPT" | grep -qE "TODO|FIXME|TBD"; then
  echo "error: ckpt_path is still a placeholder: $CKPT" >&2
  echo "       set a real path in: $LEAF" >&2
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "error: ckpt_path file does not exist on disk: $CKPT" >&2
  echo "       referenced by:                          $LEAF" >&2
  exit 1
fi
echo "[ckpt] $CKPT"

mkdir -p "$RUN_ROOT/slurm"

echo "[stage] composing $ORGANELLE/$MODEL/ipsc_confocal${OVERWRITE:+ + overwrite}"
uv run python applications/dynacell/tools/submit_benchmark_job.py \
  "$LEAF" $OVERWRITE --dry-run >/dev/null

resolved=$(ls -t "$RUN_ROOT/resolved/predict_${JOB_NAME}_"*.yml 2>/dev/null | head -1)
if [ -z "$resolved" ]; then
  echo "error: no resolved yaml found for job_name=$JOB_NAME" >&2
  exit 1
fi

# Pick the runner: unbuffer keeps Lightning's TQDM progress bar visible
# in the log (Python sees a pseudo-tty); fall back to plain uv if missing
# OR if unbuffer is on PATH but its Tcl Expect runtime isn't installed
# (some HPC nodes ship the wrapper without the package, so a PATH check
# alone is insufficient). Probe by running it on `true` and checking the
# exit code.
if command -v unbuffer >/dev/null 2>&1 && unbuffer true 2>/dev/null; then
  RUNNER=(unbuffer uv run dynacell predict)
else
  RUNNER=(uv run dynacell predict)
  echo "[warn] 'unbuffer' unavailable or broken — TQDM progress bar will be hidden in logs"
fi

TS=$(date +%Y%m%d-%H%M%S)
LOG="$RUN_ROOT/slurm/local_${TS}_${ORGANELLE}_${MODEL}_ipsc_confocal.log"

echo "[run] resolved=$(basename "$resolved")"
echo "[run] log=$LOG"
"${RUNNER[@]}" -c "$resolved" >"$LOG" 2>&1
echo "[done] $LOG"
