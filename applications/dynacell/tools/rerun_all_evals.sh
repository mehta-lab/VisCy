#!/bin/bash
# Submit one sbatch job per (organelle, model, train_set, test_set) tuple to
# rerun evaluations under the new torch-fidelity + linear-probe pipeline.
#
# Each sbatch job runs every per-plate leaf in series for its tuple, passing
# `--regen-metrics` (force_recompute.final_metrics=true) so cached GT masks,
# CP regionprops, and deep features stay intact while metrics + per-cell
# embedding NPZs are rewritten.
#
# Usage:
#   rerun_all_evals.sh --dry-run     # print sbatch lines, do not submit
#   rerun_all_evals.sh --submit      # submit all 79 jobs
#   rerun_all_evals.sh --submit --filter <pattern>
#                                    # only tuples whose composite key matches
#                                    # `<pattern>` (egrep) — e.g. 'joint|a549'
#
# Composite key format: `<organelle>_<model>_<train>_<test>` (lowercase).

set -euo pipefail

MODE=""
FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --submit)  MODE=submit;  shift ;;
    --filter)  FILTER="$2"; shift 2 ;;
    --filter=*) FILTER="${1#--filter=}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "usage: $0 [--dry-run | --submit] [--filter <pattern>]" >&2
  exit 2
fi

VISCY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$VISCY_ROOT"

# Standard tuples — evaluate_batch.sh handles the leaf-glob discovery and
# delegates to submit_evaluation_batch.py. Format:
#   <organelle> <model> <train_short> <test_short>
read -r -d '' STANDARD_TUPLES <<'EOF' || true
er celldiff ipsc a549
er fcmae_vscyto3d_pretrained a549 a549
er fcmae_vscyto3d_pretrained a549 ipsc
er fcmae_vscyto3d_pretrained ipsc a549
er fcmae_vscyto3d_pretrained ipsc ipsc
er fcmae_vscyto3d_pretrained joint a549
er fcmae_vscyto3d_pretrained joint ipsc
er fcmae_vscyto3d_scratch a549 a549
er fcmae_vscyto3d_scratch a549 ipsc
er fcmae_vscyto3d_scratch ipsc a549
er fcmae_vscyto3d_scratch ipsc ipsc
er fcmae_vscyto3d_scratch joint a549
er fcmae_vscyto3d_scratch joint ipsc
er fnet3d_paper a549 a549
er fnet3d_paper a549 ipsc
er fnet3d_paper ipsc a549
er fnet3d_paper ipsc ipsc
er unetvit3d ipsc a549
er unetvit3d ipsc ipsc
membrane celldiff ipsc a549
membrane celldiff joint a549
membrane celldiff joint ipsc
membrane fcmae_vscyto3d_pretrained ipsc a549
membrane fcmae_vscyto3d_pretrained ipsc ipsc
membrane fcmae_vscyto3d_pretrained joint a549
membrane fcmae_vscyto3d_pretrained joint ipsc
membrane fcmae_vscyto3d_scratch a549 a549
membrane fcmae_vscyto3d_scratch a549 ipsc
membrane fcmae_vscyto3d_scratch ipsc a549
membrane fcmae_vscyto3d_scratch ipsc ipsc
membrane fcmae_vscyto3d_scratch joint a549
membrane fcmae_vscyto3d_scratch joint ipsc
membrane fnet3d_paper a549 a549
membrane fnet3d_paper a549 ipsc
membrane fnet3d_paper ipsc a549
membrane fnet3d_paper ipsc ipsc
membrane fnet3d_paper joint a549
membrane fnet3d_paper joint ipsc
membrane unetvit3d ipsc a549
membrane unetvit3d ipsc ipsc
mito celldiff ipsc a549
mito fcmae_vscyto3d_pretrained ipsc a549
mito fcmae_vscyto3d_pretrained ipsc ipsc
mito fcmae_vscyto3d_pretrained joint a549
mito fcmae_vscyto3d_pretrained joint ipsc
mito fcmae_vscyto3d_scratch a549 a549
mito fcmae_vscyto3d_scratch a549 ipsc
mito fcmae_vscyto3d_scratch ipsc a549
mito fcmae_vscyto3d_scratch ipsc ipsc
mito fcmae_vscyto3d_scratch joint a549
mito fcmae_vscyto3d_scratch joint ipsc
mito fnet3d_paper a549 a549
mito fnet3d_paper a549 ipsc
mito fnet3d_paper ipsc a549
mito fnet3d_paper ipsc ipsc
mito unetvit3d ipsc a549
mito unetvit3d ipsc ipsc
nucleus celldiff ipsc a549
nucleus fcmae_vscyto3d_pretrained a549 a549
nucleus fcmae_vscyto3d_pretrained a549 ipsc
nucleus fcmae_vscyto3d_pretrained ipsc a549
nucleus fcmae_vscyto3d_pretrained ipsc ipsc
nucleus fcmae_vscyto3d_scratch a549 a549
nucleus fcmae_vscyto3d_scratch a549 ipsc
nucleus fcmae_vscyto3d_scratch ipsc a549
nucleus fcmae_vscyto3d_scratch ipsc ipsc
nucleus fcmae_vscyto3d_scratch joint a549
nucleus fcmae_vscyto3d_scratch joint ipsc
nucleus fnet3d_paper a549 a549
nucleus fnet3d_paper a549 ipsc
nucleus fnet3d_paper ipsc a549
nucleus fnet3d_paper ipsc ipsc
nucleus fnet3d_paper joint a549
nucleus fnet3d_paper joint ipsc
nucleus unetvit3d ipsc a549
EOF

# CellDiff iPSC predict configs are variant-suffixed
# (`predict__ipsc_confocal__iterative.yml` etc). save_paths collapses all
# three variants to one paper key (`celldiff_iterative`), so we pick the
# canonical `__iterative` leaf per organelle for the rerun.
declare -A CELLDIFF_IPSC_LEAVES=(
  [er]="applications/dynacell/configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/predict__ipsc_confocal__iterative.yml"
  [mito]="applications/dynacell/configs/benchmarks/virtual_staining/mito/celldiff/ipsc_confocal/predict__ipsc_confocal__iterative.yml"
  [nucleus]="applications/dynacell/configs/benchmarks/virtual_staining/nucleus/celldiff/ipsc_confocal/predict__ipsc_confocal__iterative.yml"
  [membrane]="applications/dynacell/configs/benchmarks/virtual_staining/membrane/celldiff/ipsc_confocal/predict__ipsc_confocal__iterative.yml"
)

# Build the dispatch table.
DISPATCH=()
JOBKEYS=()

while IFS= read -r line; do
  [ -z "$line" ] && continue
  set -- $line
  org=$1; model=$2; train=$3; test=$4
  key="${org}_${model}_${train}_${test}"
  cmd="applications/dynacell/tools/evaluate_batch.sh $org $model $train $test --regen-metrics"
  if [ -n "$FILTER" ] && ! echo "$key" | grep -Eq "$FILTER"; then
    continue
  fi
  DISPATCH+=("$cmd")
  JOBKEYS+=("$key")
done <<< "$STANDARD_TUPLES"

# CellDiff iPSC × 4 organelles (variant leaves).
for org in er mito nucleus membrane; do
  leaf="${CELLDIFF_IPSC_LEAVES[$org]}"
  key="${org}_celldiff_ipsc_ipsc"
  cmd="uv run python applications/dynacell/tools/submit_evaluation_batch.py $leaf --regen-metrics"
  if [ -n "$FILTER" ] && ! echo "$key" | grep -Eq "$FILTER"; then
    continue
  fi
  DISPATCH+=("$cmd")
  JOBKEYS+=("$key")
done

echo "[rerun] ${#DISPATCH[@]} sbatch jobs to ${MODE}"
if [ "$MODE" = dry-run ]; then
  for i in "${!DISPATCH[@]}"; do
    printf '  %3d  %-50s  %s\n' "$((i+1))" "${JOBKEYS[$i]}" "${DISPATCH[$i]}"
  done
  exit 0
fi

# Submit (paced — sleep 1s between submissions to keep the SLURM controller calm).
N=${#DISPATCH[@]}
SUCCEEDED=0
FAILED=()
for i in "${!DISPATCH[@]}"; do
  key="${JOBKEYS[$i]}"
  cmd="${DISPATCH[$i]}"
  printf '[%3d/%d] %s\n' "$((i+1))" "$N" "$key"
  if eval "$cmd"; then
    SUCCEEDED=$((SUCCEEDED+1))
  else
    FAILED+=("$key")
  fi
  sleep 1
done

echo "[rerun] submitted=$SUCCEEDED/$N failed=${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[rerun] failures:" >&2
  for f in "${FAILED[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
