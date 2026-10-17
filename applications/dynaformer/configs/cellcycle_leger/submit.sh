#!/bin/bash
# Queue the fit of each Leger experiment leaf and, after it succeeds, its evaluation.
#
# Usage:  ./submit.sh [sbatch options for the fit, e.g. --time=1:00:00 --dependency=afterok:123] -- leaf.yml [leaf.yml ...]
set -euo pipefail
cd "$(dirname "$0")"
FIT_OPTS=()
while [ "$1" != "--" ]; do FIT_OPTS+=("$1"); shift; done
shift
for leaf in "$@"; do
  name="${leaf%.yml}"
  fit=$(sbatch --parsable --job-name="ccl_fit_$name" "${FIT_OPTS[@]}" run_fit.slurm "$leaf")
  eval=$(sbatch --parsable --job-name="ccl_eval_$name" --dependency="afterok:$fit" run_eval.slurm "$leaf")
  echo "$leaf: fit $fit, eval $eval"
done
