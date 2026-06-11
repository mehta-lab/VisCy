#!/bin/bash
# Submit the two-phase instance-AP re-eval of the nucleus & membrane grouped
# buckets (instance-AP unification follow-up). Phase 1 warms the GT instance-mask
# cache for both organelles; phase 2 reads it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_instance_reeval.slurm"
mkdir -p /hpc/mydata/alex.kalinin/logs/reeval

J1=$(sbatch --parsable --array=0-1 "$LAUNCHER" gtwarm)
echo "Phase 1 (gtwarm: nucleus+membrane ipsc_trained) submitted: array job $J1"

J2=$(sbatch --parsable --dependency="afterany:${J1}" --array=0-3 "$LAUNCHER" eval)
echo "Phase 2 (eval: nucleus/membrane joint+a549_trained) submitted: array job $J2 (afterany:${J1})"

echo "Watch:   squeue -j ${J1},${J2} -o '%.18i %.9P %.22j %.8T %.10M %.6D %R'"
echo "Logs:    /hpc/mydata/alex.kalinin/logs/reeval/instance_reeval_${J1}_* and _${J2}_*"
