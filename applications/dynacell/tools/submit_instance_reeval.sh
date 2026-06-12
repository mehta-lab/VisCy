#!/bin/bash
# Submit the two-phase instance-AP re-eval of the grouped nucleus/membrane
# buckets. Phase 1 warms the GT instance-mask cache (ipsc_trained leaves fold
# every test set, so they produce the full GT cache); phase 2 reads it.
#
# Optional first arg restricts to one organelle; optional second arg selects a
# leaf variant (carved leaves live as grouped/<bucket>_carved/):
#   submit_instance_reeval.sh                   # both nucleus + membrane
#   submit_instance_reeval.sh membrane          # membrane only (no-carve)
#   submit_instance_reeval.sh membrane carved   # membrane carved (subtract_nuclei=true)
#   submit_instance_reeval.sh nucleus           # nucleus only
set -euo pipefail

ORGANELLE=${1:-all}  # all | nucleus | membrane
VARIANT=${2:-}       # "" (no-carve) | carved

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_instance_reeval.slurm"
mkdir -p /hpc/mydata/alex.kalinin/logs/reeval

# Array sizes must match the leaf counts run_instance_reeval.slurm builds per
# (phase, organelle): gtwarm = 1 ipsc_trained leaf/organelle; eval = 2/organelle.
case "$ORGANELLE" in
  all)               GTWARM_ARRAY=0-1; EVAL_ARRAY=0-3 ;;
  nucleus|membrane)  GTWARM_ARRAY=0-0; EVAL_ARRAY=0-1 ;;
  *) echo "Unknown organelle '$ORGANELLE' (expected all|nucleus|membrane)" >&2; exit 2 ;;
esac

J1=$(sbatch --parsable --array="$GTWARM_ARRAY" "$LAUNCHER" gtwarm "$ORGANELLE" "$VARIANT")
echo "Phase 1 (gtwarm: ${ORGANELLE}${VARIANT:+ $VARIANT} ipsc_trained) submitted: array job $J1"

J2=$(sbatch --parsable --dependency="afterany:${J1}" --array="$EVAL_ARRAY" "$LAUNCHER" eval "$ORGANELLE" "$VARIANT")
echo "Phase 2 (eval: ${ORGANELLE}${VARIANT:+ $VARIANT} joint+a549_trained) submitted: array job $J2 (afterany:${J1})"

echo "Watch:   squeue -j ${J1},${J2} -o '%.18i %.9P %.22j %.8T %.10M %.6D %R'"
echo "Logs:    /hpc/mydata/alex.kalinin/logs/reeval/instance_reeval_${J1}_* and _${J2}_*"
