#!/bin/bash
# Submit the focus-aware re-eval of the vscyto3d-ablations track (PR #468 follow-up),
# gated on the main focus campaign so the shared GT caches are fully warm first.
#
#   32 leaves -> 32 array tasks, 1 eval/GPU. Each reads the warm per-dataset GT
#   cache (focus planes / GLCM-CP / focus-tagged GT deep features / masks / in-focus
#   GT instance plane) and computes only its PRED side, so they fan out fully in
#   parallel. afterany on the campaign jobs guarantees GT is built before we read it.
#
# Override the gating job IDs by passing them as args, e.g.
#   tools/submit_ablations_focus.sh 33946313 33946314 33948203
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_ablations_focus.slurm"
mkdir -p /hpc/mydata/alex.kalinin/logs/reeval

# Main focus-campaign jobs to wait on (gtwarm + eval + nucleus resubmit).
DEPS=("$@")
if [[ ${#DEPS[@]} -eq 0 ]]; then
  DEPS=(33946313 33946314 33948203)
fi
DEP_SPEC="afterany:$(IFS=:; echo "${DEPS[*]}")"

J=$(sbatch --parsable --dependency="$DEP_SPEC" --array=0-31 "$LAUNCHER")
echo "Ablations focus re-eval submitted: array job $J (32 tasks, 1 eval/GPU) — $DEP_SPEC"
echo "Watch:   squeue -j ${J} -o '%.18i %.9P %.22j %.8T %.10M %.6D %R'"
echo "Logs:    /hpc/mydata/alex.kalinin/logs/reeval/ablations_focus_${J}_*"
