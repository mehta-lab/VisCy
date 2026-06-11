#!/bin/bash
# Submit the two-phase focus-aware re-eval campaign (PR #468).
#
#   Phase 1 (gtwarm): 4 *_ipsc_trained leaves -> 4 tasks (1 eval/GPU). Produces
#     every GT-side cache (focus / GLCM-CP / focus-slab GT deep features / masks /
#     in-focus instance plane).
#   Phase 2 (eval): 16 leaves -> 16 tasks (1 eval/GPU). Reads the warm GT caches,
#     so it fans out fully in parallel. Gated on phase 1 with afterany (phase 2 can
#     self-heal any un-warmed GT via compute-at-eval-time, so a single phase-1
#     failure must not block the whole fan-out).
#
# 1 eval/GPU (2/GPU was too slow on 40 GB A100); free cluster -> wide fan-out.
# 14 h job limit, inside the maintenance reservation (gpu-* reserved 06-12 ~09:05).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_focus_campaign.slurm"
mkdir -p /hpc/mydata/alex.kalinin/logs/reeval

J1=$(sbatch --parsable --array=0-3 "$LAUNCHER" gtwarm)
echo "Phase 1 (gtwarm) submitted: array job $J1 (4 tasks = 4 ipsc_trained leaves, 1 eval/GPU)"

J2=$(sbatch --parsable --dependency=afterany:"$J1" --array=0-15%16 "$LAUNCHER" eval)
echo "Phase 2 (eval) submitted:   array job $J2 (16 tasks = 16 leaves, 1 eval/GPU) — afterany:$J1"

echo
echo "Watch:   squeue -j ${J1},${J2} -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'"
echo "Logs:    /hpc/mydata/alex.kalinin/logs/reeval/focus_reeval_*"
