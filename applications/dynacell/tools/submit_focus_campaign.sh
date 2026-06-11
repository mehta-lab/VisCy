#!/bin/bash
# Submit the two-phase focus-aware re-eval campaign (PR #468).
#
#   Phase 1 (gtwarm): 4 *_ipsc_trained leaves -> 2 tasks (2 evals/GPU). Produces
#     every GT-side cache (focus / GLCM-CP / focus-slab GT deep features / masks /
#     in-focus instance plane).
#   Phase 2 (eval): 16 leaves -> 8 tasks (2 evals/GPU). Reads the warm GT caches,
#     so it fans out fully in parallel. Gated on phase 1 with afterany (phase 2 can
#     self-heal any un-warmed GT via compute-at-eval-time, so a single phase-1
#     failure must not block the whole fan-out).
#
# Cluster is free + maintenance in ~12 h -> 12 h job limit, max concurrency.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${HERE}/run_focus_campaign.slurm"
mkdir -p /hpc/mydata/alex.kalinin/logs/reeval

J1=$(sbatch --parsable --array=0-1%2 "$LAUNCHER" gtwarm)
echo "Phase 1 (gtwarm) submitted: array job $J1 (2 tasks, 4 ipsc_trained leaves)"

J2=$(sbatch --parsable --dependency=afterany:"$J1" --array=0-7%8 "$LAUNCHER" eval)
echo "Phase 2 (eval) submitted:   array job $J2 (8 tasks, 16 leaves) — afterany:$J1"

echo
echo "Watch:   squeue -j ${J1},${J2} -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'"
echo "Logs:    /hpc/mydata/alex.kalinin/logs/reeval/focus_reeval_*"
