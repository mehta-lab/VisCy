#!/usr/bin/env bash
# Handoff script: resubmit the 4 a549-only ER/MITO FCMAE training jobs
# from a higher-fairshare account.
#
# These 4 leaves (er + mito × scratch + pretrained, single-store
# a549_mantis training) initially failed under @alex.kalinin's account
# because the mmap_preload path assumed uniform T per FOV; the a549
# mantis pooled stores mix T=5/7/10 per condition. Fix is on
# dynacell-models @ dcfedfd2 ("refactor(viscy-data): compute mmap T
# offsets once per setup_fit") — that commit (or later) MUST be at
# HEAD or jobs will crash again at prepare_data.
#
# Why a different submitter: alex.kalinin is currently #3 by cluster
# NormUsage, so fairshare is pushing the resubmits ~7-58 hours into
# the future. Submitting from a less-used account gets better
# priority.
#
# Pre-flight (run once per user, not per script invocation):
#
#   1. Clone / pull the repo onto your /hpc/mydata/<user>/ checkout,
#      switch to dynacell-models, and verify the fix is in:
#
#         git checkout dynacell-models && git pull
#         git log --oneline -5 | grep dcfedfd2  # must show the commit
#
#   2. Set up the venv if not already (uv lives behind Lmod on this
#      cluster — load it first):
#
#         ml uv
#         uv venv -p 3.13
#         uv sync --all-packages --all-extras
#
#   3. Wandb credentials (configs use WandbLogger -> czi.wandb.io,
#      project=dynacell, entity=computational_imaging):
#
#         wandb login --relogin   # paste a key with that entity
#
#   4. Confirm group write access on the prod output root:
#
#         touch /hpc/projects/comp.micro/virtual_staining/models/dynacell/a549_mantis/.write_test \
#           && rm /hpc/projects/comp.micro/virtual_staining/models/dynacell/a549_mantis/.write_test
#
#   5. Coordinate with alex.kalinin so HE cancels job IDs 31910346,
#      31910356, 31910360, 31910371 BEFORE this script runs — those
#      target the same run_root and wandb runs would collide.
#
# Usage (from your VisCy repo root):
#
#       bash applications/dynacell/tools/handoff_resubmit_a549_only_fcmae.sh
#
# What it does: composes + sbatches the 4 leaves via the standard
# launcher (submit_benchmark_job.py), echoing each JID and the
# whole-cohort scheduling table at the end.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# uv lives behind Lmod on this cluster; load it explicitly so the script
# works whether or not the user has already done `ml uv` interactively.
ml uv

# Sanity guards before we hand work to sbatch.
if [[ ! -d .venv ]]; then
  echo "error: no .venv at $REPO_ROOT — run \`uv venv -p 3.13 && uv sync --all-packages --all-extras\` first" >&2
  exit 1
fi
if ! git merge-base --is-ancestor dcfedfd2 HEAD 2>/dev/null; then
  echo "error: HEAD does not contain commit dcfedfd2 (heterogeneous-T fix) — \`git checkout dynacell-models && git pull\` first" >&2
  exit 1
fi

CONFIGS=(
  applications/dynacell/configs/benchmarks/virtual_staining/er/fcmae_vscyto3d_scratch/a549_mantis/train.yml
  applications/dynacell/configs/benchmarks/virtual_staining/er/fcmae_vscyto3d_pretrained/a549_mantis/train.yml
  applications/dynacell/configs/benchmarks/virtual_staining/mito/fcmae_vscyto3d_scratch/a549_mantis/train.yml
  applications/dynacell/configs/benchmarks/virtual_staining/mito/fcmae_vscyto3d_pretrained/a549_mantis/train.yml
)

JIDS=()
for c in "${CONFIGS[@]}"; do
  jid=$(uv run python applications/dynacell/tools/submit_benchmark_job.py "$c" --parsable | tail -1)
  echo "submitted $jid  $c"
  JIDS+=("$jid")
done

echo
echo "=== submitted ${#JIDS[@]} jobs — current schedule ==="
squeue -j "$(IFS=,; echo "${JIDS[*]}")" -o "%.10i %.42j %.8T %.6M %.20S %.16R" --sort=S
