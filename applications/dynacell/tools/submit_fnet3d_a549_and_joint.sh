#!/bin/bash
# Submit fnet3d_paper training for all four organelles (er, mito, membrane,
# nucleus) on both a549-only and joint (iPSC + a549) train sets.
#
# Each leaf has been audited against the latest fixes:
#   * mmap_preload uses BasicIndexer (commit 6ec0d6f7), not oindex —
#     ~6x lower peak host RAM than the pre-fix path
#   * runtime_shared.yml exports PYTORCH_ALLOC_CONF=expandable_segments:True
#     (commit d97c23b7) so all 8 jobs run with frag-resilient CUDA alloc
#   * joint leaves carry batch_size=6 in the shared _hcs_init_args anchor
#     (commit 7a884b56) so BatchedConcatDataModule lands at the same
#     effective on-GPU batch=48 (6 * num_samples=8) as single-set fnet3d
#   * all 8 leaves request --mem=512G via the launcher.sbatch.mem override
#
# Resolved sbatch: 1 GPU (any), 32 cpus, 512G mem, 20-day wall.
#
# Usage:
#   bash applications/dynacell/tools/submit_fnet3d_a549_and_joint.sh
#
# Pin to repo root regardless of caller cwd.
set -euo pipefail

REPO_ROOT=/hpc/mydata/alex.kalinin/VisCy
cd "$REPO_ROOT"

CFG_BASE="applications/dynacell/configs/benchmarks/virtual_staining"

# 8 leaves: 4 organelles × {a549_mantis, joint_ipsc_confocal_a549_mantis}.
LEAVES=(
  "er/fnet3d_paper/a549_mantis/train.yml"
  "mito/fnet3d_paper/a549_mantis/train.yml"
  "membrane/fnet3d_paper/a549_mantis/train.yml"
  "nucleus/fnet3d_paper/a549_mantis/train.yml"
  "er/fnet3d_paper/joint_ipsc_confocal_a549_mantis/train.yml"
  "mito/fnet3d_paper/joint_ipsc_confocal_a549_mantis/train.yml"
  "membrane/fnet3d_paper/joint_ipsc_confocal_a549_mantis/train.yml"
  "nucleus/fnet3d_paper/joint_ipsc_confocal_a549_mantis/train.yml"
)

echo "Submitting ${#LEAVES[@]} fnet3d jobs..."
echo
for leaf in "${LEAVES[@]}"; do
  cfg="$CFG_BASE/$leaf"
  if [[ ! -f "$cfg" ]]; then
    echo "  ✗ MISSING: $cfg" >&2
    exit 1
  fi
  printf "  %-60s : " "$leaf"
  uv run --no-sync python applications/dynacell/tools/submit_benchmark_job.py "$cfg" 2>&1 \
    | grep -E "Submitted|error" \
    | head -1
done
echo
echo "Done. Use 'squeue -u $USER -o \"%.10i %.32j %.8T %.10M %.20R\"' to monitor."
