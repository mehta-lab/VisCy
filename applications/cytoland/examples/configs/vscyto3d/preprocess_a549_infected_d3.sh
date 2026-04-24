#!/bin/bash
# One-shot preprocess for the A549 Mantis infection dataset (D3).
#
# Writes per-FOV and per-timepoint normalization statistics into the
# zarr's .zattrs for the three channels actually used by the VSCyto3D
# finetune (Phase3D source + raw mCherry / raw Cy5 targets). D1 and D2
# already have normalization_metadata and do not need preprocessing.
#
# Usage:
#   bash applications/cytoland/examples/configs/vscyto3d/preprocess_a549_infected_d3.sh
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
mkdir -p .tmp/preprocess_logs

uv run viscy preprocess \
  --data_path /hpc/projects/intracellular_dashboard/organelle_dynamics/2026_03_26_A549_CAAX_H2B_DENV_ZIKV/2-assemble/2026_03_26_A549_CAAX_H2B_DENV_ZIKV.zarr \
  --channel_names+ "Phase3D" \
  --channel_names+ "raw mCherry EX561 EM600-37" \
  --channel_names+ "raw Cy5 EX639 EM698-70" \
  --num_workers 16 \
  2>&1 | tee .tmp/preprocess_logs/d3_preprocess.log
