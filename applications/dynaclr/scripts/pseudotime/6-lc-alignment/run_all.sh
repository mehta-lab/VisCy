#!/usr/bin/env bash
# Two-stage LC-alignment pipeline for one model.
#
# Stage 1 — register_by_lc.py: train viral_sensor LC, write
#           <dataset>_lc_registration.parquet per cell with t_LC_star.
# Stage 2 — evaluate_registered_organelle.py: shift Phase3D embeddings by
#           -t_LC_star, score LC_unshifted vs LC_registered.
#
# Usage:
#   bash applications/dynaclr/scripts/pseudotime/6-lc-alignment/run_all.sh
#   MODEL=<other> bash .../run_all.sh
#
# Adding more datasets later: extend the recipes
#   configs/recipes/datasets_zikv_register.yml
#   configs/recipes/datasets_zikv_register_organelle_phase3d.yml
# and (if needed) add a new organelle channel leaf following the
# zikv_register_organelle_phase3d.yml pattern.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNACLR_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE="$(cd "$DYNACLR_DIR/../.." && pwd)"

PYBIN="${PYBIN:-$WORKSPACE/.venv/bin/python}"
MODEL="${MODEL:-DynaCLR-2D-MIP-BagOfChannels-single-marker-fix-shuffler}"

if [[ ! -x "$PYBIN" ]]; then
  echo "ERROR: python not found at $PYBIN" >&2
  exit 1
fi

cd "$DYNACLR_DIR"

CONFIG_DIR="scripts/pseudotime/6-lc-alignment/configs/${MODEL}"
OUT_DIR="scripts/pseudotime/6-lc-alignment/out/${MODEL}"

# Stage 1
echo "=== Stage 1: register_by_lc → ${OUT_DIR}/zikv_register ==="
"$PYBIN" scripts/pseudotime/6-lc-alignment/register_by_lc.py \
  --config     "${CONFIG_DIR}/zikv_register.yml" \
  --output-dir "${OUT_DIR}/zikv_register"

# Stage 2 — Phase3D classification with LC-derived registered HPI
echo "=== Stage 2: evaluate_registered_organelle (Phase3D) → ${OUT_DIR}/zikv_register_organelle_phase3d ==="
"$PYBIN" scripts/pseudotime/6-lc-alignment/evaluate_registered_organelle.py \
  --config     "${CONFIG_DIR}/zikv_register_organelle_phase3d.yml" \
  --output-dir "${OUT_DIR}/zikv_register_organelle_phase3d"

# Stage B — per-cell organelle event timing
for organelle in sec61 g3bp1; do
  echo "=== Stage B: measure_event_timing (${organelle}) → ${OUT_DIR}/zikv_timing_${organelle} ==="
  "$PYBIN" scripts/pseudotime/6-lc-alignment/measure_event_timing.py \
    --config     "${CONFIG_DIR}/zikv_timing_${organelle}.yml" \
    --output-dir "${OUT_DIR}/zikv_timing_${organelle}"
done

echo
echo "=== Done. Outputs: ==="
echo "  ${OUT_DIR}/zikv_register/registration_summary.csv"
echo "  ${OUT_DIR}/zikv_register_organelle_phase3d/report.md"
echo "  ${OUT_DIR}/zikv_timing_sec61/timing_summary.csv"
echo "  ${OUT_DIR}/zikv_timing_g3bp1/timing_summary.csv"
