#!/bin/bash
# Direct eval launcher for vscyto3d-ablations leaves that bypass the
# submit_evaluation_job.py / submit_evaluation_batch.py helpers.
#
# Usage:
#   sbatch applications/dynacell/tools/run_eval_direct.slurm <leaf_path_no_ext> [extra_overrides...]
#
# <leaf_path_no_ext>: Hydra-style path relative to the leaf search root,
#                     no file extension. The dynacell CLI resolves it via
#                     ``leaf=<leaf_path_no_ext>``.
#
# Single vs grouped is decided from the leaf basename: leaves named
# ``eval_grouped`` (current grouped convention, incl. the instance-AP
# buckets) or ``eval__a549_mantis`` (legacy vscyto3d-ablations grouped)
# carry a ``conditions:`` list and invoke ``dynacell evaluate-grouped``;
# all others invoke ``dynacell evaluate``.
#
# Extra positional args after the leaf path are passed verbatim as Hydra
# overrides (e.g. ``limit_positions=2 force_recompute.final_metrics=true``).

#SBATCH --job-name=eval_direct
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint="a40|l40s|a100|h100|h200"

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: sbatch $0 <leaf_path_no_ext> [extra_overrides...]" >&2
  exit 2
fi

LEAF="$1"
shift
EXTRA_OVERRIDES=("$@")

# Cap BLAS/OMP threads to the SLURM allocation before any C extension loads.
# applications/dynacell/CLAUDE.md "Eval runtime / parallelism".
export DYNACELL_THREADS_PER_WORKER="${SLURM_CPUS_PER_TASK:-16}"

# SLURM stages this script in a spool dir, so ``BASH_SOURCE[0]`` does not
# point at the source tree. ``SLURM_SUBMIT_DIR`` is the directory where
# ``sbatch`` was invoked — by convention that's the worktree root. Fall
# back to BASH_SOURCE[0] for direct ``bash run_eval_direct.slurm`` use
# outside SLURM (testing).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO="${SLURM_SUBMIT_DIR}"
else
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi
cd "${REPO}"

ml uv

# ``segmentation.backend: cpdino`` is the eval.yaml default, and the cpdino stack
# (cubic + cellpose 4.2.x) does not co-install with the workspace .venv, where
# ``segmentation_cpdino`` dies on
# ``ImportError: cannot import name 'segment_cellpose' from 'cubic.segmentation'``.
# Route uv at the dedicated eval venv and read it read-only (``--no-sync``), the
# same pattern as build_cpdino_seg_cleaned.slurm. Point DYNACELL_EVAL_VENV
# elsewhere to use a different environment.
#
# The default MUST hold the cubic version pyproject declares. The older
# ``cpdino-eval`` (cubic 0.8.0a2) was this default through 2026-07-29 even after
# the 0.9.0a1 venv existed, so jobs that did not export DYNACELL_EVAL_VENV kept
# silently producing 0.8.0a2 FSC/FRC/Spectral_PCC values — 62 of the 121
# surviving eval logs. ``check_cubic_pin`` now fails such a run loudly, but the
# default still has to be right.
export UV_PROJECT_ENVIRONMENT="${DYNACELL_EVAL_VENV:-/hpc/mydata/alex.kalinin/VisCy-venvs/cpdino-eval-cubic090a1}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/hpc/projects/comp.micro/virtual_staining/models/dynacell/evaluation/hf_cache}"
export PYTHONUNBUFFERED=1

# Decide single vs grouped from leaf basename. Grouped leaves carry a
# ``conditions:`` list: ``eval_grouped`` (current convention, incl. the
# instance-AP buckets) and ``eval__a549_mantis`` (legacy ablations).
LEAF_BASENAME=$(basename "${LEAF}")
case "${LEAF_BASENAME}" in
  eval_grouped|eval__a549_mantis) SUBCMD=evaluate-grouped ;;
  *) SUBCMD=evaluate ;;
esac

echo "[run_eval_direct] subcmd=${SUBCMD} leaf=${LEAF} overrides=${EXTRA_OVERRIDES[*]:-<none>}"
echo "[run_eval_direct] venv=${UV_PROJECT_ENVIRONMENT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync dynacell "${SUBCMD}" "leaf=${LEAF}" "${EXTRA_OVERRIDES[@]}"
