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
# ``eval__a549_mantis`` invoke ``dynacell evaluate-grouped`` (the leaf's
# ``conditions:`` list covers mock+denv+zikv in one process); all others
# invoke ``dynacell evaluate``.
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

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO}"

ml uv 2>/dev/null || true

# Decide single vs grouped from leaf basename.
LEAF_BASENAME=$(basename "${LEAF}")
if [ "${LEAF_BASENAME}" = "eval__a549_mantis" ]; then
  SUBCMD=evaluate-grouped
else
  SUBCMD=evaluate
fi

echo "[run_eval_direct] subcmd=${SUBCMD} leaf=${LEAF} overrides=${EXTRA_OVERRIDES[*]:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run dynacell "${SUBCMD}" "leaf=${LEAF}" "${EXTRA_OVERRIDES[@]}"
