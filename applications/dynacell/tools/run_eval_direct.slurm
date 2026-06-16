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

# Decide single vs grouped from leaf basename. Grouped leaves carry a
# ``conditions:`` list: ``eval_grouped`` (current convention, incl. the
# instance-AP buckets) and ``eval__a549_mantis`` (legacy ablations).
LEAF_BASENAME=$(basename "${LEAF}")
case "${LEAF_BASENAME}" in
  eval_grouped|eval__a549_mantis) SUBCMD=evaluate-grouped ;;
  *) SUBCMD=evaluate ;;
esac

echo "[run_eval_direct] subcmd=${SUBCMD} leaf=${LEAF} overrides=${EXTRA_OVERRIDES[*]:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run dynacell "${SUBCMD}" "leaf=${LEAF}" "${EXTRA_OVERRIDES[@]}"
